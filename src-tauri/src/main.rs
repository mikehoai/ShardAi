// Prevents an additional console window on Windows in release builds
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};
use tauri::{
    menu::{Menu, MenuItem},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    Manager, RunEvent, State, WindowEvent,
};

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

#[cfg(unix)]
use std::os::unix::process::CommandExt as UnixCommandExt;

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;

#[cfg(unix)]
const SIGKILL: i32 = 9;

#[cfg(unix)]
extern "C" {
    fn kill(pid: i32, sig: i32) -> i32;
}

const LOCAL_ENGINE_ADDR: &str = "127.0.0.1:8000";
// PyInstaller --onefile bundles PyTorch/Transformers (~200 MB) and can take 30-45s on cold Windows Defender unpack
const STARTUP_TIMEOUT: Duration = Duration::from_secs(75);

/// Thread-safe container holding the background Python FastAPI child process and shutdown state.
pub struct PythonCoreState {
    pub child: Mutex<Option<Child>>,
    pub shutting_down: AtomicBool,
}

impl Default for PythonCoreState {
    fn default() -> Self {
        Self {
            child: Mutex::new(None),
            shutting_down: AtomicBool::new(false),
        }
    }
}

impl PythonCoreState {
    /// Marks the desktop shell as shutting down so background boot threads refuse to spawn.
    pub fn mark_shutting_down(&self) {
        self.shutting_down.store(true, Ordering::SeqCst);
    }

    /// Returns true once application shutdown has been initiated.
    pub fn is_shutting_down(&self) -> bool {
        self.shutting_down.load(Ordering::SeqCst)
    }

    /// Takes and terminates any tracked child process under the mutex, recovering from poison on panic.
    fn terminate_child_inner(&self) -> Option<u32> {
        let mut guard = self
            .child
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.take().map(kill_and_reap_child)
    }
}

impl Drop for PythonCoreState {
    fn drop(&mut self) {
        self.mark_shutting_down();
        let _ = self.terminate_child_inner();
    }
}

/// Resolves the `core-engine` directory whether running from `src-tauri/`, `target/release/`, or workspace root.
fn resolve_core_engine_dir() -> Result<PathBuf, String> {
    if let Ok(env_dir) = std::env::var("SHARDAI_CORE_DIR") {
        let trimmed = env_dir.trim();
        if !trimmed.is_empty() {
            let p = PathBuf::from(trimmed);
            if p.join("main.py").exists() {
                return p.canonicalize().map_err(|e| e.to_string());
            }
        }
    }

    let mut candidates = vec![
        PathBuf::from("../core-engine"),
        PathBuf::from("core-engine"),
        PathBuf::from("../../core-engine"),
        PathBuf::from("../../../core-engine"),
        PathBuf::from("../../../../core-engine"),
    ];

    if let Ok(exe_path) = std::env::current_exe() {
        let mut cur = exe_path.parent();
        while let Some(dir) = cur {
            candidates.push(dir.join("core-engine"));
            candidates.push(dir.join("shardai-node").join("core-engine"));
            candidates.push(dir.join("..").join("core-engine"));
            cur = dir.parent();
        }
    }

    if let Ok(cwd) = std::env::current_dir() {
        let mut cur = Some(cwd.as_path());
        while let Some(dir) = cur {
            candidates.push(dir.join("core-engine"));
            candidates.push(dir.join("shardai-node").join("core-engine"));
            candidates.push(dir.join("..").join("core-engine"));
            cur = dir.parent();
        }
    }

    for candidate in candidates {
        if candidate.join("main.py").exists() {
            return candidate
                .canonicalize()
                .map_err(|e| format!("Failed to canonicalize core-engine path: {e}"));
        }
    }
    Err("Could not locate 'core-engine/main.py'. Ensure the directory structure is intact.".to_string())
}

/// Resolves the preferred Python interpreter (local `.venv` first, then system Python) and reports whether a venv was matched.
fn resolve_python_executable(engine_dir: &Path) -> (PathBuf, bool) {
    #[cfg(target_os = "windows")]
    let venv_candidates = [
        engine_dir.join(".venv").join("Scripts").join("python.exe"),
        engine_dir.join("..").join(".venv").join("Scripts").join("python.exe"),
    ];

    #[cfg(not(target_os = "windows"))]
    let venv_candidates = [
        engine_dir.join(".venv").join("bin").join("python"),
        engine_dir.join("..").join(".venv").join("bin").join("python"),
    ];

    for candidate in venv_candidates {
        if candidate.exists() {
            return (candidate, true);
        }
    }

    #[cfg(target_os = "windows")]
    {
        (PathBuf::from("python"), false)
    }
    #[cfg(not(target_os = "windows"))]
    {
        (PathBuf::from("python3"), false)
    }
}

/// Configures background command stdio, Windows console suppression, and Unix process group isolation.
fn configure_background_command(cmd: &mut Command) {
    cmd.stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    #[cfg(target_os = "windows")]
    {
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    #[cfg(unix)]
    {
        // Spawn in a dedicated process group (PGID = child PID) so PyInstaller --onefile
        // bootloaders and their forked Python servers can be killed together via -pgid.
        cmd.process_group(0);
    }
}

/// Forcefully kills a child process (and its process tree/group) and reaps its exit status.
fn kill_and_reap_child(mut child: Child) -> u32 {
    let pid = child.id();

    #[cfg(target_os = "windows")]
    {
        // Kill the entire process tree on Windows so Uvicorn child workers do not become orphans
        let _ = Command::new("taskkill")
            .args(["/F", "/T", "/PID", &pid.to_string()])
            .creation_flags(CREATE_NO_WINDOW)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }

    #[cfg(unix)]
    {
        // Kill the entire Unix process group (-pgid) so forked child processes are terminated
        if let Ok(pgid) = i32::try_from(pid) {
            if pgid > 0 {
                unsafe {
                    let _ = kill(-pgid, SIGKILL);
                }
            }
        }
    }

    let _ = child.kill();
    let _ = child.wait();
    pid
}

/// Terminates the background Python process and reaps its exit status to prevent zombie processes.
fn terminate_child_process(state: &PythonCoreState) -> Result<String, String> {
    match state.terminate_child_inner() {
        Some(pid) => Ok(format!("Python core process (PID {pid}) stopped cleanly.")),
        None => Ok("Python core process is not running.".to_string()),
    }
}

/// Marks the state as shutting down and terminates any tracked child process tree.
fn shutdown_python_core(state: &PythonCoreState) -> Result<String, String> {
    state.mark_shutting_down();
    terminate_child_process(state)
}

/// Resolves a packaged PyInstaller Tauri Sidecar binary placed next to the desktop executable.
fn resolve_sidecar_executable() -> Option<PathBuf> {
    let exe_dir = std::env::current_exe().ok()?.parent()?.to_path_buf();

    #[cfg(target_os = "windows")]
    let candidates = [
        exe_dir.join("core-engine.exe"),
        exe_dir.join("binaries").join("core-engine-x86_64-pc-windows-msvc.exe"),
    ];

    #[cfg(not(target_os = "windows"))]
    let candidates = [
        exe_dir.join("core-engine"),
        exe_dir.join("binaries").join("core-engine"),
        exe_dir.join("binaries").join("core-engine-x86_64-unknown-linux-gnu"),
        exe_dir.join("binaries").join("core-engine-aarch64-apple-darwin"),
        exe_dir.join("binaries").join("core-engine-x86_64-apple-darwin"),
    ];

    candidates.into_iter().find(|path| path.exists())
}

#[derive(Debug, PartialEq, Eq)]
enum CoreLaunchTarget {
    Sidecar(PathBuf),
    Python {
        engine_dir: PathBuf,
        python_bin: PathBuf,
        is_venv: bool,
    },
}

/// Returns true when Python execution should take precedence over a bundled sidecar (debug builds or explicit env overrides).
fn should_prefer_python(
    is_debug: bool,
    env_core_dir: Option<&str>,
    env_use_python: Option<&str>,
) -> bool {
    if is_debug {
        return true;
    }

    if env_core_dir.is_some_and(|val| !val.trim().is_empty()) {
        return true;
    }

    env_use_python.is_some_and(|val| {
        matches!(
            val.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes"
        )
    })
}

/// Chooses between bundled sidecar and Python core engine based on build mode and explicit overrides.
fn select_core_launch_target(
    prefer_python: bool,
    sidecar_bin: Option<PathBuf>,
    engine_dir: Result<PathBuf, String>,
) -> Result<CoreLaunchTarget, String> {
    if prefer_python {
        if let Ok(dir) = engine_dir {
            let (python_bin, is_venv) = resolve_python_executable(&dir);
            return Ok(CoreLaunchTarget::Python {
                engine_dir: dir,
                python_bin,
                is_venv,
            });
        }
        if let Some(sidecar) = sidecar_bin {
            return Ok(CoreLaunchTarget::Sidecar(sidecar));
        }
    } else {
        if let Some(sidecar) = sidecar_bin {
            return Ok(CoreLaunchTarget::Sidecar(sidecar));
        }
        if let Ok(dir) = engine_dir {
            let (python_bin, is_venv) = resolve_python_executable(&dir);
            return Ok(CoreLaunchTarget::Python {
                engine_dir: dir,
                python_bin,
                is_venv,
            });
        }
    }

    Err("Neither bundled sidecar binary nor Python core-engine runtime could be located.".to_string())
}

/// Verifies that a system Python fallback can actually import core runtime dependencies before spawning daemon mode.
fn verify_system_python_runtime(python_bin: &Path, engine_dir: &Path) -> Result<(), String> {
    let mut probe = Command::new(python_bin);
    probe
        .current_dir(engine_dir)
        .args(["-c", "import fastapi, uvicorn"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    #[cfg(target_os = "windows")]
    {
        probe.creation_flags(CREATE_NO_WINDOW);
    }

    match probe.status() {
        Ok(status) if status.success() => Ok(()),
        Ok(status) => Err(format!(
            "Bundled sidecar and local '.venv' were not found, and system Python ({:?}) exited with {status} while checking required packages (fastapi, uvicorn).",
            python_bin
        )),
        Err(err) => Err(format!(
            "Bundled sidecar and local '.venv' were not found, and system Python ({:?}) could not be executed: {err}",
            python_bin
        )),
    }
}

/// Checks whether `127.0.0.1:8000` is currently accepting HTTP connections.
fn is_engine_listening() -> bool {
    LOCAL_ENGINE_ADDR
        .parse::<SocketAddr>()
        .map(|addr| TcpStream::connect_timeout(&addr, Duration::from_millis(150)).is_ok())
        .unwrap_or(false)
}

/// Checks shutdown state under the held child mutex right before and after spawning.
fn spawn_and_register_child(
    state: &PythonCoreState,
    slot: &mut Option<Child>,
    mut cmd: Command,
    launch_label: &str,
) -> Result<String, String> {
    if state.is_shutting_down() {
        return Err("Application is shutting down; aborting Python core spawn.".to_string());
    }

    let child = cmd
        .spawn()
        .map_err(|e| format!("Failed to spawn Python core ({}): {e}", launch_label))?;

    if state.is_shutting_down() {
        let pid = kill_and_reap_child(child);
        return Err(format!(
            "Application began shutting down during spawn; terminated Python core (PID {pid})."
        ));
    }

    let pid = child.id();
    *slot = Some(child);
    Ok(format!("Spawned Python core ({launch_label}, PID {pid})."))
}

/// Ensures the sidecar (or dev Python core) process is spawned without blocking on HTTP readiness.
fn ensure_core_spawned(state: &PythonCoreState) -> Result<String, String> {
    if state.is_shutting_down() {
        return Err("Application is shutting down; refusing to start Python core.".to_string());
    }

    if is_engine_listening() {
        return Ok("Python core is already listening.".to_string());
    }

    let mut guard = state
        .child
        .lock()
        .map_err(|_| "Failed to acquire lock on PythonCoreState".to_string())?;

    if state.is_shutting_down() {
        return Err("Application is shutting down; refusing to spawn Python core.".to_string());
    }

    if let Some(existing_child) = guard.as_mut() {
        match existing_child.try_wait() {
            Ok(None) => {
                return Ok("Python core process is starting up.".to_string());
            }
            _ => {
                *guard = None;
            }
        }
    }

    let env_core_dir = std::env::var("SHARDAI_CORE_DIR").ok();
    let env_use_python = std::env::var("SHARDAI_USE_PYTHON").ok();
    let prefer_python = should_prefer_python(
        cfg!(debug_assertions),
        env_core_dir.as_deref(),
        env_use_python.as_deref(),
    );

    let target = select_core_launch_target(
        prefer_python,
        resolve_sidecar_executable(),
        resolve_core_engine_dir(),
    )?;

    let (mut cmd, launch_label) = match target {
        CoreLaunchTarget::Sidecar(sidecar_bin) => (Command::new(&sidecar_bin), format!("{:?}", sidecar_bin)),
        CoreLaunchTarget::Python {
            engine_dir,
            python_bin,
            is_venv,
        } => {
            if !is_venv {
                verify_system_python_runtime(&python_bin, &engine_dir)?;
            }
            let mut c = Command::new(&python_bin);
            c.current_dir(&engine_dir).arg("main.py");
            (c, format!("{:?}", python_bin))
        }
    };

    cmd.args(["--mode", "daemon", "--port", "8000"])
        .env("SHARDAI_ENGINE_PORT", "8000");
    configure_background_command(&mut cmd);

    spawn_and_register_child(state, &mut guard, cmd, &launch_label)
}

/// Polls until the local engine starts listening, exits prematurely, shuts down, or times out.
/// On timeout, terminates and reaps the hung child without setting `shutting_down` so retries succeed.
fn wait_for_engine_or_terminate<F>(
    state: &PythonCoreState,
    timeout: Duration,
    poll_interval: Duration,
    mut check_listening: F,
) -> Result<String, String>
where
    F: FnMut() -> bool,
{
    let start_instant = Instant::now();
    loop {
        if state.is_shutting_down() {
            let _ = terminate_child_process(state);
            return Err("Application is shutting down before Python core became ready.".to_string());
        }

        if check_listening() {
            return Ok(format!("Python core is ready on http://{LOCAL_ENGINE_ADDR}."));
        }

        // Re-check process health without holding the Mutex during sleep
        {
            let mut guard = state
                .child
                .lock()
                .map_err(|_| "Failed to acquire lock on PythonCoreState".to_string())?;
            if let Some(child) = guard.as_mut() {
                if let Ok(Some(exit_status)) = child.try_wait() {
                    *guard = None;
                    return Err(format!(
                        "Python core exited prematurely with status: {exit_status}"
                    ));
                }
            }
        }

        let elapsed = start_instant.elapsed();
        if elapsed >= timeout {
            break;
        }

        let remaining = timeout.saturating_sub(elapsed);
        thread::sleep(poll_interval.min(remaining));
    }

    let _ = terminate_child_process(state);
    Err(format!(
        "Timed out waiting for Python core to listen on http://{LOCAL_ENGINE_ADDR}."
    ))
}

/// Spawns the bundled PyInstaller sidecar (release) or `core-engine/main.py` (dev) and waits for `127.0.0.1:8000`.
#[tauri::command]
async fn start_python_core(app_handle: tauri::AppHandle) -> Result<String, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app_handle.state::<PythonCoreState>();
        ensure_core_spawned(&state)?;
        wait_for_engine_or_terminate(
            &state,
            STARTUP_TIMEOUT,
            Duration::from_millis(250),
            is_engine_listening,
        )
    })
    .await
    .map_err(|e| format!("Failed joining core startup task: {e}"))?
}

/// Kills the background Python core process on user request or application shutdown.
#[tauri::command]
fn stop_python_core(state: State<'_, PythonCoreState>) -> Result<String, String> {
    terminate_child_process(&state)
}

#[cfg(target_os = "windows")]
#[repr(C)]
struct MemoryStatusEx {
    dw_length: u32,
    dw_memory_load: u32,
    ull_total_phys: u64,
    ull_avail_phys: u64,
    ull_total_page_file: u64,
    ull_avail_page_file: u64,
    ull_total_virtual: u64,
    ull_avail_virtual: u64,
    ull_avail_extended_virtual: u64,
}

#[cfg(target_os = "windows")]
extern "system" {
    fn GlobalMemoryStatusEx(lp_buffer: *mut MemoryStatusEx) -> i32;
}

/// Returns the host OS's currently free physical RAM in gigabytes (GB).
#[tauri::command]
fn get_available_ram_gb() -> Result<f64, String> {
    #[cfg(target_os = "windows")]
    {
        let mut status = MemoryStatusEx {
            dw_length: std::mem::size_of::<MemoryStatusEx>() as u32,
            dw_memory_load: 0,
            ull_total_phys: 0,
            ull_avail_phys: 0,
            ull_total_page_file: 0,
            ull_avail_page_file: 0,
            ull_total_virtual: 0,
            ull_avail_virtual: 0,
            ull_avail_extended_virtual: 0,
        };
        let ok = unsafe { GlobalMemoryStatusEx(&mut status as *mut MemoryStatusEx) };
        if ok == 0 {
            return Err("GlobalMemoryStatusEx failed to read Windows physical memory.".to_string());
        }
        let gb = (status.ull_avail_phys as f64) / (1024.0 * 1024.0 * 1024.0);
        return Ok((gb * 100.0).round() / 100.0);
    }

    #[cfg(target_os = "linux")]
    {
        let content = std::fs::read_to_string("/proc/meminfo")
            .map_err(|e| format!("Failed to read /proc/meminfo: {e}"))?;
        for line in content.lines() {
            if let Some(rest) = line.strip_prefix("MemAvailable:") {
                let kb_str = rest.split_whitespace().next().unwrap_or("0");
                let kb: f64 = kb_str
                    .parse()
                    .map_err(|e| format!("Failed to parse MemAvailable: {e}"))?;
                let gb = (kb * 1024.0) / (1024.0 * 1024.0 * 1024.0);
                return Ok((gb * 100.0).round() / 100.0);
            }
        }
        return Err("MemAvailable not found in /proc/meminfo.".to_string());
    }

    #[cfg(target_os = "macos")]
    {
        let output = Command::new("vm_stat")
            .output()
            .map_err(|e| format!("Failed to execute vm_stat: {e}"))?;
        let text = String::from_utf8_lossy(&output.stdout);
        let mut free_pages: u64 = 0;
        for line in text.lines() {
            if line.contains("Pages free:") || line.contains("Pages inactive:") {
                if let Some(val_part) = line.split(':').nth(1) {
                    let num_str = val_part.trim().trim_end_matches('.');
                    free_pages += num_str.parse::<u64>().unwrap_or(0);
                }
            }
        }
        let avail_bytes = (free_pages as f64) * 4096.0;
        let gb = avail_bytes / (1024.0 * 1024.0 * 1024.0);
        return Ok((gb * 100.0).round() / 100.0);
    }
}

#[tauri::command]
fn window_minimize(window: tauri::Window) -> Result<(), String> {
    window.minimize().map_err(|e| e.to_string())
}

#[tauri::command]
fn window_toggle_maximize(window: tauri::Window) -> Result<bool, String> {
    let is_max = window.is_maximized().map_err(|e| e.to_string())?;
    if is_max {
        window.unmaximize().map_err(|e| e.to_string())?;
        Ok(false)
    } else {
        window.maximize().map_err(|e| e.to_string())?;
        Ok(true)
    }
}

#[tauri::command]
fn window_close(app_handle: tauri::AppHandle) -> Result<(), String> {
    let state = app_handle.state::<PythonCoreState>();
    let _ = shutdown_python_core(&state);
    app_handle.exit(0);
    Ok(())
}

#[tauri::command]
fn window_start_drag(window: tauri::Window) -> Result<(), String> {
    window.start_dragging().map_err(|e| e.to_string())
}

fn main() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(PythonCoreState::default())
        .setup(|app| {
            // 1. Auto-boot core-engine sidecar in the background as soon as the desktop app opens
            let app_handle = app.handle().clone();
            std::thread::spawn(move || {
                let state = app_handle.state::<PythonCoreState>();
                let _ = ensure_core_spawned(&state);
            });

            // 2. Native System Tray icon & menu (Windows taskbar tray / macOS menu bar)
            let show_item = MenuItem::with_id(app, "tray_show", "Show ShardAi", true, None::<&str>)?;
            let quit_item = MenuItem::with_id(app, "tray_quit", "Quit ShardAi", true, None::<&str>)?;
            let tray_menu = Menu::with_items(app, &[&show_item, &quit_item])?;

            let mut tray_builder = TrayIconBuilder::new()
                .menu(&tray_menu)
                .tooltip("ShardAi — P2P AI Gateway")
                .on_menu_event(|app, event| match event.id.as_ref() {
                    "tray_show" => {
                        if let Some(win) = app.get_webview_window("main") {
                            let _ = win.show();
                            let _ = win.unminimize();
                            let _ = win.set_focus();
                        }
                    }
                    "tray_quit" => {
                        let state = app.state::<PythonCoreState>();
                        let _ = shutdown_python_core(&state);
                        app.exit(0);
                    }
                    _ => {}
                })
                .on_tray_icon_event(|tray, event| {
                    if let TrayIconEvent::Click {
                        button: MouseButton::Left,
                        button_state: MouseButtonState::Up,
                        ..
                    } = event
                    {
                        let app = tray.app_handle();
                        if let Some(win) = app.get_webview_window("main") {
                            let _ = win.show();
                            let _ = win.unminimize();
                            let _ = win.set_focus();
                        }
                    }
                });

            if let Some(icon) = app.default_window_icon().cloned() {
                tray_builder = tray_builder.icon(icon);
            }
            tray_builder.build(app)?;

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            start_python_core,
            stop_python_core,
            get_available_ram_gb,
            window_minimize,
            window_toggle_maximize,
            window_close,
            window_start_drag
        ])
        .on_window_event(|window, event| {
            if matches!(
                event,
                WindowEvent::CloseRequested { .. } | WindowEvent::Destroyed
            ) {
                let state = window.app_handle().state::<PythonCoreState>();
                let _ = terminate_child_process(&state);
            }
        })
        .build(tauri::generate_context!())
        .expect("error while building ShardAi Tauri application");

    app.run(|app_handle, event| {
        if let RunEvent::Exit = event {
            let state = app_handle.state::<PythonCoreState>();
            let _ = shutdown_python_core(&state);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_hardcoded_workstation_paths_in_source() {
        let source = include_str!("main.rs");
        let forbidden_workspace = ["ShardAi-", "Workspace"].concat();
        let forbidden_drive_path = ["d:\\\\", "ShardAi"].concat();
        assert!(
            !source.contains(&forbidden_workspace),
            "main.rs must not contain hardcoded workstation folder names"
        );
        assert!(
            !source.contains(&forbidden_drive_path),
            "main.rs must not contain hardcoded local drive paths"
        );
    }

    #[test]
    fn default_capabilities_include_dialog_permission() {
        let caps_json = include_str!("../capabilities/default.json");
        let parsed: serde_json::Value =
            serde_json::from_str(caps_json).expect("capabilities/default.json must be valid JSON");
        let perms = parsed["permissions"]
            .as_array()
            .expect("permissions must be a JSON array");
        assert!(
            perms.iter().any(|item| item.as_str() == Some("dialog:default")),
            "capabilities/default.json must include dialog:default"
        );
    }

    #[test]
    fn test_release_prioritizes_sidecar_over_python() {
        assert!(!should_prefer_python(false, None, None));
        assert!(!should_prefer_python(false, Some("   "), Some("0")));

        let sidecar = PathBuf::from("/opt/shardai/core-engine");
        let engine_dir = PathBuf::from("/opt/shardai/src/core-engine");

        let selected = select_core_launch_target(
            should_prefer_python(false, None, None),
            Some(sidecar.clone()),
            Ok(engine_dir),
        )
        .expect("expected launch target in release mode");

        assert_eq!(selected, CoreLaunchTarget::Sidecar(sidecar));
    }

    #[test]
    fn test_debug_and_env_overrides_prioritize_python() {
        assert!(should_prefer_python(true, None, None));
        assert!(should_prefer_python(false, Some("/custom/core-engine"), None));
        assert!(should_prefer_python(false, None, Some("1")));
        assert!(should_prefer_python(false, None, Some("TRUE")));
        assert!(should_prefer_python(false, None, Some("yes")));

        let temp_root = std::env::temp_dir().join(format!(
            "shardai_tauri_priority_test_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&temp_root);
        std::fs::create_dir_all(&temp_root).expect("create temp engine dir");

        let sidecar = PathBuf::from("/opt/shardai/core-engine");
        let selected = select_core_launch_target(
            should_prefer_python(false, None, Some("1")),
            Some(sidecar),
            Ok(temp_root.clone()),
        )
        .expect("expected python target when SHARDAI_USE_PYTHON=1");

        match selected {
            CoreLaunchTarget::Python { engine_dir, .. } => {
                assert_eq!(engine_dir, temp_root);
            }
            other => panic!("expected Python target, got {other:?}"),
        }

        let _ = std::fs::remove_dir_all(&temp_root);
    }

    #[test]
    fn test_resolve_python_executable_prefers_local_venv_over_system_fallback() {
        let base_dir = std::env::temp_dir().join(format!(
            "shardai_tauri_venv_test_{}",
            std::process::id()
        ));
        let engine_dir = base_dir.join("core-engine");
        let _ = std::fs::remove_dir_all(&base_dir);
        std::fs::create_dir_all(&engine_dir).expect("create temp core-engine dir");

        let (fallback_bin, fallback_is_venv) = resolve_python_executable(&engine_dir);
        assert!(!fallback_is_venv);
        #[cfg(target_os = "windows")]
        assert_eq!(fallback_bin, PathBuf::from("python"));
        #[cfg(not(target_os = "windows"))]
        assert_eq!(fallback_bin, PathBuf::from("python3"));

        #[cfg(target_os = "windows")]
        let venv_python = engine_dir.join(".venv").join("Scripts").join("python.exe");
        #[cfg(not(target_os = "windows"))]
        let venv_python = engine_dir.join(".venv").join("bin").join("python");

        std::fs::create_dir_all(venv_python.parent().expect("venv parent"))
            .expect("create venv parent");
        std::fs::write(&venv_python, b"").expect("write dummy venv python");

        let (resolved_bin, is_venv) = resolve_python_executable(&engine_dir);
        assert!(is_venv);
        assert_eq!(resolved_bin, venv_python);

        let _ = std::fs::remove_dir_all(&base_dir);
    }

    #[test]
    fn test_select_core_launch_target_returns_error_when_both_missing() {
        let res = select_core_launch_target(false, None, Err("missing".to_string()));
        assert!(res.is_err());
    }

    #[test]
    fn dialog_plugin_initializes() {
        let _plugin: tauri::plugin::TauriPlugin<tauri::Wry> = tauri_plugin_dialog::init();
    }

    fn spawn_dummy_long_running_child() -> Child {
        #[cfg(target_os = "windows")]
        let mut cmd = {
            let mut c = Command::new("ping");
            c.args(["-n", "30", "127.0.0.1"]);
            c
        };
        #[cfg(not(target_os = "windows"))]
        let mut cmd = {
            let mut c = Command::new("sleep");
            c.arg("30");
            c
        };

        configure_background_command(&mut cmd);
        cmd.spawn().expect("spawn dummy long-running child process")
    }

    fn spawn_dummy_failing_child() -> Child {
        #[cfg(target_os = "windows")]
        let mut cmd = {
            let mut c = Command::new("cmd");
            c.args(["/C", "exit", "1"]);
            c
        };
        #[cfg(not(target_os = "windows"))]
        let mut cmd = {
            let mut c = Command::new("sh");
            c.args(["-c", "exit 1"]);
            c
        };

        configure_background_command(&mut cmd);
        cmd.spawn().expect("spawn dummy failing child process")
    }

    fn is_process_alive(pid: u32) -> bool {
        #[cfg(target_os = "windows")]
        {
            let mut cmd = Command::new("tasklist");
            cmd.args(["/FI", &format!("PID eq {pid}"), "/FO", "CSV", "/NH"])
                .creation_flags(CREATE_NO_WINDOW);
            let output = cmd.output().expect("run tasklist to inspect PID");
            let stdout = String::from_utf8_lossy(&output.stdout);
            stdout.contains(&format!("\"{pid}\""))
        }
        #[cfg(unix)]
        {
            match i32::try_from(pid) {
                Ok(pgid) if pgid > 0 => unsafe { kill(pgid, 0) == 0 },
                _ => false,
            }
        }
    }

    #[test]
    fn test_ensure_core_spawned_aborts_immediately_when_shutting_down() {
        let state = PythonCoreState::default();
        assert!(!state.is_shutting_down());

        state.mark_shutting_down();
        assert!(state.is_shutting_down());

        let err = ensure_core_spawned(&state).expect_err("expected shutdown rejection");
        assert!(
            err.contains("shutting down"),
            "expected shutdown message, got: {err}"
        );
        assert!(
            state.child.lock().expect("lock child mutex").is_none(),
            "child must not be spawned during shutdown"
        );

        let mut slot: Option<Child> = None;
        let cmd = Command::new("nonexistent-binary-must-not-spawn");
        let spawn_err = spawn_and_register_child(&state, &mut slot, cmd, "dummy")
            .expect_err("expected pre-spawn mutex check to abort");
        assert!(
            spawn_err.contains("shutting down"),
            "expected pre-spawn shutdown message, got: {spawn_err}"
        );
        assert!(slot.is_none());
    }

    #[test]
    fn test_drop_python_core_state_kills_and_reaps_child() {
        let state = PythonCoreState::default();
        let mut child = spawn_dummy_long_running_child();
        let pid = child.id();

        assert!(
            matches!(child.try_wait(), Ok(None)),
            "dummy child must be running before Drop"
        );
        {
            let mut guard = state.child.lock().expect("lock child mutex");
            *guard = Some(child);
        }
        assert!(
            is_process_alive(pid),
            "dummy child PID {pid} should be alive before Drop"
        );

        drop(state);

        assert!(
            !is_process_alive(pid),
            "dummy child PID {pid} must be terminated and reaped when PythonCoreState drops"
        );
    }

    #[test]
    fn test_drop_python_core_state_recovers_from_poisoned_mutex() {
        let state = std::sync::Arc::new(PythonCoreState::default());
        let state_clone = std::sync::Arc::clone(&state);
        let (tx, rx) = std::sync::mpsc::channel();

        let handle = thread::spawn(move || {
            let mut guard = state_clone.child.lock().expect("lock child mutex");
            let child = spawn_dummy_long_running_child();
            let pid = child.id();
            *guard = Some(child);
            tx.send(pid).expect("send child PID");
            panic!("simulated panic while holding PythonCoreState mutex");
        });

        let pid = rx.recv().expect("receive spawned child PID");
        assert!(handle.join().is_err(), "panicking thread should fail join");
        assert!(
            state.child.is_poisoned(),
            "child mutex should be poisoned after panic"
        );
        assert!(
            is_process_alive(pid),
            "dummy child PID {pid} should still be alive prior to state Drop"
        );

        drop(state);

        assert!(
            !is_process_alive(pid),
            "dummy child PID {pid} must be killed and reaped on Drop even after mutex poison"
        );
    }

    #[test]
    fn test_startup_timeout_terminates_and_reaps_hung_child_so_retries_succeed() {
        let state = PythonCoreState::default();
        let mut hung_child = spawn_dummy_long_running_child();
        let hung_pid = hung_child.id();

        assert!(
            matches!(hung_child.try_wait(), Ok(None)),
            "hung dummy child must be running"
        );
        {
            let mut guard = state.child.lock().expect("lock child mutex");
            *guard = Some(hung_child);
        }
        assert!(is_process_alive(hung_pid));

        let timeout_err = wait_for_engine_or_terminate(
            &state,
            Duration::from_millis(60),
            Duration::from_millis(15),
            || false,
        )
        .expect_err("expected startup timeout error");

        assert!(
            timeout_err.contains("Timed out"),
            "unexpected timeout error: {timeout_err}"
        );
        assert!(
            !state.is_shutting_down(),
            "startup timeout must keep shutting_down=false so retries are allowed"
        );
        assert!(
            state.child.lock().expect("lock child mutex").is_none(),
            "hung child handle must be cleared on startup timeout"
        );
        assert!(
            !is_process_alive(hung_pid),
            "hung child PID {hung_pid} must be killed and reaped on startup timeout"
        );

        // Verify a subsequent retry is not blocked by the previous hung child
        let retry_child = spawn_dummy_long_running_child();
        let retry_pid = retry_child.id();
        {
            let mut guard = state.child.lock().expect("lock child mutex");
            *guard = Some(retry_child);
        }
        let ready_msg = wait_for_engine_or_terminate(
            &state,
            Duration::from_millis(200),
            Duration::from_millis(10),
            || true,
        )
        .expect("retry after startup timeout should succeed when listening");
        assert!(ready_msg.contains("ready"));

        let _ = terminate_child_process(&state);
        assert!(!is_process_alive(retry_pid));
    }

    #[test]
    fn test_terminate_child_process_preserves_restart_ability_unlike_shutdown() {
        let state = PythonCoreState::default();
        let child = spawn_dummy_long_running_child();
        let pid = child.id();
        {
            let mut guard = state.child.lock().expect("lock child mutex");
            *guard = Some(child);
        }

        let stop_msg = terminate_child_process(&state).expect("stop_python_core cleanup succeeds");
        assert!(stop_msg.contains("stopped cleanly"));
        assert!(
            !state.is_shutting_down(),
            "stop_python_core must not set shutting_down=true"
        );
        assert!(!is_process_alive(pid));

        let shutdown_msg = shutdown_python_core(&state).expect("shutdown helper succeeds");
        assert!(shutdown_msg.contains("not running"));
        assert!(state.is_shutting_down());
    }

    #[test]
    fn test_wait_for_engine_aborts_on_premature_exit_and_shutdown() {
        let state = PythonCoreState::default();
        let failing_child = spawn_dummy_failing_child();
        {
            let mut guard = state.child.lock().expect("lock child mutex");
            *guard = Some(failing_child);
        }
        let exit_err = wait_for_engine_or_terminate(
            &state,
            Duration::from_secs(5),
            Duration::from_millis(10),
            || false,
        )
        .expect_err("expected premature exit error");
        assert!(
            exit_err.contains("exited prematurely"),
            "unexpected premature exit error: {exit_err}"
        );
        assert!(state.child.lock().expect("lock child mutex").is_none());

        let child = spawn_dummy_long_running_child();
        let pid = child.id();
        {
            let mut guard = state.child.lock().expect("lock child mutex");
            *guard = Some(child);
        }
        state.mark_shutting_down();
        let started = Instant::now();
        let shutdown_err = wait_for_engine_or_terminate(
            &state,
            Duration::from_secs(30),
            Duration::from_millis(10),
            || true,
        )
        .expect_err("expected early abort on shutdown");
        assert!(shutdown_err.contains("shutting down"));
        assert!(started.elapsed() < Duration::from_secs(5));
        assert!(!is_process_alive(pid));
    }
}
