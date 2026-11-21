// Prevents an additional console window on Windows in release builds
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::net::{SocketAddr, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};
use tauri::{
    menu::{Menu, MenuItem},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    Manager, State, WindowEvent,
};

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;

const LOCAL_ENGINE_ADDR: &str = "127.0.0.1:8000";
// PyInstaller --onefile bundles PyTorch/Transformers (~200 MB) and can take 30-45s on cold Windows Defender unpack
const STARTUP_TIMEOUT: Duration = Duration::from_secs(75);

/// Thread-safe container holding the background Python FastAPI child process.
pub struct PythonCoreState {
    pub child: Mutex<Option<Child>>,
}

impl Default for PythonCoreState {
    fn default() -> Self {
        Self {
            child: Mutex::new(None),
        }
    }
}

/// Resolves the `core-engine` directory whether running from `src-tauri/`, `target/release/`, or workspace root.
fn resolve_core_engine_dir() -> Result<PathBuf, String> {
    if let Ok(env_dir) = std::env::var("SHARDAI_CORE_DIR") {
        let p = PathBuf::from(env_dir);
        if p.join("main.py").exists() {
            return p.canonicalize().map_err(|e| e.to_string());
        }
    }

    let mut candidates = vec![
        PathBuf::from("../core-engine"),
        PathBuf::from("core-engine"),
        PathBuf::from("../../core-engine"),
        PathBuf::from("../../../core-engine"),
        PathBuf::from("../../../../core-engine"),
        PathBuf::from("d:\\ShardAi-Workspace\\shardai-node\\core-engine"),
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

/// Resolves the preferred Python interpreter (local `.venv` first, then system Python).
fn resolve_python_executable(engine_dir: &PathBuf) -> PathBuf {
    #[cfg(target_os = "windows")]
    let venv_candidates = [
        engine_dir.join(".venv").join("Scripts").join("python.exe"),
        engine_dir.join("..").join(".venv").join("Scripts").join("python.exe"),
        PathBuf::from("d:\\ShardAi-Workspace\\shardai-node\\core-engine\\.venv\\Scripts\\python.exe"),
    ];

    #[cfg(not(target_os = "windows"))]
    let venv_candidates = [
        engine_dir.join(".venv").join("bin").join("python"),
        engine_dir.join("..").join(".venv").join("bin").join("python"),
    ];

    for candidate in venv_candidates {
        if candidate.exists() {
            return candidate;
        }
    }

    #[cfg(target_os = "windows")]
    {
        PathBuf::from("python")
    }
    #[cfg(not(target_os = "windows"))]
    {
        PathBuf::from("python3")
    }
}

/// Terminates the background Python process and reaps its exit status to prevent zombie processes.
fn terminate_child_process(state: &PythonCoreState) -> Result<String, String> {
    let mut guard = state
        .child
        .lock()
        .map_err(|_| "Failed to acquire lock on PythonCoreState".to_string())?;

    if let Some(mut child) = guard.take() {
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

        let _ = child.kill();
        let _ = child.wait();
        Ok(format!("Python core process (PID {pid}) stopped cleanly."))
    } else {
        Ok("Python core process is not running.".to_string())
    }
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
    ];

    candidates.into_iter().find(|path| path.exists())
}

/// Checks whether `127.0.0.1:8000` is currently accepting HTTP connections.
fn is_engine_listening() -> bool {
    LOCAL_ENGINE_ADDR
        .parse::<SocketAddr>()
        .map(|addr| TcpStream::connect_timeout(&addr, Duration::from_millis(150)).is_ok())
        .unwrap_or(false)
}

/// Ensures the sidecar (or dev Python core) process is spawned without blocking on HTTP readiness.
fn ensure_core_spawned(state: &PythonCoreState) -> Result<String, String> {
    if is_engine_listening() {
        return Ok("Python core is already listening.".to_string());
    }

    let mut guard = state
        .child
        .lock()
        .map_err(|_| "Failed to acquire lock on PythonCoreState".to_string())?;

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

    let (mut cmd, launch_label) = if let Ok(engine_dir) = resolve_core_engine_dir() {
        let python_bin = resolve_python_executable(&engine_dir);
        let mut c = Command::new(&python_bin);
        c.current_dir(&engine_dir)
            .args(["main.py", "--mode", "daemon", "--port", "8000"])
            .env("SHARDAI_ENGINE_PORT", "8000")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        (c, format!("{:?}", python_bin))
    } else if let Some(sidecar_bin) = resolve_sidecar_executable() {
        let mut c = Command::new(&sidecar_bin);
        c.args(["--mode", "daemon", "--port", "8000"])
            .env("SHARDAI_ENGINE_PORT", "8000")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        (c, format!("{:?}", sidecar_bin))
    } else {
        return Err("Neither Python environment nor sidecar binary could be located.".to_string());
    };

    #[cfg(target_os = "windows")]
    {
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    let child = cmd
        .spawn()
        .map_err(|e| format!("Failed to spawn Python core ({}): {e}", launch_label))?;
    let pid = child.id();
    *guard = Some(child);
    Ok(format!("Spawned Python core ({launch_label}, PID {pid})."))
}

/// Spawns the bundled PyInstaller sidecar (release) or `core-engine/main.py` (dev) and waits for `127.0.0.1:8000`.
#[tauri::command]
async fn start_python_core(app_handle: tauri::AppHandle) -> Result<String, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app_handle.state::<PythonCoreState>();
        ensure_core_spawned(&state)?;

        let start_instant = Instant::now();
        while start_instant.elapsed() < STARTUP_TIMEOUT {
            if is_engine_listening() {
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

            thread::sleep(Duration::from_millis(250));
        }

        Err(format!(
            "Timed out waiting for Python core to listen on http://{LOCAL_ENGINE_ADDR}."
        ))
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
    let _ = terminate_child_process(&state);
    app_handle.exit(0);
    Ok(())
}

#[tauri::command]
fn window_start_drag(window: tauri::Window) -> Result<(), String> {
    window.start_dragging().map_err(|e| e.to_string())
}

fn main() {
    tauri::Builder::default()
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
                        let _ = terminate_child_process(&state);
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
        .run(tauri::generate_context!())
        .expect("error while running ShardAi Tauri application");
}

#[cfg(test)]
mod tests {
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
    fn dialog_plugin_initializes() {
        let _plugin: tauri::plugin::TauriPlugin<tauri::Wry> = tauri_plugin_dialog::init();
    }
}
