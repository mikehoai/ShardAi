// Prevents an additional console window on Windows in release builds
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::net::{SocketAddr, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};
use tauri::{Manager, State, WindowEvent};

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;

const LOCAL_ENGINE_ADDR: &str = "127.0.0.1:8000";
const STARTUP_TIMEOUT: Duration = Duration::from_secs(10);

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

/// Resolves the `core-engine` directory whether running from `src-tauri/` or the workspace root.
fn resolve_core_engine_dir() -> Result<PathBuf, String> {
    let candidates = [PathBuf::from("../core-engine"), PathBuf::from("core-engine")];
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

/// Spawns `core-engine/main.py` in the background and waits until `127.0.0.1:8000` is reachable.
#[tauri::command]
fn start_python_core(state: State<'_, PythonCoreState>) -> Result<String, String> {
    let mut guard = state
        .child
        .lock()
        .map_err(|_| "Failed to acquire lock on PythonCoreState".to_string())?;

    // Check if an existing process is still alive
    if let Some(existing_child) = guard.as_mut() {
        match existing_child.try_wait() {
            Ok(None) => {
                return Ok("Python core is already running.".to_string());
            }
            _ => {
                *guard = None;
            }
        }
    }

    let engine_dir = resolve_core_engine_dir()?;
    let python_bin = resolve_python_executable(&engine_dir);

    let mut cmd = Command::new(&python_bin);
    cmd.current_dir(&engine_dir)
        .arg("main.py")
        .env("SHARDAI_ENGINE_PORT", "8000")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    #[cfg(target_os = "windows")]
    {
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("Failed to spawn Python core ({:?}): {e}", python_bin))?;

    let pid = child.id();
    let socket_addr: SocketAddr = LOCAL_ENGINE_ADDR
        .parse()
        .map_err(|e| format!("Invalid local engine socket address: {e}"))?;

    // Poll until FastAPI binds to 127.0.0.1:8000 or the process exits prematurely
    let start_instant = Instant::now();
    while start_instant.elapsed() < STARTUP_TIMEOUT {
        if TcpStream::connect_timeout(&socket_addr, Duration::from_millis(150)).is_ok() {
            *guard = Some(child);
            return Ok(format!(
                "Python core started on http://{LOCAL_ENGINE_ADDR} (PID {pid})."
            ));
        }

        if let Ok(Some(exit_status)) = child.try_wait() {
            return Err(format!(
                "Python core exited prematurely with status: {exit_status}"
            ));
        }

        thread::sleep(Duration::from_millis(150));
    }

    // Startup timed out; ensure the hung process is killed and reaped
    let _ = child.kill();
    let _ = child.wait();
    Err(format!(
        "Timed out waiting for Python core to listen on http://{LOCAL_ENGINE_ADDR}."
    ))
}

/// Kills the background Python core process on user request or application shutdown.
#[tauri::command]
fn stop_python_core(state: State<'_, PythonCoreState>) -> Result<String, String> {
    terminate_child_process(&state)
}

fn main() {
    tauri::Builder::default()
        .manage(PythonCoreState::default())
        .invoke_handler(tauri::generate_handler![
            start_python_core,
            stop_python_core
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
