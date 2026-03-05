use tauri::Manager;
use std::net::TcpStream;
use std::process::{Command, Child, Stdio};
use std::sync::Mutex;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use std::thread;

struct BackendProcess(Mutex<Option<Child>>);

impl Drop for BackendProcess {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.0.lock() {
            if let Some(ref mut child) = *guard {
                let _ = child.kill();
                let _ = child.wait();
                println!("Backend process terminated.");
            }
        }
    }
}

/// Check if backend is already running on the given port.
fn is_port_in_use(port: u16) -> bool {
    TcpStream::connect_timeout(
        &format!("127.0.0.1:{}", port).parse().unwrap(),
        Duration::from_millis(200),
    ).is_ok()
}

/// Wait for backend to become responsive (up to timeout).
fn wait_for_backend(port: u16, timeout: Duration) -> bool {
    let start = Instant::now();
    while start.elapsed() < timeout {
        if is_port_in_use(port) {
            return true;
        }
        thread::sleep(Duration::from_millis(300));
    }
    false
}

/// Find the project root directory (contains pyproject.toml).
fn find_project_root() -> Option<PathBuf> {
    let candidates: Vec<Option<PathBuf>> = vec![
        // current working directory
        std::env::current_dir().ok(),
        // exe dir -> go up multiple levels (dev: target/release/ or target/debug/)
        std::env::current_exe().ok().and_then(|p| p.parent().map(|p| p.to_path_buf())),
        std::env::current_exe().ok().and_then(|p| p.parent()?.parent().map(|p| p.to_path_buf())),
        std::env::current_exe().ok().and_then(|p| p.parent()?.parent()?.parent().map(|p| p.to_path_buf())),
        std::env::current_exe().ok().and_then(|p| p.parent()?.parent()?.parent()?.parent().map(|p| p.to_path_buf())),
        std::env::current_exe().ok().and_then(|p| p.parent()?.parent()?.parent()?.parent()?.parent().map(|p| p.to_path_buf())),
    ];

    for candidate in candidates.into_iter().flatten() {
        if candidate.join("pyproject.toml").exists() {
            println!("Project root found: {}", candidate.display());
            return Some(candidate);
        }
    }

    None
}

fn start_backend() -> Option<Child> {
    let port: u16 = 8000;

    // If backend is already running, don't start another one
    if is_port_in_use(port) {
        println!("Backend already running on port {}", port);
        return None;
    }

    let project_root = match find_project_root() {
        Some(root) => root,
        None => {
            eprintln!("Could not find project root (no pyproject.toml found).");
            eprintln!("Please start backend manually: python -m uvicorn src.api.main:app --port 8000");
            return None;
        }
    };

    // Start backend with stdout/stderr piped (no flashing console window)
    let child = Command::new("python")
        .args(["-m", "uvicorn", "src.api.main:app", "--host", "127.0.0.1", "--port", "8000"])
        .current_dir(&project_root)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .ok();

    match &child {
        Some(_) => {
            println!("Starting backend from: {}", project_root.display());
            // Wait up to 10 seconds for backend to become responsive
            if wait_for_backend(port, Duration::from_secs(10)) {
                println!("Backend ready on http://127.0.0.1:{}", port);
            } else {
                eprintln!("Backend process started but not responding yet. It may still be loading.");
            }
        }
        None => {
            eprintln!("Failed to spawn backend process. Is Python installed?");
            eprintln!("  Start manually: cd {} && python -m uvicorn src.api.main:app --port 8000", project_root.display());
        }
    }

    child
}

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! RegenTwin is ready.", name)
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            let backend = start_backend();
            app.manage(BackendProcess(Mutex::new(backend)));
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![greet])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
