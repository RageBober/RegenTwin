use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};
use tauri::Manager;

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

fn is_port_in_use(port: u16) -> bool {
    TcpStream::connect_timeout(
        &format!("127.0.0.1:{}", port).parse().unwrap(),
        Duration::from_millis(200),
    )
    .is_ok()
}

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

fn find_project_root() -> Option<PathBuf> {
    let candidates: Vec<Option<PathBuf>> = vec![
        std::env::current_dir().ok(),
        std::env::current_exe().ok().and_then(|p| p.parent().map(|p| p.to_path_buf())),
        std::env::current_exe().ok().and_then(|p| p.parent()?.parent().map(|p| p.to_path_buf())),
        std::env::current_exe().ok().and_then(|p| p.parent()?.parent()?.parent().map(|p| p.to_path_buf())),
        std::env::current_exe().ok().and_then(|p| p.parent()?.parent()?.parent()?.parent().map(|p| p.to_path_buf())),
    ];

    for candidate in candidates.into_iter().flatten() {
        if candidate.join("pyproject.toml").exists() {
            println!("Project root found: {}", candidate.display());
            return Some(candidate);
        }
    }
    None
}

fn local_python_candidates(root: &Path) -> Vec<PathBuf> {
    vec![
        // Bundled venv (production .msi: tauri.conf.json bundle.resources)
        root.join("venv").join("Scripts").join("python.exe"),
        root.join("venv").join("bin").join("python"),
        // Dev / source .venv
        root.join(".venv").join("Scripts").join("python.exe"),
        root.join(".venv").join("bin").join("python"),
    ]
}

/// Bundled resources directory (Tauri 2): рядом с .exe в установленной MSI это
/// `<install>/resources/`. Для dev-режима возвращает None.
fn bundled_resource_root() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let parent = exe.parent()?.to_path_buf();
    if parent.join("resources").exists() {
        Some(parent.join("resources"))
    } else {
        None
    }
}

fn spawn_uv_backend(project_root: &Path) -> Option<Child> {
    Command::new("uv")
        .args([
            "run",
            "uvicorn",
            "src.api.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ])
        .current_dir(project_root)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .ok()
}

fn spawn_python_backend(project_root: &Path, python: &Path) -> Option<Child> {
    Command::new(python)
        .args(["-m", "uvicorn", "src.api.main:app", "--host", "127.0.0.1", "--port", "8000"])
        .current_dir(project_root)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .ok()
}

fn start_backend() -> Option<Child> {
    let port: u16 = 8000;

    if is_port_in_use(port) {
        println!("Backend already running on port {}", port);
        return None;
    }

    let project_root = match find_project_root() {
        Some(root) => root,
        None => {
            eprintln!("Could not find project root (no pyproject.toml found).");
            eprintln!("Please start backend manually: uv run uvicorn src.api.main:app --port 8000");
            return None;
        }
    };

    // Production-first: попытаться запустить из bundled resources/.venv (без uv).
    let child = bundled_resource_root()
        .and_then(|res_root| {
            for python in local_python_candidates(&res_root) {
                if python.exists() {
                    if let Some(child) = spawn_python_backend(&project_root, &python) {
                        println!("Backend started from bundled venv: {}", python.display());
                        return Some(child);
                    }
                }
            }
            None
        })
        // Dev fallback: пробуем uv (если установлен в PATH).
        .or_else(|| spawn_uv_backend(&project_root))
        .or_else(|| {
            for python in local_python_candidates(&project_root) {
                if python.exists() {
                    if let Some(child) = spawn_python_backend(&project_root, &python) {
                        return Some(child);
                    }
                }
            }
            None
        })
        .or_else(|| spawn_python_backend(&project_root, Path::new("python")));

    match &child {
        Some(_) => {
            println!("Starting backend from: {}", project_root.display());
            if wait_for_backend(port, Duration::from_secs(10)) {
                println!("Backend ready on http://127.0.0.1:{}", port);
            } else {
                eprintln!("Backend process started but not responding yet. It may still be loading.");
            }
        }
        None => {
            eprintln!("Failed to spawn backend process. Tried: uv, project-local .venv, python.");
            eprintln!(
                "  Start manually: cd {} && uv run uvicorn src.api.main:app --port 8000",
                project_root.display()
            );
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
