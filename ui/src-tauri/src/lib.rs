use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tauri::Manager;

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

// CreateProcess flags. Without these, spawning python.exe (a console-subsystem
// app) from a windowless Tauri parent makes Windows allocate a brand-new console
// for it. That console window is what the user saw on launch — closing it sent
// CTRL_CLOSE_EVENT to python.exe and killed the backend.
#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;
#[cfg(target_os = "windows")]
const CREATE_NEW_PROCESS_GROUP: u32 = 0x00000200;

type SharedChild = Arc<Mutex<Option<Child>>>;

struct BackendProcess(SharedChild);

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
        // Production: portable Python runtime bundled in MSI (tauri.conf.json
        // bundle.resources → python-runtime/). Real python.exe, not a venv shim.
        root.join("python-runtime").join("python.exe"),
        root.join("python-runtime").join("bin").join("python"),
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

#[cfg(target_os = "windows")]
fn apply_windows_spawn_flags(cmd: &mut Command) {
    cmd.creation_flags(CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP);
}

#[cfg(not(target_os = "windows"))]
fn apply_windows_spawn_flags(_cmd: &mut Command) {}

fn spawn_uv_backend(project_root: &Path) -> Option<Child> {
    let mut cmd = Command::new("uv");
    cmd.args([
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
    .stderr(Stdio::piped());
    apply_windows_spawn_flags(&mut cmd);
    cmd.spawn().ok()
}

fn backend_log_paths(project_root: &Path) -> (PathBuf, PathBuf) {
    let log_dir = project_root.join("logs");
    let _ = std::fs::create_dir_all(&log_dir);
    (
        log_dir.join("backend-stdout.log"),
        log_dir.join("backend-stderr.log"),
    )
}

fn spawn_python_backend(project_root: &Path, python: &Path) -> Option<Child> {
    let (stdout_path, stderr_path) = backend_log_paths(project_root);
    let stdout_file = std::fs::File::create(&stdout_path).ok()?;
    let stderr_file = std::fs::File::create(&stderr_path).ok()?;
    println!("Backend logs: {} | {}", stdout_path.display(), stderr_path.display());

    let mut cmd = Command::new(python);
    cmd.args(["-m", "uvicorn", "src.api.main:app", "--host", "127.0.0.1", "--port", "8000"])
        .current_dir(project_root)
        .stdout(Stdio::from(stdout_file))
        .stderr(Stdio::from(stderr_file));
    apply_windows_spawn_flags(&mut cmd);
    cmd.spawn().ok()
}

fn ensure_data_dir(root: &Path) {
    let data_dir = root.join("data");
    if !data_dir.exists() {
        if let Err(e) = std::fs::create_dir_all(&data_dir) {
            eprintln!("Could not create data dir at {}: {}", data_dir.display(), e);
        }
    }
}

fn start_backend() -> Option<Child> {
    let port: u16 = 8000;

    if is_port_in_use(port) {
        println!("Backend already running on port {}", port);
        return None;
    }

    // В production project_root = bundled resources/ (содержит src/, venv/, pyproject.toml).
    // В dev — обычный поиск pyproject.toml вверх от cwd / exe.
    let bundled = bundled_resource_root();
    let project_root = bundled.clone().or_else(find_project_root).or_else(|| {
        eprintln!("Could not find project root (no pyproject.toml found).");
        eprintln!("Please start backend manually: uv run uvicorn src.api.main:app --port 8000");
        None
    })?;

    ensure_data_dir(&project_root);

    // Production-first: bundled portable Python (resources/python-runtime/python.exe).
    let child = bundled
        .as_ref()
        .and_then(|res_root| {
            for python in local_python_candidates(res_root) {
                if python.exists() {
                    if let Some(child) = spawn_python_backend(&project_root, &python) {
                        println!("Backend started from bundled runtime: {}", python.display());
                        return Some(child);
                    }
                }
            }
            None
        })
        // Dev fallback: project-local .venv.
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
        // Dev fallback: uv в PATH.
        .or_else(|| spawn_uv_backend(&project_root))
        // Last resort: системный python.
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

/// Watchdog: раз в 5 секунд проверяет, что child жив и порт слушается.
/// Если backend упал — пытается перезапустить, не более 3 раз за 60 секунд.
fn spawn_backend_watchdog(state: SharedChild) {
    thread::spawn(move || {
        let port: u16 = 8000;
        let mut restart_count: u32 = 0;
        let mut window_start = Instant::now();
        loop {
            thread::sleep(Duration::from_secs(5));

            let dead = {
                let mut guard = match state.lock() {
                    Ok(g) => g,
                    Err(_) => continue,
                };
                match guard.as_mut() {
                    Some(child) => matches!(child.try_wait(), Ok(Some(_))),
                    None => true,
                }
            };

            if !dead {
                continue;
            }
            if is_port_in_use(port) {
                continue;
            }

            if window_start.elapsed() > Duration::from_secs(60) {
                restart_count = 0;
                window_start = Instant::now();
            }
            if restart_count >= 3 {
                eprintln!("Watchdog: backend died 3x in 60s, giving up until next window");
                continue;
            }
            restart_count += 1;
            eprintln!("Watchdog: backend dead, restart attempt {}", restart_count);

            let new_child = start_backend();
            if let Ok(mut guard) = state.lock() {
                *guard = new_child;
            }
        }
    });
}

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! RegenTwin is ready.", name)
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            // В production DevTools доступны через Ctrl+Shift+I (feature "devtools").
            // Открываем автоматически только в debug билдах.
            #[cfg(debug_assertions)]
            if let Some(window) = app.get_webview_window("main") {
                window.open_devtools();
            }
            let backend = start_backend();
            let state: SharedChild = Arc::new(Mutex::new(backend));
            spawn_backend_watchdog(Arc::clone(&state));
            app.manage(BackendProcess(state));
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![greet])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
