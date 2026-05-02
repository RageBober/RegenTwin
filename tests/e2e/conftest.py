"""Фикстуры для E2E тестов против живого uvicorn-процесса.

live_api_server поднимает реальный FastAPI-сервер в subprocess на свободном порту,
с изолированной SQLite/upload/results-директорией. http_client — сессионный httpx
клиент. sample_fcs_path — путь к сгенерированной FCS-фикстуре.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
HEALTH_PATH = "/api/v1/health"
STARTUP_TIMEOUT_SECONDS = 45.0
SHUTDOWN_TIMEOUT_SECONDS = 15.0


@dataclass(frozen=True)
class LiveServer:
    base_url: str
    host: str
    port: int
    upload_dir: Path
    results_dir: Path
    database_url: str

    def ws_url(self, simulation_id: str) -> str:
        return f"ws://{self.host}:{self.port}/api/v1/simulate/{simulation_id}/ws"


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _ensure_fixture() -> Path:
    target = FIXTURES_DIR / "sample.fcs"
    if not target.exists():
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "generate_e2e_fixtures.py")],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to generate FCS fixture:\nstdout={result.stdout}\nstderr={result.stderr}"
            )
    if not target.exists():
        raise FileNotFoundError(f"FCS fixture was not generated at {target}")
    return target


def _wait_for_health(base_url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{base_url}{HEALTH_PATH}", timeout=2.0)
            if response.status_code == 200:
                return
        except httpx.HTTPError as exc:
            last_error = exc
        time.sleep(0.3)
    raise TimeoutError(
        f"uvicorn did not become healthy at {base_url}{HEALTH_PATH} "
        f"within {timeout:.0f}s (last error: {last_error})"
    )


@pytest.fixture(scope="session")
def live_api_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[LiveServer]:
    """Запускает реальный uvicorn-процесс с изолированным окружением."""
    _ensure_fixture()

    host = "127.0.0.1"
    port = _find_free_port()
    base_url = f"http://{host}:{port}"

    workdir = tmp_path_factory.mktemp("e2e_server")
    upload_dir = workdir / "uploads"
    results_dir = workdir / "results"
    db_path = workdir / "e2e.db"
    upload_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # SQLite URL на Windows должен быть forward-slash
    database_url = f"sqlite:///{db_path.as_posix()}"

    env = os.environ.copy()
    env.update(
        {
            "REGENTWIN_HOST": host,
            "REGENTWIN_PORT": str(port),
            "REGENTWIN_UPLOAD_DIR": str(upload_dir),
            "REGENTWIN_RESULTS_DIR": str(results_dir),
            "REGENTWIN_DATABASE_URL": database_url,
            "REGENTWIN_USE_CELERY": "false",
            "PYTHONUNBUFFERED": "1",
        }
    )

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.api.main:app",
        "--host",
        host,
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]

    process = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    log_buffer: deque[str] = deque(maxlen=500)

    def _drain_stdout() -> None:
        assert process.stdout is not None
        for raw in iter(process.stdout.readline, b""):
            try:
                log_buffer.append(raw.decode("utf-8", errors="replace").rstrip())
            except Exception:
                pass

    drainer = threading.Thread(target=_drain_stdout, name="uvicorn-log-drainer", daemon=True)
    drainer.start()

    try:
        _wait_for_health(base_url, STARTUP_TIMEOUT_SECONDS)
    except Exception:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise RuntimeError(
            "uvicorn failed to start.\n--- server output ---\n" + "\n".join(log_buffer)
        ) from None

    server = LiveServer(
        base_url=base_url,
        host=host,
        port=port,
        upload_dir=upload_dir,
        results_dir=results_dir,
        database_url=database_url,
    )

    try:
        yield server
    finally:
        process.terminate()
        try:
            process.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


@pytest.fixture(scope="session")
def http_client(live_api_server: LiveServer) -> Iterator[httpx.Client]:
    """Session-scoped httpx client pointing at the live server."""
    with httpx.Client(base_url=live_api_server.base_url, timeout=120.0) as client:
        yield client


@pytest.fixture(scope="session")
def sample_fcs_path() -> Path:
    """Path to the shared FCS fixture used by upload tests."""
    return _ensure_fixture()


def poll_until_done(
    client: httpx.Client,
    simulation_id: str,
    *,
    timeout: float = 120.0,
    interval: float = 0.5,
) -> dict:
    """Polls /api/v1/simulate/{id} until status is terminal. Returns the final payload."""
    deadline = time.monotonic() + timeout
    last_payload: dict = {}
    while time.monotonic() < deadline:
        try:
            response = client.get(f"/api/v1/simulate/{simulation_id}", timeout=10.0)
        except httpx.ReadTimeout:
            time.sleep(interval)
            continue
        response.raise_for_status()
        last_payload = response.json()
        status = last_payload.get("status")
        if status in {"completed", "failed", "cancelled"}:
            return last_payload
        time.sleep(interval)
    raise TimeoutError(
        f"Simulation {simulation_id} did not finish within {timeout:.0f}s "
        f"(last status: {last_payload.get('status')!r})"
    )
