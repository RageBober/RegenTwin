"""Tests for simulation endpoints."""

from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.routes.simulate import router
from src.db.models import Base, SimulationRecord
from src.db.session import get_db

_UUID_NONEXISTENT = "00000000-0000-0000-0000-000000000000"
_UUID_SIM1 = "11111111-1111-1111-1111-111111111111"
_UUID_STATUS = "22222222-2222-2222-2222-222222222222"
_UUID_DONE = "33333333-3333-3333-3333-333333333333"
_UUID_CANCELLED = "44444444-4444-4444-4444-444444444444"


@contextmanager
def _setup():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    test_session = sessionmaker(bind=engine)

    def override_get_db():
        db = test_session()
        try:
            yield db
        finally:
            db.close()

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    try:
        yield client, test_session
    finally:
        client.close()
        engine.dispose()


def _seed_record(session_factory, **kwargs):
    db = session_factory()
    defaults = {
        "id": _UUID_SIM1,
        "mode": "extended",
        "status": "pending",
        "params_json": {},
        "created_at": datetime.now(UTC),
    }
    defaults.update(kwargs)
    record = SimulationRecord(**defaults)
    db.add(record)
    db.commit()
    db.close()
    return defaults["id"]


class TestStartSimulation:
    @patch("src.api.routes.simulate.SimulationService")
    def test_start_simulation(self, MockService: MagicMock) -> None:
        with _setup() as (client, _):
            mock_record = MagicMock()
            mock_record.id = "sim-123"
            mock_record.status = "pending"
            mock_record.mode = "extended"
            mock_record.created_at = datetime.now(UTC)

            def mock_start(req):
                return mock_record

            MockService.return_value.start_simulation = mock_start

            resp = client.post("/api/v1/simulate", json={"mode": "extended", "t_max_hours": 10.0})
            assert resp.status_code == 200
            data = resp.json()
            assert data["simulation_id"] == "sim-123"
            assert data["status"] == "pending"
            assert data["mode"] == "extended"

    @patch("src.api.routes.simulate.SimulationService")
    def test_start_with_defaults(self, MockService: MagicMock) -> None:
        with _setup() as (client, _):
            mock_record = MagicMock()
            mock_record.id = "sim-456"
            mock_record.status = "pending"
            mock_record.mode = "extended"
            mock_record.created_at = datetime.now(UTC)

            def mock_start(req):
                return mock_record

            MockService.return_value.start_simulation = mock_start

            resp = client.post("/api/v1/simulate", json={})
            assert resp.status_code == 200
            assert resp.json()["mode"] == "extended"

    @patch("src.api.routes.simulate.SimulationService")
    def test_integrated_mode_is_accepted(self, MockService: MagicMock) -> None:
        with _setup() as (client, _):
            mock_record = MagicMock()
            mock_record.id = "sim-int-1"
            mock_record.status = "pending"
            mock_record.mode = "integrated"
            mock_record.created_at = datetime.now(UTC)

            def mock_start(req):
                return mock_record

            MockService.return_value.start_simulation = mock_start

            resp = client.post("/api/v1/simulate", json={"mode": "integrated"})
            assert resp.status_code == 200
            assert resp.json()["mode"] == "integrated"


class TestGetSimulationStatus:
    def test_not_found(self) -> None:
        with _setup() as (client, _):
            resp = client.get(f"/api/v1/simulate/{_UUID_NONEXISTENT}")
            assert resp.status_code == 404

    def test_get_existing_status(self) -> None:
        with _setup() as (client, test_session):
            sim_id = _seed_record(test_session, id=_UUID_STATUS, status="running", progress=50.0)

            resp = client.get(f"/api/v1/simulate/{sim_id}")
            assert resp.status_code == 200
            data = resp.json()
            assert data["simulation_id"] == sim_id
            assert data["status"] == "running"

    def test_get_completed_status(self) -> None:
        with _setup() as (client, test_session):
            sim_id = _seed_record(
                test_session,
                id=_UUID_DONE,
                status="completed",
                progress=100.0,
                completed_at=datetime.now(UTC),
            )

            resp = client.get(f"/api/v1/simulate/{sim_id}")
            assert resp.status_code == 200
            assert resp.json()["status"] == "completed"


class TestCancelSimulation:
    def test_cancel_nonexistent(self) -> None:
        with _setup() as (client, _):
            resp = client.post(f"/api/v1/simulate/{_UUID_NONEXISTENT}/cancel")
            assert resp.status_code == 404


class TestSimulationWebsocket:
    def test_completed_simulation_emits_complete_terminal_event(self) -> None:
        with _setup() as (client, test_session):
            _seed_record(
                test_session,
                id=_UUID_DONE,
                status="completed",
                progress=100.0,
                message="Simulation completed",
                completed_at=datetime.now(UTC),
            )

            with patch("src.api.routes.simulate.SessionLocal", test_session):
                with client.websocket_connect(f"/api/v1/simulate/{_UUID_DONE}/ws") as websocket:
                    progress_msg = websocket.receive_json()
                    terminal_msg = websocket.receive_json()

            assert progress_msg["event"] == "progress"
            assert terminal_msg["event"] == "complete"

    def test_cancelled_simulation_emits_cancelled_terminal_event(self) -> None:
        with _setup() as (client, test_session):
            _seed_record(
                test_session,
                id=_UUID_CANCELLED,
                status="cancelled",
                progress=42.0,
                message="Simulation cancelled",
                completed_at=datetime.now(UTC),
            )

            with patch("src.api.routes.simulate.SessionLocal", test_session):
                with client.websocket_connect(
                    f"/api/v1/simulate/{_UUID_CANCELLED}/ws"
                ) as websocket:
                    progress_msg = websocket.receive_json()
                    terminal_msg = websocket.receive_json()

            assert progress_msg["data"]["percent"] == 42.0
            assert terminal_msg["event"] == "cancelled"


class TestAbmEndToEnd:
    """Real end-to-end ABM regression: POST simulate -> thread runs -> DB records completion.

    Protects against the regression where asyncio-backed workers were cancelled
    as soon as the originating request ended (ABM stalled at 10% in the UI).
    """

    def test_abm_simulation_completes_via_background_thread(self, tmp_path: Path) -> None:
        with _setup() as (client, test_session):

            def _new_test_session():
                return test_session()

            results_dir = tmp_path / "results"
            results_dir.mkdir(parents=True, exist_ok=True)

            with (
                patch("src.db.session.SessionLocal", _new_test_session),
                patch("src.api.routes.simulate.SessionLocal", _new_test_session),
                patch("src.api.services.simulation_service.settings.results_dir", str(results_dir)),
            ):
                resp = client.post(
                    "/api/v1/simulate",
                    json={
                        "mode": "abm",
                        "t_max_hours": 6.0,
                        "dt": 0.5,
                        "random_seed": 42,
                        "P0": 50.0,
                        "Ne0": 20.0,
                        "M1_0": 10.0,
                        "M2_0": 5.0,
                        "F0": 10.0,
                        "S0": 5.0,
                        "D0": 2.0,
                    },
                )
                assert resp.status_code == 200, resp.text
                sim_id = resp.json()["simulation_id"]

                terminal_statuses = {"completed", "failed", "cancelled"}
                deadline = time.monotonic() + 120.0
                final_status: str | None = None
                last_progress = 0.0
                while time.monotonic() < deadline:
                    status_resp = client.get(f"/api/v1/simulate/{sim_id}")
                    assert status_resp.status_code == 200
                    payload = status_resp.json()
                    last_progress = payload.get("progress", 0.0)
                    if payload["status"] in terminal_statuses:
                        final_status = payload["status"]
                        break
                    time.sleep(0.5)

                assert final_status == "completed", (
                    f"ABM simulation did not complete; status={final_status}, "
                    f"progress={last_progress}"
                )
