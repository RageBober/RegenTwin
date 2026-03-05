"""Тесты для simulation endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
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


def _setup():  # type: ignore[no-untyped-def]
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine)

    def override_get_db():  # type: ignore[no-untyped-def]
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_db] = override_get_db

    return TestClient(app), TestSession


def _seed_record(session_factory, **kwargs):  # type: ignore[no-untyped-def]
    """Создать SimulationRecord напрямую в БД."""
    db = session_factory()
    defaults = {
        "id": _UUID_SIM1,
        "mode": "extended",
        "status": "pending",
        "params_json": {},
        "created_at": datetime.now(timezone.utc),
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
        client, TestSession = _setup()

        # Мокируем service.start_simulation → возвращает готовый record
        mock_record = MagicMock()
        mock_record.id = "sim-123"
        mock_record.status = "pending"
        mock_record.mode = "extended"
        mock_record.created_at = datetime.now(timezone.utc)

        instance = MockService.return_value

        async def mock_start(req):  # type: ignore[no-untyped-def]
            return mock_record

        instance.start_simulation = mock_start

        resp = client.post("/api/v1/simulate", json={
            "mode": "extended",
            "t_max_hours": 10.0,
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["simulation_id"] == "sim-123"
        assert data["status"] == "pending"
        assert data["mode"] == "extended"

    @patch("src.api.routes.simulate.SimulationService")
    def test_start_with_defaults(self, MockService: MagicMock) -> None:
        client, _ = _setup()

        mock_record = MagicMock()
        mock_record.id = "sim-456"
        mock_record.status = "pending"
        mock_record.mode = "extended"
        mock_record.created_at = datetime.now(timezone.utc)

        async def mock_start(req):  # type: ignore[no-untyped-def]
            return mock_record

        MockService.return_value.start_simulation = mock_start

        resp = client.post("/api/v1/simulate", json={})
        assert resp.status_code == 200
        assert resp.json()["mode"] == "extended"


class TestGetSimulationStatus:
    def test_not_found(self) -> None:
        client, _ = _setup()
        resp = client.get(f"/api/v1/simulate/{_UUID_NONEXISTENT}")
        assert resp.status_code == 404

    def test_get_existing_status(self) -> None:
        client, TestSession = _setup()
        sim_id = _seed_record(TestSession, id=_UUID_STATUS, status="running", progress=50.0)

        resp = client.get(f"/api/v1/simulate/{sim_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["simulation_id"] == sim_id
        assert data["status"] == "running"

    def test_get_completed_status(self) -> None:
        client, TestSession = _setup()
        sim_id = _seed_record(
            TestSession,
            id=_UUID_DONE,
            status="completed",
            progress=100.0,
            completed_at=datetime.now(timezone.utc),
        )

        resp = client.get(f"/api/v1/simulate/{sim_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"


class TestCancelSimulation:
    def test_cancel_nonexistent(self) -> None:
        client, _ = _setup()
        resp = client.post(f"/api/v1/simulate/{_UUID_NONEXISTENT}/cancel")
        assert resp.status_code == 404
