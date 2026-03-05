"""Тесты для results и export endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.routes.results import router
from src.db.models import Base, SimulationRecord
from src.db.session import get_db

_UUID_NONEXISTENT = "00000000-0000-0000-0000-000000000000"
_UUID_RUNNING = "11111111-1111-1111-1111-111111111111"
_UUID_DONE = "22222222-2222-2222-2222-222222222222"
_UUID_PENDING = "33333333-3333-3333-3333-333333333333"


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


def _seed_completed(session_factory, sim_id=_UUID_DONE):  # type: ignore[no-untyped-def]
    db = session_factory()
    record = SimulationRecord(
        id=sim_id,
        mode="extended",
        status="completed",
        progress=100.0,
        params_json={},
        result_path=f"data/results/{sim_id}.npz",
        created_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
    )
    db.add(record)
    db.commit()
    db.close()


MOCK_TRAJECTORY_DATA = {
    "times": [0.0, 1.0, 2.0],
    "variables": {
        "P": [500.0, 450.0, 400.0],
        "Ne": [200.0, 180.0, 160.0],
        "M1": [100.0, 90.0, 80.0],
        "M2": [10.0, 15.0, 20.0],
        "F": [50.0, 55.0, 60.0],
    },
}


class TestGetResults:
    def test_not_found(self) -> None:
        client, _ = _setup()
        resp = client.get(f"/api/v1/results/{_UUID_NONEXISTENT}")
        assert resp.status_code == 404

    def test_not_completed(self) -> None:
        client, TestSession = _setup()
        db = TestSession()
        record = SimulationRecord(
            id=_UUID_RUNNING,
            mode="extended",
            status="running",
            params_json={},
            created_at=datetime.now(timezone.utc),
        )
        db.add(record)
        db.commit()
        db.close()

        resp = client.get(f"/api/v1/results/{_UUID_RUNNING}")
        assert resp.status_code == 400

    @patch("src.api.services.simulation_service.SimulationService.load_trajectory")
    def test_get_completed_results(self, mock_load) -> None:  # type: ignore[no-untyped-def]
        mock_load.return_value = MOCK_TRAJECTORY_DATA
        client, TestSession = _setup()
        _seed_completed(TestSession)

        resp = client.get(f"/api/v1/results/{_UUID_DONE}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["simulation_id"] == _UUID_DONE
        assert data["mode"] == "extended"
        assert len(data["times"]) == 3
        assert "P" in data["variables"]


class TestExportResults:
    def test_not_found(self) -> None:
        client, _ = _setup()
        resp = client.post(f"/api/v1/export/{_UUID_NONEXISTENT}", json={"format": "csv"})
        assert resp.status_code == 404

    def test_not_completed(self) -> None:
        client, TestSession = _setup()
        db = TestSession()
        record = SimulationRecord(
            id=_UUID_PENDING,
            mode="extended",
            status="pending",
            params_json={},
            created_at=datetime.now(timezone.utc),
        )
        db.add(record)
        db.commit()
        db.close()

        resp = client.post(f"/api/v1/export/{_UUID_PENDING}", json={"format": "pdf"})
        assert resp.status_code == 400
