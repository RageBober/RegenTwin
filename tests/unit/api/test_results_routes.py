"""Tests for results and export endpoints."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
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
_UUID_ABM = "44444444-4444-4444-4444-444444444444"


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


def _seed_completed(session_factory, sim_id=_UUID_DONE, mode="extended"):
    db = session_factory()
    record = SimulationRecord(
        id=sim_id,
        mode=mode,
        status="completed",
        progress=100.0,
        params_json={},
        result_path=f"data/results/{sim_id}.npz",
        created_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
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

MOCK_ABM_DATA = {
    "mode": "abm",
    "times": [0.0, 24.0, 48.0],
    "variables": {
        "stem": [10.0, 12.0, 14.0],
        "macro": [20.0, 18.0, 16.0],
        "fibro": [5.0, 8.0, 11.0],
    },
    "metadata": {"snapshot_count": 3, "supported_exports": ["csv"]},
}


class TestGetResults:
    def test_not_found(self) -> None:
        with _setup() as (client, _):
            resp = client.get(f"/api/v1/results/{_UUID_NONEXISTENT}")
            assert resp.status_code == 404

    def test_not_completed(self) -> None:
        with _setup() as (client, test_session):
            db = test_session()
            record = SimulationRecord(
                id=_UUID_RUNNING,
                mode="extended",
                status="running",
                params_json={},
                created_at=datetime.now(UTC),
            )
            db.add(record)
            db.commit()
            db.close()

            resp = client.get(f"/api/v1/results/{_UUID_RUNNING}")
            assert resp.status_code == 400

    @patch("src.api.services.simulation_service.SimulationService.load_trajectory")
    def test_get_completed_results(self, mock_load) -> None:
        mock_load.return_value = MOCK_TRAJECTORY_DATA
        with _setup() as (client, test_session):
            _seed_completed(test_session)

            resp = client.get(f"/api/v1/results/{_UUID_DONE}")
            assert resp.status_code == 200
            data = resp.json()
            assert data["simulation_id"] == _UUID_DONE
            assert data["mode"] == "extended"
            assert len(data["times"]) == 3
            assert "P" in data["variables"]


class TestExportResults:
    def test_not_found(self) -> None:
        with _setup() as (client, _):
            resp = client.post(f"/api/v1/export/{_UUID_NONEXISTENT}", json={"format": "csv"})
            assert resp.status_code == 404

    def test_not_completed(self) -> None:
        with _setup() as (client, test_session):
            db = test_session()
            record = SimulationRecord(
                id=_UUID_PENDING,
                mode="extended",
                status="pending",
                params_json={},
                created_at=datetime.now(UTC),
            )
            db.add(record)
            db.commit()
            db.close()

            resp = client.post(f"/api/v1/export/{_UUID_PENDING}", json={"format": "pdf"})
            assert resp.status_code == 400

    @patch("src.api.services.simulation_service.SimulationService.load_trajectory")
    def test_csv_export_supports_abm_results(self, mock_load) -> None:
        mock_load.return_value = MOCK_ABM_DATA
        with _setup() as (client, test_session):
            _seed_completed(test_session, sim_id=_UUID_ABM, mode="abm")

            resp = client.post(f"/api/v1/export/{_UUID_ABM}", json={"format": "csv"})
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/csv")
            assert "stem" in resp.text

    @patch("src.api.services.simulation_service.SimulationService.load_trajectory")
    def test_png_export_rejects_abm_results(self, mock_load) -> None:
        mock_load.return_value = MOCK_ABM_DATA
        with _setup() as (client, test_session):
            _seed_completed(test_session, sim_id=_UUID_ABM, mode="abm")

            resp = client.post(f"/api/v1/export/{_UUID_ABM}", json={"format": "png"})
            assert resp.status_code == 501
