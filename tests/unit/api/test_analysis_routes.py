"""Тесты для analysis endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.routes.analysis import router
from src.db.models import AnalysisRecord, Base
from src.db.session import get_db

_UUID_NONEXISTENT = "00000000-0000-0000-0000-000000000000"
_UUID_EXISTING = "11111111-1111-1111-1111-111111111111"


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


class TestSensitivityEndpoint:
    @patch("src.api.routes.analysis.AnalysisService")
    def test_start_sensitivity(self, MockService: MagicMock) -> None:
        client, _ = _setup()

        mock_record = MagicMock()
        mock_record.id = "ana-1"
        mock_record.analysis_type = "sensitivity"
        mock_record.status = "pending"
        mock_record.created_at = datetime.now(timezone.utc)

        async def mock_run(req):  # type: ignore[no-untyped-def]
            return mock_record

        MockService.return_value.run_sensitivity = mock_run

        resp = client.post("/api/v1/analysis/sensitivity", json={
            "parameters": ["r", "K"],
            "method": "sobol",
            "n_samples": 64,
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["analysis_id"] == "ana-1"
        assert data["analysis_type"] == "sensitivity"
        assert data["status"] == "pending"


class TestEstimationEndpoint:
    @patch("src.api.routes.analysis.AnalysisService")
    def test_start_estimation(self, MockService: MagicMock) -> None:
        client, _ = _setup()

        mock_record = MagicMock()
        mock_record.id = "ana-2"
        mock_record.analysis_type = "estimation"
        mock_record.status = "pending"
        mock_record.created_at = datetime.now(timezone.utc)

        async def mock_run(req):  # type: ignore[no-untyped-def]
            return mock_record

        MockService.return_value.run_estimation = mock_run

        resp = client.post("/api/v1/analysis/estimation", json={
            "upload_id": "upl-123",
            "target_variable": "F",
            "method": "mcmc",
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["analysis_id"] == "ana-2"
        assert data["analysis_type"] == "estimation"

    def test_estimation_missing_upload_id(self) -> None:
        client, _ = _setup()
        resp = client.post("/api/v1/analysis/estimation", json={
            "method": "mcmc",
        })
        assert resp.status_code == 422


class TestGetAnalysisStatus:
    def test_not_found(self) -> None:
        client, _ = _setup()
        resp = client.get(f"/api/v1/analysis/{_UUID_NONEXISTENT}")
        assert resp.status_code == 404

    def test_get_existing(self) -> None:
        client, TestSession = _setup()

        db = TestSession()
        record = AnalysisRecord(
            id=_UUID_EXISTING,
            analysis_type="sensitivity",
            status="completed",
            progress=100.0,
            params_json={"method": "sobol"},
            result_json={"S1": [0.5, 0.3], "parameters": ["r", "K"]},
            created_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )
        db.add(record)
        db.commit()
        db.close()

        resp = client.get(f"/api/v1/analysis/{_UUID_EXISTING}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["analysis_id"] == _UUID_EXISTING
        assert data["status"] == "completed"
        assert data["result"]["S1"] == [0.5, 0.3]
