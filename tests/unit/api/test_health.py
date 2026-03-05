"""Тесты для health endpoint."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.health import router


def _make_client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestHealthEndpoint:
    def test_returns_200(self) -> None:
        client = _make_client()
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_response_schema(self) -> None:
        client = _make_client()
        data = _make_client().get("/api/v1/health").json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"
        assert isinstance(data["uptime_seconds"], float)

    def test_uptime_is_positive(self) -> None:
        client = _make_client()
        data = client.get("/api/v1/health").json()
        assert data["uptime_seconds"] >= 0
