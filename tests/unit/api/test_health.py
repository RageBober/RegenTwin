"""Тесты для health endpoints.

Покрывают `/api/v1/health`, `/api/v1/health/live` и `/api/v1/health/ready`.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.models.schemas import HealthCheckDetail
from src.api.routes.health import router


def _make_client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _patch_checks(
    db: HealthCheckDetail, celery: HealthCheckDetail, redis: HealthCheckDetail
) -> Any:
    """Удобный context manager для мокирования всех трёх health-helpers."""
    return patch.multiple(
        "src.api.routes.health",
        check_db=lambda: _async_return(db),
        check_celery=lambda: _async_return(celery),
        check_redis=lambda: _async_return(redis),
    )


async def _async_return(value: HealthCheckDetail) -> HealthCheckDetail:
    return value


class TestHealthAggregated:
    def test_returns_200_when_all_ok(self) -> None:
        client = _make_client()
        with _patch_checks(
            db=HealthCheckDetail(status="ok", latency_ms=1.2),
            celery=HealthCheckDetail(status="skipped", message="use_celery=False"),
            redis=HealthCheckDetail(status="skipped", message="use_celery=False"),
        ):
            resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"
        assert data["uptime_seconds"] >= 0
        assert set(data["checks"].keys()) == {"db", "celery", "redis"}
        assert data["checks"]["db"]["status"] == "ok"
        assert data["checks"]["celery"]["status"] == "skipped"

    def test_unhealthy_when_db_down(self) -> None:
        client = _make_client()
        with _patch_checks(
            db=HealthCheckDetail(status="unhealthy", message="connection refused"),
            celery=HealthCheckDetail(status="skipped"),
            redis=HealthCheckDetail(status="skipped"),
        ):
            resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "unhealthy"
        assert data["checks"]["db"]["status"] == "unhealthy"

    def test_degraded_propagates(self) -> None:
        client = _make_client()
        with _patch_checks(
            db=HealthCheckDetail(status="ok"),
            celery=HealthCheckDetail(status="degraded", message="slow"),
            redis=HealthCheckDetail(status="ok"),
        ):
            resp = client.get("/api/v1/health")
        assert resp.json()["status"] == "degraded"


class TestHealthLive:
    def test_always_returns_ok_without_db_check(self) -> None:
        client = _make_client()
        # Даже если все зависимости легли, /live должен вернуть ok.
        with _patch_checks(
            db=HealthCheckDetail(status="unhealthy", message="dead"),
            celery=HealthCheckDetail(status="unhealthy"),
            redis=HealthCheckDetail(status="unhealthy"),
        ):
            resp = client.get("/api/v1/health/live")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["checks"] == {}


class TestHealthReady:
    def test_returns_200_when_db_ok(self) -> None:
        client = _make_client()
        with _patch_checks(
            db=HealthCheckDetail(status="ok", latency_ms=0.5),
            celery=HealthCheckDetail(status="skipped"),
            redis=HealthCheckDetail(status="skipped"),
        ):
            resp = client.get("/api/v1/health/ready")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_returns_503_when_db_unhealthy(self) -> None:
        client = _make_client()
        with _patch_checks(
            db=HealthCheckDetail(status="unhealthy", message="connection refused"),
            celery=HealthCheckDetail(status="skipped"),
            redis=HealthCheckDetail(status="skipped"),
        ):
            resp = client.get("/api/v1/health/ready")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "unhealthy"
