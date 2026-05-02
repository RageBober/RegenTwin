"""Smoke-уровневые E2E против живого uvicorn: health + CORS."""

from __future__ import annotations

import httpx
import pytest


@pytest.mark.e2e
def test_health_returns_ok(http_client: httpx.Client) -> None:
    response = http_client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "version" in payload
    assert "uptime_seconds" in payload
    assert payload["uptime_seconds"] >= 0


@pytest.mark.e2e
def test_cors_preflight_for_vite_origin(http_client: httpx.Client) -> None:
    response = http_client.request(
        "OPTIONS",
        "/api/v1/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code in (200, 204)
    allow_origin = response.headers.get("access-control-allow-origin", "")
    assert allow_origin == "http://localhost:5173"


@pytest.mark.e2e
def test_cors_actual_request_includes_origin_header(http_client: httpx.Client) -> None:
    response = http_client.get(
        "/api/v1/health",
        headers={"Origin": "http://localhost:5173"},
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
