"""E2E для экспорта результатов (CSV/PNG/PDF по simulation_id и inline)."""

from __future__ import annotations

import httpx
import pytest

from tests.e2e.conftest import poll_until_done


@pytest.fixture(scope="module")
def completed_extended_simulation_id(http_client: httpx.Client) -> str:
    payload = {
        "mode": "extended",
        "t_max_hours": 12.0,
        "dt": 0.5,
        "random_seed": 42,
    }
    response = http_client.post("/api/v1/simulate", json=payload)
    response.raise_for_status()
    simulation_id = response.json()["simulation_id"]
    final = poll_until_done(http_client, simulation_id, timeout=120.0)
    assert final["status"] == "completed"
    return simulation_id


@pytest.mark.e2e
def test_export_csv_by_simulation_id(
    http_client: httpx.Client, completed_extended_simulation_id: str
) -> None:
    response = http_client.post(
        f"/api/v1/export/{completed_extended_simulation_id}",
        json={"format": "csv"},
    )
    assert response.status_code == 200, response.text
    assert "text/csv" in response.headers.get("content-type", "")
    body = response.text
    header = body.splitlines()[0]
    assert header.startswith("time,")
    assert "P" in header or "N" in header


@pytest.mark.e2e
def test_export_png_by_simulation_id(
    http_client: httpx.Client, completed_extended_simulation_id: str
) -> None:
    response = http_client.post(
        f"/api/v1/export/{completed_extended_simulation_id}",
        json={"format": "png"},
    )
    assert response.status_code == 200, response.text
    assert response.headers.get("content-type") == "image/png"
    assert len(response.content) > 0
    assert response.content[:8].startswith(b"\x89PNG")


@pytest.mark.e2e
def test_export_unknown_simulation_returns_404(http_client: httpx.Client) -> None:
    response = http_client.post(
        "/api/v1/export/00000000-0000-0000-0000-000000000000",
        json={"format": "csv"},
    )
    assert response.status_code == 404


@pytest.mark.e2e
def test_viz_export_csv_inline(http_client: httpx.Client) -> None:
    response = http_client.post(
        "/api/viz/export/csv",
        json={"simulation": {"t_max_hours": 12.0, "dt": 0.5, "random_seed": 42}},
    )
    assert response.status_code == 200, response.text
    assert "text/csv" in response.headers.get("content-type", "")
    assert response.text.startswith("time,") or "," in response.text.splitlines()[0]
