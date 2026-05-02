"""E2E для mode=mvp: полный цикл simulate → status → results."""

from __future__ import annotations

import httpx
import pytest

from tests.e2e.conftest import poll_until_done


@pytest.mark.e2e
def test_simulate_mvp_completes_and_returns_results(http_client: httpx.Client) -> None:
    payload = {
        "mode": "mvp",
        "t_max_hours": 24.0,
        "dt": 0.5,
        "random_seed": 42,
        "P0": 500.0,
        "Ne0": 200.0,
        "M1_0": 100.0,
        "M2_0": 10.0,
        "F0": 50.0,
        "S0": 40.0,
        "C_TNF0": 10.0,
        "C_IL10_0": 0.5,
    }
    response = http_client.post("/api/v1/simulate", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    simulation_id = body["simulation_id"]
    assert body["mode"] == "mvp"
    assert body["status"] in {"pending", "running"}

    final = poll_until_done(http_client, simulation_id, timeout=90.0)
    assert final["status"] == "completed", final
    assert final["progress"] == pytest.approx(100.0)

    results = http_client.get(f"/api/v1/results/{simulation_id}")
    assert results.status_code == 200, results.text
    data = results.json()
    assert data["simulation_id"] == simulation_id
    assert data["mode"] == "mvp"
    assert isinstance(data["times"], list) and len(data["times"]) > 1
    variables = data["variables"]
    assert "N" in variables
    assert "C" in variables
    assert len(variables["N"]) == len(data["times"])
    assert all(value >= 0 for value in variables["N"])


@pytest.mark.e2e
def test_simulate_mvp_rejects_invalid_params(http_client: httpx.Client) -> None:
    response = http_client.post(
        "/api/v1/simulate",
        json={"mode": "mvp", "t_max_hours": 24.0, "dt": 50.0},
    )
    assert response.status_code == 422


@pytest.mark.e2e
def test_simulate_rejects_unknown_mode(http_client: httpx.Client) -> None:
    response = http_client.post(
        "/api/v1/simulate",
        json={"mode": "nonsense", "t_max_hours": 24.0},
    )
    assert response.status_code == 422
