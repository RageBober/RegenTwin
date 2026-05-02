"""E2E для Monte Carlo ансамбля (n_trajectories > 1)."""

from __future__ import annotations

import httpx
import pytest

from tests.e2e.conftest import poll_until_done


@pytest.mark.e2e
@pytest.mark.e2e_slow
def test_simulate_extended_monte_carlo_ensemble(http_client: httpx.Client) -> None:
    payload = {
        "mode": "extended",
        "t_max_hours": 12.0,
        "dt": 0.5,
        "random_seed": 42,
        "n_trajectories": 5,
    }
    response = http_client.post("/api/v1/simulate", json=payload)
    assert response.status_code == 200, response.text
    simulation_id = response.json()["simulation_id"]

    final = poll_until_done(http_client, simulation_id, timeout=600.0)
    assert final["status"] == "completed", final

    results = http_client.get(f"/api/v1/results/{simulation_id}")
    assert results.status_code == 200, results.text
    data = results.json()
    metadata = data.get("metadata") or {}
    assert metadata.get("n_trajectories", 1) == 5
    assert metadata.get("n_successful", 0) >= 1
    assert metadata.get("extended_mc") is True

    variables = set(data["variables"].keys())
    assert "mean_N" in variables
    assert "std_N" in variables
    assert any(v.startswith("mean_P") or v == "mean_F" for v in variables)


@pytest.mark.e2e
def test_simulate_mvp_monte_carlo_returns_statistics(http_client: httpx.Client) -> None:
    payload = {
        "mode": "mvp",
        "t_max_hours": 12.0,
        "dt": 0.5,
        "random_seed": 42,
        "n_trajectories": 3,
    }
    response = http_client.post("/api/v1/simulate", json=payload)
    assert response.status_code == 200, response.text
    simulation_id = response.json()["simulation_id"]

    final = poll_until_done(http_client, simulation_id, timeout=240.0)
    assert final["status"] == "completed", final

    results = http_client.get(f"/api/v1/results/{simulation_id}")
    assert results.status_code == 200, results.text
    data = results.json()
    metadata = data.get("metadata") or {}
    assert metadata.get("n_trajectories", 1) == 3
    variables = set(data["variables"].keys())
    assert {"mean_N", "std_N", "mean_C", "std_C"} <= variables
