"""E2E для mode=integrated (SDE + ABM multi-scale coupling)."""

from __future__ import annotations

import httpx
import pytest

from tests.e2e.conftest import poll_until_done


@pytest.mark.e2e
@pytest.mark.e2e_slow
def test_simulate_integrated_completes_with_sde_variables(http_client: httpx.Client) -> None:
    payload = {
        "mode": "integrated",
        "t_max_hours": 12.0,
        "dt": 0.5,
        "random_seed": 42,
    }
    response = http_client.post("/api/v1/simulate", json=payload)
    assert response.status_code == 200, response.text
    simulation_id = response.json()["simulation_id"]
    assert response.json()["mode"] == "integrated"

    final = poll_until_done(http_client, simulation_id, timeout=600.0)
    assert final["status"] == "completed", final

    results = http_client.get(f"/api/v1/results/{simulation_id}")
    assert results.status_code == 200, results.text
    data = results.json()
    assert data["mode"] == "integrated"
    variables = set(data["variables"].keys())

    assert {"N", "C"} <= variables
    time_points = len(data["times"])
    for key in ("N", "C"):
        assert len(data["variables"][key]) == time_points

    metadata = data.get("metadata") or {}
    assert metadata.get("abm_snapshot_count", 0) > 0

    status = http_client.get(f"/api/v1/simulate/{simulation_id}")
    assert status.status_code == 200
    assert status.json()["status"] == "completed"
