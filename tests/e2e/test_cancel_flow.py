"""E2E для кооперативной отмены симуляции."""

from __future__ import annotations

import time

import httpx
import pytest

from tests.e2e.conftest import poll_until_done


@pytest.mark.e2e
@pytest.mark.e2e_slow
def test_cancel_running_simulation(http_client: httpx.Client) -> None:
    payload = {
        "mode": "extended",
        "t_max_hours": 720.0,
        "dt": 0.1,
        "random_seed": 42,
        "n_trajectories": 50,
    }
    response = http_client.post("/api/v1/simulate", json=payload)
    assert response.status_code == 200
    simulation_id = response.json()["simulation_id"]

    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        status = http_client.get(f"/api/v1/simulate/{simulation_id}").json()
        if status.get("status") == "running" and status.get("progress", 0.0) > 5.0:
            break
        if status.get("status") in {"completed", "failed", "cancelled"}:
            pytest.skip(f"Simulation finished before cancel could fire: {status['status']}")
        time.sleep(0.1)

    cancel = http_client.post(f"/api/v1/simulate/{simulation_id}/cancel")
    assert cancel.status_code == 200, cancel.text
    assert cancel.json()["status"] == "cancelling"

    final = poll_until_done(http_client, simulation_id, timeout=120.0)
    assert final["status"] in {"cancelled", "completed"}, final


@pytest.mark.e2e
def test_cancel_unknown_simulation_returns_404(http_client: httpx.Client) -> None:
    response = http_client.post("/api/v1/simulate/00000000-0000-0000-0000-000000000000/cancel")
    assert response.status_code == 404


@pytest.mark.e2e
def test_cancel_invalid_uuid_returns_400(http_client: httpx.Client) -> None:
    response = http_client.post("/api/v1/simulate/not-a-uuid/cancel")
    assert response.status_code == 400
