"""E2E для mode=extended: 20-переменная SDE + цитокины + ECM."""

from __future__ import annotations

import httpx
import pytest

from tests.e2e.conftest import poll_until_done

_EXPECTED_CELL_VARS = {"P", "Ne", "M1", "M2", "F", "S"}
_EXPECTED_CYTOKINE_VARS = {"C_TNF", "C_IL10", "C_PDGF", "C_VEGF", "C_TGFb"}
_EXPECTED_ECM_VARS = {"rho_collagen", "rho_fibrin", "C_MMP"}


@pytest.mark.e2e
def test_simulate_extended_completes_with_all_variables(http_client: httpx.Client) -> None:
    payload = {
        "mode": "extended",
        "t_max_hours": 24.0,
        "dt": 0.5,
        "random_seed": 42,
    }
    response = http_client.post("/api/v1/simulate", json=payload)
    assert response.status_code == 200, response.text
    simulation_id = response.json()["simulation_id"]

    final = poll_until_done(http_client, simulation_id, timeout=120.0)
    assert final["status"] == "completed", final

    results = http_client.get(f"/api/v1/results/{simulation_id}")
    assert results.status_code == 200, results.text
    data = results.json()
    assert data["mode"] == "extended"
    variables = set(data["variables"].keys())

    assert variables >= _EXPECTED_CELL_VARS, f"missing cells: {_EXPECTED_CELL_VARS - variables}"
    assert (
        variables >= _EXPECTED_CYTOKINE_VARS
    ), f"missing cytokines: {_EXPECTED_CYTOKINE_VARS - variables}"
    assert variables >= _EXPECTED_ECM_VARS, f"missing ECM: {_EXPECTED_ECM_VARS - variables}"

    time_points = len(data["times"])
    for key in variables:
        assert len(data["variables"][key]) == time_points


@pytest.mark.e2e
def test_simulate_extended_deterministic_with_same_seed(http_client: httpx.Client) -> None:
    payload = {
        "mode": "extended",
        "t_max_hours": 12.0,
        "dt": 0.5,
        "random_seed": 123,
    }
    ids: list[str] = []
    for _ in range(2):
        response = http_client.post("/api/v1/simulate", json=payload)
        assert response.status_code == 200
        sim_id = response.json()["simulation_id"]
        final = poll_until_done(http_client, sim_id, timeout=120.0)
        assert final["status"] == "completed"
        ids.append(sim_id)

    first = http_client.get(f"/api/v1/results/{ids[0]}").json()["variables"]
    second = http_client.get(f"/api/v1/results/{ids[1]}").json()["variables"]

    for key in _EXPECTED_CELL_VARS:
        assert first[key] == pytest.approx(second[key], rel=1e-6, abs=1e-6)
