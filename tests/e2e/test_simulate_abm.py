"""E2E для mode=abm: полный цикл simulate + spatial-визуализации через сохранённый snapshot."""

from __future__ import annotations

import httpx
import pytest

from tests.e2e.conftest import poll_until_done

_ABM_POPULATION_KEYS = {"stem", "macro", "fibro"}


@pytest.mark.e2e
@pytest.mark.e2e_slow
def test_simulate_abm_completes_and_produces_snapshots(http_client: httpx.Client) -> None:
    payload = {
        "mode": "abm",
        "t_max_hours": 5.0,
        "dt": 1.0,
        "random_seed": 42,
        "P0": 50.0,
        "Ne0": 20.0,
        "M1_0": 10.0,
        "M2_0": 5.0,
        "F0": 15.0,
        "S0": 10.0,
        "C_TNF0": 5.0,
        "C_IL10_0": 0.5,
    }
    response = http_client.post("/api/v1/simulate", json=payload)
    assert response.status_code == 200, response.text
    simulation_id = response.json()["simulation_id"]

    final = poll_until_done(http_client, simulation_id, timeout=600.0)
    assert final["status"] == "completed", final

    results = http_client.get(f"/api/v1/results/{simulation_id}")
    assert results.status_code == 200, results.text
    data = results.json()
    assert data["mode"] == "abm"
    variables = set(data["variables"].keys())
    assert variables >= _ABM_POPULATION_KEYS, f"missing: {_ABM_POPULATION_KEYS - variables}"
    metadata = data.get("metadata") or {}
    assert metadata.get("snapshot_count", 0) > 0

    heatmap = http_client.post(
        "/api/viz/spatial/heatmap",
        json={"simulation_id": simulation_id, "timestep": -1},
    )
    assert heatmap.status_code == 200, heatmap.text
    heatmap_payload = heatmap.json()
    assert "data" in heatmap_payload
    assert len(heatmap_payload["data"]) >= 1

    for color_by in ("type", "energy", "age"):
        scatter = http_client.post(
            "/api/viz/spatial/scatter",
            json={"simulation_id": simulation_id, "timestep": -1, "color_by": color_by},
        )
        assert scatter.status_code == 200, f"scatter color_by={color_by}: {scatter.text}"
        assert "data" in scatter.json()

    inflammation = http_client.post(
        "/api/viz/spatial/inflammation",
        json={"simulation_id": simulation_id, "timestep": -1},
    )
    assert inflammation.status_code == 200, inflammation.text
    assert "data" in inflammation.json()


@pytest.mark.e2e
def test_spatial_rejects_out_of_range_params(http_client: httpx.Client) -> None:
    response = http_client.post(
        "/api/viz/spatial/heatmap",
        json={"n_stem": -1, "n_macro": 30, "t_max_hours": 48.0},
    )
    assert response.status_code == 422

    too_long = http_client.post(
        "/api/viz/spatial/heatmap",
        json={"t_max_hours": 800.0},
    )
    assert too_long.status_code == 422


@pytest.mark.e2e
def test_spatial_scatter_rejects_invalid_color_by(http_client: httpx.Client) -> None:
    response = http_client.post(
        "/api/viz/spatial/scatter",
        json={"color_by": "nope"},
    )
    assert response.status_code == 422


@pytest.mark.e2e
def test_spatial_heatmap_unknown_simulation_returns_404(http_client: httpx.Client) -> None:
    response = http_client.post(
        "/api/viz/spatial/heatmap",
        json={"simulation_id": "00000000-0000-0000-0000-000000000000"},
    )
    assert response.status_code == 404
