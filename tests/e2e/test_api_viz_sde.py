"""E2E для inline SDE визуализаций (/api/viz/populations, /cytokines, /ecm, /phases, /comparison)."""

from __future__ import annotations

import httpx
import pytest

_SHORT_SIM = {
    "t_max_hours": 12.0,
    "dt": 0.5,
    "random_seed": 42,
}


@pytest.mark.e2e
@pytest.mark.parametrize("endpoint", ["populations", "cytokines", "ecm", "phases"])
def test_viz_inline_endpoints_return_plotly_json(http_client: httpx.Client, endpoint: str) -> None:
    response = http_client.post(
        f"/api/viz/{endpoint}",
        json={"simulation": _SHORT_SIM},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert "data" in payload
    assert isinstance(payload["data"], list)
    assert len(payload["data"]) >= 1


@pytest.mark.e2e
def test_viz_comparison_returns_four_scenarios(http_client: httpx.Client) -> None:
    response = http_client.post(
        "/api/viz/comparison",
        json={"simulation": _SHORT_SIM, "variable": "F"},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert len(payload["data"]) >= 1


@pytest.mark.e2e
def test_viz_cytokines_respects_layout_param(http_client: httpx.Client) -> None:
    for layout in ("overlay", "subplots"):
        response = http_client.post(
            "/api/viz/cytokines",
            json={"simulation": _SHORT_SIM, "layout": layout},
        )
        assert response.status_code == 200, f"layout={layout}: {response.text}"
