"""E2E для негативных сценариев API: 404/400/422."""

from __future__ import annotations

import httpx
import pytest


@pytest.mark.e2e
def test_get_results_unknown_simulation_returns_404(http_client: httpx.Client) -> None:
    response = http_client.get("/api/v1/results/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


@pytest.mark.e2e
def test_get_results_invalid_uuid_returns_400(http_client: httpx.Client) -> None:
    response = http_client.get("/api/v1/results/not-a-uuid")
    assert response.status_code == 400


@pytest.mark.e2e
def test_get_status_invalid_uuid_returns_400(http_client: httpx.Client) -> None:
    response = http_client.get("/api/v1/simulate/not-a-uuid")
    assert response.status_code == 400


@pytest.mark.e2e
def test_get_status_unknown_simulation_returns_404(http_client: httpx.Client) -> None:
    response = http_client.get("/api/v1/simulate/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


@pytest.mark.e2e
def test_simulate_rejects_negative_population(http_client: httpx.Client) -> None:
    response = http_client.post(
        "/api/v1/simulate",
        json={"mode": "extended", "t_max_hours": 24.0, "P0": -10.0},
    )
    assert response.status_code == 422


@pytest.mark.e2e
def test_simulate_rejects_t_max_hours_too_large(http_client: httpx.Client) -> None:
    response = http_client.post(
        "/api/v1/simulate",
        json={"mode": "extended", "t_max_hours": 100000.0},
    )
    assert response.status_code == 422


@pytest.mark.e2e
def test_simulate_rejects_dt_geq_t_max(http_client: httpx.Client) -> None:
    response = http_client.post(
        "/api/v1/simulate",
        json={"mode": "extended", "t_max_hours": 10.0, "dt": 20.0},
    )
    assert response.status_code == 422


@pytest.mark.e2e
def test_list_simulations_accepts_pagination(http_client: httpx.Client) -> None:
    response = http_client.get("/api/v1/simulations", params={"skip": 0, "limit": 10})
    assert response.status_code == 200
    assert isinstance(response.json(), list)


@pytest.mark.e2e
def test_list_simulations_rejects_invalid_limit(http_client: httpx.Client) -> None:
    response = http_client.get("/api/v1/simulations", params={"limit": 0})
    assert response.status_code == 422
