"""Тесты API endpoints визуализации.

Используем FastAPI TestClient для проверки маршрутов.
Тесты не запускают реальную симуляцию — мокаем _run_simulation.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.visualization import router
from src.core.extended_sde import ExtendedSDEState, ExtendedSDETrajectory
from src.core.parameters import ParameterSet

# Тестовое приложение
app = FastAPI()
app.include_router(router)
client = TestClient(app)


def _mock_trajectory() -> ExtendedSDETrajectory:
    """Лёгкая mock-траектория для тестов API."""
    n_steps = 20
    times = np.linspace(0, 100, n_steps)
    states = []
    for i, t in enumerate(times):
        frac = i / (n_steps - 1)
        states.append(ExtendedSDEState(
            P=500 * (1 - frac),
            Ne=200 * np.exp(-frac * 3),
            M1=100 * np.exp(-frac * 2),
            M2=150 * frac,
            F=300 * frac,
            Mf=50 * frac * np.exp(-frac),
            E=80 * frac,
            S=40 * np.exp(-frac),
            C_TNF=10 * np.exp(-frac * 3),
            C_IL10=5 * frac,
            C_PDGF=3 * np.exp(-frac),
            C_VEGF=4 * frac,
            C_TGFb=2 * (1 + frac),
            C_MCP1=6 * np.exp(-frac * 2),
            C_IL8=8 * np.exp(-frac * 3),
            rho_collagen=min(1.0, 0.1 + 0.9 * frac),
            C_MMP=2 * np.exp(-frac),
            rho_fibrin=max(0, 0.8 * (1 - frac)),
            D=5 * np.exp(-frac * 4),
            O2=80 + 20 * frac,
            t=t,
        ))
    return ExtendedSDETrajectory(times=times, states=states, params=ParameterSet())


MOCK_TRAJ = _mock_trajectory()


def _patched_run_simulation(*args, **kwargs) -> ExtendedSDETrajectory:
    return MOCK_TRAJ


def _patched_run_comparison(*args, **kwargs) -> dict[str, ExtendedSDETrajectory]:
    return {
        "Control": MOCK_TRAJ,
        "PRP": MOCK_TRAJ,
        "PEMF": MOCK_TRAJ,
        "PRP+PEMF": MOCK_TRAJ,
    }


class TestPopulationsEndpoint:
    """Тесты POST /api/viz/populations."""

    @patch("src.api.routes.visualization._run_simulation", _patched_run_simulation)
    def test_returns_200(self) -> None:
        resp = client.post("/api/viz/populations", json={})
        assert resp.status_code == 200

    @patch("src.api.routes.visualization._run_simulation", _patched_run_simulation)
    def test_returns_plotly_json(self) -> None:
        resp = client.post("/api/viz/populations", json={})
        data = resp.json()
        assert "data" in data
        assert "layout" in data

    @patch("src.api.routes.visualization._run_simulation", _patched_run_simulation)
    def test_with_variables(self) -> None:
        resp = client.post("/api/viz/populations", json={
            "simulation": {}, "variables": ["P", "F"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["data"]) == 2


class TestCytokinesEndpoint:
    """Тесты POST /api/viz/cytokines."""

    @patch("src.api.routes.visualization._run_simulation", _patched_run_simulation)
    def test_returns_200(self) -> None:
        resp = client.post("/api/viz/cytokines", json={})
        assert resp.status_code == 200

    @patch("src.api.routes.visualization._run_simulation", _patched_run_simulation)
    def test_overlay_mode(self) -> None:
        resp = client.post("/api/viz/cytokines", json={"layout": "overlay"})
        data = resp.json()
        assert len(data["data"]) == 7

    @patch("src.api.routes.visualization._run_simulation", _patched_run_simulation)
    def test_subplots_mode(self) -> None:
        resp = client.post("/api/viz/cytokines", json={"layout": "subplots"})
        assert resp.status_code == 200


class TestECMEndpoint:
    """Тесты POST /api/viz/ecm."""

    @patch("src.api.routes.visualization._run_simulation", _patched_run_simulation)
    def test_returns_200(self) -> None:
        resp = client.post("/api/viz/ecm", json={})
        assert resp.status_code == 200

    @patch("src.api.routes.visualization._run_simulation", _patched_run_simulation)
    def test_has_3_traces(self) -> None:
        resp = client.post("/api/viz/ecm", json={})
        data = resp.json()
        assert len(data["data"]) == 3


class TestPhasesEndpoint:
    """Тесты POST /api/viz/phases."""

    @patch("src.api.routes.visualization._run_simulation", _patched_run_simulation)
    def test_returns_200(self) -> None:
        resp = client.post("/api/viz/phases", json={})
        assert resp.status_code == 200


class TestComparisonEndpoint:
    """Тесты POST /api/viz/comparison."""

    @patch("src.api.routes.visualization._run_comparison", _patched_run_comparison)
    def test_returns_200(self) -> None:
        resp = client.post("/api/viz/comparison", json={})
        assert resp.status_code == 200

    @patch("src.api.routes.visualization._run_comparison", _patched_run_comparison)
    def test_four_traces(self) -> None:
        resp = client.post("/api/viz/comparison", json={})
        data = resp.json()
        assert len(data["data"]) == 4


class TestExportCSV:
    """Тесты POST /api/viz/export/csv."""

    @patch("src.api.routes.visualization._run_simulation", _patched_run_simulation)
    def test_returns_csv(self) -> None:
        resp = client.post("/api/viz/export/csv", json={})
        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]

    @patch("src.api.routes.visualization._run_simulation", _patched_run_simulation)
    def test_csv_has_header(self) -> None:
        resp = client.post("/api/viz/export/csv", json={})
        text = resp.text
        first_line = text.split("\n")[0]
        assert "time" in first_line
        assert "P" in first_line


class TestExportPNG:
    """Тесты POST /api/viz/export/png."""

    @patch("src.api.routes.visualization._run_simulation", _patched_run_simulation)
    def test_returns_png(self) -> None:
        resp = client.post("/api/viz/export/png", json={})
        assert resp.status_code == 200
        assert "image/png" in resp.headers["content-type"]

    @patch("src.api.routes.visualization._run_simulation", _patched_run_simulation)
    def test_png_not_empty(self) -> None:
        resp = client.post("/api/viz/export/png", json={})
        assert len(resp.content) > 100


class TestExportPDF:
    """Тесты POST /api/viz/export/pdf."""

    @patch("src.api.routes.visualization._run_simulation", _patched_run_simulation)
    def test_returns_pdf(self) -> None:
        resp = client.post("/api/viz/export/pdf", json={})
        assert resp.status_code == 200
        assert "application/pdf" in resp.headers["content-type"]

    @patch("src.api.routes.visualization._run_simulation", _patched_run_simulation)
    def test_pdf_not_empty(self) -> None:
        resp = client.post("/api/viz/export/pdf", json={})
        assert len(resp.content) > 100
