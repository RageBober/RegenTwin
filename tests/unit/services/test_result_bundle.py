"""Regression tests for persisted result bundles."""

from __future__ import annotations

import numpy as np

from src.api.services.result_bundle import (
    build_abm_snapshot,
    load_result_bundle,
    save_result_bundle,
)
from src.core.abm_model import ABMConfig, ABMSnapshot, ABMTrajectory, AgentState


def _make_snapshot() -> ABMSnapshot:
    agents = [
        AgentState(
            agent_id=7,
            agent_type="macro",
            x=10.0,
            y=15.0,
            age=24.0,
            division_count=3,
            energy=0.8,
            alive=True,
        ),
    ]
    return ABMSnapshot(
        t=24.0,
        agents=agents,
        cytokine_field=np.ones((4, 4), dtype=np.float64),
        ecm_field=np.zeros((4, 4), dtype=np.float64),
    )


def test_abm_result_bundle_round_trip_preserves_division_count(tmp_path) -> None:
    result_path = tmp_path / "abm-results.npz"
    trajectory = ABMTrajectory(snapshots=[_make_snapshot()], config=ABMConfig())

    save_result_bundle(result_path, "abm", trajectory)
    result = load_result_bundle(result_path, default_mode="abm", include_snapshots=True)
    snapshot = build_abm_snapshot(result)

    assert snapshot.agents[0].division_count == 3
    assert snapshot.agents[0].agent_type == "macro"


def test_build_abm_snapshot_defaults_missing_division_count() -> None:
    result = {
        "mode": "abm",
        "times": [24.0],
        "variables": {"macro": [1.0]},
        "metadata": {"snapshot_count": 1},
        "snapshots": [
            {
                "t": 24.0,
                "x": [10.0],
                "y": [20.0],
                "type": ["macro"],
                "energy": [0.5],
                "age": [12.0],
                "cytokine_field": [[1.0, 0.0], [0.0, 1.0]],
                "ecm_field": [[0.0, 0.0], [0.0, 0.0]],
            },
        ],
    }

    snapshot = build_abm_snapshot(result)

    assert snapshot.agents[0].division_count == 0
    assert snapshot.agents[0].energy == 0.5
