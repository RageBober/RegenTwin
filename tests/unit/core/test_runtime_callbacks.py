"""Regression tests for cooperative progress and cancellation hooks."""

from __future__ import annotations

import pytest

from src.core.abm_model import ABMConfig, ABMModel
from src.core.extended_sde import ExtendedSDEModel, ExtendedSDEState
from src.core.parameters import ParameterSet
from src.core.sde_model import SDEConfig, SDEModel
from src.data.parameter_extraction import ModelParameters


class RuntimeCancelledError(RuntimeError):
    """Test-only cooperative cancellation error."""


def _model_params() -> ModelParameters:
    return ModelParameters(
        n0=100.0,
        c0=10.0,
        stem_cell_fraction=0.1,
        macrophage_fraction=0.2,
        apoptotic_fraction=0.05,
        inflammation_level=0.5,
    )


def test_sde_model_reports_progress_and_supports_mid_run_cancellation() -> None:
    progress_steps: list[int] = []
    model = SDEModel(config=SDEConfig(dt=1.0, t_max=5.0), random_seed=42)

    def on_progress(current_step: int, total_steps: int) -> None:
        assert total_steps == 5
        progress_steps.append(current_step)

    def cancel_callback() -> None:
        if len(progress_steps) >= 2:
            raise RuntimeCancelledError("stop sde")

    with pytest.raises(RuntimeCancelledError):
        model.simulate(
            _model_params(),
            progress_callback=on_progress,
            cancel_callback=cancel_callback,
        )

    assert progress_steps == [1, 2]


def test_extended_sde_model_reports_progress_and_supports_mid_run_cancellation() -> None:
    progress_steps: list[int] = []
    params = ParameterSet()
    params.dt = 1.0
    params.t_max = 5.0
    model = ExtendedSDEModel(params=params, rng_seed=42)

    def on_progress(current_step: int, total_steps: int) -> None:
        assert total_steps == 5
        progress_steps.append(current_step)

    def cancel_callback() -> None:
        if len(progress_steps) >= 2:
            raise RuntimeCancelledError("stop extended")

    with pytest.raises(RuntimeCancelledError):
        model.simulate(
            ExtendedSDEState(P=100.0),
            progress_callback=on_progress,
            cancel_callback=cancel_callback,
        )

    assert progress_steps == [1, 2]


def test_abm_model_reports_progress_and_supports_mid_run_cancellation() -> None:
    progress_steps: list[int] = []
    config = ABMConfig(
        dt=1.0,
        t_max=3.0,
        initial_stem_cells=2,
        initial_macrophages=1,
        initial_fibroblasts=1,
        grid_resolution=10.0,
        space_size=(20.0, 20.0),
    )
    model = ABMModel(config=config, random_seed=42)

    def on_progress(current_step: int, total_steps: int) -> None:
        assert total_steps == 3
        progress_steps.append(current_step)

    def cancel_callback() -> None:
        if len(progress_steps) >= 1:
            raise RuntimeCancelledError("stop abm")

    with pytest.raises(RuntimeCancelledError):
        model.simulate(
            _model_params(),
            snapshot_interval=1.0,
            progress_callback=on_progress,
            cancel_callback=cancel_callback,
        )

    assert progress_steps == [1]
