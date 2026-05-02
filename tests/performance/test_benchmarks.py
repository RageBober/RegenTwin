"""Бенчмарки тяжёлых модулей RegenTwin.

Запуск:
    uv run pytest tests/performance/test_benchmarks.py --benchmark-only

В CI ИСКЛЮЧАЮТСЯ через `-m "not benchmark"` — числа нестабильны на shared runner.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.abm_model import ABMModel
from src.core.extended_sde import ExtendedSDEModel
from src.core.monte_carlo import MonteCarloConfig, MonteCarloSimulator
from src.data.parameter_extraction import ModelParameters

pytestmark = pytest.mark.benchmark


def _default_model_params() -> ModelParameters:
    return ModelParameters(
        n0=10000.0,
        stem_cell_fraction=0.05,
        macrophage_fraction=0.03,
        apoptotic_fraction=0.02,
        c0=10.0,
        inflammation_level=0.5,
    )


@pytest.mark.benchmark(group="sde-small", min_rounds=3, warmup=False)
def test_extended_sde_small(benchmark, small_sde_params, initial_sde_state):
    """Extended SDE: t_max=24ч (~2400 шагов)."""

    def run() -> None:
        model = ExtendedSDEModel(params=small_sde_params, rng_seed=42)
        model.simulate(initial_sde_state)

    benchmark(run)


@pytest.mark.benchmark(group="sde-large", min_rounds=3, warmup=False)
def test_extended_sde_large(benchmark, large_sde_params, initial_sde_state):
    """Extended SDE: t_max=72ч (~7200 шагов)."""

    def run() -> None:
        model = ExtendedSDEModel(params=large_sde_params, rng_seed=42)
        model.simulate(initial_sde_state)

    benchmark(run)


@pytest.mark.benchmark(group="abm-small", min_rounds=2, warmup=False)
def test_abm_small(benchmark, abm_config_small, model_params):
    """ABM: 100 макс. агентов, t_max=24ч."""

    def run() -> None:
        model = ABMModel(config=abm_config_small, random_seed=42)
        model.simulate(model_params, snapshot_interval=24.0)

    benchmark(run)


@pytest.mark.benchmark(group="abm-medium", min_rounds=2, warmup=False)
def test_abm_medium(benchmark, abm_config_medium, model_params):
    """ABM: 500 макс. агентов, t_max=24ч."""

    def run() -> None:
        model = ABMModel(config=abm_config_medium, random_seed=42)
        model.simulate(model_params, snapshot_interval=24.0)

    benchmark(run)


@pytest.mark.benchmark(group="mc-serial", min_rounds=2, warmup=False)
def test_mc_serial_4(benchmark, small_sde_params, initial_sde_state):
    """Monte Carlo последовательно, 4 траектории extended-SDE."""
    cfg = MonteCarloConfig(
        n_trajectories=4,
        model_type="extended",
        extended_params=small_sde_params,
        extended_initial_state=initial_sde_state,
        n_jobs=1,
        use_multiprocessing=False,
        base_seed=42,
    )

    def run() -> None:
        simulator = MonteCarloSimulator(config=cfg)
        simulator.run(_default_model_params())

    benchmark(run)


@pytest.mark.benchmark(group="mc-parallel", min_rounds=2, warmup=False)
def test_mc_parallel_4(benchmark, small_sde_params, initial_sde_state):
    """Monte Carlo параллельно (n_jobs=cpu-1), 4 траектории extended-SDE."""
    import os

    n_jobs = max(1, (os.cpu_count() or 2) - 1)
    cfg = MonteCarloConfig(
        n_trajectories=4,
        model_type="extended",
        extended_params=small_sde_params,
        extended_initial_state=initial_sde_state,
        n_jobs=n_jobs,
        use_multiprocessing=True,
        base_seed=42,
    )

    def run() -> None:
        simulator = MonteCarloSimulator(config=cfg)
        simulator.run(_default_model_params())

    benchmark(run)


@pytest.mark.benchmark(group="sensitivity-sobol", min_rounds=1, warmup=False)
def test_sensitivity_sobol_small(benchmark):
    """Sobol sensitivity: маленькая выборка для проверки масштабирования."""
    from SALib.analyze import sobol as sobol_analyze
    from SALib.sample import saltelli

    problem = {
        "num_vars": 4,
        "names": ["x1", "x2", "x3", "x4"],
        "bounds": [[0.0, 1.0]] * 4,
    }

    def ishigami(samples: np.ndarray) -> np.ndarray:
        a, b = 7.0, 0.1
        return (
            np.sin(samples[:, 0])
            + a * np.sin(samples[:, 1]) ** 2
            + b * samples[:, 2] ** 4 * np.sin(samples[:, 0])
        )

    def run() -> None:
        # N=64 → 64 * (2*4 + 2) = 640 evaluations
        param_values = saltelli.sample(problem, N=64, calc_second_order=True)
        outputs = ishigami(param_values)
        sobol_analyze.analyze(problem, outputs, calc_second_order=True, print_to_console=False)

    benchmark(run)
