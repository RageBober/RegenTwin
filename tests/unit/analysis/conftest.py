"""Фикстуры для тестирования модуля валидации моделей.

Синтетические данные для ExtendedSDETrajectory, TimeSeriesData,
MonteCarloResults, SobolResult, MorrisResult, EstimationResult,
ValidationConfig и PhaseBreakpoint.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.validation import PhaseBreakpoint, ValidationConfig
from src.core.extended_sde import ExtendedSDEState, ExtendedSDETrajectory
from src.core.monte_carlo import MonteCarloConfig, MonteCarloResults, TrajectoryResult
from src.core.parameter_estimation import (
    ConvergenceDiagnostics,
    EstimationConfig,
    EstimationResult,
)
from src.core.parameters import ParameterSet
from src.core.sde_model import SDEConfig, SDEState, SDETrajectory
from src.core.sensitivity_analysis import MorrisResult, SobolResult
from src.data.dataset_loader import TimeSeriesData

# ── ExtendedSDETrajectory ──────────────────────────────────────────────────


@pytest.fixture
def mock_extended_trajectory() -> ExtendedSDETrajectory:
    """20-переменная траектория: 100 шагов, реалистичная динамика 0-720 ч."""
    n_steps = 100
    times = np.linspace(0, 720, n_steps)

    states: list[ExtendedSDEState] = []
    for i, t in enumerate(times):
        frac = i / (n_steps - 1)
        states.append(
            ExtendedSDEState(
                P=max(0, 500 * (1 - frac * 2)),
                Ne=max(0, 200 * np.exp(-frac * 3) * np.sin(frac * 5 + 0.5)),
                M1=100 * np.exp(-frac * 2),
                M2=150 * (1 - np.exp(-frac * 3)),
                F=300 * (1 - np.exp(-frac * 2)),
                Mf=50 * frac * np.exp(-frac),
                E=80 * (1 - np.exp(-frac * 1.5)),
                S=40 * np.exp(-frac * 0.5),
                C_TNF=10 * np.exp(-frac * 3),
                C_IL10=5 * (1 - np.exp(-frac * 2)),
                C_PDGF=3 * np.exp(-frac),
                C_VEGF=4 * (1 - np.exp(-frac * 1.5)),
                C_TGFb=2 * (1 + frac),
                C_MCP1=6 * np.exp(-frac * 2),
                C_IL8=8 * np.exp(-frac * 3),
                rho_collagen=min(1.0, 0.1 + 0.9 * frac),
                C_MMP=2 * np.exp(-frac),
                rho_fibrin=max(0, 0.8 * (1 - frac * 1.5)),
                D=5 * np.exp(-frac * 4),
                O2=80 + 20 * frac,
                t=t,
            )
        )

    return ExtendedSDETrajectory(
        times=times,
        states=states,
        params=ParameterSet(),
    )


# ── TimeSeriesData ─────────────────────────────────────────────────────────


@pytest.fixture
def mock_time_series_data(
    mock_extended_trajectory: ExtendedSDETrajectory,
) -> TimeSeriesData:
    """Синтетические наблюдения: F, M1, M2, Ne на 20 временных точках.

    Создаются из траектории + небольшой шум (seed=42).
    """
    rng = np.random.default_rng(42)
    n_obs = 20
    obs_times = np.linspace(0, 720, n_obs)

    values: dict[str, np.ndarray] = {}
    for var in ["F", "M1", "M2", "Ne"]:
        clean = mock_extended_trajectory.get_variable(var)
        # Интерполяция на obs_times
        clean_interp = np.interp(obs_times, mock_extended_trajectory.times, clean)
        # Добавить шум ≤ 5% от среднего
        noise_scale = 0.05 * max(float(np.mean(np.abs(clean_interp))), 1.0)
        values[var] = np.maximum(0.0, clean_interp + rng.normal(0, noise_scale, n_obs))

    return TimeSeriesData(
        time_points=obs_times,
        values=values,
        units=dict.fromkeys(values, "cells/mm²"),
    )


# ── MonteCarloResults ──────────────────────────────────────────────────────


@pytest.fixture
def mock_mc_results(mock_extended_trajectory: ExtendedSDETrajectory) -> MonteCarloResults:
    """20 траекторий MC с заполненными variable_quantiles для F, M1, M2, Ne."""
    n_traj = 20
    n_steps = 100
    times = np.linspace(0, 30, n_steps)
    sde_config = SDEConfig(dt=0.01, t_max=30.0)

    trajectories: list[TrajectoryResult] = []
    all_N = np.zeros((n_traj, n_steps))
    all_C = np.zeros((n_traj, n_steps))

    for i in range(n_traj):
        rng = np.random.default_rng(42 + i)
        n_values = 5000 + 15000 * (1 - np.exp(-times / 10)) + rng.normal(0, 500, n_steps)
        c_values = 5 * np.exp(-times / 15) + rng.normal(0, 0.2, n_steps)
        n_values = np.maximum(n_values, 0)
        c_values = np.maximum(c_values, 0)
        all_N[i] = n_values
        all_C[i] = c_values

        sde_traj = SDETrajectory(
            times=times.copy(),
            N_values=n_values.copy(),
            C_values=c_values.copy(),
            therapy_markers={"prp": np.zeros(n_steps, dtype=bool)},
            config=sde_config,
            initial_state=SDEState(t=0.0, N=float(n_values[0]), C=float(c_values[0])),
        )
        trajectories.append(
            TrajectoryResult(
                trajectory_id=i,
                random_seed=42 + i,
                sde_trajectory=sde_traj,
                final_N=float(n_values[-1]),
                final_C=float(c_values[-1]),
                max_N=float(n_values.max()),
                growth_rate=float((n_values[-1] - n_values[0]) / max(n_values[0], 1.0)),
                success=True,
                computation_time=0.1,
            )
        )

    quantiles_list = [0.05, 0.25, 0.5, 0.75, 0.95]

    # Synthetic variable arrays для 20-var extended
    traj_times = mock_extended_trajectory.times
    var_quantiles: dict[str, dict[float, np.ndarray]] = {}
    for var in ["F", "M1", "M2", "Ne"]:
        base = mock_extended_trajectory.get_variable(var)
        rng_v = np.random.default_rng(99)
        ensemble = np.array(
            [base + rng_v.normal(0, 0.1 * np.abs(base).mean() + 1, len(base)) for _ in range(20)]
        )
        var_quantiles[var] = {q: np.quantile(ensemble, q, axis=0) for q in quantiles_list}

    return MonteCarloResults(
        trajectories=trajectories,
        config=MonteCarloConfig(n_trajectories=n_traj, sde_config=sde_config),
        times=traj_times,
        mean_N=np.mean(all_N, axis=0),
        std_N=np.std(all_N, axis=0),
        mean_C=np.mean(all_C, axis=0),
        std_C=np.std(all_C, axis=0),
        quantiles_N={q: np.quantile(all_N, q, axis=0) for q in quantiles_list},
        quantiles_C={q: np.quantile(all_C, q, axis=0) for q in quantiles_list},
        variable_quantiles=var_quantiles,
        n_successful=n_traj,
        n_failed=0,
        total_computation_time=2.0,
    )


# ── SobolResult / MorrisResult ─────────────────────────────────────────────


@pytest.fixture
def mock_sobol_result() -> SobolResult:
    """Sobol result: 8 параметров."""
    names = ["r_F", "r_M1", "K_F", "d_Ne", "alpha_PDGF", "beta_TNF", "gamma_IL10", "sigma_noise"]
    rng = np.random.default_rng(42)
    s1 = np.array([0.35, 0.25, 0.15, 0.10, 0.05, 0.04, 0.03, 0.01])
    st = np.array([0.45, 0.30, 0.20, 0.15, 0.08, 0.06, 0.04, 0.02])
    return SobolResult(
        parameter_names=names,
        S1=s1,
        ST=st,
        S1_conf=rng.uniform(0.01, 0.05, len(names)),
        ST_conf=rng.uniform(0.01, 0.05, len(names)),
        output_variable="F",
        n_samples=1024,
        n_model_runs=10240,
        elapsed_seconds=42.0,
    )


@pytest.fixture
def mock_morris_result() -> MorrisResult:
    """Morris result: 8 параметров (те же, что в Sobol)."""
    names = ["r_F", "r_M1", "K_F", "d_Ne", "alpha_PDGF", "beta_TNF", "gamma_IL10", "sigma_noise"]
    mu_star = np.array([5.0, 3.5, 2.0, 1.5, 0.4, 0.3, 0.2, 0.15])
    sigma = np.array([3.0, 4.0, 1.0, 0.5, 0.3, 0.25, 0.15, 0.1])
    mu = np.array([4.5, -2.0, 1.8, 1.2, 0.3, -0.2, 0.1, 0.1])
    mu_star_conf = np.array([0.5, 0.4, 0.3, 0.2, 0.1, 0.08, 0.05, 0.04])
    return MorrisResult(
        parameter_names=names,
        mu=mu,
        mu_star=mu_star,
        sigma=sigma,
        mu_star_conf=mu_star_conf,
        output_variable="F",
        n_trajectories=50,
        n_levels=4,
        n_model_runs=450,
        elapsed_seconds=15.0,
    )


# ── EstimationResult ───────────────────────────────────────────────────────


@pytest.fixture
def mock_estimation_result() -> EstimationResult:
    """Bayesian estimation result: 3 параметра, без inference_data."""
    rng = np.random.default_rng(42)
    n_samples = 4000

    posterior_samples = {
        "r_F": rng.normal(0.05, 0.01, n_samples),
        "r_M1": rng.normal(0.03, 0.005, n_samples),
        "K_F": rng.normal(5000, 500, n_samples),
    }
    diagnostics = ConvergenceDiagnostics(
        rhat={"r_F": 1.01, "r_M1": 1.02, "K_F": 1.03},
        ess_bulk={"r_F": 3500.0, "r_M1": 2800.0, "K_F": 2000.0},
        ess_tail={"r_F": 2500.0, "r_M1": 2000.0, "K_F": 1500.0},
        converged=True,
        warnings=[],
    )
    return EstimationResult(
        method="bayesian_pymc",
        point_estimates={"r_F": 0.05, "r_M1": 0.03, "K_F": 5000.0},
        ci_lower={"r_F": 0.03, "r_M1": 0.02, "K_F": 4100.0},
        ci_upper={"r_F": 0.07, "r_M1": 0.04, "K_F": 5900.0},
        posterior_samples=posterior_samples,
        diagnostics=diagnostics,
        inference_data=None,  # без ArviZ InferenceData → MC fallback path
        config=EstimationConfig(
            n_chains=4,
            n_samples=1000,
            observed_variables=["F", "M1", "M2"],
        ),
        n_observations=100,
        n_estimated_params=3,
        elapsed_seconds=60.0,
    )


# ── ValidationConfig ───────────────────────────────────────────────────────


@pytest.fixture
def mock_validation_config() -> ValidationConfig:
    """Стандартный конфиг с dtw_variables=[F, M1, M2, Ne]."""
    return ValidationConfig(
        run_dtw_crps=True,
        run_ppc=True,
        run_phase_timing=True,
        run_sensitivity_ranking=True,
        dtw_variables=["F", "M1", "M2", "Ne"],
        ppc_variables=["F", "M1", "M2"],
        hdi_prob=0.95,
    )


# ── PhaseBreakpoint ────────────────────────────────────────────────────────


@pytest.fixture
def mock_observed_breakpoints() -> list[PhaseBreakpoint]:
    """3 точки разрыва для сравнения с обнаруженными."""
    return [
        PhaseBreakpoint(
            time_hours=6.0, phase_before="hemostasis", phase_after="inflammation", confidence=0.9
        ),
        PhaseBreakpoint(
            time_hours=96.0,
            phase_before="inflammation",
            phase_after="proliferation",
            confidence=0.85,
        ),
        PhaseBreakpoint(
            time_hours=504.0, phase_before="proliferation", phase_after="remodeling", confidence=0.8
        ),
    ]
