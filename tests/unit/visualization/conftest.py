"""Фикстуры для тестирования модуля визуализации.

Синтетические данные для ExtendedSDETrajectory, ABMTrajectory,
MonteCarloResults и PhaseIndicators.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.abm_model import ABMConfig, ABMSnapshot, ABMTrajectory, AgentState
from src.core.extended_sde import ExtendedSDEState, ExtendedSDETrajectory
from src.core.monte_carlo import (
    MonteCarloConfig,
    MonteCarloResults,
    TrajectoryResult,
)
from src.core.parameters import ParameterSet
from src.core.sde_model import SDEConfig, SDETrajectory
from src.core.wound_phases import PhaseIndicators, WoundPhase


@pytest.fixture
def mock_extended_trajectory() -> ExtendedSDETrajectory:
    """20-переменная траектория: 100 шагов, реалистичная динамика."""
    n_steps = 100
    times = np.linspace(0, 720, n_steps)  # 0-720 часов (30 дней)

    states: list[ExtendedSDEState] = []
    for i, t in enumerate(times):
        frac = i / (n_steps - 1)
        states.append(ExtendedSDEState(
            # Клетки: тромбоциты падают, фибробласты растут
            P=max(0, 500 * (1 - frac * 2)),
            Ne=max(0, 200 * np.exp(-frac * 3) * np.sin(frac * 5 + 0.5)),
            M1=100 * np.exp(-frac * 2),
            M2=150 * (1 - np.exp(-frac * 3)),
            F=300 * (1 - np.exp(-frac * 2)),
            Mf=50 * frac * np.exp(-frac),
            E=80 * (1 - np.exp(-frac * 1.5)),
            S=40 * np.exp(-frac * 0.5),
            # Цитокины: TNF падает, IL-10 растёт
            C_TNF=10 * np.exp(-frac * 3),
            C_IL10=5 * (1 - np.exp(-frac * 2)),
            C_PDGF=3 * np.exp(-frac),
            C_VEGF=4 * (1 - np.exp(-frac * 1.5)),
            C_TGFb=2 * (1 + frac),
            C_MCP1=6 * np.exp(-frac * 2),
            C_IL8=8 * np.exp(-frac * 3),
            # ECM: коллаген растёт, фибрин падает
            rho_collagen=min(1.0, 0.1 + 0.9 * frac),
            C_MMP=2 * np.exp(-frac),
            rho_fibrin=max(0, 0.8 * (1 - frac * 1.5)),
            # Вспомогательные
            D=5 * np.exp(-frac * 4),
            O2=80 + 20 * frac,
            t=t,
        ))

    return ExtendedSDETrajectory(
        times=times,
        states=states,
        params=ParameterSet(),
    )


def _make_agents(n: int, agent_type: str, rng: np.random.Generator) -> list[AgentState]:
    """Создание N агентов данного типа со случайными координатами."""
    return [
        AgentState(
            agent_id=i,
            agent_type=agent_type,
            x=rng.uniform(0, 100),
            y=rng.uniform(0, 100),
            age=rng.uniform(0, 48),
            division_count=rng.integers(0, 3),
            energy=rng.uniform(0.3, 1.0),
            alive=True,
        )
        for i in range(n)
    ]


@pytest.fixture
def mock_abm_snapshot() -> ABMSnapshot:
    """Один ABM snapshot: 50 агентов, 10x10 поля."""
    rng = np.random.default_rng(42)

    agents: list[AgentState] = []
    agent_id = 0
    for atype, count in [("stem", 15), ("macro", 15), ("fibro", 20)]:
        for ag in _make_agents(count, atype, rng):
            agents.append(AgentState(
                agent_id=agent_id,
                agent_type=ag.agent_type,
                x=ag.x, y=ag.y,
                age=ag.age,
                division_count=ag.division_count,
                energy=ag.energy,
                alive=True,
            ))
            agent_id += 1

    return ABMSnapshot(
        t=168.0,  # 1 неделя
        agents=agents,
        cytokine_field=rng.uniform(0, 5, size=(10, 10)),
        ecm_field=rng.uniform(0, 1, size=(10, 10)),
    )


@pytest.fixture
def mock_abm_trajectory(mock_abm_snapshot: ABMSnapshot) -> ABMTrajectory:
    """ABM траектория: 10 снимков."""
    rng = np.random.default_rng(42)
    snapshots: list[ABMSnapshot] = []
    for i in range(10):
        t = i * 24.0  # каждые 24 часа
        n_agents = 50 + i * 5
        agents = []
        agent_id = 0
        for atype, base_count in [("stem", 15), ("macro", 15), ("fibro", 20 + i * 2)]:
            for ag in _make_agents(base_count, atype, rng):
                agents.append(AgentState(
                    agent_id=agent_id,
                    agent_type=ag.agent_type,
                    x=ag.x, y=ag.y,
                    age=ag.age,
                    division_count=ag.division_count,
                    energy=ag.energy,
                    alive=True,
                ))
                agent_id += 1

        snapshots.append(ABMSnapshot(
            t=t,
            agents=agents,
            cytokine_field=rng.uniform(0, 5, size=(10, 10)),
            ecm_field=rng.uniform(0, 1, size=(10, 10)),
        ))

    return ABMTrajectory(snapshots=snapshots, config=ABMConfig())


@pytest.fixture
def mock_mc_results() -> MonteCarloResults:
    """Monte Carlo результаты: 20 траекторий."""
    n_traj = 20
    n_steps = 100
    times = np.linspace(0, 30, n_steps)  # 30 дней

    trajectories: list[TrajectoryResult] = []
    all_N = np.zeros((n_traj, n_steps))
    all_C = np.zeros((n_traj, n_steps))

    sde_config = SDEConfig(dt=0.01, t_max=30.0)

    for i in range(n_traj):
        rng = np.random.default_rng(42 + i)
        noise = rng.normal(0, 500, n_steps).cumsum() * 0.01
        n_values = 5000 + 15000 * (1 - np.exp(-times / 10)) + noise
        n_values = np.maximum(n_values, 0)
        c_values = 5 * np.exp(-times / 15) + rng.normal(0, 0.2, n_steps)
        c_values = np.maximum(c_values, 0)

        all_N[i] = n_values
        all_C[i] = c_values

        sde_traj = SDETrajectory(
            times=times.copy(),
            N_values=n_values.copy(),
            C_values=c_values.copy(),
            therapy_markers={"prp": np.zeros(n_steps, dtype=bool)},
            config=sde_config,
            initial_state=None,
        )

        trajectories.append(TrajectoryResult(
            trajectory_id=i,
            random_seed=42 + i,
            sde_trajectory=sde_traj,
            final_N=float(n_values[-1]),
            final_C=float(c_values[-1]),
            max_N=float(n_values.max()),
            growth_rate=float((n_values[-1] - n_values[0]) / n_values[0]),
            success=True,
            computation_time=0.1,
        ))

    mean_N = np.mean(all_N, axis=0)
    std_N = np.std(all_N, axis=0)
    mean_C = np.mean(all_C, axis=0)
    std_C = np.std(all_C, axis=0)

    quantiles_N = {
        q: np.quantile(all_N, q, axis=0)
        for q in [0.05, 0.25, 0.5, 0.75, 0.95]
    }
    quantiles_C = {
        q: np.quantile(all_C, q, axis=0)
        for q in [0.05, 0.25, 0.5, 0.75, 0.95]
    }

    return MonteCarloResults(
        trajectories=trajectories,
        config=MonteCarloConfig(n_trajectories=n_traj, sde_config=sde_config),
        times=times,
        mean_N=mean_N,
        std_N=std_N,
        mean_C=mean_C,
        std_C=std_C,
        quantiles_N=quantiles_N,
        quantiles_C=quantiles_C,
        n_successful=n_traj,
        n_failed=0,
        total_computation_time=2.0,
    )


@pytest.fixture
def mock_phase_indicators() -> list[PhaseIndicators]:
    """Фазовые индикаторы для 100 временных шагов: H → I → P → R."""
    indicators: list[PhaseIndicators] = []
    n_steps = 100

    for i in range(n_steps):
        frac = i / (n_steps - 1)

        if frac < 0.05:
            phase = WoundPhase.HEMOSTASIS
            confidence = 0.9
            cells = ["P", "Ne"]
            cytos = ["PDGF", "TGFb"]
        elif frac < 0.25:
            phase = WoundPhase.INFLAMMATION
            confidence = 0.85
            cells = ["Ne", "M1", "P"]
            cytos = ["TNF", "IL8", "MCP1"]
        elif frac < 0.70:
            phase = WoundPhase.PROLIFERATION
            confidence = 0.8
            cells = ["F", "M2", "E"]
            cytos = ["VEGF", "IL10", "PDGF"]
        else:
            phase = WoundPhase.REMODELING
            confidence = 0.75
            cells = ["F", "Mf", "E"]
            cytos = ["TGFb", "IL10", "VEGF"]

        indicators.append(PhaseIndicators(
            phase=phase,
            confidence=confidence,
            dominant_cells=cells,
            dominant_cytokines=cytos,
            phase_progress=frac,
        ))

    return indicators
