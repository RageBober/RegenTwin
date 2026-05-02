"""Фикстуры для performance/benchmark тестов.

Все генераторы используют фиксированный seed для воспроизводимости.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.abm_model import ABMConfig
from src.core.extended_sde import ExtendedSDEState
from src.core.parameters import ParameterSet
from src.data.parameter_extraction import ModelParameters

BENCH_SEED: int = 42


@pytest.fixture(scope="session")
def small_sde_params() -> ParameterSet:
    """ParameterSet с укороченным t_max для быстрых SDE-бенчей."""
    p = ParameterSet()
    p.t_max = 24.0  # 1 день, dt=0.01 → 2400 шагов
    return p


@pytest.fixture(scope="session")
def large_sde_params() -> ParameterSet:
    """ParameterSet с t_max=72ч (3 дня) для нагруженного SDE-бенча."""
    p = ParameterSet()
    p.t_max = 72.0  # 3 дня, dt=0.01 → 7200 шагов
    return p


@pytest.fixture(scope="session")
def initial_sde_state() -> ExtendedSDEState:
    """Начальное состояние 20-переменной SDE с биологически осмысленными значениями."""
    return ExtendedSDEState(
        P=200.0,
        Ne=100.0,
        M1=50.0,
        M2=10.0,
        F=80.0,
        Mf=5.0,
        E=20.0,
        S=10.0,
        C_TNF=5.0,
        C_IL10=1.0,
        C_PDGF=2.0,
        C_VEGF=2.0,
        C_TGFb=1.0,
        C_MCP1=3.0,
        C_IL8=4.0,
        rho_collagen=0.1,
        C_MMP=0.5,
        rho_fibrin=0.5,
        D=1.0,
        O2=40.0,
        t=0.0,
    )


@pytest.fixture(scope="session")
def model_params() -> ModelParameters:
    """ModelParameters для запуска ABM."""
    return ModelParameters(
        n0=10000.0,
        stem_cell_fraction=0.05,
        macrophage_fraction=0.03,
        apoptotic_fraction=0.02,
        c0=10.0,
        inflammation_level=0.5,
    )


def _abm_config(t_max_hours: float, max_agents: int) -> ABMConfig:
    cfg = ABMConfig()
    cfg.t_max = t_max_hours
    cfg.max_agents = max_agents
    return cfg


@pytest.fixture(scope="session")
def abm_config_small() -> ABMConfig:
    """ABM config: 100 max agents, 24h."""
    return _abm_config(t_max_hours=24.0, max_agents=100)


@pytest.fixture(scope="session")
def abm_config_medium() -> ABMConfig:
    """ABM config: 500 max agents, 24h."""
    return _abm_config(t_max_hours=24.0, max_agents=500)


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Фиксированный RNG для воспроизводимости параметрических выборок."""
    return np.random.default_rng(BENCH_SEED)
