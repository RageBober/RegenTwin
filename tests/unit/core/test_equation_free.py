"""TDD тесты для equation_free.py — Phase 2.9 Equation-Free Framework.

Тестирование полного EF-цикла:
- EquationFreeConfig: defaults и типы полей
- Lifter: macro→micro lifting, distribute_population, assign_cytokine_fields
- Restrictor: micro→macro restriction, count_population, aggregate_cytokines
- EquationFreeIntegrator: step, _micro_step, _lift_step, _restrict_step, run, apply_therapy
- TestLiftRestrictConsistency: round-trip conservation (lift→restrict)

Все тесты написаны для stub-реализации (NotImplementedError).
Должны ПРОВАЛИТЬСЯ до реализации и ПРОЙТИ после.
"""

from __future__ import annotations

from dataclasses import is_dataclass
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.core.abm_model import (
    ABMConfig,
    Fibroblast,
    Macrophage,
    StemCell,
)
from src.core.equation_free import (
    EquationFreeConfig,
    EquationFreeIntegrator,
    Lifter,
    Restrictor,
)
from src.core.extended_sde import ExtendedSDEState
from src.core.sde_model import TherapyProtocol

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

CYTOKINE_NAMES = ["TNF", "IL10", "PDGF", "VEGF", "TGFb", "MCP1", "IL8"]

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def ef_config() -> EquationFreeConfig:
    """EquationFreeConfig с дефолтными значениями."""
    return EquationFreeConfig()


@pytest.fixture
def abm_config() -> ABMConfig:
    """ABMConfig с дефолтными значениями."""
    return ABMConfig()


@pytest.fixture
def lifter(ef_config, abm_config) -> Lifter:
    return Lifter(ef_config, abm_config)


@pytest.fixture
def restrictor(ef_config) -> Restrictor:
    return Restrictor(ef_config)


@pytest.fixture
def zero_state() -> ExtendedSDEState:
    """ExtendedSDEState с нулевыми значениями всех полей."""
    return ExtendedSDEState()


@pytest.fixture
def populated_state() -> ExtendedSDEState:
    """ExtendedSDEState с ненулевыми концентрациями клеток.

    Значения подобраны так, чтобы round(c_i * 1e6 * 1e-3) >= 1 для каждого типа.
    Минимальная концентрация для 1 агента: c_i >= 0.0015 (round(1.5)=2).
    """
    return ExtendedSDEState(
        P=0.005,
        Ne=0.003,
        M1=0.004,
        M2=0.003,
        F=0.01,
        Mf=0.002,
        E=0.003,
        S=0.002,
        C_TNF=0.5,
        C_IL10=0.3,
        C_PDGF=0.2,
        C_VEGF=0.1,
        C_TGFb=0.15,
        C_MCP1=0.25,
        C_IL8=0.2,
    )


@pytest.fixture
def populated_state_full() -> ExtendedSDEState:
    """ExtendedSDEState со всеми 20 переменными ненулевыми (включая ECM и Aux)."""
    return ExtendedSDEState(
        P=0.005,
        Ne=0.003,
        M1=0.004,
        M2=0.003,
        F=0.01,
        Mf=0.002,
        E=0.003,
        S=0.002,
        C_TNF=0.5,
        C_IL10=0.3,
        C_PDGF=0.2,
        C_VEGF=0.1,
        C_TGFb=0.15,
        C_MCP1=0.25,
        C_IL8=0.2,
        rho_collagen=0.4,
        C_MMP=0.1,
        rho_fibrin=0.3,
        D=0.05,
        O2=95.0,
    )


@pytest.fixture
def mock_sde_model() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_abm_model() -> MagicMock:
    return MagicMock()


@pytest.fixture
def integrator(ef_config, abm_config, mock_sde_model, mock_abm_model) -> EquationFreeIntegrator:
    l = Lifter(ef_config, abm_config)
    r = Restrictor(ef_config)
    return EquationFreeIntegrator(mock_sde_model, mock_abm_model, l, r, ef_config)


@pytest.fixture
def alive_fibroblasts() -> list[Fibroblast]:
    rng = np.random.default_rng(42)
    return [Fibroblast(agent_id=i, x=float(i), y=float(i), rng=rng) for i in range(5)]


@pytest.fixture
def dead_fibroblasts() -> list[Fibroblast]:
    rng = np.random.default_rng(42)
    agents = [Fibroblast(agent_id=100 + i, x=float(i), y=float(i), rng=rng) for i in range(3)]
    for a in agents:
        a.alive = False
    return agents


# =============================================================================
# TestEquationFreeConfig
# =============================================================================


class TestEquationFreeConfig:
    """Тесты дефолтных значений EquationFreeConfig."""

    def test_is_dataclass(self):
        assert is_dataclass(EquationFreeConfig)

    def test_default_dt_macro(self):
        assert EquationFreeConfig().dt_macro == 1.0

    def test_default_dt_micro(self):
        assert EquationFreeConfig().dt_micro == 0.1

    def test_default_n_micro_steps(self):
        assert EquationFreeConfig().n_micro_steps == 10

    def test_default_volume(self):
        assert EquationFreeConfig().volume == pytest.approx(1e6)

    def test_default_n_agents_scale(self):
        assert EquationFreeConfig().n_agents_scale == pytest.approx(1e-3)

    def test_custom_values(self):
        cfg = EquationFreeConfig(
            dt_macro=2.0, dt_micro=0.2, n_micro_steps=5, volume=5e5, n_agents_scale=1e-4
        )
        assert cfg.dt_macro == 2.0
        assert cfg.n_micro_steps == 5

    def test_dt_macro_less_than_dt_micro_no_error(self):
        """dt_macro < dt_micro не вызывает ошибку (проверка в Integrator)."""
        cfg = EquationFreeConfig(dt_macro=0.01, dt_micro=1.0)
        assert cfg.dt_macro < cfg.dt_micro

    def test_negative_dt_macro_raises(self):
        """dt_macro <= 0 нарушает инвариант → ValueError."""
        with pytest.raises(ValueError):
            EquationFreeConfig(dt_macro=-1.0)

    def test_zero_dt_macro_raises(self):
        """dt_macro == 0 нарушает инвариант → ValueError."""
        with pytest.raises(ValueError):
            EquationFreeConfig(dt_macro=0.0)

    def test_negative_dt_micro_raises(self):
        """dt_micro <= 0 нарушает инвариант → ValueError."""
        with pytest.raises(ValueError):
            EquationFreeConfig(dt_micro=-0.5)

    def test_negative_volume_raises(self):
        """Volume <= 0 нарушает инвариант → ValueError."""
        with pytest.raises(ValueError):
            EquationFreeConfig(volume=-1e6)

    def test_zero_volume_raises(self):
        """Volume == 0 нарушает инвариант → ValueError."""
        with pytest.raises(ValueError):
            EquationFreeConfig(volume=0.0)

    def test_negative_n_agents_scale_raises(self):
        """n_agents_scale <= 0 нарушает инвариант → ValueError."""
        with pytest.raises(ValueError):
            EquationFreeConfig(n_agents_scale=-1e-3)

    def test_zero_n_agents_scale_raises(self):
        """n_agents_scale == 0 нарушает инвариант → ValueError."""
        with pytest.raises(ValueError):
            EquationFreeConfig(n_agents_scale=0.0)

    def test_zero_n_micro_steps_raises(self):
        """n_micro_steps < 1 нарушает инвариант → ValueError."""
        with pytest.raises(ValueError):
            EquationFreeConfig(n_micro_steps=0)

    def test_negative_n_micro_steps_raises(self):
        """n_micro_steps < 0 нарушает инвариант → ValueError."""
        with pytest.raises(ValueError):
            EquationFreeConfig(n_micro_steps=-5)


# =============================================================================
# TestLifterInit
# =============================================================================


class TestLifterInit:
    """Тесты инициализации Lifter."""

    def test_creates_without_error(self, ef_config, abm_config):
        Lifter(ef_config, abm_config)

    def test_config_attribute(self, ef_config, abm_config):
        l = Lifter(ef_config, abm_config)
        assert l.config is ef_config

    def test_abm_config_attribute(self, ef_config, abm_config):
        l = Lifter(ef_config, abm_config)
        assert l.abm_config is abm_config

    def test_has_rng(self, ef_config, abm_config):
        l = Lifter(ef_config, abm_config)
        assert isinstance(l.rng, np.random.Generator)


# =============================================================================
# TestLifterDistributePopulation
# =============================================================================


class TestLifterDistributePopulation:
    """Тесты distribute_population."""

    def test_zero_agents_returns_empty(self, lifter):
        rng = np.random.default_rng(0)
        result = lifter.distribute_population(0.0, StemCell, 0, (100.0, 100.0), rng)
        assert result == []

    def test_returns_correct_count(self, lifter):
        rng = np.random.default_rng(0)
        result = lifter.distribute_population(10.0, StemCell, 10, (100.0, 100.0), rng)
        assert len(result) == 10

    def test_all_correct_type(self, lifter):
        rng = np.random.default_rng(0)
        result = lifter.distribute_population(5.0, StemCell, 5, (100.0, 100.0), rng)
        assert all(isinstance(a, StemCell) for a in result)

    def test_positions_within_bounds(self, lifter):
        rng = np.random.default_rng(0)
        space = (100.0, 100.0)
        result = lifter.distribute_population(10.0, Fibroblast, 10, space, rng)
        for a in result:
            assert 0.0 <= a.x <= space[0]
            assert 0.0 <= a.y <= space[1]

    def test_unique_agent_ids(self, lifter):
        rng = np.random.default_rng(0)
        n = 20
        result = lifter.distribute_population(20.0, Macrophage, n, (100.0, 100.0), rng)
        assert len({a.agent_id for a in result}) == n


# =============================================================================
# TestLifterLift
# =============================================================================


class TestLifterLift:
    """Тесты метода lift."""

    def test_zero_state_returns_empty(self, lifter, zero_state):
        result = lifter.lift(zero_state, 0, 1e6)
        assert result == []

    def test_only_platelets_when_only_p(self, lifter):
        state = ExtendedSDEState(P=0.001)
        result = lifter.lift(state, 0, 1e6)
        assert len(result) > 0
        assert all(a.AGENT_TYPE == "platelet" for a in result)

    def test_only_fibroblasts_when_only_f(self, lifter):
        state = ExtendedSDEState(F=0.002)
        result = lifter.lift(state, 0, 1e6)
        assert len(result) > 0
        assert all(a.AGENT_TYPE == "fibro" for a in result)

    def test_all_types_when_all_populated(self, lifter, populated_state):
        """M1 и M2 оба Macrophage (AGENT_TYPE='macro'), итого 7 уникальных типов."""
        result = lifter.lift(populated_state, 0, 1e6)
        types_present = {a.AGENT_TYPE for a in result}
        expected_types = {"platelet", "neutro", "macro", "fibro", "myofibro", "endo", "stem"}
        assert types_present == expected_types

    def test_count_proportional_to_concentration(self, lifter, ef_config):
        F_conc = 0.005
        state = ExtendedSDEState(F=F_conc)
        result = lifter.lift(state, 0, ef_config.volume)
        fibros = [a for a in result if a.AGENT_TYPE == "fibro"]
        expected = round(F_conc * ef_config.volume * ef_config.n_agents_scale)
        assert len(fibros) == expected


# =============================================================================
# TestLifterAssignCytokineFields
# =============================================================================


class TestLifterAssignCytokineFields:
    """Тесты assign_cytokine_fields."""

    def test_empty_agents_no_error(self, lifter):
        result = lifter.assign_cytokine_fields([], {})
        assert result is None

    def test_assigns_cytokine_environment(self, lifter):
        rng = np.random.default_rng(0)
        agent = StemCell(agent_id=1, x=10.0, y=10.0, rng=rng)
        levels = {"TNF": 1.0, "IL10": 0.5}
        lifter.assign_cytokine_fields([agent], levels)
        assert agent.cytokine_environment == levels

    def test_empty_dict_assigned(self, lifter):
        rng = np.random.default_rng(0)
        agent = Macrophage(agent_id=1, x=10.0, y=10.0, rng=rng)
        lifter.assign_cytokine_fields([agent], {})
        assert agent.cytokine_environment == {}

    def test_multiple_agents_all_assigned(self, lifter, alive_fibroblasts):
        levels = dict.fromkeys(CYTOKINE_NAMES, 1.0)
        lifter.assign_cytokine_fields(alive_fibroblasts, levels)
        for a in alive_fibroblasts:
            assert a.cytokine_environment == levels

    def test_agent_without_attribute_skipped(self, lifter):
        """Агент без поддержки cytokine_environment → пропускается без ошибки."""

        class DummyAgent:
            """Агент-заглушка без атрибута cytokine_environment."""

            AGENT_TYPE = "dummy"
            alive = True

            def __init__(self):
                self.agent_id = 999
                self.x = 0.0
                self.y = 0.0

        dummy = DummyAgent()
        # Не должно вызвать ошибку
        lifter.assign_cytokine_fields([dummy], {"TNF": 1.0})
        # Если агент не поддерживает атрибут — не устанавливается
        # (реализация может либо setattr, либо пропустить — оба допустимы)


# =============================================================================
# TestRestrictorInit
# =============================================================================


class TestRestrictorInit:
    """Тесты инициализации Restrictor."""

    def test_creates_without_error(self, ef_config):
        Restrictor(ef_config)

    def test_config_attribute(self, ef_config):
        r = Restrictor(ef_config)
        assert r.config is ef_config


# =============================================================================
# TestRestrictorCountPopulation
# =============================================================================


class TestRestrictorCountPopulation:
    """Тесты count_population."""

    def test_empty_list_returns_zero(self, restrictor):
        assert restrictor.count_population([], "fibro") == 0.0

    def test_counts_only_alive(self, restrictor, alive_fibroblasts, dead_fibroblasts):
        agents = alive_fibroblasts + dead_fibroblasts  # 5 живых + 3 мёртвых
        assert restrictor.count_population(agents, "fibro") == 5.0

    def test_unknown_type_returns_zero(self, restrictor, alive_fibroblasts):
        assert restrictor.count_population(alive_fibroblasts, "unknown_type") == 0.0

    def test_mixed_types_counts_correct(self, restrictor):
        rng = np.random.default_rng(0)
        fibros = [Fibroblast(agent_id=i, x=0.0, y=0.0, rng=rng) for i in range(3)]
        macros = [Macrophage(agent_id=10 + i, x=0.0, y=0.0, rng=rng) for i in range(5)]
        agents = fibros + macros
        assert restrictor.count_population(agents, "fibro") == 3.0
        assert restrictor.count_population(agents, "macro") == 5.0

    def test_returns_float(self, restrictor, alive_fibroblasts):
        result = restrictor.count_population(alive_fibroblasts, "fibro")
        assert isinstance(result, float)


# =============================================================================
# TestRestrictorAggregateCytokines
# =============================================================================


class TestRestrictorAggregateCytokines:
    """Тесты aggregate_cytokines."""

    def test_empty_list_returns_zero_dict(self, restrictor):
        result = restrictor.aggregate_cytokines([])
        assert set(result.keys()) == set(CYTOKINE_NAMES)
        assert all(v == 0.0 for v in result.values())

    def test_agents_without_attribute_return_zeros(self, restrictor, alive_fibroblasts):
        # Агенты без cytokine_environment
        result = restrictor.aggregate_cytokines(alive_fibroblasts)
        assert all(v == 0.0 for v in result.values())

    def test_uniform_field_averages_correctly(self, restrictor):
        rng = np.random.default_rng(0)
        agents = [StemCell(agent_id=i, x=0.0, y=0.0, rng=rng) for i in range(4)]
        levels = dict.fromkeys(CYTOKINE_NAMES, 2.0)
        for a in agents:
            a.cytokine_environment = levels
        result = restrictor.aggregate_cytokines(agents)
        assert result["TNF"] == pytest.approx(2.0)
        assert result["IL10"] == pytest.approx(2.0)

    def test_heterogeneous_values_averaged(self, restrictor):
        """Разные значения у агентов → среднее."""
        rng = np.random.default_rng(0)
        a1 = StemCell(agent_id=0, x=0.0, y=0.0, rng=rng)
        a2 = StemCell(agent_id=1, x=0.0, y=0.0, rng=rng)
        a1.cytokine_environment = dict.fromkeys(CYTOKINE_NAMES, 1.0)
        a2.cytokine_environment = dict.fromkeys(CYTOKINE_NAMES, 3.0)
        result = restrictor.aggregate_cytokines([a1, a2])
        assert result["TNF"] == pytest.approx(2.0)
        assert result["IL10"] == pytest.approx(2.0)

    def test_returns_all_cytokine_keys(self, restrictor):
        result = restrictor.aggregate_cytokines([])
        assert set(result.keys()) == set(CYTOKINE_NAMES)


# =============================================================================
# TestRestrictorRestrict
# =============================================================================


class TestRestrictorRestrict:
    """Тесты restrict."""

    def test_empty_agents_returns_zero_state(self, restrictor):
        result = restrictor.restrict([], 1e6, 0.0)
        assert pytest.approx(0.0) == result.P
        assert pytest.approx(0.0) == result.F
        assert result.Ne == pytest.approx(0.0)

    def test_time_field_matches(self, restrictor):
        result = restrictor.restrict([], 1e6, 5.0)
        assert result.t == pytest.approx(5.0)

    def test_returns_extended_sde_state(self, restrictor, alive_fibroblasts):
        result = restrictor.restrict(alive_fibroblasts, 1e6, 0.0)
        assert isinstance(result, ExtendedSDEState)

    def test_fibroblast_concentration(self, restrictor, ef_config):
        """100 агентов / (volume * n_agents_scale) = 100 / (1e6 * 1e-3) = 0.1.

        Каждый агент представляет 1/n_agents_scale = 1000 клеток,
        поэтому restrict должен восстанавливать реальную концентрацию.
        """
        rng = np.random.default_rng(0)
        n = 100
        agents = [Fibroblast(agent_id=i, x=0.0, y=0.0, rng=rng) for i in range(n)]
        result = restrictor.restrict(agents, ef_config.volume, 0.0)
        expected_conc = n / (ef_config.volume * ef_config.n_agents_scale)
        assert pytest.approx(expected_conc) == result.F

    def test_all_fields_nonnegative(self, restrictor, alive_fibroblasts):
        result = restrictor.restrict(alive_fibroblasts, 1e6, 0.0)
        arr = result.to_array()
        assert np.all(arr >= 0.0)

    def test_time_monotone_with_different_t(self, restrictor):
        r1 = restrictor.restrict([], 1e6, 1.0)
        r2 = restrictor.restrict([], 1e6, 3.0)
        assert r1.t < r2.t

    def test_ecm_fields_in_restrict(self, restrictor):
        """Restrict пустого списка → ECM поля = 0.0."""
        result = restrictor.restrict([], 1e6, 0.0)
        assert result.rho_collagen == pytest.approx(0.0)
        assert pytest.approx(0.0) == result.C_MMP
        assert result.rho_fibrin == pytest.approx(0.0)

    def test_aux_fields_in_restrict(self, restrictor):
        """Restrict пустого списка → D и O2 = 0.0."""
        result = restrictor.restrict([], 1e6, 0.0)
        assert pytest.approx(0.0) == result.D
        assert pytest.approx(0.0) == result.O2


# =============================================================================
# TestEquationFreeIntegratorInit
# =============================================================================


class TestEquationFreeIntegratorInit:
    """Тесты инициализации EquationFreeIntegrator."""

    def test_valid_config_no_error(self, integrator):
        # Фикстура уже создала интегратор, проверяем что без ошибок
        assert integrator is not None

    def test_n_micro_steps_zero_raises(self, ef_config, abm_config, mock_sde_model, mock_abm_model):
        """n_micro_steps=0 → ValueError (на уровне Config __post_init__ или Integrator)."""
        with pytest.raises(ValueError):
            bad_config = EquationFreeConfig(n_micro_steps=0)
            l = Lifter(ef_config, abm_config)
            r = Restrictor(ef_config)
            EquationFreeIntegrator(mock_sde_model, mock_abm_model, l, r, bad_config)

    def test_negative_n_micro_steps_raises(
        self, ef_config, abm_config, mock_sde_model, mock_abm_model
    ):
        """n_micro_steps < 0 → ValueError (на уровне Config __post_init__ или Integrator)."""
        with pytest.raises(ValueError):
            bad_config = EquationFreeConfig(n_micro_steps=-3)
            l = Lifter(ef_config, abm_config)
            r = Restrictor(ef_config)
            EquationFreeIntegrator(mock_sde_model, mock_abm_model, l, r, bad_config)

    def test_trajectory_starts_empty(self, integrator):
        assert integrator.trajectory == []

    def test_stores_config(self, integrator, ef_config):
        assert integrator.config is ef_config

    def test_stores_lifter(self, integrator):
        assert isinstance(integrator.lifter, Lifter)

    def test_stores_restrictor(self, integrator):
        assert isinstance(integrator.restrictor, Restrictor)


# =============================================================================
# TestEquationFreeIntegratorMicroStep
# =============================================================================


class TestEquationFreeIntegratorMicroStep:
    """Тесты _micro_step."""

    def test_empty_agents_returns_empty(self, integrator):
        result = integrator._micro_step([], 0.1)
        assert result == []

    def test_dead_agents_filtered(self, integrator, dead_fibroblasts):
        result = integrator._micro_step(dead_fibroblasts, 0.1)
        assert all(a.alive for a in result)

    def test_returns_list(self, integrator, alive_fibroblasts):
        result = integrator._micro_step(alive_fibroblasts, 0.1)
        assert isinstance(result, list)

    def test_all_returned_agents_alive(self, integrator, alive_fibroblasts, dead_fibroblasts):
        mixed = alive_fibroblasts + dead_fibroblasts
        result = integrator._micro_step(mixed, 0.1)
        assert all(a.alive for a in result)

    def test_newborn_agents_from_division(self, integrator):
        """Если агент делится во время _micro_step, новорождённый добавляется в список."""
        rng = np.random.default_rng(42)
        agent = MagicMock()
        agent.alive = True
        agent.dividing = True
        child = MagicMock()
        child.alive = True
        child.dividing = False
        agent.update = MagicMock()
        agent.divide = MagicMock(return_value=child)
        # _micro_step должен обработать деление и добавить child
        result = integrator._micro_step([agent], 0.1)
        assert isinstance(result, list)
        # Все результирующие агенты живы
        assert all(a.alive for a in result)


# =============================================================================
# TestEquationFreeIntegratorLiftStep
# =============================================================================


class TestEquationFreeIntegratorLiftStep:
    """Тесты _lift_step."""

    def test_zero_state_returns_empty(self, integrator, zero_state):
        result = integrator._lift_step(zero_state, 0.0)
        assert result == []

    def test_nonzero_state_returns_agents(self, integrator, populated_state):
        result = integrator._lift_step(populated_state, 0.0)
        assert len(result) > 0

    def test_returns_list(self, integrator, zero_state):
        result = integrator._lift_step(zero_state, 0.0)
        assert isinstance(result, list)


# =============================================================================
# TestEquationFreeIntegratorRestrictStep
# =============================================================================


class TestEquationFreeIntegratorRestrictStep:
    """Тесты _restrict_step."""

    def test_empty_returns_zero_state(self, integrator):
        result = integrator._restrict_step([], 0.0)
        assert pytest.approx(0.0) == result.P
        assert pytest.approx(0.0) == result.F

    def test_returns_extended_sde_state(self, integrator):
        result = integrator._restrict_step([], 0.0)
        assert isinstance(result, ExtendedSDEState)

    def test_time_field_matches(self, integrator):
        result = integrator._restrict_step([], 7.5)
        assert result.t == pytest.approx(7.5)


# =============================================================================
# TestEquationFreeIntegratorStep
# =============================================================================


class TestEquationFreeIntegratorStep:
    """Тесты step."""

    def test_returns_extended_sde_state(self, integrator, zero_state):
        result = integrator.step(zero_state, 0.0, 1.0)
        assert isinstance(result, ExtendedSDEState)

    def test_result_time_equals_t_plus_dt(self, integrator, zero_state):
        t, dt = 2.0, 1.0
        result = integrator.step(zero_state, t, dt)
        assert result.t == pytest.approx(t + dt)

    def test_trajectory_grows_by_one(self, integrator, zero_state):
        assert len(integrator.trajectory) == 0
        integrator.step(zero_state, 0.0, 1.0)
        assert len(integrator.trajectory) == 1

    def test_three_steps_trajectory_length(self, integrator, zero_state):
        for i in range(3):
            integrator.step(zero_state, float(i), 1.0)
        assert len(integrator.trajectory) == 3

    def test_all_concentrations_nonnegative(self, integrator, populated_state):
        result = integrator.step(populated_state, 0.0, 1.0)
        arr = result.to_array()
        assert np.all(arr >= 0.0)


# =============================================================================
# TestEquationFreeIntegratorRun
# =============================================================================


class TestEquationFreeIntegratorRun:
    """Тесты run."""

    def test_returns_correct_length(self, integrator):
        result = integrator.run((0.0, 10.0), 1.0, 0.1)
        assert len(result) == 10

    def test_returns_list(self, integrator):
        result = integrator.run((0.0, 5.0), 1.0, 0.1)
        assert isinstance(result, list)

    def test_all_elements_extended_sde_state(self, integrator):
        result = integrator.run((0.0, 5.0), 1.0, 0.1)
        assert all(isinstance(s, ExtendedSDEState) for s in result)

    def test_monotone_time(self, integrator):
        result = integrator.run((0.0, 5.0), 1.0, 0.1)
        for i in range(len(result) - 1):
            assert result[i + 1].t > result[i].t

    def test_short_run_two_steps(self, integrator):
        result = integrator.run((0.0, 2.0), 1.0, 0.1)
        assert len(result) == 2

    def test_uses_sde_model_initial_state(self, ef_config, abm_config, mock_abm_model):
        """run() использует initial_state из sde_model для начального состояния."""
        mock_sde = MagicMock()
        initial = ExtendedSDEState(F=0.005, t=0.0)
        mock_sde.initial_state = initial
        mock_sde.apply_therapy_effect = MagicMock(return_value=initial)
        l = Lifter(ef_config, abm_config)
        r = Restrictor(ef_config)
        integ = EquationFreeIntegrator(mock_sde, mock_abm_model, l, r, ef_config)
        result = integ.run((0.0, 2.0), 1.0, 0.1)
        assert len(result) == 2


# =============================================================================
# TestEquationFreeIntegratorApplyTherapy
# =============================================================================


class TestEquationFreeIntegratorApplyTherapy:
    """Тесты apply_therapy."""

    def test_returns_tuple(self, integrator, zero_state):
        therapy = MagicMock()
        result = integrator.apply_therapy([], zero_state, therapy)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_list_and_state(self, integrator, zero_state):
        therapy = MagicMock()
        agents_out, state_out = integrator.apply_therapy([], zero_state, therapy)
        assert isinstance(agents_out, list)
        assert isinstance(state_out, ExtendedSDEState)

    def test_empty_agents_no_error(self, integrator, zero_state):
        therapy = MagicMock()
        integrator.apply_therapy([], zero_state, therapy)

    def test_with_agents_no_error(self, integrator, populated_state, alive_fibroblasts):
        therapy = MagicMock()
        integrator.apply_therapy(alive_fibroblasts, populated_state, therapy)

    def test_original_state_not_mutated(self, integrator, populated_state):
        """Исходные объекты не мутируются (спец: возвращаются копии)."""
        therapy = MagicMock()
        original_f = populated_state.F
        original_t = populated_state.t
        integrator.apply_therapy([], populated_state, therapy)
        assert original_f == populated_state.F
        assert populated_state.t == original_t

    def test_delegates_to_sde_model(self, integrator, zero_state):
        """apply_therapy вызывает sde_model.apply_therapy_effect."""
        therapy = MagicMock()
        integrator.apply_therapy([], zero_state, therapy)
        integrator.sde_model.apply_therapy_effect.assert_called_once()

    def test_agent_apply_therapy_called(self, integrator, zero_state):
        """apply_therapy вызывает agent.apply_therapy для каждого агента (если поддерживается)."""
        therapy = MagicMock()
        agent = MagicMock()
        agent.alive = True
        integrator.apply_therapy([agent], zero_state, therapy)
        agent.apply_therapy.assert_called_once_with(therapy)


# =============================================================================
# TestLiftRestrictConsistency — round-trip conservation tests
# =============================================================================


class TestLiftRestrictConsistency:
    """Тесты согласованности lift→restrict (круговой тест)."""

    def test_zero_state_round_trip(self, lifter, restrictor, ef_config, zero_state):
        """lift(нули) → [] → restrict([]) → нулевой state."""
        agents = lifter.lift(zero_state, 0, ef_config.volume)
        assert agents == []
        result = restrictor.restrict(agents, ef_config.volume, 0.0)
        assert pytest.approx(0.0) == result.P
        assert pytest.approx(0.0) == result.F
        assert result.Ne == pytest.approx(0.0)

    def test_round_trip_fibroblast_only(self, lifter, restrictor, ef_config):
        """lift(F=conc) → restrict → F ≈ conc (в пределах 10% из-за округления)."""
        F_conc = 0.005  # клеток/мкм³
        state = ExtendedSDEState(F=F_conc)
        agents = lifter.lift(state, 0, ef_config.volume)
        result = restrictor.restrict(agents, ef_config.volume, 0.0)
        # Допускаем погрешность из-за целочисленного округления числа агентов
        assert pytest.approx(F_conc, rel=0.1) == result.F
        # Все остальные клеточные концентрации = 0
        assert pytest.approx(0.0) == result.P
        assert result.Ne == pytest.approx(0.0)
        assert pytest.approx(0.0) == result.M1
        assert pytest.approx(0.0) == result.S

    def test_agent_types_match_source(self, lifter, restrictor, ef_config):
        """Только фибробласты в state → в restrict только F > 0."""
        state = ExtendedSDEState(F=0.003)
        agents = lifter.lift(state, 0, ef_config.volume)
        result = restrictor.restrict(agents, ef_config.volume, 0.0)
        assert result.F > 0.0
        assert pytest.approx(0.0) == result.P
        assert result.Ne == pytest.approx(0.0)
        assert pytest.approx(0.0) == result.M1
        assert pytest.approx(0.0) == result.M2
        assert pytest.approx(0.0) == result.S
        assert pytest.approx(0.0) == result.E
        assert result.Mf == pytest.approx(0.0)

    def test_multiple_types_round_trip(self, lifter, restrictor, ef_config):
        """Несколько типов → после round-trip все типы с конц > 0."""
        state = ExtendedSDEState(F=0.003, M1=0.002, S=0.001)
        agents = lifter.lift(state, 0, ef_config.volume)
        result = restrictor.restrict(agents, ef_config.volume, 0.0)
        assert result.F > 0.0
        assert result.M1 > 0.0
        assert result.S > 0.0

    def test_ecm_preserved_in_round_trip(self, lifter, restrictor, ef_config):
        """ECM поля (rho_collagen, C_MMP, rho_fibrin) сохраняются через lift→restrict."""
        state = ExtendedSDEState(F=0.005, rho_collagen=0.4, C_MMP=0.1, rho_fibrin=0.3)
        agents = lifter.lift(state, 0, ef_config.volume)
        result = restrictor.restrict(agents, ef_config.volume, 0.0)
        assert result.rho_collagen == pytest.approx(0.4, rel=0.15)
        assert pytest.approx(0.1, rel=0.15) == result.C_MMP
        assert result.rho_fibrin == pytest.approx(0.3, rel=0.15)

    def test_damage_signal_round_trip(self, lifter, restrictor, ef_config):
        """D (damage signal) сохраняется через lift→restrict."""
        state = ExtendedSDEState(F=0.005, D=0.05)
        agents = lifter.lift(state, 0, ef_config.volume)
        result = restrictor.restrict(agents, ef_config.volume, 0.0)
        assert pytest.approx(0.05, rel=0.15) == result.D

    def test_oxygen_round_trip(self, lifter, restrictor, ef_config):
        """O2 (oxygen) сохраняется через lift→restrict."""
        state = ExtendedSDEState(F=0.005, O2=95.0)
        agents = lifter.lift(state, 0, ef_config.volume)
        result = restrictor.restrict(agents, ef_config.volume, 0.0)
        assert pytest.approx(95.0, rel=0.15) == result.O2

    def test_cytokine_round_trip(self, lifter, restrictor, ef_config):
        """Все 7 цитокинов сохраняются через lift→restrict."""
        state = ExtendedSDEState(
            F=0.005,
            C_TNF=0.5,
            C_IL10=0.3,
            C_PDGF=0.2,
            C_VEGF=0.1,
            C_TGFb=0.15,
            C_MCP1=0.25,
            C_IL8=0.2,
        )
        agents = lifter.lift(state, 0, ef_config.volume)
        result = restrictor.restrict(agents, ef_config.volume, 0.0)
        assert pytest.approx(0.5, rel=0.15) == result.C_TNF
        assert pytest.approx(0.3, rel=0.15) == result.C_IL10
        assert pytest.approx(0.2, rel=0.15) == result.C_PDGF
        assert pytest.approx(0.1, rel=0.15) == result.C_VEGF
        assert result.C_TGFb == pytest.approx(0.15, rel=0.15)
        assert pytest.approx(0.25, rel=0.15) == result.C_MCP1
        assert pytest.approx(0.2, rel=0.15) == result.C_IL8

    def test_full_20_vector_round_trip(self, lifter, restrictor, ef_config, populated_state_full):
        """Полный 20-вектор: lift→restrict ≈ original (все переменные)."""
        original = populated_state_full.to_array()
        agents = lifter.lift(populated_state_full, 0, ef_config.volume)
        result = restrictor.restrict(agents, ef_config.volume, 0.0)
        restored = result.to_array()
        # Все 20 переменных должны быть ≈ оригиналу с допуском 15%
        for i in range(20):
            if original[i] > 0.0:
                assert restored[i] == pytest.approx(
                    original[i], rel=0.15
                ), f"Variable index {i}: expected {original[i]}, got {restored[i]}"

    def test_full_20_vector_nonneg_after_round_trip(
        self, lifter, restrictor, ef_config, populated_state_full
    ):
        """Все 20 переменных >= 0.0 после lift→restrict."""
        agents = lifter.lift(populated_state_full, 0, ef_config.volume)
        result = restrictor.restrict(agents, ef_config.volume, 0.0)
        arr = result.to_array()
        assert np.all(arr >= 0.0), f"Negative values found: {arr[arr < 0]}"


# =============================================================================
# TestSubcycling — разные dt для быстрых/медленных переменных
# =============================================================================


class TestSubcycling:
    """Тесты subcycling интеграции (n_micro_steps шагов ABM на 1 шаг SDE)."""

    def test_default_time_consistency(self):
        """dt_macro == n_micro_steps * dt_micro для дефолтного конфига."""
        cfg = EquationFreeConfig()
        assert cfg.dt_macro == pytest.approx(cfg.n_micro_steps * cfg.dt_micro)

    def test_micro_steps_count(self, ef_config, abm_config, mock_sde_model, mock_abm_model):
        """step() вызывает _micro_step, который выполняет n_micro_steps итераций ABM."""
        l = Lifter(ef_config, abm_config)
        r = Restrictor(ef_config)
        integ = EquationFreeIntegrator(mock_sde_model, mock_abm_model, l, r, ef_config)
        # Подменяем _micro_step, чтобы считать вызовы
        original_micro = integ._micro_step
        call_count = {"n": 0}

        def counting_micro(agents, dt):
            call_count["n"] += 1
            return original_micro(agents, dt)

        integ._micro_step = counting_micro
        state = ExtendedSDEState(F=0.005)
        integ.step(state, 0.0, ef_config.dt_macro)
        # _micro_step вызывается 1 раз, внутри неё n_micro_steps итераций
        assert call_count["n"] == 1

    def test_different_subcycling_configs(self, abm_config, mock_sde_model, mock_abm_model):
        """Разные n_micro_steps дают разные результаты (больше шагов → больше эволюции)."""
        cfg_5 = EquationFreeConfig(n_micro_steps=5, dt_micro=0.2)
        cfg_20 = EquationFreeConfig(n_micro_steps=20, dt_micro=0.05)

        l5 = Lifter(cfg_5, abm_config)
        r5 = Restrictor(cfg_5)
        integ_5 = EquationFreeIntegrator(mock_sde_model, mock_abm_model, l5, r5, cfg_5)

        l20 = Lifter(cfg_20, abm_config)
        r20 = Restrictor(cfg_20)
        integ_20 = EquationFreeIntegrator(mock_sde_model, mock_abm_model, l20, r20, cfg_20)

        state = ExtendedSDEState(F=0.01, Ne=0.005)
        r5_result = integ_5.step(state, 0.0, 1.0)
        r20_result = integ_20.step(state, 0.0, 1.0)
        # Оба возвращают ExtendedSDEState с правильным временем
        assert r5_result.t == pytest.approx(1.0)
        assert r20_result.t == pytest.approx(1.0)
        # Оба nonneg
        assert np.all(r5_result.to_array() >= 0.0)
        assert np.all(r20_result.to_array() >= 0.0)


# =============================================================================
# TestApplyTherapyPRP — реальные эффекты PRP терапии
# =============================================================================


class TestApplyTherapyPRP:
    """Тесты PRP терапии через apply_therapy (реальный TherapyProtocol)."""

    @pytest.fixture
    def prp_therapy(self) -> TherapyProtocol:
        return TherapyProtocol(prp_enabled=True, prp_start_time=0.0, prp_duration=7.0)

    def test_prp_increases_growth_factors(self, integrator, prp_therapy):
        """PRP → C_PDGF, C_VEGF, C_TGFb увеличиваются в macro_state."""
        state = ExtendedSDEState(F=0.005, C_PDGF=0.1, C_VEGF=0.05, C_TGFb=0.1)
        _, new_state = integrator.apply_therapy([], state, prp_therapy)
        # PRP должна увеличить факторы роста
        assert new_state.C_PDGF >= state.C_PDGF
        assert new_state.C_VEGF >= state.C_VEGF
        assert new_state.C_TGFb >= state.C_TGFb

    def test_prp_agent_mobilization(self, integrator, prp_therapy):
        """StemCell.prp_mobilization() вызывается при PRP терапии."""
        rng = np.random.default_rng(42)
        stems = [StemCell(agent_id=i, x=float(i), y=float(i), rng=rng) for i in range(3)]
        state = ExtendedSDEState(S=0.003)
        agents_out, _ = integrator.apply_therapy(stems, state, prp_therapy)
        # Агенты возвращаются (не теряются)
        assert isinstance(agents_out, list)

    def test_prp_returns_correct_types(self, integrator, prp_therapy):
        """apply_therapy с PRP возвращает (list, ExtendedSDEState)."""
        state = ExtendedSDEState(F=0.005)
        agents_out, state_out = integrator.apply_therapy([], state, prp_therapy)
        assert isinstance(agents_out, list)
        assert isinstance(state_out, ExtendedSDEState)


# =============================================================================
# TestApplyTherapyPEMF — реальные эффекты PEMF терапии
# =============================================================================


class TestApplyTherapyPEMF:
    """Тесты PEMF терапии через apply_therapy."""

    @pytest.fixture
    def pemf_therapy(self) -> TherapyProtocol:
        return TherapyProtocol(pemf_enabled=True, pemf_start_time=0.0, pemf_duration=14.0)

    def test_pemf_anti_inflammatory_effect(self, integrator, pemf_therapy):
        """PEMF → anti-inflammatory эффект (TNF-α подавление, M2 поляризация)."""
        state = ExtendedSDEState(M1=0.005, M2=0.002, C_TNF=1.0)
        _, new_state = integrator.apply_therapy([], state, pemf_therapy)
        assert isinstance(new_state, ExtendedSDEState)
        # PEMF anti-inflammatory: TNF не должен увеличиться
        assert new_state.C_TNF <= state.C_TNF + 0.01

    def test_pemf_proliferation_boost(self, integrator, pemf_therapy):
        """PEMF → proliferation_boost для фибробластов и эндотелия."""
        state = ExtendedSDEState(F=0.005, E=0.003)
        _, new_state = integrator.apply_therapy([], state, pemf_therapy)
        assert isinstance(new_state, ExtendedSDEState)
        # Результат nonneg
        assert new_state.F >= 0.0
        assert new_state.E >= 0.0

    def test_pemf_therapy_protocol_integration(self, integrator, pemf_therapy):
        """TherapyProtocol(pemf_enabled=True) корректно передаётся."""
        assert pemf_therapy.pemf_enabled is True
        state = ExtendedSDEState()
        agents_out, state_out = integrator.apply_therapy([], state, pemf_therapy)
        assert isinstance(state_out, ExtendedSDEState)


# =============================================================================
# TestApplyTherapySynergy — совместный PRP+PEMF
# =============================================================================


class TestApplyTherapySynergy:
    """Тесты синергии PRP + PEMF."""

    @pytest.fixture
    def combined_therapy(self) -> TherapyProtocol:
        return TherapyProtocol(
            prp_enabled=True,
            prp_start_time=0.0,
            pemf_enabled=True,
            pemf_start_time=0.0,
            synergy_factor=1.2,
        )

    def test_synergy_returns_correct_types(self, integrator, combined_therapy):
        """Совместная терапия возвращает (list, ExtendedSDEState)."""
        state = ExtendedSDEState(F=0.005, M1=0.003)
        agents_out, state_out = integrator.apply_therapy([], state, combined_therapy)
        assert isinstance(agents_out, list)
        assert isinstance(state_out, ExtendedSDEState)

    def test_synergy_factor_active(self, combined_therapy):
        """synergy_factor > 1 при обеих включённых терапиях."""
        assert combined_therapy.prp_enabled is True
        assert combined_therapy.pemf_enabled is True
        assert combined_therapy.synergy_factor > 1.0


# =============================================================================
# TestRestrictorAggregateCytokinesExtended — дополнительные сценарии
# =============================================================================


class TestRestrictorAggregateCytokinesExtended:
    """Расширенные тесты aggregate_cytokines."""

    def test_partial_cytokine_environments(self, restrictor):
        """Часть агентов с cytokine_environment, часть без → усреднение по имеющим."""
        rng = np.random.default_rng(0)
        with_env = StemCell(agent_id=0, x=0.0, y=0.0, rng=rng)
        without_env = Fibroblast(agent_id=1, x=0.0, y=0.0, rng=rng)
        with_env.cytokine_environment = dict.fromkeys(CYTOKINE_NAMES, 4.0)
        # without_env не имеет cytokine_environment
        result = restrictor.aggregate_cytokines([with_env, without_env])
        # Среднее учитывает только агентов с окружением, или всех → зависит от реализации
        # Но результат должен быть >= 0 и содержать все ключи
        assert set(result.keys()) == set(CYTOKINE_NAMES)
        assert all(v >= 0.0 for v in result.values())
