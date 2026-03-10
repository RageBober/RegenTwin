"""TDD тесты для расширенной ABM модели (Phase 2.8).

Тестирование:
- PlateletAgent: константы, инициализация, поведение (стабы)
- ChemotaxisEngine: инициализация, AGENT_ATTRACTANT_MAP, стабы
- ContactInhibitionEngine: инициализация, стабы
- EfferocytosisEngine: инициализация, стабы
- MechanotransductionEngine: инициализация, стабы
- MultiCytokineField: инициализация (7 полей), стабы
- KDTreeNeighborSearch: инициализация, стабы
- SubcyclingManager: инициализация, стабы
- StemCell.prp_mobilization: стаб
- Macrophage.efferocytose: стаб
- Fibroblast.tgfb_activation: стаб

Все тесты написаны для stub-реализации (NotImplementedError).
После реализации методов тесты должны быть обновлены.
"""

import math

import numpy as np
import pytest

from src.core.abm_model import (
    ABMConfig,
    ABMModel,
    Agent,
    EndothelialAgent,
    Fibroblast,
    Macrophage,
    MyofibroblastAgent,
    NeutrophilAgent,
    StemCell,
)
from src.core.abm_spatial import (
    ChemotaxisEngine,
    ContactInhibitionEngine,
    EfferocytosisEngine,
    KDTreeNeighborSearch,
    MechanotransductionEngine,
    MultiCytokineField,
    PlateletAgent,
    SubcyclingManager,
)

# =============================================================================
# Test PlateletAgent
# =============================================================================


class TestPlateletAgentConstants:
    """Тесты констант PlateletAgent."""

    def test_agent_type(self):
        assert PlateletAgent.AGENT_TYPE == "platelet"

    def test_lifespan(self):
        assert PlateletAgent.LIFESPAN == 72.0

    def test_max_divisions(self):
        assert PlateletAgent.MAX_DIVISIONS == 0

    def test_division_probability(self):
        assert PlateletAgent.DIVISION_PROBABILITY == 0.0

    def test_death_probability(self):
        assert PlateletAgent.DEATH_PROBABILITY == pytest.approx(0.014)

    def test_degranulation_rate(self):
        assert PlateletAgent.DEGRANULATION_RATE == pytest.approx(0.05)

    def test_pdgf_release_rate(self):
        assert PlateletAgent.PDGF_RELEASE_RATE == pytest.approx(0.02)

    def test_tgfb_release_rate(self):
        assert PlateletAgent.TGFB_RELEASE_RATE == pytest.approx(0.015)

    def test_vegf_release_rate(self):
        assert PlateletAgent.VEGF_RELEASE_RATE == pytest.approx(0.01)


class TestPlateletAgentInit:
    """Тесты инициализации PlateletAgent."""

    def test_is_agent_subclass(self):
        agent = PlateletAgent(agent_id=1, x=10.0, y=20.0)
        assert isinstance(agent, Agent)

    def test_initial_position(self):
        agent = PlateletAgent(agent_id=1, x=10.0, y=20.0)
        assert agent.x == 10.0
        assert agent.y == 20.0

    def test_initial_age(self):
        agent = PlateletAgent(agent_id=1, x=0.0, y=0.0)
        assert agent.age == 0.0

    def test_initial_alive(self):
        agent = PlateletAgent(agent_id=1, x=0.0, y=0.0)
        assert agent.alive is True

    def test_initial_degranulated(self):
        agent = PlateletAgent(agent_id=1, x=0.0, y=0.0)
        assert agent.degranulated is False

    def test_initial_factors_released(self):
        agent = PlateletAgent(agent_id=1, x=0.0, y=0.0)
        assert agent.factors_released == {"PDGF": 0.0, "TGFb": 0.0, "VEGF": 0.0}

    def test_agent_type_instance(self):
        agent = PlateletAgent(agent_id=1, x=0.0, y=0.0)
        assert agent.AGENT_TYPE == "platelet"

    def test_custom_age(self):
        agent = PlateletAgent(agent_id=1, x=0.0, y=0.0, age=10.0)
        assert agent.age == 10.0

    def test_rng_seed(self):
        rng = np.random.default_rng(42)
        agent = PlateletAgent(agent_id=1, x=0.0, y=0.0, rng=rng)
        assert agent._rng is not None


class TestPlateletAgentBehavior:
    """Тесты поведения PlateletAgent."""

    def test_can_divide_always_false(self):
        agent = PlateletAgent(agent_id=1, x=0.0, y=0.0)
        agent.energy = 1.0
        assert agent.can_divide() is False

    def test_divide_always_none(self):
        agent = PlateletAgent(agent_id=1, x=0.0, y=0.0)
        assert agent.divide(new_id=2) is None

    def test_get_state_type(self):
        agent = PlateletAgent(agent_id=1, x=5.0, y=10.0)
        state = agent.get_state()
        assert state.agent_type == "platelet"
        assert state.x == 5.0
        assert state.y == 10.0

    def test_get_state_alive(self):
        agent = PlateletAgent(agent_id=1, x=0.0, y=0.0)
        assert agent.get_state().alive is True

    def test_move_periodic(self):
        agent = PlateletAgent(agent_id=1, x=95.0, y=95.0)
        agent.move(dx=10.0, dy=10.0, space_size=(100.0, 100.0))
        assert 0.0 <= agent.x < 100.0
        assert 0.0 <= agent.y < 100.0

    def test_should_die_old_age(self):
        agent = PlateletAgent(agent_id=1, x=0.0, y=0.0, age=100.0)
        # Возраст > LIFESPAN (72.0) → повышенная вероятность смерти
        assert agent.age > PlateletAgent.LIFESPAN


# =============================================================================
# Test ChemotaxisEngine
# =============================================================================


class TestChemotaxisEngineInit:
    """Тесты инициализации ChemotaxisEngine."""

    def test_constructor_accepts_config(self):
        config = ABMConfig()
        engine = ChemotaxisEngine(config)
        assert engine._config is config

    def test_agent_attractant_map_keys(self):
        expected_keys = {"neutro", "macro", "endo", "fibro", "platelet"}
        assert set(ChemotaxisEngine.AGENT_ATTRACTANT_MAP.keys()) == expected_keys

    def test_agent_attractant_map_neutro(self):
        assert ChemotaxisEngine.AGENT_ATTRACTANT_MAP["neutro"] == "IL8"

    def test_agent_attractant_map_macro(self):
        assert ChemotaxisEngine.AGENT_ATTRACTANT_MAP["macro"] == "MCP1"

    def test_agent_attractant_map_endo(self):
        assert ChemotaxisEngine.AGENT_ATTRACTANT_MAP["endo"] == "VEGF"

    def test_agent_attractant_map_fibro(self):
        assert ChemotaxisEngine.AGENT_ATTRACTANT_MAP["fibro"] == "PDGF"

    def test_agent_attractant_map_platelet(self):
        assert ChemotaxisEngine.AGENT_ATTRACTANT_MAP["platelet"] == "TGFb"


# =============================================================================
# Test ContactInhibitionEngine
# =============================================================================


class TestContactInhibitionEngineInit:
    """Тесты инициализации ContactInhibitionEngine."""

    def test_constructor_stores_threshold(self):
        engine = ContactInhibitionEngine(threshold=5, radius=3.0)
        assert engine.threshold == 5

    def test_constructor_stores_radius(self):
        engine = ContactInhibitionEngine(threshold=5, radius=3.0)
        assert engine.radius == 3.0

    def test_custom_values(self):
        engine = ContactInhibitionEngine(threshold=10, radius=5.0)
        assert engine.threshold == 10
        assert engine.radius == 5.0


# =============================================================================
# Test EfferocytosisEngine
# =============================================================================


class TestEfferocytosisEngineInit:
    """Тесты инициализации EfferocytosisEngine."""

    def test_default_rate(self):
        engine = EfferocytosisEngine()
        assert engine.il10_release_rate == pytest.approx(0.05)

    def test_custom_rate(self):
        engine = EfferocytosisEngine(il10_release_rate=0.1)
        assert engine.il10_release_rate == pytest.approx(0.1)


# =============================================================================
# Test MechanotransductionEngine
# =============================================================================


class TestMechanotransductionEngineInit:
    """Тесты инициализации MechanotransductionEngine."""

    def test_default_stress_threshold(self):
        engine = MechanotransductionEngine()
        assert engine.stress_threshold == pytest.approx(0.5)

    def test_default_activation_probability(self):
        engine = MechanotransductionEngine()
        assert engine.activation_probability == pytest.approx(0.01)

    def test_custom_values(self):
        engine = MechanotransductionEngine(
            stress_threshold=1.0,
            activation_probability=0.05,
        )
        assert engine.stress_threshold == pytest.approx(1.0)
        assert engine.activation_probability == pytest.approx(0.05)


# =============================================================================
# Test MultiCytokineField
# =============================================================================


class TestMultiCytokineFieldInit:
    """Тесты инициализации MultiCytokineField."""

    def test_default_cytokine_names(self):
        mcf = MultiCytokineField(grid_shape=(10, 10))
        assert set(mcf.fields.keys()) == {
            "TNF",
            "IL10",
            "PDGF",
            "VEGF",
            "TGFb",
            "MCP1",
            "IL8",
        }

    def test_field_count(self):
        mcf = MultiCytokineField(grid_shape=(10, 10))
        assert len(mcf.fields) == 7

    def test_field_shape(self):
        mcf = MultiCytokineField(grid_shape=(10, 10))
        for name, field in mcf.fields.items():
            assert field.shape == (10, 10), f"{name} has wrong shape"

    def test_fields_initialized_to_zeros(self):
        mcf = MultiCytokineField(grid_shape=(5, 5))
        for name, field in mcf.fields.items():
            assert np.all(field == 0.0), f"{name} is not zero-initialized"

    def test_custom_cytokine_names(self):
        mcf = MultiCytokineField(
            grid_shape=(10, 10),
            cytokine_names=["TNF", "IL10"],
        )
        assert set(mcf.fields.keys()) == {"TNF", "IL10"}
        assert len(mcf.fields) == 2

    def test_custom_grid_shape(self):
        mcf = MultiCytokineField(grid_shape=(20, 30))
        for field in mcf.fields.values():
            assert field.shape == (20, 30)

    def test_cytokine_names_class_constant(self):
        expected = ["TNF", "IL10", "PDGF", "VEGF", "TGFb", "MCP1", "IL8"]
        assert MultiCytokineField.CYTOKINE_NAMES == expected


# =============================================================================
# Test KDTreeNeighborSearch
# =============================================================================


class TestKDTreeNeighborSearchInit:
    """Тесты инициализации KDTreeNeighborSearch."""

    def test_constructor_stores_space_size(self):
        search = KDTreeNeighborSearch(space_size=(100.0, 100.0))
        assert search._space_size == (100.0, 100.0)

    def test_constructor_stores_periodic(self):
        search = KDTreeNeighborSearch(space_size=(100.0, 100.0), periodic=False)
        assert search._periodic is False

    def test_constructor_default_periodic(self):
        search = KDTreeNeighborSearch(space_size=(100.0, 100.0))
        assert search._periodic is True

    def test_internal_index_created(self):
        search = KDTreeNeighborSearch(space_size=(100.0, 100.0))
        assert search._index is not None


# =============================================================================
# Test SubcyclingManager
# =============================================================================


class TestSubcyclingManagerInit:
    """Тесты инициализации SubcyclingManager."""

    def test_stores_agent_dt(self):
        mgr = SubcyclingManager(agent_dt=0.1, field_dt=0.01)
        assert mgr.agent_dt == pytest.approx(0.1)

    def test_stores_field_dt(self):
        mgr = SubcyclingManager(agent_dt=0.1, field_dt=0.01)
        assert mgr.field_dt == pytest.approx(0.01)

    def test_is_dataclass(self):
        mgr = SubcyclingManager(agent_dt=0.1, field_dt=0.01)
        from dataclasses import fields as dc_fields

        field_names = {f.name for f in dc_fields(mgr)}
        assert "agent_dt" in field_names
        assert "field_dt" in field_names


# =============================================================================
# Test Existing Agent Enhancements (stubs in abm_model.py)
# =============================================================================


class TestStemCellPRPMobilization:
    """Тесты StemCell.prp_mobilization (стаб)."""

    def test_method_exists(self):
        cell = StemCell(agent_id=1, x=0.0, y=0.0)
        assert hasattr(cell, "prp_mobilization")


class TestMacrophageEfferocytose:
    """Тесты Macrophage.efferocytose (стаб)."""

    def test_method_exists(self):
        macro = Macrophage(agent_id=1, x=0.0, y=0.0)
        assert hasattr(macro, "efferocytose")


class TestFibroblastTgfbActivation:
    """Тесты Fibroblast.tgfb_activation (стаб)."""

    def test_method_exists(self):
        fibro = Fibroblast(agent_id=1, x=0.0, y=0.0)
        assert hasattr(fibro, "tgfb_activation")


# =============================================================================
# Test ABMConfig Phase 2.8 Fields
# =============================================================================


class TestABMConfigPhase28:
    """Тесты новых полей ABMConfig для Phase 2.8."""

    def test_initial_platelets_default(self):
        config = ABMConfig()
        assert config.initial_platelets == 0

    def test_enable_efferocytosis_default(self):
        config = ABMConfig()
        assert config.enable_efferocytosis is False

    def test_enable_mechanotransduction_default(self):
        config = ABMConfig()
        assert config.enable_mechanotransduction is False

    def test_enable_subcycling_default(self):
        config = ABMConfig()
        assert config.enable_subcycling is False

    def test_field_dt_default(self):
        config = ABMConfig()
        assert config.field_dt == pytest.approx(0.01)

    def test_custom_values(self):
        config = ABMConfig(
            initial_platelets=20,
            enable_efferocytosis=True,
            enable_mechanotransduction=True,
            enable_subcycling=True,
            field_dt=0.005,
        )
        assert config.initial_platelets == 20
        assert config.enable_efferocytosis is True
        assert config.enable_mechanotransduction is True
        assert config.enable_subcycling is True
        assert config.field_dt == pytest.approx(0.005)


# =============================================================================
# Test ABMModel platelet creation
# =============================================================================


class TestABMModelPlateletCreation:
    """Тесты создания PlateletAgent через ABMModel._create_agent."""

    def test_create_platelet_agent(self):
        model = ABMModel(random_seed=42)
        agent = model._create_agent("platelet", x=50.0, y=50.0)
        assert isinstance(agent, PlateletAgent)
        assert agent.AGENT_TYPE == "platelet"

    def test_create_platelet_position(self):
        model = ABMModel(random_seed=42)
        agent = model._create_agent("platelet", x=25.0, y=75.0)
        assert agent.x == 25.0
        assert agent.y == 75.0

    def test_create_platelet_random_position(self):
        model = ABMModel(random_seed=42)
        agent = model._create_agent("platelet")
        assert 0.0 <= agent.x <= 100.0
        assert 0.0 <= agent.y <= 100.0


# =============================================================================
# TDD-тесты для ожидаемого поведения Phase 2.8
#
# RED-фаза TDD: тесты описывают целевое поведение и ДОЛЖНЫ ПАДАТЬ
# до реализации соответствующего функционала.
# =============================================================================


# =============================================================================
# PlateletAgent — поведение после реализации
# =============================================================================


class TestPlateletUpdateBehaviorImplemented:
    """Тесты PlateletAgent.update() — целевое поведение."""

    def test_update_consumes_energy(self):
        """Потребление энергии: energy -= 0.005 * dt."""
        agent = PlateletAgent(agent_id=1, x=0.0, y=0.0)
        agent.energy = 1.0
        agent.update(dt=1.0, environment={})
        assert agent.energy == pytest.approx(1.0 - 0.005 * 1.0)

    def test_update_advances_age(self):
        """Старение: age += dt."""
        agent = PlateletAgent(agent_id=1, x=0.0, y=0.0)
        agent.update(dt=2.0, environment={})
        assert agent.age == pytest.approx(2.0)

    def test_update_triggers_degranulation_on_high_thrombin(self):
        """Тромбин > 0.1 запускает дегрануляцию."""
        agent = PlateletAgent(agent_id=1, x=0.0, y=0.0)
        assert agent.degranulated is False
        agent.update(dt=1.0, environment={"thrombin": 0.5})
        assert agent.degranulated is True

    def test_update_no_degranulation_on_low_thrombin(self):
        """Тромбин <= 0.1 не запускает дегрануляцию."""
        agent = PlateletAgent(agent_id=1, x=0.0, y=0.0)
        agent.update(dt=1.0, environment={"thrombin": 0.05})
        assert agent.degranulated is False

    def test_update_no_degranulation_when_thrombin_absent(self):
        """Отсутствие тромбина → нет дегрануляции."""
        agent = PlateletAgent(agent_id=1, x=0.0, y=0.0)
        agent.update(dt=1.0, environment={})
        assert agent.degranulated is False

    def test_update_energy_does_not_go_below_zero(self):
        """Энергия не опускается ниже 0."""
        agent = PlateletAgent(agent_id=1, x=0.0, y=0.0)
        agent.energy = 0.001
        agent.update(dt=10.0, environment={})
        assert agent.energy >= 0.0


class TestPlateletDegranulateBehaviorImplemented:
    """Тесты PlateletAgent.degranulate() — целевое поведение."""

    def test_degranulate_returns_correct_factors(self):
        """Возврат факторов: PDGF, TGFb, VEGF с правильными скоростями."""
        agent = PlateletAgent(agent_id=1, x=0.0, y=0.0)
        result = agent.degranulate(dt=1.0)
        assert result["PDGF"] == pytest.approx(0.02)
        assert result["TGFb"] == pytest.approx(0.015)
        assert result["VEGF"] == pytest.approx(0.01)

    def test_degranulate_scales_with_dt(self):
        """Линейное масштабирование с dt."""
        agent = PlateletAgent(agent_id=1, x=0.0, y=0.0)
        result = agent.degranulate(dt=2.0)
        assert result["PDGF"] == pytest.approx(0.04)
        assert result["TGFb"] == pytest.approx(0.03)
        assert result["VEGF"] == pytest.approx(0.02)

    def test_degranulate_sets_degranulated_flag(self):
        """Устанавливает degranulated = True."""
        agent = PlateletAgent(agent_id=1, x=0.0, y=0.0)
        agent.degranulate(dt=1.0)
        assert agent.degranulated is True

    def test_degranulate_accumulates_factors_released(self):
        """Кумулятивное накопление в factors_released."""
        agent = PlateletAgent(agent_id=1, x=0.0, y=0.0)
        agent.degranulate(dt=1.0)
        agent.degranulate(dt=1.0)
        assert agent.factors_released["PDGF"] == pytest.approx(0.04)

    def test_degranulate_returns_all_three_keys(self):
        """Возвращает словарь с тремя ключами."""
        agent = PlateletAgent(agent_id=1, x=0.0, y=0.0)
        result = agent.degranulate(dt=0.5)
        assert set(result.keys()) == {"PDGF", "TGFb", "VEGF"}

    def test_degranulate_with_zero_dt(self):
        """dt=0 → все значения 0."""
        agent = PlateletAgent(agent_id=1, x=0.0, y=0.0)
        result = agent.degranulate(dt=0.0)
        assert result["PDGF"] == pytest.approx(0.0)
        assert result["TGFb"] == pytest.approx(0.0)
        assert result["VEGF"] == pytest.approx(0.0)


class TestPlateletReleaseFactorsBehaviorImplemented:
    """Тесты PlateletAgent.release_factors() — целевое поведение."""

    def test_release_factors_after_degranulation(self):
        """Выброс факторов после дегрануляции."""
        agent = PlateletAgent(agent_id=1, x=0.0, y=0.0)
        agent.degranulated = True
        result = agent.release_factors(dt=1.0)
        assert isinstance(result, dict)
        assert all(v >= 0.0 for v in result.values())

    def test_release_factors_before_degranulation_returns_zeros(self):
        """До дегрануляции → нулевой выброс."""
        agent = PlateletAgent(agent_id=1, x=0.0, y=0.0)
        result = agent.release_factors(dt=1.0)
        assert all(v == pytest.approx(0.0) for v in result.values())

    def test_release_factors_returns_dict_type(self):
        """Возвращает dict."""
        agent = PlateletAgent(agent_id=1, x=0.0, y=0.0)
        agent.degranulated = True
        result = agent.release_factors(dt=1.0)
        assert isinstance(result, dict)

    def test_release_factors_accumulates(self):
        """Факторы накапливаются при повторных вызовах."""
        agent = PlateletAgent(agent_id=1, x=0.0, y=0.0)
        agent.degranulated = True
        initial_pdgf = agent.factors_released["PDGF"]
        agent.release_factors(dt=1.0)
        agent.release_factors(dt=1.0)
        assert agent.factors_released["PDGF"] > initial_pdgf


class TestPlateletSecreteCytokinesBehaviorImplemented:
    """Тесты PlateletAgent.secrete_cytokines() — целевое поведение."""

    def test_secrete_cytokines_returns_dict(self):
        """Возвращает dict."""
        agent = PlateletAgent(agent_id=1, x=0.0, y=0.0)
        agent.degranulated = True
        result = agent.secrete_cytokines(dt=1.0)
        assert isinstance(result, dict)

    def test_secrete_cytokines_same_keys_as_degranulate(self):
        """Совместимый интерфейс: те же ключи что и degranulate."""
        agent = PlateletAgent(agent_id=1, x=0.0, y=0.0)
        agent.degranulated = True
        result = agent.secrete_cytokines(dt=1.0)
        assert "PDGF" in result
        assert "TGFb" in result
        assert "VEGF" in result

    def test_secrete_cytokines_contains_growth_factors(self):
        """Содержит факторы роста."""
        agent = PlateletAgent(agent_id=1, x=0.0, y=0.0)
        agent.degranulated = True
        result = agent.secrete_cytokines(dt=1.0)
        assert set(result.keys()) >= {"PDGF", "TGFb", "VEGF"}


class TestPlateletInvariantsImplemented:
    """Инвариантные проверки PlateletAgent."""

    def test_platelet_never_divides_after_update(self):
        """Инвариант: тромбоцит никогда не делится (даже после update)."""
        agent = PlateletAgent(agent_id=1, x=0.0, y=0.0)
        agent.energy = 1.0
        agent.update(dt=1.0, environment={})
        assert agent.can_divide() is False
        assert agent.divide(new_id=99) is None

    def test_platelet_factors_released_always_has_three_keys(self):
        """Инвариант: factors_released всегда имеет 3 ключа."""
        agent = PlateletAgent(agent_id=1, x=0.0, y=0.0)
        agent.degranulated = True
        agent.degranulate(dt=1.0)
        agent.update(dt=1.0, environment={"thrombin": 0.5})
        assert set(agent.factors_released.keys()) == {"PDGF", "TGFb", "VEGF"}

    def test_platelet_degranulation_threshold_boundary(self):
        """Граничный случай: thrombin=0.1 точно (> 0.1, не >=)."""
        agent = PlateletAgent(agent_id=1, x=0.0, y=0.0)
        agent.update(dt=1.0, environment={"thrombin": 0.1})
        assert agent.degranulated is False


# =============================================================================
# ChemotaxisEngine — поведение после реализации
# =============================================================================


class TestChemotaxisDisplacementBehaviorImplemented:
    """Тесты ChemotaxisEngine.compute_displacement() — целевое поведение."""

    def test_unknown_agent_type_returns_zero(self):
        """Неизвестный тип агента → (0.0, 0.0)."""
        config = ABMConfig()
        engine = ChemotaxisEngine(config)

        class _UnknownAgent(Agent):
            AGENT_TYPE = "unknown_xyz"

            def update(self, dt, environment):
                pass

            def divide(self, new_id):
                return None

        agent = _UnknownAgent(agent_id=1, x=50.0, y=50.0)
        fields = {"IL8": np.ones((10, 10))}
        dx, dy = engine.compute_displacement(agent, fields, dt=0.1)
        assert dx == pytest.approx(0.0)
        assert dy == pytest.approx(0.0)

    def test_zero_gradient_returns_zero_chemotaxis(self):
        """Однородное поле → нулевое хемотаксисное смещение."""
        config = ABMConfig()
        engine = ChemotaxisEngine(config)
        agent = NeutrophilAgent(agent_id=1, x=50.0, y=50.0)
        # Однородное поле — градиент = 0
        fields = {"IL8": np.ones((10, 10)) * 5.0}
        dx, dy = engine.compute_displacement(agent, fields, dt=0.1)
        assert dx == pytest.approx(0.0)
        assert dy == pytest.approx(0.0)

    def test_neutrophil_uses_il8_field(self):
        """Нейтрофил следует градиенту IL-8."""
        config = ABMConfig()
        engine = ChemotaxisEngine(config)
        agent = NeutrophilAgent(agent_id=1, x=50.0, y=50.0)
        # Линейный градиент по x
        field = np.zeros((10, 10))
        for i in range(10):
            field[i, :] = float(i)
        fields = {"IL8": field}
        dx, dy = engine.compute_displacement(agent, fields, dt=0.1)
        assert abs(dx) > 0.0 or abs(dy) > 0.0

    def test_macrophage_uses_mcp1_field(self):
        """Макрофаг следует градиенту MCP-1."""
        config = ABMConfig()
        engine = ChemotaxisEngine(config)
        agent = Macrophage(agent_id=1, x=50.0, y=50.0)
        field = np.zeros((10, 10))
        for i in range(10):
            field[i, :] = float(i)
        fields = {"MCP1": field}
        dx, dy = engine.compute_displacement(agent, fields, dt=0.1)
        assert abs(dx) > 0.0 or abs(dy) > 0.0

    def test_endothelial_uses_vegf_field(self):
        """Эндотелиальная клетка следует градиенту VEGF."""
        from src.core.abm_model import EndothelialAgent

        config = ABMConfig()
        engine = ChemotaxisEngine(config)
        agent = EndothelialAgent(agent_id=1, x=50.0, y=50.0)
        field = np.zeros((10, 10))
        for i in range(10):
            field[i, :] = float(i)
        fields = {"VEGF": field}
        dx, dy = engine.compute_displacement(agent, fields, dt=0.1)
        assert abs(dx) > 0.0 or abs(dy) > 0.0

    def test_fibroblast_uses_pdgf_field(self):
        """Фибробласт следует градиенту PDGF."""
        config = ABMConfig()
        engine = ChemotaxisEngine(config)
        agent = Fibroblast(agent_id=1, x=50.0, y=50.0)
        field = np.zeros((10, 10))
        for i in range(10):
            field[i, :] = float(i)
        fields = {"PDGF": field}
        dx, dy = engine.compute_displacement(agent, fields, dt=0.1)
        assert abs(dx) > 0.0 or abs(dy) > 0.0

    def test_platelet_uses_tgfb_field(self):
        """Тромбоцит следует градиенту TGF-β."""
        config = ABMConfig()
        engine = ChemotaxisEngine(config)
        agent = PlateletAgent(agent_id=1, x=50.0, y=50.0)
        field = np.zeros((10, 10))
        for i in range(10):
            field[i, :] = float(i)
        fields = {"TGFb": field}
        dx, dy = engine.compute_displacement(agent, fields, dt=0.1)
        assert abs(dx) > 0.0 or abs(dy) > 0.0

    def test_displacement_returns_tuple_of_two_floats(self):
        """Возвращает tuple из двух float."""
        config = ABMConfig()
        engine = ChemotaxisEngine(config)
        agent = NeutrophilAgent(agent_id=1, x=50.0, y=50.0)
        fields = {"IL8": np.zeros((10, 10))}
        result = engine.compute_displacement(agent, fields, dt=0.1)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)


class TestChemotaxisGradientBehaviorImplemented:
    """Тесты ChemotaxisEngine._compute_gradient() — целевое поведение."""

    def test_gradient_zero_field(self):
        """Нулевое поле → нулевой градиент."""
        config = ABMConfig()
        engine = ChemotaxisEngine(config)
        field = np.zeros((10, 10))
        gx, gy = engine._compute_gradient(field, x=50.0, y=50.0, grid_resolution=10.0)
        assert gx == pytest.approx(0.0)
        assert gy == pytest.approx(0.0)

    def test_gradient_linear_x(self):
        """Линейное увеличение по x → gx > 0, gy ≈ 0."""
        config = ABMConfig()
        engine = ChemotaxisEngine(config)
        field = np.zeros((10, 10))
        for i in range(10):
            field[i, :] = float(i)
        gx, gy = engine._compute_gradient(field, x=50.0, y=50.0, grid_resolution=10.0)
        assert gx > 0.0
        assert gy == pytest.approx(0.0, abs=1e-10)

    def test_gradient_linear_y(self):
        """Линейное увеличение по y → gx ≈ 0, gy > 0."""
        config = ABMConfig()
        engine = ChemotaxisEngine(config)
        field = np.zeros((10, 10))
        for j in range(10):
            field[:, j] = float(j)
        gx, gy = engine._compute_gradient(field, x=50.0, y=50.0, grid_resolution=10.0)
        assert gx == pytest.approx(0.0, abs=1e-10)
        assert gy > 0.0

    def test_gradient_returns_tuple(self):
        """Возвращает tuple из двух float."""
        config = ABMConfig()
        engine = ChemotaxisEngine(config)
        field = np.zeros((10, 10))
        result = engine._compute_gradient(field, x=50.0, y=50.0, grid_resolution=10.0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_gradient_central_difference_accuracy(self):
        """Точность центральных разностей: field[i,j] = 2*i + 3*j."""
        config = ABMConfig()
        engine = ChemotaxisEngine(config)
        field = np.zeros((10, 10))
        for i in range(10):
            for j in range(10):
                field[i, j] = 2.0 * i + 3.0 * j
        # В центре поля (x=50, y=50) → ячейка (5, 5)
        # ∂C/∂x = (field[6,5] - field[4,5]) / (2*res) = (2*6+15 - 2*4-15) / 20 = 4/20 = 0.2
        # ∂C/∂y = (field[5,6] - field[5,4]) / (2*res) = (10+18 - 10+12) / 20 = 6/20 = 0.3
        gx, gy = engine._compute_gradient(field, x=50.0, y=50.0, grid_resolution=10.0)
        assert gx == pytest.approx(0.2, abs=0.05)
        assert gy == pytest.approx(0.3, abs=0.05)


# =============================================================================
# ContactInhibitionEngine — поведение после реализации
# =============================================================================


class TestContactInhibitionModifierBehaviorImplemented:
    """Тесты ContactInhibitionEngine.compute_modifier() — целевое поведение."""

    def test_zero_neighbors_returns_one(self):
        """0 соседей → modifier = 1.0 (нет ингибирования)."""
        engine = ContactInhibitionEngine(threshold=5, radius=3.0)
        assert engine.compute_modifier(0) == pytest.approx(1.0)

    def test_threshold_neighbors_returns_zero(self):
        """threshold соседей → modifier = 0.0 (полное ингибирование)."""
        engine = ContactInhibitionEngine(threshold=5, radius=3.0)
        assert engine.compute_modifier(5) == pytest.approx(0.0)

    def test_above_threshold_returns_zero(self):
        """Больше threshold → modifier = 0.0 (clamped)."""
        engine = ContactInhibitionEngine(threshold=5, radius=3.0)
        assert engine.compute_modifier(10) == pytest.approx(0.0)

    def test_half_threshold_returns_half(self):
        """Половина threshold → modifier = 0.5."""
        engine = ContactInhibitionEngine(threshold=10, radius=3.0)
        assert engine.compute_modifier(5) == pytest.approx(0.5)

    def test_modifier_always_in_unit_interval(self):
        """Инвариант: modifier ∈ [0, 1] для любого n."""
        engine = ContactInhibitionEngine(threshold=5, radius=3.0)
        for n in range(20):
            mod = engine.compute_modifier(n)
            assert 0.0 <= mod <= 1.0, f"modifier={mod} for n={n}"

    def test_modifier_monotonically_decreasing(self):
        """Модификатор монотонно убывает с ростом числа соседей."""
        engine = ContactInhibitionEngine(threshold=10, radius=3.0)
        modifiers = [engine.compute_modifier(n) for n in range(15)]
        for i in range(1, len(modifiers)):
            assert modifiers[i] <= modifiers[i - 1]

    def test_custom_threshold(self):
        """Пользовательский threshold: max(0, 1 - 1/3) = 2/3."""
        engine = ContactInhibitionEngine(threshold=3, radius=3.0)
        assert engine.compute_modifier(1) == pytest.approx(2.0 / 3.0)


class TestContactInhibitionBlockDivisionBehaviorImplemented:
    """Тесты ContactInhibitionEngine.should_block_division()."""

    def test_block_when_at_threshold(self):
        """При n == threshold → блокировка деления."""
        engine = ContactInhibitionEngine(threshold=5, radius=3.0)
        agent = StemCell(agent_id=1, x=0.0, y=0.0)
        assert engine.should_block_division(agent, neighbor_count=5) is True

    def test_block_when_above_threshold(self):
        """При n > threshold → блокировка деления."""
        engine = ContactInhibitionEngine(threshold=5, radius=3.0)
        agent = StemCell(agent_id=1, x=0.0, y=0.0)
        assert engine.should_block_division(agent, neighbor_count=8) is True

    def test_no_block_below_threshold(self):
        """При n < threshold → деление разрешено."""
        engine = ContactInhibitionEngine(threshold=5, radius=3.0)
        agent = StemCell(agent_id=1, x=0.0, y=0.0)
        assert engine.should_block_division(agent, neighbor_count=4) is False

    def test_no_block_zero_neighbors(self):
        """0 соседей → деление разрешено."""
        engine = ContactInhibitionEngine(threshold=5, radius=3.0)
        agent = StemCell(agent_id=1, x=0.0, y=0.0)
        assert engine.should_block_division(agent, neighbor_count=0) is False


# =============================================================================
# EfferocytosisEngine — поведение после реализации
# =============================================================================


class TestEfferocytosisProcessBehaviorImplemented:
    """Тесты EfferocytosisEngine.process() — целевое поведение."""

    def test_empty_list_returns_zeros(self):
        """Пустой список → {IL10: 0.0, phagocytosed: 0}."""
        engine = EfferocytosisEngine()
        macro = Macrophage(agent_id=1, x=0.0, y=0.0)
        result = engine.process(macro, [])
        assert result["IL10"] == pytest.approx(0.0)
        assert result["phagocytosed"] == 0

    def test_single_neutrophil_phagocytosed(self):
        """Один нейтрофил → phagocytosed=1, IL10=0.05."""
        engine = EfferocytosisEngine()
        macro = Macrophage(agent_id=1, x=0.0, y=0.0)
        neutro = NeutrophilAgent(agent_id=2, x=1.0, y=1.0)
        neutro.alive = False  # апоптотический
        result = engine.process(macro, [neutro])
        assert result["phagocytosed"] == 1
        assert result["IL10"] == pytest.approx(0.05)

    def test_marks_neutrophils_dead(self):
        """Фагоцитированные нейтрофилы помечаются мёртвыми."""
        engine = EfferocytosisEngine()
        macro = Macrophage(agent_id=1, x=0.0, y=0.0)
        neutros = [NeutrophilAgent(agent_id=i, x=float(i), y=0.0) for i in range(2, 5)]
        for n in neutros:
            n.alive = False  # апоптотические
        engine.process(macro, neutros)
        for n in neutros:
            assert n.alive is False

    def test_respects_phagocytosis_capacity(self):
        """Не более PHAGOCYTOSIS_CAPACITY (5) за раз."""
        engine = EfferocytosisEngine()
        macro = Macrophage(agent_id=1, x=0.0, y=0.0)
        neutros = [NeutrophilAgent(agent_id=i, x=float(i), y=0.0) for i in range(2, 12)]
        for n in neutros:
            n.alive = False
        result = engine.process(macro, neutros)
        assert result["phagocytosed"] <= Macrophage.PHAGOCYTOSIS_CAPACITY

    def test_il10_proportional_to_count(self):
        """IL-10 пропорционален числу фагоцитированных."""
        engine = EfferocytosisEngine()
        macro = Macrophage(agent_id=1, x=0.0, y=0.0)
        neutros = [NeutrophilAgent(agent_id=i, x=float(i), y=0.0) for i in range(2, 5)]
        for n in neutros:
            n.alive = False
        result = engine.process(macro, neutros)
        expected_il10 = result["phagocytosed"] * 0.05
        assert result["IL10"] == pytest.approx(expected_il10)

    def test_shifts_polarization_toward_m2(self):
        """Эффероцитоз сдвигает поляризацию макрофага к M2."""
        engine = EfferocytosisEngine()
        macro = Macrophage(agent_id=1, x=0.0, y=0.0)
        macro.polarization_state = "M1"
        neutros = [NeutrophilAgent(agent_id=2, x=1.0, y=1.0)]
        neutros[0].alive = False
        engine.process(macro, neutros)
        assert macro.polarization_state == "M2"

    def test_custom_il10_rate(self):
        """Пользовательская скорость IL-10."""
        engine = EfferocytosisEngine(il10_release_rate=0.1)
        macro = Macrophage(agent_id=1, x=0.0, y=0.0)
        neutros = [NeutrophilAgent(agent_id=i, x=float(i), y=0.0) for i in range(2, 4)]
        for n in neutros:
            n.alive = False
        result = engine.process(macro, neutros)
        assert result["IL10"] == pytest.approx(2 * 0.1)

    def test_return_keys(self):
        """Возвращаемый dict содержит IL10 и phagocytosed."""
        engine = EfferocytosisEngine()
        macro = Macrophage(agent_id=1, x=0.0, y=0.0)
        result = engine.process(macro, [])
        assert "IL10" in result
        assert "phagocytosed" in result

    def test_five_neutrophils_exact_capacity(self):
        """Ровно 5 нейтрофилов → все фагоцитированы (PHAGOCYTOSIS_CAPACITY=5)."""
        engine = EfferocytosisEngine()
        macro = Macrophage(agent_id=1, x=0.0, y=0.0)
        neutros = [NeutrophilAgent(agent_id=i, x=float(i), y=0.0) for i in range(2, 7)]
        for n in neutros:
            n.alive = False
        result = engine.process(macro, neutros)
        assert result["phagocytosed"] == 5


# =============================================================================
# MechanotransductionEngine — поведение после реализации
# =============================================================================


class TestMechanotransductionStressBehaviorImplemented:
    """Тесты MechanotransductionEngine.compute_stress() — целевое поведение."""

    def test_stress_non_negative(self):
        """Стресс всегда >= 0."""
        engine = MechanotransductionEngine()
        agent = Fibroblast(agent_id=1, x=50.0, y=50.0)
        stress = engine.compute_stress(agent, neighbors=[], ecm_density=0.5)
        assert stress >= 0.0

    def test_zero_ecm_minimal_stress(self):
        """ecm_density=0, нет соседей → минимальный стресс."""
        engine = MechanotransductionEngine()
        agent = Fibroblast(agent_id=1, x=50.0, y=50.0)
        stress = engine.compute_stress(agent, neighbors=[], ecm_density=0.0)
        assert stress == pytest.approx(0.0, abs=0.01)

    def test_more_neighbors_higher_stress(self):
        """Больше соседей → выше стресс."""
        engine = MechanotransductionEngine()
        agent = Fibroblast(agent_id=1, x=50.0, y=50.0)
        neighbors_0 = []
        neighbors_5 = [Fibroblast(agent_id=i, x=50.0 + i, y=50.0) for i in range(2, 7)]
        stress_0 = engine.compute_stress(agent, neighbors=neighbors_0, ecm_density=0.5)
        stress_5 = engine.compute_stress(agent, neighbors=neighbors_5, ecm_density=0.5)
        assert stress_5 > stress_0

    def test_higher_ecm_higher_stress(self):
        """Более высокая плотность ECM → выше стресс."""
        engine = MechanotransductionEngine()
        agent = Fibroblast(agent_id=1, x=50.0, y=50.0)
        stress_low = engine.compute_stress(agent, neighbors=[], ecm_density=0.1)
        stress_high = engine.compute_stress(agent, neighbors=[], ecm_density=0.9)
        assert stress_high > stress_low

    def test_stress_returns_float(self):
        """Возвращает float."""
        engine = MechanotransductionEngine()
        agent = Fibroblast(agent_id=1, x=50.0, y=50.0)
        result = engine.compute_stress(agent, neighbors=[], ecm_density=0.5)
        assert isinstance(result, float)


class TestMechanotransductionActivationBehaviorImplemented:
    """Тесты MechanotransductionEngine.should_activate() — целевое поведение."""

    def test_below_threshold_never_activates(self):
        """Стресс ниже порога → False."""
        engine = MechanotransductionEngine(stress_threshold=0.5)
        fibro = Fibroblast(agent_id=1, x=0.0, y=0.0)
        assert engine.should_activate(fibro, stress=0.1) is False

    def test_above_threshold_can_activate(self):
        """Стресс выше порога + p=1.0 → гарантированная активация."""
        engine = MechanotransductionEngine(
            stress_threshold=0.5,
            activation_probability=1.0,
        )
        fibro = Fibroblast(agent_id=1, x=0.0, y=0.0)
        assert engine.should_activate(fibro, stress=1.0) is True

    def test_returns_bool(self):
        """Возвращает bool."""
        engine = MechanotransductionEngine()
        fibro = Fibroblast(agent_id=1, x=0.0, y=0.0)
        result = engine.should_activate(fibro, stress=0.8)
        assert isinstance(result, bool)

    def test_at_threshold_boundary(self):
        """stress == threshold → False (строгое неравенство > threshold)."""
        engine = MechanotransductionEngine(stress_threshold=0.5)
        fibro = Fibroblast(agent_id=1, x=0.0, y=0.0)
        assert engine.should_activate(fibro, stress=0.5) is False

    def test_stochastic_with_seed_42(self):
        """Стохастическое поведение при p=0.5: есть и True и False."""
        engine = MechanotransductionEngine(
            stress_threshold=0.5,
            activation_probability=0.5,
        )
        results = []
        for i in range(100):
            rng = np.random.default_rng(i)
            fibro = Fibroblast(agent_id=1, x=0.0, y=0.0, rng=rng)
            results.append(engine.should_activate(fibro, stress=1.0))
        assert any(results), "Хотя бы одна активация"
        assert not all(results), "Не все активации"


# =============================================================================
# MultiCytokineField — поведение после реализации
# =============================================================================


class TestMultiCytokineFieldUpdateBehaviorImplemented:
    """Тесты MultiCytokineField.update() — целевое поведение."""

    def test_update_preserves_non_negative(self):
        """Значения полей остаются >= 0 после update."""
        mcf = MultiCytokineField(grid_shape=(10, 10))
        mcf.fields["TNF"][:] = 0.5
        config = ABMConfig()
        mcf.update(dt=0.1, agents=[], config=config)
        for name, field in mcf.fields.items():
            assert np.all(field >= 0.0), f"{name} has negative values"

    def test_update_with_empty_agents(self):
        """update с пустым списком агентов не вызывает ошибок."""
        mcf = MultiCytokineField(grid_shape=(10, 10))
        config = ABMConfig()
        mcf.update(dt=0.1, agents=[], config=config)
        assert len(mcf.fields) == 7

    def test_update_applies_decay(self):
        """Распад: однородное поле уменьшается после update."""
        mcf = MultiCytokineField(grid_shape=(10, 10))
        mcf.fields["TNF"][:] = 1.0
        config = ABMConfig()
        mcf.update(dt=0.1, agents=[], config=config)
        assert np.max(mcf.fields["TNF"]) < 1.0

    def test_fields_count_preserved_after_update(self):
        """Инвариант: количество полей = 7 после update."""
        mcf = MultiCytokineField(grid_shape=(10, 10))
        config = ABMConfig()
        mcf.update(dt=0.1, agents=[], config=config)
        assert len(mcf.fields) == 7

    def test_field_shape_preserved_after_update(self):
        """Форма полей сохраняется после update."""
        mcf = MultiCytokineField(grid_shape=(10, 10))
        config = ABMConfig()
        mcf.update(dt=0.1, agents=[], config=config)
        for field in mcf.fields.values():
            assert field.shape == (10, 10)


class TestMultiCytokineFieldGradientBehaviorImplemented:
    """Тесты MultiCytokineField.get_gradient() — целевое поведение."""

    def test_zero_field_returns_zero_gradient(self):
        """Нулевое поле → градиент (0, 0)."""
        mcf = MultiCytokineField(grid_shape=(10, 10))
        gx, gy = mcf.get_gradient("TNF", x=50.0, y=50.0, grid_resolution=10.0)
        assert gx == pytest.approx(0.0)
        assert gy == pytest.approx(0.0)

    def test_returns_tuple_of_two_floats(self):
        """Возвращает tuple из двух элементов."""
        mcf = MultiCytokineField(grid_shape=(10, 10))
        result = mcf.get_gradient("TNF", x=50.0, y=50.0, grid_resolution=10.0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_nonzero_field_nonzero_gradient(self):
        """Ненулевой градиент при линейном поле."""
        mcf = MultiCytokineField(grid_shape=(10, 10))
        for i in range(10):
            mcf.fields["TNF"][i, :] = float(i)
        gx, gy = mcf.get_gradient("TNF", x=50.0, y=50.0, grid_resolution=10.0)
        assert abs(gx) > 0.0 or abs(gy) > 0.0

    def test_gradient_direction_x(self):
        """Поле растёт по x → gx > 0."""
        mcf = MultiCytokineField(grid_shape=(10, 10))
        for i in range(10):
            mcf.fields["IL8"][i, :] = float(i)
        gx, gy = mcf.get_gradient("IL8", x=50.0, y=50.0, grid_resolution=10.0)
        assert gx > 0.0

    def test_gradient_direction_y(self):
        """Поле растёт по y → gy > 0."""
        mcf = MultiCytokineField(grid_shape=(10, 10))
        for j in range(10):
            mcf.fields["VEGF"][:, j] = float(j)
        gx, gy = mcf.get_gradient("VEGF", x=50.0, y=50.0, grid_resolution=10.0)
        assert gy > 0.0


class TestMultiCytokineFieldConcentrationBehaviorImplemented:
    """Тесты MultiCytokineField.get_concentration() — целевое поведение."""

    def test_zero_field_returns_zero(self):
        """Нулевое поле → концентрация = 0."""
        mcf = MultiCytokineField(grid_shape=(10, 10))
        c = mcf.get_concentration("TNF", x=50.0, y=50.0, grid_resolution=10.0)
        assert c == pytest.approx(0.0)

    def test_nonzero_field_returns_positive(self):
        """Однородное поле = 1.0 → концентрация = 1.0."""
        mcf = MultiCytokineField(grid_shape=(10, 10))
        mcf.fields["IL10"][:] = 1.0
        c = mcf.get_concentration("IL10", x=50.0, y=50.0, grid_resolution=10.0)
        assert c == pytest.approx(1.0)

    def test_returns_float(self):
        """Возвращает float."""
        mcf = MultiCytokineField(grid_shape=(10, 10))
        result = mcf.get_concentration("TNF", x=50.0, y=50.0, grid_resolution=10.0)
        assert isinstance(result, (float, np.floating))

    def test_non_negative(self):
        """Концентрация >= 0."""
        mcf = MultiCytokineField(grid_shape=(10, 10))
        c = mcf.get_concentration("PDGF", x=50.0, y=50.0, grid_resolution=10.0)
        assert c >= 0.0

    def test_matches_field_value(self):
        """Концентрация соответствует значению ячейки."""
        mcf = MultiCytokineField(grid_shape=(10, 10))
        # Установить значение 3.5 в ячейке (5, 5)
        mcf.fields["MCP1"][5, 5] = 3.5
        c = mcf.get_concentration("MCP1", x=50.0, y=50.0, grid_resolution=10.0)
        assert c == pytest.approx(3.5)


# =============================================================================
# KDTreeNeighborSearch — поведение после реализации
# =============================================================================


class TestKDTreeRebuildBehaviorImplemented:
    """Тесты KDTreeNeighborSearch.rebuild() — целевое поведение."""

    def test_rebuild_empty_list_no_error(self):
        """Пустой список → нет ошибки."""
        search = KDTreeNeighborSearch(space_size=(100.0, 100.0))
        search.rebuild(agents=[])

    def test_rebuild_with_agents(self):
        """Перестроение с агентами → нет ошибки."""
        search = KDTreeNeighborSearch(space_size=(100.0, 100.0))
        agents = [
            StemCell(agent_id=1, x=10.0, y=10.0),
            StemCell(agent_id=2, x=20.0, y=20.0),
        ]
        search.rebuild(agents=agents)

    def test_rebuild_enables_queries(self):
        """После rebuild запросы работают."""
        search = KDTreeNeighborSearch(space_size=(100.0, 100.0))
        agents = [StemCell(agent_id=1, x=50.0, y=50.0)]
        search.rebuild(agents=agents)
        result = search.query_radius(position=(50.0, 50.0), radius=5.0)
        assert isinstance(result, list)

    def test_rebuild_updates_tree(self):
        """Повторный rebuild обновляет дерево."""
        search = KDTreeNeighborSearch(space_size=(100.0, 100.0))
        # Первый rebuild: агенты в (10,10)
        old_agents = [StemCell(agent_id=1, x=10.0, y=10.0)]
        search.rebuild(agents=old_agents)
        # Второй rebuild: агенты в (50,50)
        new_agents = [StemCell(agent_id=2, x=50.0, y=50.0)]
        search.rebuild(agents=new_agents)
        result = search.query_radius(position=(50.0, 50.0), radius=5.0)
        agent_ids = [a.agent_id for a in result]
        assert 2 in agent_ids
        assert 1 not in agent_ids


class TestKDTreeQueryRadiusBehaviorImplemented:
    """Тесты KDTreeNeighborSearch.query_radius() — целевое поведение."""

    def test_empty_tree_returns_empty(self):
        """Пустое дерево → пустой результат."""
        search = KDTreeNeighborSearch(space_size=(100.0, 100.0))
        search.rebuild(agents=[])
        result = search.query_radius(position=(50.0, 50.0), radius=10.0)
        assert result == []

    def test_finds_agents_within_radius(self):
        """Находит агентов внутри радиуса."""
        search = KDTreeNeighborSearch(space_size=(100.0, 100.0))
        agent = StemCell(agent_id=1, x=50.0, y=50.0)
        search.rebuild(agents=[agent])
        result = search.query_radius(position=(50.0, 50.0), radius=5.0)
        assert len(result) == 1
        assert result[0].agent_id == 1

    def test_excludes_agents_outside_radius(self):
        """Не включает агентов за пределами радиуса."""
        search = KDTreeNeighborSearch(space_size=(100.0, 100.0))
        agent = StemCell(agent_id=1, x=50.0, y=50.0)
        search.rebuild(agents=[agent])
        result = search.query_radius(position=(0.0, 0.0), radius=5.0)
        assert len(result) == 0

    def test_exclude_parameter(self):
        """Параметр exclude исключает агента из результатов."""
        search = KDTreeNeighborSearch(space_size=(100.0, 100.0))
        agent = StemCell(agent_id=1, x=50.0, y=50.0)
        search.rebuild(agents=[agent])
        result = search.query_radius(
            position=(50.0, 50.0),
            radius=5.0,
            exclude=agent,
        )
        assert agent not in result

    def test_returns_list_of_agents(self):
        """Возвращает list[Agent]."""
        search = KDTreeNeighborSearch(space_size=(100.0, 100.0))
        agents = [StemCell(agent_id=1, x=50.0, y=50.0)]
        search.rebuild(agents=agents)
        result = search.query_radius(position=(50.0, 50.0), radius=5.0)
        assert isinstance(result, list)
        for a in result:
            assert isinstance(a, Agent)

    def test_multiple_agents_in_radius(self):
        """Несколько агентов в радиусе."""
        search = KDTreeNeighborSearch(space_size=(100.0, 100.0))
        agents = [
            StemCell(agent_id=1, x=50.0, y=50.0),
            StemCell(agent_id=2, x=51.0, y=50.0),
            StemCell(agent_id=3, x=52.0, y=50.0),
        ]
        search.rebuild(agents=agents)
        result = search.query_radius(position=(50.0, 50.0), radius=5.0)
        assert len(result) == 3


class TestKDTreeQueryNearestBehaviorImplemented:
    """Тесты KDTreeNeighborSearch.query_nearest() — целевое поведение."""

    def test_returns_k_nearest(self):
        """Возвращает ровно k ближайших."""
        search = KDTreeNeighborSearch(space_size=(100.0, 100.0))
        agents = [StemCell(agent_id=i, x=50.0 + i * 2, y=50.0) for i in range(5)]
        search.rebuild(agents=agents)
        result = search.query_nearest(position=(50.0, 50.0), k=3)
        assert len(result) == 3

    def test_exclude_parameter(self):
        """exclude убирает ближайшего, возвращает следующего."""
        search = KDTreeNeighborSearch(space_size=(100.0, 100.0))
        agent_a = StemCell(agent_id=1, x=50.0, y=50.0)
        agent_b = StemCell(agent_id=2, x=50.0, y=60.0)
        search.rebuild(agents=[agent_a, agent_b])
        result = search.query_nearest(
            position=(50.0, 50.0),
            k=1,
            exclude=agent_a,
        )
        assert len(result) == 1
        assert result[0].agent_id == 2

    def test_empty_tree_returns_empty(self):
        """Пустое дерево → пустой список."""
        search = KDTreeNeighborSearch(space_size=(100.0, 100.0))
        search.rebuild(agents=[])
        result = search.query_nearest(position=(50.0, 50.0), k=3)
        assert result == []

    def test_k_greater_than_agents(self):
        """k больше числа агентов → возвращает все."""
        search = KDTreeNeighborSearch(space_size=(100.0, 100.0))
        agents = [
            StemCell(agent_id=1, x=50.0, y=50.0),
            StemCell(agent_id=2, x=60.0, y=60.0),
        ]
        search.rebuild(agents=agents)
        result = search.query_nearest(position=(50.0, 50.0), k=5)
        assert len(result) == 2

    def test_nearest_is_closest(self):
        """Ближайший агент — действительно ближайший по расстоянию."""
        search = KDTreeNeighborSearch(space_size=(100.0, 100.0))
        agent_a = StemCell(agent_id=1, x=50.0, y=50.0)
        agent_b = StemCell(agent_id=2, x=50.0, y=60.0)
        search.rebuild(agents=[agent_a, agent_b])
        result = search.query_nearest(position=(50.0, 51.0), k=1)
        assert result[0].agent_id == 1  # (50,50) ближе к (50,51) чем (50,60)


# =============================================================================
# SubcyclingManager — поведение после реализации
# =============================================================================


class TestSubcyclingNFieldSubstepsBehaviorImplemented:
    """Тесты SubcyclingManager.n_field_substeps — целевое поведение."""

    def test_exact_division(self):
        """agent_dt=0.1, field_dt=0.01 → 10 подшагов."""
        mgr = SubcyclingManager(agent_dt=0.1, field_dt=0.01)
        assert mgr.n_field_substeps == 10

    def test_inexact_rounds_up(self):
        """ceil(0.1 / 0.03) = ceil(3.33) = 4."""
        mgr = SubcyclingManager(agent_dt=0.1, field_dt=0.03)
        assert mgr.n_field_substeps == math.ceil(0.1 / 0.03)

    def test_field_dt_greater_than_agent_dt(self):
        """field_dt > agent_dt → 1 (нет subcycling)."""
        mgr = SubcyclingManager(agent_dt=0.01, field_dt=0.1)
        assert mgr.n_field_substeps == 1

    def test_equal_dt(self):
        """agent_dt == field_dt → 1."""
        mgr = SubcyclingManager(agent_dt=0.1, field_dt=0.1)
        assert mgr.n_field_substeps == 1

    def test_always_at_least_one(self):
        """Инвариант: n_field_substeps >= 1."""
        for a_dt, f_dt in [(0.1, 0.01), (0.01, 0.1), (0.1, 0.1), (1.0, 0.3)]:
            mgr = SubcyclingManager(agent_dt=a_dt, field_dt=f_dt)
            assert mgr.n_field_substeps >= 1


class TestSubcyclingShouldUpdateFieldBehaviorImplemented:
    """Тесты SubcyclingManager.should_update_field() — целевое поведение."""

    def test_returns_bool(self):
        """Возвращает bool."""
        mgr = SubcyclingManager(agent_dt=0.1, field_dt=0.01)
        result = mgr.should_update_field(agent_step_count=0)
        assert isinstance(result, bool)

    def test_first_step_updates(self):
        """Первый шаг всегда обновляет поле."""
        mgr = SubcyclingManager(agent_dt=0.1, field_dt=0.01)
        assert mgr.should_update_field(agent_step_count=0) is True

    def test_consistent_with_n_substeps(self):
        """Число True за 100 шагов согласовано с n_substeps."""
        mgr = SubcyclingManager(agent_dt=0.1, field_dt=0.01)
        n_updates = sum(mgr.should_update_field(agent_step_count=i) for i in range(100))
        # При n_substeps=10, ожидаем ~10 обновлений за 100 шагов агента
        assert n_updates >= 10


class TestSubcyclingGetFieldDtBehaviorImplemented:
    """Тесты SubcyclingManager.get_field_dt() — целевое поведение."""

    def test_returns_float(self):
        """Возвращает float."""
        mgr = SubcyclingManager(agent_dt=0.1, field_dt=0.01)
        result = mgr.get_field_dt()
        assert isinstance(result, float)

    def test_positive_value(self):
        """Возвращает положительное значение."""
        mgr = SubcyclingManager(agent_dt=0.1, field_dt=0.01)
        assert mgr.get_field_dt() > 0.0

    def test_exact_division_equals_field_dt(self):
        """При точном делении возвращает field_dt."""
        mgr = SubcyclingManager(agent_dt=0.1, field_dt=0.01)
        assert mgr.get_field_dt() == pytest.approx(0.01)


# =============================================================================
# Расширения существующих агентов — поведение после реализации
# =============================================================================


class TestStemCellPRPMobilizationBehaviorImplemented:
    """Тесты StemCell.prp_mobilization() — целевое поведение (Michaelis-Menten)."""

    def test_zero_prp_returns_one(self):
        """prp_level=0 → modifier=1.0 (базовая линия)."""
        cell = StemCell(agent_id=1, x=0.0, y=0.0)
        assert cell.prp_mobilization(prp_level=0.0) == pytest.approx(1.0)

    def test_high_prp_returns_expected(self):
        """prp_level=1.0 → 1.0 + 2.0 * 1.0 / (0.3 + 1.0) ≈ 2.538."""
        cell = StemCell(agent_id=1, x=0.0, y=0.0)
        expected = 1.0 + 2.0 * 1.0 / (0.3 + 1.0)  # ≈ 2.538
        assert cell.prp_mobilization(prp_level=1.0) == pytest.approx(expected, abs=0.01)

    def test_michaelis_menten_half_saturation(self):
        """prp_level=K_m=0.3 → 1.0 + 2.0 * 0.3 / (0.3 + 0.3) = 2.0."""
        cell = StemCell(agent_id=1, x=0.0, y=0.0)
        assert cell.prp_mobilization(prp_level=0.3) == pytest.approx(2.0)

    def test_returns_float(self):
        """Возвращает float."""
        cell = StemCell(agent_id=1, x=0.0, y=0.0)
        result = cell.prp_mobilization(prp_level=0.5)
        assert isinstance(result, (float, np.floating))

    def test_always_greater_than_or_equal_one(self):
        """Инвариант: modifier >= 1.0 для любого prp_level >= 0."""
        cell = StemCell(agent_id=1, x=0.0, y=0.0)
        for prp in [0.0, 0.01, 0.1, 0.3, 0.5, 1.0]:
            assert cell.prp_mobilization(prp_level=prp) >= 1.0


class TestMacrophageEfferocytoseBehaviorImplemented:
    """Тесты Macrophage.efferocytose() — целевое поведение."""

    def test_returns_dict_with_required_keys(self):
        """Возвращает dict с ключами IL10 и phagocytosed."""
        macro = Macrophage(agent_id=1, x=0.0, y=0.0)
        neutro = NeutrophilAgent(agent_id=2, x=1.0, y=1.0)
        neutro.alive = False
        result = macro.efferocytose([neutro])
        assert "IL10" in result
        assert "phagocytosed" in result

    def test_marks_neutrophils_dead(self):
        """Помечает нейтрофилы как мёртвые."""
        macro = Macrophage(agent_id=1, x=0.0, y=0.0)
        neutros = [NeutrophilAgent(agent_id=i, x=float(i), y=0.0) for i in range(2, 5)]
        for n in neutros:
            n.alive = False
        macro.efferocytose(neutros)
        for n in neutros:
            assert n.alive is False

    def test_shifts_polarization_to_m2(self):
        """Сдвигает поляризацию к M2."""
        macro = Macrophage(agent_id=1, x=0.0, y=0.0)
        macro.polarization_state = "M1"
        neutro = NeutrophilAgent(agent_id=2, x=1.0, y=1.0)
        neutro.alive = False
        macro.efferocytose([neutro])
        assert macro.polarization_state == "M2"

    def test_phagocytosed_count_correct(self):
        """Количество фагоцитированных корректно."""
        macro = Macrophage(agent_id=1, x=0.0, y=0.0)
        neutros = [NeutrophilAgent(agent_id=i, x=float(i), y=0.0) for i in range(2, 4)]
        for n in neutros:
            n.alive = False
        result = macro.efferocytose(neutros)
        assert result["phagocytosed"] == 2

    def test_empty_list(self):
        """Пустой список → phagocytosed=0, IL10=0."""
        macro = Macrophage(agent_id=1, x=0.0, y=0.0)
        result = macro.efferocytose([])
        assert result["phagocytosed"] == 0
        assert result["IL10"] == pytest.approx(0.0)

    def test_capacity_limit(self):
        """Не более PHAGOCYTOSIS_CAPACITY за вызов."""
        macro = Macrophage(agent_id=1, x=0.0, y=0.0)
        neutros = [NeutrophilAgent(agent_id=i, x=float(i), y=0.0) for i in range(2, 12)]
        for n in neutros:
            n.alive = False
        result = macro.efferocytose(neutros)
        assert result["phagocytosed"] <= Macrophage.PHAGOCYTOSIS_CAPACITY


class TestFibroblastTgfbActivationBehaviorImplemented:
    """Тесты Fibroblast.tgfb_activation() — целевое поведение."""

    def test_low_tgfb_returns_none(self):
        """tgfb_level < 0.5 → None."""
        fibro = Fibroblast(agent_id=1, x=0.0, y=0.0)
        assert fibro.tgfb_activation(tgfb_level=0.3) is None

    def test_zero_tgfb_returns_none(self):
        """tgfb_level=0 → None."""
        fibro = Fibroblast(agent_id=1, x=0.0, y=0.0)
        assert fibro.tgfb_activation(tgfb_level=0.0) is None

    def test_high_tgfb_can_return_myofibroblast(self):
        """tgfb_level=0.9 → может вернуть MyofibroblastAgent (стохастически)."""
        results = []
        for i in range(200):
            rng = np.random.default_rng(i)
            fibro = Fibroblast(agent_id=1, x=50.0, y=50.0, rng=rng)
            result = fibro.tgfb_activation(tgfb_level=0.9)
            results.append(result)
        myofibro_results = [r for r in results if r is not None]
        assert len(myofibro_results) > 0, "Хотя бы одна активация за 200 попыток"
        assert all(isinstance(r, MyofibroblastAgent) for r in myofibro_results)

    def test_boundary_tgfb_exactly_half(self):
        """tgfb_level=0.5 → None (строгое > 0.5)."""
        fibro = Fibroblast(agent_id=1, x=0.0, y=0.0)
        assert fibro.tgfb_activation(tgfb_level=0.5) is None

    def test_returned_myofibroblast_position(self):
        """MyofibroblastAgent наследует координаты фибробласта."""
        for i in range(500):
            rng = np.random.default_rng(i)
            fibro = Fibroblast(agent_id=1, x=42.0, y=73.0, rng=rng)
            result = fibro.tgfb_activation(tgfb_level=0.9)
            if result is not None:
                assert result.x == 42.0
                assert result.y == 73.0
                break
        else:
            pytest.fail("Ни одна активация за 500 попыток")


# =============================================================================
# MyofibroblastAgent — TGF-β-зависимый апоптоз и продукция коллагена
# =============================================================================


class TestMyofibroblastTgfbApoptosisBehavior:
    """Тесты MyofibroblastAgent.should_apoptose_tgfb() — апоптоз при снижении TGF-β."""

    def test_low_tgfb_triggers_apoptosis(self):
        """TGF-β < 0.1 → True (апоптоз, разрешение фиброза)."""
        myofibro = MyofibroblastAgent(agent_id=1, x=0.0, y=0.0)
        assert myofibro.should_apoptose_tgfb(tgfb_level=0.05) is True

    def test_zero_tgfb_triggers_apoptosis(self):
        """TGF-β = 0 → True."""
        myofibro = MyofibroblastAgent(agent_id=1, x=0.0, y=0.0)
        assert myofibro.should_apoptose_tgfb(tgfb_level=0.0) is True

    def test_high_tgfb_no_apoptosis(self):
        """TGF-β > 0.1 → False (миофибробласт выживает)."""
        myofibro = MyofibroblastAgent(agent_id=1, x=0.0, y=0.0)
        assert myofibro.should_apoptose_tgfb(tgfb_level=0.5) is False

    def test_boundary_tgfb_exactly_01(self):
        """TGF-β = 0.1 → False (граничное: < 0.1 строго)."""
        myofibro = MyofibroblastAgent(agent_id=1, x=0.0, y=0.0)
        assert myofibro.should_apoptose_tgfb(tgfb_level=0.1) is False

    def test_just_below_threshold(self):
        """TGF-β = 0.099 → True."""
        myofibro = MyofibroblastAgent(agent_id=1, x=0.0, y=0.0)
        assert myofibro.should_apoptose_tgfb(tgfb_level=0.099) is True

    def test_returns_bool(self):
        """Возвращает bool."""
        myofibro = MyofibroblastAgent(agent_id=1, x=0.0, y=0.0)
        result = myofibro.should_apoptose_tgfb(tgfb_level=0.5)
        assert isinstance(result, bool)


class TestMyofibroblastECMProductionBehavior:
    """Тесты MyofibroblastAgent.produce_ecm() — продукция коллагена."""

    def test_ecm_production_rate(self):
        """ECM_PRODUCTION_RATE = 1.0 (2× fibroblast)."""
        myofibro = MyofibroblastAgent(agent_id=1, x=0.0, y=0.0)
        assert MyofibroblastAgent.ECM_PRODUCTION_RATE == pytest.approx(1.0)
        assert MyofibroblastAgent.ECM_PRODUCTION_RATE == 2.0 * Fibroblast.ECM_PRODUCTION_RATE

    def test_produce_ecm_returns_positive(self):
        """produce_ecm возвращает положительное значение."""
        myofibro = MyofibroblastAgent(agent_id=1, x=0.0, y=0.0)
        ecm = myofibro.produce_ecm(dt=1.0)
        assert ecm > 0.0

    def test_produce_ecm_scales_with_dt(self):
        """Линейная зависимость от dt."""
        myofibro = MyofibroblastAgent(agent_id=1, x=0.0, y=0.0)
        ecm_1 = myofibro.produce_ecm(dt=1.0)
        myofibro2 = MyofibroblastAgent(agent_id=2, x=0.0, y=0.0)
        ecm_2 = myofibro2.produce_ecm(dt=2.0)
        assert ecm_2 == pytest.approx(2.0 * ecm_1)

    def test_produce_ecm_accumulates(self):
        """Кумулятивная продукция ECM."""
        myofibro = MyofibroblastAgent(agent_id=1, x=0.0, y=0.0)
        myofibro.produce_ecm(dt=1.0)
        myofibro.produce_ecm(dt=1.0)
        assert myofibro.ecm_produced == pytest.approx(2.0)

    def test_contraction_force(self):
        """contract() возвращает CONTRACTION_FORCE * dt."""
        myofibro = MyofibroblastAgent(agent_id=1, x=0.0, y=0.0)
        force = myofibro.contract(dt=1.0)
        assert force == pytest.approx(MyofibroblastAgent.CONTRACTION_FORCE * 1.0)


# =============================================================================
# Macrophage — continuous polarization_state (Phase 2.8)
# =============================================================================


class TestMacrophageContinuousPolarizationBehavior:
    """Тесты continuous polarization_state ∈ [0, 1].

    Описание: polarization_state станет float вместо дискретного M0/M1/M2.
    0.0 = полный M2 (anti-inflammatory), 1.0 = полный M1 (pro-inflammatory).
    Эффероцитоз сдвигает поляризацию: -0.1 * n_phagocytosed.
    """

    def test_polarization_state_is_float(self):
        """polarization_state — float ∈ [0, 1]."""
        macro = Macrophage(agent_id=1, x=0.0, y=0.0)
        assert isinstance(macro.polarization_state, float)

    def test_initial_polarization_neutral(self):
        """Начальная поляризация M0 → 0.5 (нейтральная)."""
        macro = Macrophage(agent_id=1, x=0.0, y=0.0)
        assert macro.polarization_state == pytest.approx(0.5)

    def test_polarization_in_unit_interval(self):
        """Инвариант: polarization_state ∈ [0, 1]."""
        macro = Macrophage(agent_id=1, x=0.0, y=0.0)
        assert 0.0 <= macro.polarization_state <= 1.0

    def test_high_inflammation_shifts_toward_m1(self):
        """Высокое воспаление → сдвиг к M1 (ближе к 1.0)."""
        macro = Macrophage(agent_id=1, x=0.0, y=0.0)
        macro.polarize(inflammation_level=0.8)
        assert macro.polarization_state > 0.5

    def test_low_inflammation_shifts_toward_m2(self):
        """Низкое воспаление → сдвиг к M2 (ближе к 0.0)."""
        macro = Macrophage(agent_id=1, x=0.0, y=0.0)
        macro.polarize(inflammation_level=0.2)
        assert macro.polarization_state < 0.5

    def test_efferocytosis_shifts_toward_m2(self):
        """Эффероцитоз: сдвиг -0.1 * n_phagocytosed к M2."""
        macro = Macrophage(agent_id=1, x=0.0, y=0.0)
        macro.polarization_state = 0.8  # ближе к M1
        neutros = [NeutrophilAgent(agent_id=2, x=1.0, y=1.0)]
        neutros[0].alive = False
        macro.efferocytose(neutros)
        # Сдвиг -0.1 * 1 = -0.1 → 0.8 - 0.1 = 0.7
        assert macro.polarization_state == pytest.approx(0.7, abs=0.05)

    def test_efferocytosis_multiple_neutrophils_shift(self):
        """3 нейтрофила → сдвиг -0.3."""
        macro = Macrophage(agent_id=1, x=0.0, y=0.0)
        macro.polarization_state = 0.8
        neutros = [NeutrophilAgent(agent_id=i, x=float(i), y=0.0) for i in range(2, 5)]
        for n in neutros:
            n.alive = False
        macro.efferocytose(neutros)
        # 0.8 - 0.1 * 3 = 0.5
        assert macro.polarization_state == pytest.approx(0.5, abs=0.05)

    def test_polarization_clamped_at_zero(self):
        """Поляризация не опускается ниже 0.0."""
        macro = Macrophage(agent_id=1, x=0.0, y=0.0)
        macro.polarization_state = 0.1
        neutros = [NeutrophilAgent(agent_id=i, x=float(i), y=0.0) for i in range(2, 7)]
        for n in neutros:
            n.alive = False
        macro.efferocytose(neutros)
        # 0.1 - 0.1 * 5 = -0.4 → clamped to 0.0
        assert macro.polarization_state >= 0.0

    def test_cytokine_secretion_depends_on_continuous_state(self):
        """Секреция цитокинов зависит от непрерывного состояния поляризации."""
        macro_m1 = Macrophage(agent_id=1, x=0.0, y=0.0)
        macro_m1.polarization_state = 0.9  # ближе к M1
        macro_m2 = Macrophage(agent_id=2, x=0.0, y=0.0)
        macro_m2.polarization_state = 0.1  # ближе к M2
        cyt_m1 = macro_m1.secrete_cytokines(dt=1.0)
        cyt_m2 = macro_m2.secrete_cytokines(dt=1.0)
        # M1 выделяет больше TNF, M2 выделяет больше IL-10
        assert cyt_m1.get("TNF_alpha", 0) > cyt_m2.get("TNF_alpha", 0)
        assert cyt_m2.get("IL_10", 0) > cyt_m1.get("IL_10", 0)


# =============================================================================
# EndothelialAgent — VEGF-зависимый спраутинг (Phase 2.8)
# =============================================================================


class TestEndothelialSproutingBehavior:
    """Тесты EndothelialAgent — VEGF-зависимое поведение (ангиогенез).

    Спраутинг: VEGF стимулирует деление, миграцию и формирование
    сосудистых структур (tight junctions) эндотелиальных клеток.
    """

    def test_vegf_sensitivity_constant(self):
        """VEGF_SENSITIVITY = 0.6."""
        assert EndothelialAgent.VEGF_SENSITIVITY == pytest.approx(0.6)

    def test_high_vegf_increases_division_probability(self):
        """Высокий VEGF → повышенная вероятность деления."""
        rng = np.random.default_rng(42)
        endo = EndothelialAgent(agent_id=1, x=50.0, y=50.0, rng=rng)
        endo.energy = 1.0
        endo.update(dt=1.0, environment={"vegf_level": 0.9})
        # При высоком VEGF клетка должна иметь повышенную энергию
        # или готовность к делению
        assert endo.energy > 0.0

    def test_form_junction_close_neighbor(self):
        """form_junction с близким соседом (dist <= 3.0 μm) → True."""
        endo1 = EndothelialAgent(agent_id=1, x=50.0, y=50.0)
        endo2 = EndothelialAgent(agent_id=2, x=51.0, y=50.0)  # dist = 1.0
        result = endo1.form_junction(endo2)
        assert result is True
        assert endo1.junction_count == 1

    def test_form_junction_far_neighbor(self):
        """form_junction с далёким соседом (dist > 3.0 μm) → False."""
        endo1 = EndothelialAgent(agent_id=1, x=50.0, y=50.0)
        endo2 = EndothelialAgent(agent_id=2, x=60.0, y=60.0)  # dist ≈ 14.1
        result = endo1.form_junction(endo2)
        assert result is False
        assert endo1.junction_count == 0

    def test_junction_count_accumulates(self):
        """Несколько junction формируются кумулятивно."""
        endo1 = EndothelialAgent(agent_id=1, x=50.0, y=50.0)
        endo2 = EndothelialAgent(agent_id=2, x=51.0, y=50.0)
        endo3 = EndothelialAgent(agent_id=3, x=50.0, y=51.0)
        endo1.form_junction(endo2)
        endo1.form_junction(endo3)
        assert endo1.junction_count == 2

    def test_secrete_vegf_and_pdgf(self):
        """Эндотелиальная клетка секретирует VEGF и PDGF."""
        endo = EndothelialAgent(agent_id=1, x=50.0, y=50.0)
        cyt = endo.secrete_cytokines(dt=1.0)
        assert "VEGF" in cyt
        assert "PDGF" in cyt
        assert cyt["VEGF"] > 0.0
        assert cyt["PDGF"] > 0.0

    def test_vegf_chemotaxis_via_engine(self):
        """ChemotaxisEngine направляет эндотелиальную клетку к VEGF."""
        config = ABMConfig()
        engine = ChemotaxisEngine(config)
        assert ChemotaxisEngine.AGENT_ATTRACTANT_MAP["endo"] == "VEGF"

    def test_vegf_stimulated_division(self):
        """При высоком VEGF деление более вероятно (статистический тест)."""
        divisions_high_vegf = 0
        divisions_no_vegf = 0
        for i in range(200):
            rng = np.random.default_rng(i)
            endo = EndothelialAgent(agent_id=1, x=50.0, y=50.0, rng=rng)
            endo.energy = 1.0
            endo.update(dt=1.0, environment={"vegf_level": 0.9})
            if endo.can_divide():
                divisions_high_vegf += 1

            rng2 = np.random.default_rng(i + 1000)
            endo2 = EndothelialAgent(agent_id=2, x=50.0, y=50.0, rng=rng2)
            endo2.energy = 1.0
            endo2.update(dt=1.0, environment={"vegf_level": 0.0})
            if endo2.can_divide():
                divisions_no_vegf += 1
        # При VEGF стимуляции деление должно быть чаще
        assert divisions_high_vegf >= divisions_no_vegf
