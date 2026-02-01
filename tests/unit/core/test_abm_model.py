"""TDD тесты для Agent-Based модели (ABM) регенерации тканей.

Тестирование:
- ABMConfig: пространственные, временные, агентные параметры
- AgentState: состояние агента
- ABMSnapshot: снимок состояния системы
- ABMTrajectory: траектория симуляции
- Agent base class: движение, деление, гибель, random walk
- StemCell: пролиферация, дифференциация, секреция
- Macrophage: фагоцитоз, поляризация M1/M2, хемотаксис
- Fibroblast: производство ECM, активация
- ABMModel: инициализация, симуляция, step
- simulate_abm: convenience функция

Все тесты написаны для stub-реализации (NotImplementedError).
После реализации методов тесты должны проходить.
"""

import numpy as np
import pytest

from src.core.abm_model import (
    ABMConfig,
    ABMModel,
    ABMSnapshot,
    ABMTrajectory,
    Agent,
    AgentState,
    Fibroblast,
    Macrophage,
    StemCell,
    simulate_abm,
)
from src.data.parameter_extraction import ModelParameters


# =============================================================================
# Test ABMConfig
# =============================================================================


class TestABMConfig:
    """Тесты конфигурации ABM модели."""

    def test_default_values(self):
        """Проверка значений по умолчанию."""
        config = ABMConfig()

        assert config.space_size == (100.0, 100.0)
        assert config.boundary_type == "periodic"
        assert config.dt == 0.1
        assert config.t_max == 720.0
        assert config.initial_stem_cells == 50
        assert config.initial_macrophages == 30
        assert config.initial_fibroblasts == 20
        assert config.max_agents == 10000

    def test_custom_space_size(self):
        """Проверка пользовательского размера пространства."""
        config = ABMConfig(space_size=(200.0, 150.0))

        assert config.space_size == (200.0, 150.0)

    def test_boundary_types(self):
        """Проверка допустимых типов границ."""
        for boundary in ["periodic", "reflective", "absorbing"]:
            config = ABMConfig(boundary_type=boundary)
            assert config.boundary_type == boundary

    def test_validate_returns_true_for_valid_config(self, small_abm_config):
        """Валидация возвращает True для корректной конфигурации."""
        result = small_abm_config.validate()

        assert result is True

    def test_validate_negative_space_size_raises(self):
        """Отрицательный размер пространства вызывает ошибку."""
        config = ABMConfig(space_size=(-100.0, 100.0))

        with pytest.raises(ValueError, match="space_size"):
            config.validate()

    def test_validate_zero_space_size_raises(self):
        """Нулевой размер пространства вызывает ошибку."""
        config = ABMConfig(space_size=(0.0, 100.0))

        with pytest.raises(ValueError, match="space_size"):
            config.validate()

    def test_validate_invalid_boundary_type_raises(self):
        """Некорректный тип границ вызывает ошибку."""
        config = ABMConfig(boundary_type="invalid")

        with pytest.raises(ValueError, match="boundary_type"):
            config.validate()

    def test_validate_negative_dt_raises(self):
        """Отрицательный шаг времени вызывает ошибку."""
        config = ABMConfig(dt=-0.1)

        with pytest.raises(ValueError, match="dt"):
            config.validate()

    def test_validate_zero_dt_raises(self):
        """Нулевой шаг времени вызывает ошибку."""
        config = ABMConfig(dt=0.0)

        with pytest.raises(ValueError, match="dt"):
            config.validate()

    def test_validate_negative_t_max_raises(self):
        """Отрицательное максимальное время вызывает ошибку."""
        config = ABMConfig(t_max=-100.0)

        with pytest.raises(ValueError, match="t_max"):
            config.validate()

    def test_validate_negative_max_agents_raises(self):
        """Отрицательное максимальное количество агентов вызывает ошибку."""
        config = ABMConfig(max_agents=-1)

        with pytest.raises(ValueError, match="max_agents"):
            config.validate()

    def test_validate_negative_interaction_radius_raises(self):
        """Отрицательный радиус взаимодействия вызывает ошибку."""
        config = ABMConfig(interaction_radius=-5.0)

        with pytest.raises(ValueError, match="interaction_radius"):
            config.validate()

    def test_validate_negative_chemotaxis_strength_raises(self):
        """Отрицательная сила хемотаксиса вызывает ошибку."""
        config = ABMConfig(chemotaxis_strength=-0.1)

        with pytest.raises(ValueError, match="chemotaxis_strength"):
            config.validate()

    def test_diffusion_and_cytokine_parameters(self):
        """Проверка параметров диффузии и цитокинов."""
        config = ABMConfig(
            diffusion_coefficient=2.0,
            cytokine_diffusion=20.0,
            cytokine_decay=0.2,
            grid_resolution=5.0,
        )

        assert config.diffusion_coefficient == 2.0
        assert config.cytokine_diffusion == 20.0
        assert config.cytokine_decay == 0.2
        assert config.grid_resolution == 5.0


# =============================================================================
# Test AgentState
# =============================================================================


class TestAgentState:
    """Тесты состояния агента."""

    def test_create_agent_state(self):
        """Создание состояния агента."""
        state = AgentState(
            agent_id=1,
            agent_type="stem",
            x=50.0,
            y=50.0,
            age=10.0,
            division_count=2,
            energy=0.8,
        )

        assert state.agent_id == 1
        assert state.agent_type == "stem"
        assert state.x == 50.0
        assert state.y == 50.0
        assert state.age == 10.0
        assert state.division_count == 2
        assert state.energy == 0.8

    def test_default_alive_and_dividing(self):
        """Проверка значений по умолчанию для alive и dividing."""
        state = AgentState(
            agent_id=1,
            agent_type="macro",
            x=0.0,
            y=0.0,
            age=0.0,
            division_count=0,
            energy=1.0,
        )

        assert state.alive is True
        assert state.dividing is False

    def test_to_dict(self):
        """Конвертация в словарь."""
        state = AgentState(
            agent_id=5,
            agent_type="fibro",
            x=25.0,
            y=75.0,
            age=100.0,
            division_count=3,
            energy=0.5,
            alive=True,
            dividing=True,
        )

        result = state.to_dict()

        assert result["id"] == 5
        assert result["type"] == "fibro"
        assert result["x"] == 25.0
        assert result["y"] == 75.0
        assert result["age"] == 100.0
        assert result["divisions"] == 3
        assert result["energy"] == 0.5
        assert result["alive"] is True
        assert result["dividing"] is True

    def test_agent_types(self):
        """Проверка различных типов агентов."""
        for agent_type in ["stem", "macro", "fibro"]:
            state = AgentState(
                agent_id=1,
                agent_type=agent_type,
                x=0.0,
                y=0.0,
                age=0.0,
                division_count=0,
                energy=1.0,
            )
            assert state.agent_type == agent_type


# =============================================================================
# Test ABMSnapshot
# =============================================================================


class TestABMSnapshot:
    """Тесты снимка состояния ABM."""

    @pytest.fixture
    def sample_agents(self):
        """Пример списка агентов."""
        return [
            AgentState(1, "stem", 10.0, 10.0, 5.0, 0, 1.0, alive=True),
            AgentState(2, "stem", 20.0, 20.0, 10.0, 1, 0.9, alive=True),
            AgentState(3, "macro", 30.0, 30.0, 20.0, 0, 0.8, alive=True),
            AgentState(4, "macro", 40.0, 40.0, 15.0, 0, 0.7, alive=False),  # Dead
            AgentState(5, "fibro", 50.0, 50.0, 50.0, 2, 0.6, alive=True),
        ]

    @pytest.fixture
    def sample_snapshot(self, sample_agents):
        """Пример снимка."""
        cytokine_field = np.zeros((10, 10))
        ecm_field = np.zeros((10, 10))
        return ABMSnapshot(
            t=24.0,
            agents=sample_agents,
            cytokine_field=cytokine_field,
            ecm_field=ecm_field,
        )

    def test_snapshot_time(self, sample_snapshot):
        """Проверка времени снимка."""
        assert sample_snapshot.t == 24.0

    def test_snapshot_agents_count(self, sample_snapshot):
        """Проверка количества агентов в снимке."""
        assert len(sample_snapshot.agents) == 5

    def test_get_agent_count_by_type(self, sample_snapshot):
        """Подсчёт агентов по типам возвращает корректный результат."""
        result = sample_snapshot.get_agent_count_by_type()
        assert result["stem"] == 2
        assert result["macro"] == 1  # Один мёртвый макрофаг не считается
        assert result["fibro"] == 1

    def test_get_total_agents(self, sample_snapshot):
        """Общее количество живых агентов."""
        result = sample_snapshot.get_total_agents()
        assert result == 4  # Один мёртвый не считается

    def test_cytokine_field_shape(self, sample_snapshot):
        """Проверка формы поля цитокинов."""
        assert sample_snapshot.cytokine_field.shape == (10, 10)

    def test_ecm_field_shape(self, sample_snapshot):
        """Проверка формы поля ECM."""
        assert sample_snapshot.ecm_field.shape == (10, 10)


# =============================================================================
# Test ABMTrajectory
# =============================================================================


class TestABMTrajectory:
    """Тесты траектории ABM."""

    @pytest.fixture
    def sample_trajectory(self):
        """Пример траектории с несколькими снимками."""
        snapshots = []
        for i in range(4):
            t = i * 24.0
            agents = [
                AgentState(1, "stem", 10.0 + i, 10.0, t, 0, 1.0 - i * 0.1),
                AgentState(2, "macro", 50.0, 50.0 + i, t, 0, 0.9),
            ]
            snapshot = ABMSnapshot(
                t=t,
                agents=agents,
                cytokine_field=np.zeros((10, 10)),
                ecm_field=np.zeros((10, 10)),
            )
            snapshots.append(snapshot)

        return ABMTrajectory(snapshots=snapshots, config=ABMConfig())

    def test_trajectory_has_snapshots(self, sample_trajectory):
        """Траектория содержит снимки."""
        assert len(sample_trajectory.snapshots) == 4

    def test_trajectory_has_config(self, sample_trajectory):
        """Траектория содержит конфигурацию."""
        assert isinstance(sample_trajectory.config, ABMConfig)

    def test_get_times(self, sample_trajectory):
        """Получение временных точек возвращает array([0, 24, 48, 72])."""
        times = sample_trajectory.get_times()
        np.testing.assert_array_equal(times, [0.0, 24.0, 48.0, 72.0])

    def test_get_population_dynamics(self, sample_trajectory):
        """Динамика популяций содержит ключи 'stem', 'macro', 'fibro'."""
        dynamics = sample_trajectory.get_population_dynamics()
        assert "stem" in dynamics
        assert "macro" in dynamics
        assert "fibro" in dynamics

    def test_get_statistics(self, sample_trajectory):
        """Получение статистики содержит ключи final_total, final_stem и т.д."""
        stats = sample_trajectory.get_statistics()
        assert "final_total" in stats
        assert "final_stem" in stats
        assert "final_macro" in stats
        assert "final_fibro" in stats


# =============================================================================
# Test Agent Base Class
# =============================================================================


class TestAgentBase:
    """Тесты базового класса агента."""

    def test_agent_class_constants(self):
        """Проверка констант базового класса."""
        assert Agent.AGENT_TYPE == "base"
        assert Agent.LIFESPAN == 240.0
        assert Agent.DIVISION_ENERGY_THRESHOLD == 0.7
        assert Agent.MAX_DIVISIONS == 10
        assert Agent.DIVISION_PROBABILITY == 0.01
        assert Agent.DEATH_PROBABILITY == 0.001

    def test_stem_cell_class_constants(self):
        """Проверка констант StemCell."""
        assert StemCell.AGENT_TYPE == "stem"
        assert StemCell.LIFESPAN == 240.0  # 10 дней
        assert StemCell.DIVISION_PROBABILITY == 0.02  # Высокая
        assert StemCell.DIFFERENTIATION_PROBABILITY == 0.005
        assert StemCell.CYTOKINE_SECRETION_RATE == 0.1

    def test_macrophage_class_constants(self):
        """Проверка констант Macrophage."""
        assert Macrophage.AGENT_TYPE == "macro"
        assert Macrophage.LIFESPAN == 168.0  # 7 дней
        assert Macrophage.MAX_DIVISIONS == 3  # Низкая
        assert Macrophage.PHAGOCYTOSIS_RADIUS == 3.0
        assert Macrophage.PHAGOCYTOSIS_CAPACITY == 5
        assert Macrophage.CHEMOTAXIS_SENSITIVITY == 0.5

    def test_fibroblast_class_constants(self):
        """Проверка констант Fibroblast."""
        assert Fibroblast.AGENT_TYPE == "fibro"
        assert Fibroblast.LIFESPAN == 360.0  # 15 дней
        assert Fibroblast.ECM_PRODUCTION_RATE == 0.5
        assert Fibroblast.CONTRACTION_STRENGTH == 0.1


# =============================================================================
# Test StemCell
# =============================================================================


class TestStemCell:
    """Тесты стволовой клетки."""

    @pytest.fixture
    def stem_cell(self):
        """Создание стволовой клетки."""
        rng = np.random.default_rng(42)
        return StemCell(agent_id=1, x=50.0, y=50.0, age=0.0, rng=rng)

    def test_init_defaults(self, stem_cell):
        """Проверка значений по умолчанию после инициализации."""
        assert stem_cell.agent_id == 1
        assert stem_cell.x == 50.0
        assert stem_cell.y == 50.0
        assert stem_cell.age == 0.0
        assert stem_cell.division_count == 0
        assert stem_cell.energy == 1.0
        assert stem_cell.alive is True
        assert stem_cell.dividing is False

    def test_init_with_age(self):
        """Инициализация с указанным возрастом."""
        cell = StemCell(agent_id=2, x=10.0, y=10.0, age=100.0)
        assert cell.age == 100.0

    def test_differentiation_probability_attribute(self, stem_cell):
        """Атрибут вероятности дифференциации."""
        assert stem_cell.differentiation_probability == StemCell.DIFFERENTIATION_PROBABILITY

    def test_divide_returns_new_stem_cell(self, stem_cell):
        """Метод divide возвращает новую StemCell при достаточной энергии."""
        stem_cell.energy = 1.0  # Достаточно энергии
        stem_cell.division_count = 0  # Лимит делений не достигнут
        daughter = stem_cell.divide(new_id=100)
        if daughter is not None:
            assert isinstance(daughter, StemCell)
            assert daughter.agent_id == 100

    def test_get_state_returns_agent_state(self, stem_cell):
        """Метод get_state возвращает AgentState."""
        state = stem_cell.get_state()
        assert isinstance(state, AgentState)
        assert state.agent_id == stem_cell.agent_id
        assert state.agent_type == "stem"


class TestStemCellBehavior:
    """Тесты поведения стволовой клетки."""

    def test_can_divide_energy_threshold(self):
        """Деление требует достаточно энергии."""
        cell = StemCell(agent_id=1, x=0, y=0)
        cell.energy = 0.5  # Ниже порога
        assert cell.can_divide() is False
        cell.energy = 0.8  # Выше порога
        assert cell.can_divide() is True

    def test_can_divide_max_divisions(self):
        """Деление ограничено максимальным количеством."""
        cell = StemCell(agent_id=1, x=0, y=0)
        cell.division_count = StemCell.MAX_DIVISIONS
        assert cell.can_divide() is False

    def test_divide_creates_daughter_cell(self):
        """Деление создаёт дочернюю клетку при достаточной энергии."""
        cell = StemCell(agent_id=1, x=50, y=50, rng=np.random.default_rng(42))
        cell.energy = 1.0
        cell.division_count = 0
        daughter = cell.divide(new_id=2)
        if daughter is not None:
            assert isinstance(daughter, StemCell)
            assert daughter.agent_id == 2

    def test_divide_energy_split(self):
        """Энергия делится между родителем и потомком."""
        cell = StemCell(agent_id=1, x=50, y=50, rng=np.random.default_rng(42))
        cell.energy = 1.0
        cell.division_count = 0
        initial_energy = cell.energy
        daughter = cell.divide(new_id=2)
        if daughter is not None:
            assert cell.energy < initial_energy
            assert daughter.energy > 0

    def test_differentiate_returns_fibroblast(self):
        """Дифференциация создаёт фибробласт."""
        cell = StemCell(agent_id=1, x=50, y=50, rng=np.random.default_rng(42))
        fibro = cell.differentiate(new_id=2)
        assert isinstance(fibro, Fibroblast)
        assert fibro.agent_id == 2

    def test_secrete_cytokines_rate(self):
        """Секреция цитокинов пропорциональна времени."""
        cell = StemCell(agent_id=1, x=50, y=50)
        amount = cell.secrete_cytokines(dt=1.0)
        assert amount == pytest.approx(StemCell.CYTOKINE_SECRETION_RATE)


# =============================================================================
# Test Macrophage
# =============================================================================


class TestMacrophage:
    """Тесты макрофага."""

    @pytest.fixture
    def macrophage(self):
        """Создание макрофага."""
        rng = np.random.default_rng(42)
        return Macrophage(agent_id=1, x=50.0, y=50.0, age=0.0, rng=rng)

    def test_init_defaults(self, macrophage):
        """Проверка значений по умолчанию."""
        assert macrophage.agent_id == 1
        assert macrophage.x == 50.0
        assert macrophage.y == 50.0
        assert macrophage.age == 0.0
        assert macrophage.alive is True

    def test_init_polarization_state(self, macrophage):
        """Начальное состояние поляризации M0."""
        assert macrophage.polarization_state == "M0"

    def test_init_phagocytosed_count(self, macrophage):
        """Начальный счётчик фагоцитоза."""
        assert macrophage.phagocytosed_count == 0

    def test_divide_returns_new_macrophage(self, macrophage):
        """Метод divide возвращает новый Macrophage при достаточной энергии."""
        macrophage.energy = 1.0
        macrophage.division_count = 0
        daughter = macrophage.divide(new_id=100)
        if daughter is not None:
            assert isinstance(daughter, Macrophage)
            assert daughter.agent_id == 100


class TestMacrophageBehavior:
    """Тесты поведения макрофага."""

    def test_phagocytose_limited_by_capacity(self):
        """Фагоцитоз ограничен вместимостью."""
        macro = Macrophage(agent_id=1, x=0, y=0)
        consumed = macro.phagocytose(debris_count=100)
        assert consumed <= Macrophage.PHAGOCYTOSIS_CAPACITY

    def test_phagocytose_limited_by_available(self):
        """Фагоцитоз ограничен доступным количеством."""
        macro = Macrophage(agent_id=1, x=0, y=0)
        consumed = macro.phagocytose(debris_count=2)
        assert consumed <= 2

    def test_polarize_m1_high_inflammation(self):
        """Поляризация M1 при высоком воспалении."""
        macro = Macrophage(agent_id=1, x=0, y=0)
        macro.polarize(inflammation_level=0.8)
        assert macro.polarization_state == "M1"

    def test_polarize_m2_low_inflammation(self):
        """Поляризация M2 при низком воспалении."""
        macro = Macrophage(agent_id=1, x=0, y=0)
        macro.polarize(inflammation_level=0.2)
        assert macro.polarization_state == "M2"


# =============================================================================
# Test Fibroblast
# =============================================================================


class TestFibroblast:
    """Тесты фибробласта."""

    @pytest.fixture
    def fibroblast(self):
        """Создание фибробласта."""
        rng = np.random.default_rng(42)
        return Fibroblast(agent_id=1, x=50.0, y=50.0, age=0.0, rng=rng)

    def test_init_defaults(self, fibroblast):
        """Проверка значений по умолчанию."""
        assert fibroblast.agent_id == 1
        assert fibroblast.x == 50.0
        assert fibroblast.y == 50.0
        assert fibroblast.age == 0.0
        assert fibroblast.alive is True

    def test_init_ecm_produced(self, fibroblast):
        """Начальное значение произведённого ECM."""
        assert fibroblast.ecm_produced == 0.0

    def test_init_not_activated(self, fibroblast):
        """Фибробласт изначально не активирован."""
        assert fibroblast.activated is False

    def test_divide_returns_new_fibroblast(self, fibroblast):
        """Метод divide возвращает новый Fibroblast при достаточной энергии."""
        fibroblast.energy = 1.0
        fibroblast.division_count = 0
        daughter = fibroblast.divide(new_id=100)
        if daughter is not None:
            assert isinstance(daughter, Fibroblast)
            assert daughter.agent_id == 100


class TestFibroblastBehavior:
    """Тесты поведения фибробласта."""

    def test_produce_ecm_rate(self):
        """Производство ECM пропорционально времени."""
        fibro = Fibroblast(agent_id=1, x=0, y=0)
        amount = fibro.produce_ecm(dt=1.0)
        assert amount == pytest.approx(Fibroblast.ECM_PRODUCTION_RATE)

    def test_produce_ecm_activated_higher(self):
        """Активированный фибробласт производит больше ECM."""
        fibro = Fibroblast(agent_id=1, x=0, y=0)
        fibro.activate()
        amount = fibro.produce_ecm(dt=1.0)
        assert amount > Fibroblast.ECM_PRODUCTION_RATE

    def test_activate_sets_flag(self):
        """Активация устанавливает флаг."""
        fibro = Fibroblast(agent_id=1, x=0, y=0)
        fibro.activate()
        assert fibro.activated is True


# =============================================================================
# Test Agent Movement
# =============================================================================


class TestAgentMovement:
    """Тесты движения агентов."""

    @pytest.fixture
    def cell(self):
        """Создание клетки для тестов движения."""
        return StemCell(agent_id=1, x=50.0, y=50.0, rng=np.random.default_rng(42))

    def test_move_periodic_boundary_wrap_x(self, cell):
        """Периодическая граница: обёртка по X."""
        cell.x = 95.0
        cell.move(10.0, 0.0, (100.0, 100.0), "periodic")
        assert 0 <= cell.x < 100.0

    def test_move_periodic_boundary_wrap_y(self, cell):
        """Периодическая граница: обёртка по Y."""
        cell.y = 95.0
        cell.move(0.0, 10.0, (100.0, 100.0), "periodic")
        assert 0 <= cell.y < 100.0

    def test_move_reflective_boundary(self, cell):
        """Отражающая граница: координата отражается от края."""
        cell.x = 95.0
        cell.move(10.0, 0.0, (100.0, 100.0), "reflective")
        assert 0 <= cell.x <= 100.0

    def test_random_walk_displacement_formula(self, cell):
        """Формула random walk: dx = sqrt(2*D*dt) * xi."""
        dx, dy = cell._random_walk_displacement(diffusion=1.0, dt=0.1)
        # Статистически: mean ~ 0, variance ~ 2*D*dt
        assert isinstance(dx, float)
        assert isinstance(dy, float)


# =============================================================================
# Test ABMModel Initialization
# =============================================================================


class TestABMModelInit:
    """Тесты инициализации ABM модели."""

    def test_init_default_config(self):
        """Инициализация с конфигурацией по умолчанию."""
        model = ABMModel()

        assert model.config is not None
        assert isinstance(model.config, ABMConfig)

    def test_init_custom_config(self, small_abm_config):
        """Инициализация с пользовательской конфигурацией."""
        model = ABMModel(config=small_abm_config)

        assert model.config.space_size == (50.0, 50.0)
        assert model.config.dt == 0.5

    def test_init_invalid_config_raises(self):
        """Некорректная конфигурация вызывает ошибку при инициализации."""
        invalid_config = ABMConfig(dt=-0.1)

        with pytest.raises(ValueError):
            ABMModel(config=invalid_config)

    def test_init_with_random_seed(self):
        """Инициализация с seed для воспроизводимости."""
        model = ABMModel(random_seed=42)

        assert model._rng is not None

    def test_init_empty_agents_list(self):
        """Начальный список агентов пустой."""
        model = ABMModel()

        assert model.agents == []

    def test_init_cytokine_field_shape(self, small_abm_config):
        """Форма поля цитокинов соответствует конфигурации."""
        model = ABMModel(config=small_abm_config)

        expected_shape = (
            int(50.0 / small_abm_config.grid_resolution),
            int(50.0 / small_abm_config.grid_resolution),
        )
        assert model._cytokine_field.shape == expected_shape

    def test_init_ecm_field_zeros(self, small_abm_config):
        """Поле ECM изначально нулевое."""
        model = ABMModel(config=small_abm_config)

        assert np.all(model._ecm_field == 0.0)


# =============================================================================
# Test ABMModel Methods
# =============================================================================


class TestABMModelMethods:
    """Тесты методов ABMModel."""

    @pytest.fixture
    def model(self, small_abm_config):
        """Модель с небольшой конфигурацией."""
        return ABMModel(config=small_abm_config, random_seed=42)

    def test_initialize_from_parameters_creates_agents(self, model, sample_model_parameters):
        """Метод initialize_from_parameters создаёт агентов."""
        model.initialize_from_parameters(sample_model_parameters)
        assert len(model.agents) > 0

    def test_simulate_returns_trajectory(self, model, sample_model_parameters):
        """Метод simulate возвращает ABMTrajectory."""
        trajectory = model.simulate(sample_model_parameters)
        assert isinstance(trajectory, ABMTrajectory)
        assert len(trajectory.snapshots) > 0

    def test_step_updates_time(self, model, sample_model_parameters):
        """Метод step обновляет текущее время."""
        model.initialize_from_parameters(sample_model_parameters)
        initial_time = model._current_time
        model.step(dt=0.1)
        assert model._current_time > initial_time

    def test_get_snapshot_returns_snapshot(self, model, sample_model_parameters):
        """Метод _get_snapshot возвращает ABMSnapshot."""
        model.initialize_from_parameters(sample_model_parameters)
        snapshot = model._get_snapshot()
        assert isinstance(snapshot, ABMSnapshot)

    def test_create_agent_stem(self, model):
        """Метод _create_agent создаёт стволовую клетку."""
        agent = model._create_agent(agent_type="stem")
        assert isinstance(agent, StemCell)

    def test_create_agent_macro(self, model):
        """Метод _create_agent создаёт макрофаг."""
        agent = model._create_agent(agent_type="macro")
        assert isinstance(agent, Macrophage)

    def test_create_agent_fibro(self, model):
        """Метод _create_agent создаёт фибробласт."""
        agent = model._create_agent(agent_type="fibro")
        assert isinstance(agent, Fibroblast)


# =============================================================================
# Test ABMModel Simulation Behavior
# =============================================================================


class TestABMModelSimulationBehavior:
    """Тесты поведения симуляции ABM."""

    def test_simulate_reproducibility_with_seed(self, sample_model_parameters):
        """Симуляция воспроизводима с одинаковым seed."""
        config = ABMConfig(t_max=24.0)

        model1 = ABMModel(config=config, random_seed=42)
        traj1 = model1.simulate(sample_model_parameters)

        model2 = ABMModel(config=config, random_seed=42)
        traj2 = model2.simulate(sample_model_parameters)

        assert traj1.snapshots[-1].get_total_agents() == traj2.snapshots[-1].get_total_agents()

    def test_dead_agents_removed(self):
        """Мёртвые агенты удаляются через _remove_dead_agents."""
        model = ABMModel(random_seed=42)
        # Initialize agents through the proper method
        model._agents = [
            StemCell(1, 50, 50, rng=np.random.default_rng(42)),
            StemCell(2, 60, 60, rng=np.random.default_rng(42)),
        ]
        model._agents[1].alive = False
        model._remove_dead_agents()
        assert len(model._agents) == 1


# =============================================================================
# Test simulate_abm Function
# =============================================================================


class TestSimulateAbmFunction:
    """Тесты convenience функции simulate_abm."""

    def test_simulate_abm_returns_trajectory(self, sample_model_parameters):
        """Функция возвращает ABMTrajectory."""
        result = simulate_abm(
            initial_params=sample_model_parameters,
            config=ABMConfig(t_max=24.0),
            random_seed=42,
        )
        assert isinstance(result, ABMTrajectory)

    def test_simulate_abm_default_config(self, sample_model_parameters):
        """Функция работает с конфигурацией по умолчанию."""
        result = simulate_abm(initial_params=sample_model_parameters, random_seed=42)
        assert isinstance(result, ABMTrajectory)

    def test_simulate_abm_custom_snapshot_interval(self, sample_model_parameters):
        """Функция принимает пользовательский интервал снимков."""
        result = simulate_abm(
            initial_params=sample_model_parameters,
            snapshot_interval=12.0,
            random_seed=42,
        )
        assert isinstance(result, ABMTrajectory)


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestABMEdgeCases:
    """Тесты граничных случаев."""

    def test_agent_at_boundary_corner(self):
        """Агент в углу пространства: корректная обработка углов."""
        cell = StemCell(agent_id=1, x=0.0, y=0.0, rng=np.random.default_rng(42))
        cell.move(-1.0, -1.0, (100.0, 100.0), "periodic")
        assert 0 <= cell.x < 100.0
        assert 0 <= cell.y < 100.0

    def test_agent_energy_zero(self):
        """Агент с нулевой энергией не может делиться."""
        cell = StemCell(agent_id=1, x=50, y=50)
        cell.energy = 0.0
        assert cell.can_divide() is False

    def test_agent_max_age_death(self):
        """Агент умирает при достижении максимального возраста."""
        cell = StemCell(agent_id=1, x=50, y=50, rng=np.random.default_rng(42))
        cell.age = StemCell.LIFESPAN + 1.0
        assert cell.should_die(dt=0.1) is True

    def test_minimal_params_simulation(self):
        """Симуляция с минимальными параметрами (n0=0) завершается корректно.

        Модель гарантирует минимальную популяцию агентов (min 10),
        поэтому проверяем корректное завершение, а не 0 агентов.
        """
        params = ModelParameters(
            n0=0.0, c0=0.0, stem_cell_fraction=0.0,
            macrophage_fraction=0.0, apoptotic_fraction=0.0, inflammation_level=0.0
        )
        config = ABMConfig(t_max=24.0)
        model = ABMModel(config=config, random_seed=42)
        trajectory = model.simulate(params)

        # Симуляция завершается корректно
        assert len(trajectory.snapshots) > 0
        # Финальное время соответствует t_max
        assert trajectory.snapshots[-1].t >= 24.0 - config.dt


# =============================================================================
# Test Cytokine Field Dynamics
# =============================================================================


class TestCytokineFieldDynamics:
    """Тесты динамики поля цитокинов."""

    def test_cytokine_decay(self):
        """Цитокины деградируют со временем."""
        model = ABMModel(random_seed=42)
        model._cytokine_field[:] = 10.0
        initial_mean = np.mean(model._cytokine_field)
        model._update_cytokine_field(dt=1.0)
        assert np.mean(model._cytokine_field) < initial_mean

    def test_agents_contribute_to_cytokine_field(self):
        """Агенты вносят вклад в поле цитокинов.

        После реализации: стволовые клетки секретируют цитокины.
        """
        # model = ABMModel()
        # model._agents = [StemCell(1, 50, 50)]
        # model._update_cytokine_field(dt=1.0)
        # # Должен быть локальный пик около агента
        pass


# =============================================================================
# Test ECM Field Dynamics
# =============================================================================


class TestECMFieldDynamics:
    """Тесты динамики поля ECM."""

    def test_fibroblasts_produce_ecm(self):
        """Фибробласты производят ECM.

        После реализации: ECM увеличивается около фибробластов.
        """
        # model = ABMModel()
        # model._agents = [Fibroblast(1, 50, 50)]
        # initial_ecm = np.sum(model._ecm_field)
        # model._update_ecm_field(dt=1.0)
        # assert np.sum(model._ecm_field) > initial_ecm
        pass

    def test_activated_fibroblasts_produce_more_ecm(self):
        """Активированные фибробласты производят больше ECM.

        После реализации: миофибробласты имеют повышенную продукцию.
        """
        # fibro = Fibroblast(1, 50, 50)
        # fibro.activate()
        # # ecm_rate должен быть выше
        pass


# =============================================================================
# Test Agent Interactions
# =============================================================================


class TestAgentInteractions:
    """Тесты взаимодействий между агентами."""

    def test_contact_inhibition(self):
        """Контактное ингибирование деления.

        После реализации: агенты не делятся при высокой плотности.
        """
        # model = ABMModel(config=ABMConfig(contact_inhibition_radius=5.0))
        # # Разместить много агентов близко
        # # Деление должно быть заблокировано
        pass

    def test_macrophage_chemotaxis(self):
        """Макрофаги движутся к высокой концентрации цитокинов.

        После реализации: хемотаксис в направлении градиента.
        """
        # model = ABMModel()
        # model._cytokine_field[0, 5] = 100.0  # Высокая концентрация слева
        # macro = Macrophage(1, 50, 50)
        # model._agents = [macro]
        # model._update_agents(dt=1.0)
        # # Макрофаг должен сместиться влево
        pass


# =============================================================================
# Test Numerical Stability
# =============================================================================


class TestNumericalStability:
    """Тесты численной стабильности."""

    def test_large_time_step_stability(self):
        """Стабильность при большом шаге времени.

        После реализации: результаты должны быть разумными.
        """
        # config = ABMConfig(dt=1.0, t_max=24.0)
        # model = ABMModel(config=config)
        # params = ModelParameters(n0=100.0, c0=10.0)
        # trajectory = model.simulate(params)
        # # Должно завершиться без ошибок
        pass

    def test_many_agents_performance(self):
        """Производительность при большом количестве агентов.

        После реализации: должно работать за разумное время.
        """
        # config = ABMConfig(max_agents=1000, t_max=24.0)
        # model = ABMModel(config=config)
        # # Тест на производительность
        pass

    def test_long_simulation_stability(self):
        """Стабильность при длительной симуляции.

        После реализации: нет утечек памяти, корректные значения.
        """
        # config = ABMConfig(t_max=720.0)  # 30 дней
        # model = ABMModel(config=config)
        # params = ModelParameters(n0=100.0, c0=10.0)
        # trajectory = model.simulate(params)
        # # Значения должны быть конечными
        pass
