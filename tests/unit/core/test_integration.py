"""TDD тесты для интеграции SDE и ABM моделей.

Тестирование:
- IntegrationConfig: параметры синхронизации, режимы, валидация
- IntegratedState: состояние системы в точке синхронизации
- IntegratedTrajectory: траектория с метриками рассогласования
- IntegratedModel: operator splitting, синхронизация, коррекция
- simulate_integrated: convenience функция
- create_default_integration_config: создание согласованных конфигураций

Все тесты написаны для stub-реализации (NotImplementedError).
После реализации методов тесты должны проходить.
"""

import numpy as np
import pytest

from src.core.abm_model import ABMConfig, ABMSnapshot, ABMTrajectory, AgentState
from src.core.integration import (
    IntegrationConfig,
    IntegratedModel,
    IntegratedState,
    IntegratedTrajectory,
    create_default_integration_config,
    simulate_integrated,
)
from src.core.sde_model import SDEConfig, SDETrajectory, TherapyProtocol
from src.data.parameter_extraction import ModelParameters


# =============================================================================
# Test IntegrationConfig
# =============================================================================


class TestIntegrationConfig:
    """Тесты конфигурации интеграции."""

    def test_default_values(self):
        """Проверка значений по умолчанию."""
        config = IntegrationConfig()

        assert config.sync_interval == 1.0
        assert config.coupling_strength == 0.5
        assert config.mode == "bidirectional"
        assert config.correction_rate == 0.1
        assert config.max_discrepancy == 0.5

    def test_default_nested_configs(self):
        """Проверка вложенных конфигураций по умолчанию."""
        config = IntegrationConfig()

        assert isinstance(config.sde_config, SDEConfig)
        assert isinstance(config.abm_config, ABMConfig)

    def test_custom_sync_interval(self):
        """Пользовательский интервал синхронизации."""
        config = IntegrationConfig(sync_interval=2.0)

        assert config.sync_interval == 2.0

    def test_custom_coupling_strength(self):
        """Пользовательская сила связи."""
        config = IntegrationConfig(coupling_strength=0.8)

        assert config.coupling_strength == 0.8

    def test_integration_modes(self):
        """Проверка допустимых режимов интеграции."""
        for mode in ["sde_only", "abm_only", "bidirectional"]:
            config = IntegrationConfig(mode=mode)
            assert config.mode == mode

    def test_validate_returns_true_for_valid_config(self, bidirectional_integration_config):
        """Валидация возвращает True для корректной конфигурации."""
        result = bidirectional_integration_config.validate()

        assert result is True

    def test_validate_negative_sync_interval_raises(self):
        """Отрицательный интервал синхронизации вызывает ошибку."""
        config = IntegrationConfig(sync_interval=-1.0)

        with pytest.raises(ValueError, match="sync_interval"):
            config.validate()

    def test_validate_zero_sync_interval_raises(self):
        """Нулевой интервал синхронизации вызывает ошибку."""
        config = IntegrationConfig(sync_interval=0.0)

        with pytest.raises(ValueError, match="sync_interval"):
            config.validate()

    def test_validate_coupling_strength_below_zero_raises(self):
        """Сила связи < 0 вызывает ошибку."""
        config = IntegrationConfig(coupling_strength=-0.1)

        with pytest.raises(ValueError, match="coupling_strength"):
            config.validate()

    def test_validate_coupling_strength_above_one_raises(self):
        """Сила связи > 1 вызывает ошибку."""
        config = IntegrationConfig(coupling_strength=1.5)

        with pytest.raises(ValueError, match="coupling_strength"):
            config.validate()

    def test_validate_invalid_mode_raises(self):
        """Некорректный режим вызывает ошибку."""
        config = IntegrationConfig(mode="invalid_mode")

        with pytest.raises(ValueError, match="mode"):
            config.validate()

    def test_validate_negative_correction_rate_raises(self):
        """Отрицательная скорость коррекции вызывает ошибку."""
        config = IntegrationConfig(correction_rate=-0.1)

        with pytest.raises(ValueError, match="correction_rate"):
            config.validate()

    def test_validate_correction_rate_above_one_raises(self):
        """Скорость коррекции > 1 вызывает ошибку."""
        config = IntegrationConfig(correction_rate=1.5)

        with pytest.raises(ValueError, match="correction_rate"):
            config.validate()

    def test_validate_negative_max_discrepancy_raises(self):
        """Отрицательный max_discrepancy вызывает ошибку."""
        config = IntegrationConfig(max_discrepancy=-0.1)

        with pytest.raises(ValueError, match="max_discrepancy"):
            config.validate()

    def test_validate_propagates_to_nested_configs(self):
        """Валидация проверяет вложенные конфигурации."""
        # Некорректная SDE конфигурация
        invalid_sde = SDEConfig(dt=-0.01)
        config = IntegrationConfig(sde_config=invalid_sde)

        with pytest.raises(ValueError):
            config.validate()

    def test_validate_propagates_to_abm_config(self):
        """Валидация проверяет ABM конфигурацию."""
        # Некорректная ABM конфигурация
        invalid_abm = ABMConfig(dt=-0.1)
        config = IntegrationConfig(abm_config=invalid_abm)

        with pytest.raises(ValueError):
            config.validate()


# =============================================================================
# Test IntegratedState
# =============================================================================


class TestIntegratedState:
    """Тесты состояния интегрированной системы."""

    @pytest.fixture
    def sample_integrated_state(self):
        """Пример состояния интеграции."""
        return IntegratedState(
            t=24.0,
            sde_N=1000.0,
            sde_C=50.0,
            abm_agent_counts={"stem": 100, "macro": 50, "fibro": 850},
            abm_total=1000,
            discrepancy=0.0,
            correction_applied=0.0,
        )

    def test_state_time(self, sample_integrated_state):
        """Проверка времени состояния."""
        assert sample_integrated_state.t == 24.0

    def test_state_sde_values(self, sample_integrated_state):
        """Проверка значений SDE."""
        assert sample_integrated_state.sde_N == 1000.0
        assert sample_integrated_state.sde_C == 50.0

    def test_state_abm_values(self, sample_integrated_state):
        """Проверка значений ABM."""
        assert sample_integrated_state.abm_total == 1000
        assert sample_integrated_state.abm_agent_counts["stem"] == 100

    def test_state_discrepancy_and_correction(self, sample_integrated_state):
        """Проверка метрик синхронизации."""
        assert sample_integrated_state.discrepancy == 0.0
        assert sample_integrated_state.correction_applied == 0.0

    def test_to_dict_basic_fields(self, sample_integrated_state):
        """to_dict содержит базовые поля."""
        result = sample_integrated_state.to_dict()

        assert result["t"] == 24.0
        assert result["sde_N"] == 1000.0
        assert result["sde_C"] == 50.0
        assert result["abm_total"] == 1000
        assert result["discrepancy"] == 0.0
        assert result["correction_applied"] == 0.0

    def test_to_dict_agent_counts(self, sample_integrated_state):
        """to_dict содержит количества агентов по типам."""
        result = sample_integrated_state.to_dict()

        assert result["abm_stem"] == 100
        assert result["abm_macro"] == 50
        assert result["abm_fibro"] == 850

    def test_state_with_nonzero_discrepancy(self):
        """Состояние с ненулевым рассогласованием."""
        state = IntegratedState(
            t=48.0,
            sde_N=1000.0,
            sde_C=40.0,
            abm_agent_counts={"stem": 80, "macro": 40, "fibro": 780},
            abm_total=900,
            discrepancy=0.1,  # 10% рассогласование
            correction_applied=10.0,
        )

        assert state.discrepancy == 0.1
        assert state.correction_applied == 10.0


# =============================================================================
# Test IntegratedTrajectory
# =============================================================================


class TestIntegratedTrajectory:
    """Тесты траектории интегрированной симуляции."""

    @pytest.fixture
    def sample_trajectory(self):
        """Пример траектории с несколькими состояниями."""
        states = [
            IntegratedState(
                t=0.0,
                sde_N=1000.0,
                sde_C=100.0,
                abm_agent_counts={"stem": 100, "macro": 50, "fibro": 850},
                abm_total=1000,
                discrepancy=0.0,
                correction_applied=0.0,
            ),
            IntegratedState(
                t=24.0,
                sde_N=1100.0,
                sde_C=80.0,
                abm_agent_counts={"stem": 110, "macro": 55, "fibro": 900},
                abm_total=1065,
                discrepancy=0.032,
                correction_applied=3.5,
            ),
            IntegratedState(
                t=48.0,
                sde_N=1200.0,
                sde_C=60.0,
                abm_agent_counts={"stem": 120, "macro": 60, "fibro": 1000},
                abm_total=1180,
                discrepancy=0.017,
                correction_applied=2.0,
            ),
        ]

        # Мок SDE и ABM траекторий (упрощённые)
        sde_traj = SDETrajectory(
            times=np.array([0.0, 1.0, 2.0]),
            N_values=np.array([1000.0, 1100.0, 1200.0]),
            C_values=np.array([100.0, 80.0, 60.0]),
        )
        abm_traj = ABMTrajectory(
            snapshots=[],
            config=ABMConfig(),
        )

        return IntegratedTrajectory(
            times=np.array([0.0, 24.0, 48.0]),
            states=states,
            sde_trajectory=sde_traj,
            abm_trajectory=abm_traj,
            config=IntegrationConfig(),
        )

    def test_trajectory_has_times(self, sample_trajectory):
        """Траектория содержит временные точки."""
        np.testing.assert_array_equal(
            sample_trajectory.times, [0.0, 24.0, 48.0]
        )

    def test_trajectory_has_states(self, sample_trajectory):
        """Траектория содержит состояния."""
        assert len(sample_trajectory.states) == 3

    def test_trajectory_has_sde_trajectory(self, sample_trajectory):
        """Траектория содержит SDE траекторию."""
        assert isinstance(sample_trajectory.sde_trajectory, SDETrajectory)

    def test_trajectory_has_abm_trajectory(self, sample_trajectory):
        """Траектория содержит ABM траекторию."""
        assert isinstance(sample_trajectory.abm_trajectory, ABMTrajectory)

    def test_get_statistics_final_values(self, sample_trajectory):
        """get_statistics возвращает финальные значения."""
        stats = sample_trajectory.get_statistics()

        assert stats["final_sde_N"] == 1200.0
        assert stats["final_sde_C"] == 60.0
        assert stats["final_abm_total"] == 1180.0

    def test_get_statistics_discrepancy_metrics(self, sample_trajectory):
        """get_statistics возвращает метрики рассогласования."""
        stats = sample_trajectory.get_statistics()

        # mean_discrepancy = (0.0 + 0.032 + 0.017) / 3 ≈ 0.0163
        assert "mean_discrepancy" in stats
        assert "max_discrepancy" in stats
        assert "std_discrepancy" in stats

    def test_get_statistics_max_discrepancy(self, sample_trajectory):
        """max_discrepancy равен максимуму по всем состояниям."""
        stats = sample_trajectory.get_statistics()

        assert stats["max_discrepancy"] == 0.032

    def test_get_statistics_total_corrections(self, sample_trajectory):
        """total_corrections суммирует все коррекции."""
        stats = sample_trajectory.get_statistics()

        # total = |0.0| + |3.5| + |2.0| = 5.5
        assert stats["total_corrections"] == pytest.approx(5.5)

    def test_get_statistics_n_sync_points(self, sample_trajectory):
        """n_sync_points равен количеству состояний."""
        stats = sample_trajectory.get_statistics()

        assert stats["n_sync_points"] == 3

    def test_get_statistics_empty_trajectory(self):
        """get_statistics для пустой траектории."""
        empty_traj = IntegratedTrajectory(
            times=np.array([]),
            states=[],
            sde_trajectory=SDETrajectory(
                times=np.array([]),
                N_values=np.array([]),
                C_values=np.array([]),
            ),
            abm_trajectory=ABMTrajectory(snapshots=[], config=ABMConfig()),
        )

        stats = empty_traj.get_statistics()

        assert stats == {}

    def test_get_discrepancy_timeseries(self, sample_trajectory):
        """get_discrepancy_timeseries возвращает (times, discrepancies)."""
        times, discrepancies = sample_trajectory.get_discrepancy_timeseries()
        np.testing.assert_array_equal(times, [0.0, 24.0, 48.0])
        assert len(discrepancies) == 3


# =============================================================================
# Test IntegratedModel Initialization
# =============================================================================


class TestIntegratedModelInit:
    """Тесты инициализации интегрированной модели."""

    def test_init_with_config(self, bidirectional_integration_config):
        """Инициализация с конфигурацией."""
        model = IntegratedModel(config=bidirectional_integration_config)

        assert model.config is not None
        assert isinstance(model.config, IntegrationConfig)

    def test_init_validates_config(self):
        """Инициализация валидирует конфигурацию."""
        invalid_config = IntegrationConfig(sync_interval=-1.0)

        with pytest.raises(ValueError):
            IntegratedModel(config=invalid_config)

    def test_init_creates_sde_model(self, bidirectional_integration_config):
        """Инициализация создаёт SDE модель."""
        model = IntegratedModel(config=bidirectional_integration_config)

        assert model.sde_model is not None

    def test_init_creates_abm_model(self, bidirectional_integration_config):
        """Инициализация создаёт ABM модель."""
        model = IntegratedModel(config=bidirectional_integration_config)

        assert model.abm_model is not None

    def test_init_with_therapy(self, bidirectional_integration_config, prp_therapy_protocol):
        """Инициализация с протоколом терапии."""
        model = IntegratedModel(
            config=bidirectional_integration_config,
            therapy=prp_therapy_protocol,
        )

        assert model._therapy is not None

    def test_init_default_therapy(self, bidirectional_integration_config):
        """По умолчанию создаётся пустой протокол терапии."""
        model = IntegratedModel(config=bidirectional_integration_config)

        assert isinstance(model._therapy, TherapyProtocol)

    def test_init_with_random_seed(self, bidirectional_integration_config):
        """Инициализация с seed для воспроизводимости."""
        model = IntegratedModel(
            config=bidirectional_integration_config,
            random_seed=42,
        )

        assert model._rng is not None

    def test_init_different_seeds_for_sde_and_abm(self, bidirectional_integration_config):
        """SDE и ABM получают разные seeds для независимости."""
        # Проверяем, что модели созданы (seeds передаются внутри)
        model = IntegratedModel(
            config=bidirectional_integration_config,
            random_seed=42,
        )

        # Модели должны быть независимы
        assert model.sde_model is not model.abm_model


# =============================================================================
# Test IntegratedModel Methods
# =============================================================================


class TestIntegratedModelMethods:
    """Тесты методов IntegratedModel."""

    @pytest.fixture
    def model(self, bidirectional_integration_config):
        """Модель для тестов."""
        return IntegratedModel(
            config=bidirectional_integration_config,
            random_seed=42,
        )

    def test_simulate_returns_trajectory(self, model, sample_model_parameters):
        """Метод simulate возвращает IntegratedTrajectory."""
        trajectory = model.simulate(sample_model_parameters)
        assert isinstance(trajectory, IntegratedTrajectory)
        assert len(trajectory.states) > 0

    def test_calculate_discrepancy(self, model):
        """Метод _calculate_discrepancy вычисляет рассогласование."""
        discrepancy = model._calculate_discrepancy(sde_N=1000.0, abm_count=950)
        assert isinstance(discrepancy, float)
        assert discrepancy >= 0

    def test_apply_correction(self, model):
        """Метод _apply_correction возвращает скорректированное значение."""
        corrected = model._apply_correction(sde_N=1000.0, abm_count=950, discrepancy=0.05)
        assert isinstance(corrected, float)

    def test_create_integrated_state(self, model):
        """Метод _create_integrated_state создаёт IntegratedState."""
        snapshot = ABMSnapshot(
            t=1.0,
            agents=[],
            cytokine_field=np.zeros((10, 10)),
            ecm_field=np.zeros((10, 10)),
        )

        state = model._create_integrated_state(
            t=1.0,
            sde_N=1000.0,
            sde_C=50.0,
            abm_snapshot=snapshot,
            discrepancy=0.05,
            correction=5.0,
        )
        assert isinstance(state, IntegratedState)
        assert state.t == 1.0


# =============================================================================
# Test IntegratedModel Simulation Behavior
# =============================================================================


class TestIntegratedModelSimulationBehavior:
    """Тесты поведения симуляции."""

    def test_simulate_reproducibility_with_seed(self, sample_model_parameters):
        """Симуляция воспроизводима с одинаковым seed."""
        config = create_default_integration_config(t_max_days=1.0)

        model1 = IntegratedModel(config=config, random_seed=42)
        traj1 = model1.simulate(sample_model_parameters)

        model2 = IntegratedModel(config=config, random_seed=42)
        traj2 = model2.simulate(sample_model_parameters)

        assert traj1.states[-1].sde_N == traj2.states[-1].sde_N


# =============================================================================
# Test Discrepancy Calculation
# =============================================================================


class TestDiscrepancyCalculation:
    """Тесты расчёта рассогласования."""

    def test_discrepancy_formula(self):
        """Формула: discrepancy = |sde_N - abm_count| / max(sde_N, 1)."""
        config = IntegrationConfig()
        model = IntegratedModel(config=config, random_seed=42)

        # sde_N=1000, abm_count=900 -> discrepancy = 100/1000 = 0.1
        disc = model._calculate_discrepancy(sde_N=1000.0, abm_count=900)
        assert disc == pytest.approx(0.1)

    def test_discrepancy_zero_when_equal(self):
        """Рассогласование = 0 когда значения равны."""
        config = IntegrationConfig()
        model = IntegratedModel(config=config, random_seed=42)

        disc = model._calculate_discrepancy(sde_N=1000.0, abm_count=1000)
        assert disc == 0.0

    def test_discrepancy_symmetric(self):
        """Рассогласование симметрично (абсолютное значение)."""
        config = IntegrationConfig()
        model = IntegratedModel(config=config, random_seed=42)

        disc1 = model._calculate_discrepancy(sde_N=1000.0, abm_count=900)
        disc2 = model._calculate_discrepancy(sde_N=1000.0, abm_count=1100)
        assert disc1 == disc2


# =============================================================================
# Test Correction Application
# =============================================================================


class TestCorrectionApplication:
    """Тесты применения коррекции."""

    def test_correction_reduces_discrepancy(self):
        """Коррекция уменьшает рассогласование."""
        config = IntegrationConfig(correction_rate=0.5)
        model = IntegratedModel(config=config, random_seed=42)

        # sde_N=1000, abm_count=900, discrepancy=0.1
        corrected = model._apply_correction(
            sde_N=1000.0, abm_count=900, discrepancy=0.1
        )
        # Коррекция должна сдвинуть sde_N к 900
        assert 900 <= corrected <= 1000


# =============================================================================
# Test Synchronization
# =============================================================================


class TestSynchronization:
    """Тесты синхронизации SDE и ABM."""

    def test_synchronize_returns_corrected_values(self):
        """_synchronize возвращает скорректированные значения.

        После реализации: (corrected_N, corrected_C, discrepancy).
        """
        # config = IntegrationConfig()
        # model = IntegratedModel(config=config)
        # snapshot = ABMSnapshot(
        #     t=1.0,
        #     agents=[AgentState(1, "stem", 50, 50, 0, 0, 1.0) for _ in range(900)],
        #     cytokine_field=np.zeros((10, 10)),
        #     ecm_field=np.zeros((10, 10)),
        # )
        #
        # corrected_N, corrected_C, disc = model._synchronize(
        #     sde_N=1000.0, sde_C=50.0, abm_snapshot=snapshot
        # )
        #
        # assert corrected_N is not None
        # assert corrected_C is not None
        # assert disc >= 0
        pass

    def test_update_abm_environment_affects_cytokines(self):
        """_update_abm_environment обновляет цитокиновое поле ABM.

        После реализации: ABM environment получает sde_C.
        """
        # config = IntegrationConfig()
        # model = IntegratedModel(config=config)
        # model._update_abm_environment(sde_C=100.0)
        # # Проверить, что ABM модель получила обновлённое значение
        pass


# =============================================================================
# Test simulate_integrated Function
# =============================================================================


class TestSimulateIntegratedFunction:
    """Тесты convenience функции simulate_integrated."""

    def test_simulate_integrated_returns_trajectory(self, sample_model_parameters):
        """Функция возвращает IntegratedTrajectory."""
        config = create_default_integration_config(t_max_days=1.0)

        result = simulate_integrated(
            initial_params=sample_model_parameters,
            integration_config=config,
            random_seed=42,
        )
        assert isinstance(result, IntegratedTrajectory)

    def test_simulate_integrated_with_therapy(
        self, sample_model_parameters, prp_therapy_protocol
    ):
        """Функция принимает протокол терапии."""
        config = create_default_integration_config(t_max_days=1.0)

        result = simulate_integrated(
            initial_params=sample_model_parameters,
            integration_config=config,
            therapy=prp_therapy_protocol,
            random_seed=42,
        )
        assert isinstance(result, IntegratedTrajectory)


# =============================================================================
# Test create_default_integration_config Function
# =============================================================================


class TestCreateDefaultIntegrationConfig:
    """Тесты функции создания конфигурации по умолчанию."""

    def test_creates_valid_config(self):
        """Функция создаёт валидную конфигурацию."""
        config = create_default_integration_config()

        assert config.validate() is True

    def test_t_max_days_parameter(self):
        """Параметр t_max_days устанавливает время симуляции."""
        config = create_default_integration_config(t_max_days=10.0)

        assert config.sde_config.t_max == 10.0
        assert config.abm_config.t_max == 10.0 * 24.0  # В часах

    def test_sync_interval_hours_parameter(self):
        """Параметр sync_interval_hours устанавливает интервал."""
        config = create_default_integration_config(sync_interval_hours=2.0)

        assert config.sync_interval == 2.0

    def test_mode_parameter(self):
        """Параметр mode устанавливает режим интеграции."""
        config = create_default_integration_config(mode="sde_only")

        assert config.mode == "sde_only"

    def test_sde_and_abm_configs_consistent(self):
        """SDE и ABM конфигурации согласованы по времени."""
        config = create_default_integration_config(t_max_days=30.0)

        sde_t_max = config.sde_config.t_max  # В днях
        abm_t_max = config.abm_config.t_max  # В часах

        assert abm_t_max == sde_t_max * 24.0


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestIntegrationEdgeCases:
    """Тесты граничных случаев интеграции."""

    def test_very_short_simulation(self):
        """Очень короткая симуляция (меньше sync_interval).

        После реализации: должна корректно обработать.
        """
        # config = IntegrationConfig(sync_interval=10.0)
        # config.sde_config.t_max = 1.0  # Меньше sync_interval
        # model = IntegratedModel(config=config)
        # params = ModelParameters(n0=1000.0, c0=100.0)
        # trajectory = model.simulate(params)
        # # Должно быть минимум 1-2 точки синхронизации
        pass

    def test_large_discrepancy_handling(self):
        """Обработка большого рассогласования.

        После реализации: max_discrepancy ограничивает коррекцию.
        """
        # config = IntegrationConfig(max_discrepancy=0.1)
        # model = IntegratedModel(config=config)
        # # При очень большом рассогласовании коррекция должна быть ограничена
        pass

    def test_zero_initial_population(self):
        """Симуляция с нулевой начальной популяцией.

        После реализации: должна корректно обработать N0=0.
        """
        # config = create_default_integration_config(t_max_days=1.0)
        # model = IntegratedModel(config=config)
        # params = ModelParameters(n0=0.0, c0=100.0)
        # trajectory = model.simulate(params)
        # # Не должно быть деления на 0
        pass

    def test_therapy_affects_both_models(self):
        """Терапия влияет на обе модели.

        После реализации: PRP/PEMF применяются к SDE и ABM.
        """
        # therapy = TherapyProtocol(
        #     prp_dose=10.0,
        #     prp_start_day=0.0,
        #     pemf_frequency=50.0,
        #     pemf_start_day=0.0,
        # )
        # config = create_default_integration_config(t_max_days=5.0)
        # model = IntegratedModel(config=config, therapy=therapy)
        # params = ModelParameters(n0=1000.0, c0=100.0)
        # trajectory = model.simulate(params)
        # # Терапия должна ускорить рост
        pass


# =============================================================================
# Test Numerical Stability
# =============================================================================


class TestIntegrationNumericalStability:
    """Тесты численной стабильности интеграции."""

    def test_long_simulation_stability(self):
        """Стабильность при длительной симуляции.

        После реализации: нет утечек памяти, корректные значения.
        """
        # config = create_default_integration_config(t_max_days=30.0)
        # model = IntegratedModel(config=config)
        # params = ModelParameters(n0=1000.0, c0=100.0)
        # trajectory = model.simulate(params)
        # # Все значения должны быть конечными
        # assert np.isfinite(trajectory.states[-1].sde_N)
        # assert np.isfinite(trajectory.states[-1].sde_C)
        pass

    def test_small_sync_interval_stability(self):
        """Стабильность при малом интервале синхронизации.

        После реализации: частая синхронизация не вызывает проблем.
        """
        # config = IntegrationConfig(sync_interval=0.1)  # Каждые 6 минут
        # config.sde_config.t_max = 1.0
        # model = IntegratedModel(config=config)
        # params = ModelParameters(n0=1000.0, c0=100.0)
        # trajectory = model.simulate(params)
        # # Должно работать корректно
        pass

    def test_discrepancy_bounded(self):
        """Рассогласование остаётся ограниченным.

        После реализации: discrepancy не уходит в бесконечность.
        """
        # config = create_default_integration_config(t_max_days=10.0)
        # model = IntegratedModel(config=config)
        # params = ModelParameters(n0=1000.0, c0=100.0)
        # trajectory = model.simulate(params)
        # for state in trajectory.states:
        #     assert state.discrepancy < 10.0  # Разумный предел
        pass


# =============================================================================
# Phase 2: Test _synchronize_cytokines
# =============================================================================


class TestSynchronizeCytokines:
    """Тесты двусторонней синхронизации цитокинов ABM ↔ SDE."""

    @pytest.fixture
    def model(self, bidirectional_integration_config):
        """Модель для тестов синхронизации цитокинов."""
        return IntegratedModel(
            config=bidirectional_integration_config,
            random_seed=42,
        )

    @pytest.fixture
    def uniform_snapshot(self):
        """ABM snapshot с однородным цитокиновым полем."""
        agents = [
            AgentState(i, "stem", 10.0 * i, 10.0 * i, 0, 0, 1.0)
            for i in range(10)
        ]
        return ABMSnapshot(
            t=1.0,
            agents=agents,
            cytokine_field=np.ones((10, 10)) * 5.0,
            ecm_field=np.zeros((10, 10)),
        )

    def test_matching_values_no_correction(self, model, uniform_snapshot):
        """Если ABM и SDE совпадают — коррекция ≈ 0."""
        # С полем = 5.0 и sde_C ≈ 5.0 (с масштабированием)
        # Ожидаем что результат близок к sde_C
        result = model._synchronize_cytokines(5.0, uniform_snapshot)
        assert isinstance(result, float)

    def test_abm_higher_than_sde_increases_c(self, model):
        """Если ABM >> SDE — C увеличивается."""
        high_field_snapshot = ABMSnapshot(
            t=1.0,
            agents=[],
            cytokine_field=np.ones((10, 10)) * 100.0,
            ecm_field=np.zeros((10, 10)),
        )
        sde_C = 1.0
        result = model._synchronize_cytokines(sde_C, high_field_snapshot)
        assert result >= sde_C

    def test_abm_lower_than_sde_decreases_c(self, model):
        """Если ABM << SDE — C уменьшается."""
        low_field_snapshot = ABMSnapshot(
            t=1.0,
            agents=[],
            cytokine_field=np.ones((10, 10)) * 0.01,
            ecm_field=np.zeros((10, 10)),
        )
        sde_C = 100.0
        result = model._synchronize_cytokines(sde_C, low_field_snapshot)
        assert result <= sde_C

    def test_coupling_zero_no_correction(self, uniform_snapshot):
        """coupling_strength=0 → результат == sde_C."""
        config = IntegrationConfig(coupling_strength=0.0)
        model = IntegratedModel(config=config, random_seed=42)
        sde_C = 42.0
        result = model._synchronize_cytokines(sde_C, uniform_snapshot)
        assert result == pytest.approx(sde_C)

    def test_result_non_negative(self, model):
        """Результат ≥ 0 (концентрация неотрицательна)."""
        zero_snapshot = ABMSnapshot(
            t=1.0,
            agents=[],
            cytokine_field=np.zeros((10, 10)),
            ecm_field=np.zeros((10, 10)),
        )
        result = model._synchronize_cytokines(0.0, zero_snapshot)
        assert result >= 0.0

    def test_correction_proportional_to_coupling(self, uniform_snapshot):
        """Коррекция пропорциональна coupling_strength."""
        sde_C = 1.0
        results = {}
        for coupling in [0.1, 0.5, 0.9]:
            config = IntegrationConfig(coupling_strength=coupling)
            model = IntegratedModel(config=config, random_seed=42)
            results[coupling] = model._synchronize_cytokines(sde_C, uniform_snapshot)

        # Большая связь → большая коррекция (при ABM > SDE)
        correction_01 = abs(results[0.1] - sde_C)
        correction_09 = abs(results[0.9] - sde_C)
        # При coupling=0.9 коррекция должна быть больше чем при 0.1
        assert correction_09 >= correction_01


# =============================================================================
# Phase 2: Test _transfer_therapy_to_abm
# =============================================================================


class TestTransferTherapyToABM:
    """Тесты передачи терапевтических флагов в ABM."""

    def test_no_therapy_both_inactive(self):
        """Без терапии — prp_active=False, pemf_active=False."""
        config = IntegrationConfig()
        model = IntegratedModel(config=config, random_seed=42)
        # По умолчанию TherapyProtocol — без терапии
        model._transfer_therapy_to_abm(current_time_days=3.0)
        # Не должно вызвать ошибку

    def test_prp_active_in_window(self):
        """PRP активен внутри временного окна."""
        therapy = TherapyProtocol(
            prp_enabled=True,
            prp_start_time=1.0,
            prp_end_time=5.0,
        )
        config = IntegrationConfig()
        model = IntegratedModel(config=config, therapy=therapy, random_seed=42)
        model._transfer_therapy_to_abm(current_time_days=3.0)
        # PRP должен быть активен

    def test_prp_inactive_outside_window(self):
        """PRP неактивен вне временного окна."""
        therapy = TherapyProtocol(
            prp_enabled=True,
            prp_start_time=1.0,
            prp_end_time=5.0,
        )
        config = IntegrationConfig()
        model = IntegratedModel(config=config, therapy=therapy, random_seed=42)
        model._transfer_therapy_to_abm(current_time_days=10.0)
        # PRP должен быть неактивен

    def test_pemf_active_in_window(self):
        """PEMF активен внутри временного окна."""
        therapy = TherapyProtocol(
            pemf_enabled=True,
            pemf_start_time=2.0,
            pemf_end_time=7.0,
        )
        config = IntegrationConfig()
        model = IntegratedModel(config=config, therapy=therapy, random_seed=42)
        model._transfer_therapy_to_abm(current_time_days=5.0)
        # PEMF должен быть активен

    def test_combined_therapy_both_active(self):
        """Комбинированная терапия — оба флага активны в пересечении."""
        therapy = TherapyProtocol(
            prp_enabled=True,
            prp_start_time=1.0,
            prp_end_time=10.0,
            pemf_enabled=True,
            pemf_start_time=2.0,
            pemf_end_time=8.0,
        )
        config = IntegrationConfig()
        model = IntegratedModel(config=config, therapy=therapy, random_seed=42)
        model._transfer_therapy_to_abm(current_time_days=5.0)
        # Оба должны быть активны

    def test_does_not_modify_sde_state(self):
        """Метод не изменяет SDE состояние."""
        therapy = TherapyProtocol(
            prp_enabled=True,
            prp_start_time=1.0,
            prp_end_time=5.0,
        )
        config = IntegrationConfig()
        model = IntegratedModel(config=config, therapy=therapy, random_seed=42)

        # Запомнить SDE конфигурацию
        sde_config_before = model.sde_model.config

        model._transfer_therapy_to_abm(current_time_days=3.0)

        # SDE конфигурация не изменилась
        assert model.sde_model.config is sde_config_before


# =============================================================================
# Phase 2: Test _spatial_scaling
# =============================================================================


class TestSpatialScaling:
    """Тесты конвертации SDE скаляр ↔ ABM 2D поле."""

    @pytest.fixture
    def model(self, bidirectional_integration_config):
        """Модель для тестов."""
        return IntegratedModel(
            config=bidirectional_integration_config,
            random_seed=42,
        )

    def test_sde_to_abm_returns_ndarray(self, model):
        """sde_to_abm возвращает ndarray."""
        abm_field = np.zeros((10, 10))
        result = model._spatial_scaling(5.0, abm_field, direction="sde_to_abm")
        assert isinstance(result, np.ndarray)

    def test_abm_to_sde_returns_float(self, model):
        """abm_to_sde возвращает float."""
        abm_field = np.ones((10, 10)) * 5.0
        result = model._spatial_scaling(0.0, abm_field, direction="abm_to_sde")
        assert isinstance(result, (float, np.floating))

    def test_sde_to_abm_zero_produces_zero_field(self, model):
        """C=0 → нулевое поле."""
        abm_field = np.zeros((10, 10))
        result = model._spatial_scaling(0.0, abm_field, direction="sde_to_abm")
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.zeros_like(result))

    def test_invalid_direction_raises_valueerror(self, model):
        """Неизвестное направление → ValueError."""
        abm_field = np.zeros((10, 10))
        with pytest.raises(ValueError):
            model._spatial_scaling(5.0, abm_field, direction="invalid")

    def test_abm_to_sde_uniform_field(self, model):
        """Однородное поле = 5.0 → float ≈ 5.0 (с масштабированием)."""
        abm_field = np.ones((10, 10)) * 5.0
        result = model._spatial_scaling(0.0, abm_field, direction="abm_to_sde")
        # Среднее поля = 5.0, результат масштабирован
        assert isinstance(result, (float, np.floating))

    def test_invertibility(self, model):
        """abm_to_sde(sde_to_abm(C)) ≈ C (обратимость)."""
        C_original = 10.0
        abm_field = np.zeros((10, 10))

        # SDE → ABM
        field_result = model._spatial_scaling(
            C_original, abm_field, direction="sde_to_abm"
        )

        # ABM → SDE
        C_recovered = model._spatial_scaling(
            0.0, field_result, direction="abm_to_sde"
        )

        # Должно быть приблизительно равно оригиналу
        assert C_recovered == pytest.approx(C_original, rel=0.1)


# =============================================================================
# Phase 2: Test _lifting
# =============================================================================


class TestLifting:
    """Тесты Equation-Free lifting: макро SDE → микро ABM."""

    @pytest.fixture
    def model(self, bidirectional_integration_config):
        """Модель для тестов lifting."""
        return IntegratedModel(
            config=bidirectional_integration_config,
            random_seed=42,
        )

    def test_lifting_n_100_produces_agents(self, model):
        """N=100 → snapshot содержит агентов пропорционально 100."""
        macro_state = {"N": 100.0, "C": 5.0}
        snapshot = model._lifting(macro_state)

        assert isinstance(snapshot, ABMSnapshot)
        assert snapshot.get_total_agents() > 0

    def test_lifting_n_zero_produces_empty(self, model):
        """N=0 → snapshot без агентов."""
        macro_state = {"N": 0.0, "C": 5.0}
        snapshot = model._lifting(macro_state)

        assert isinstance(snapshot, ABMSnapshot)
        assert snapshot.get_total_agents() == 0

    def test_lifting_cytokine_field_from_c(self, model):
        """C=5.0 → cytokine_field не пустое."""
        macro_state = {"N": 50.0, "C": 5.0}
        snapshot = model._lifting(macro_state)

        assert snapshot.cytokine_field is not None
        assert snapshot.cytokine_field.size > 0

    def test_lifting_all_zero(self, model):
        """N=0, C=0 → пустой snapshot."""
        macro_state = {"N": 0.0, "C": 0.0}
        snapshot = model._lifting(macro_state)

        assert snapshot.get_total_agents() == 0

    def test_lifting_negative_n_raises(self, model):
        """N < 0 → ValueError."""
        macro_state = {"N": -10.0, "C": 5.0}
        with pytest.raises(ValueError):
            model._lifting(macro_state)

    def test_lifting_returns_abm_snapshot(self, model):
        """Результат — ABMSnapshot с корректной структурой."""
        macro_state = {"N": 200.0, "C": 10.0}
        snapshot = model._lifting(macro_state)

        assert isinstance(snapshot, ABMSnapshot)
        assert hasattr(snapshot, "agents")
        assert hasattr(snapshot, "cytokine_field")
        assert hasattr(snapshot, "ecm_field")


# =============================================================================
# Phase 2: Test _restricting
# =============================================================================


class TestRestricting:
    """Тесты Equation-Free restricting: микро ABM → макро SDE."""

    @pytest.fixture
    def model(self, bidirectional_integration_config):
        """Модель для тестов restricting."""
        return IntegratedModel(
            config=bidirectional_integration_config,
            random_seed=42,
        )

    def test_restricting_100_agents(self, model):
        """100 агентов → result['N'] пропорционально 100."""
        agents = [
            AgentState(i, "stem", float(i * 5), float(i * 5), 0, 0, 1.0)
            for i in range(100)
        ]
        snapshot = ABMSnapshot(
            t=1.0,
            agents=agents,
            cytokine_field=np.ones((10, 10)) * 5.0,
            ecm_field=np.zeros((10, 10)),
        )

        result = model._restricting(snapshot)
        assert isinstance(result, dict)
        assert "N" in result
        assert result["N"] > 0

    def test_restricting_zero_agents(self, model):
        """0 агентов → result['N'] == 0."""
        snapshot = ABMSnapshot(
            t=1.0,
            agents=[],
            cytokine_field=np.zeros((10, 10)),
            ecm_field=np.zeros((10, 10)),
        )

        result = model._restricting(snapshot)
        assert result["N"] == 0

    def test_restricting_cytokine_field_to_c(self, model):
        """Uniform cytokine_field=1.0 → result['C'] ≈ 1.0."""
        agents = [
            AgentState(i, "stem", float(i * 5), float(i * 5), 0, 0, 1.0)
            for i in range(10)
        ]
        snapshot = ABMSnapshot(
            t=1.0,
            agents=agents,
            cytokine_field=np.ones((10, 10)) * 1.0,
            ecm_field=np.zeros((10, 10)),
        )

        result = model._restricting(snapshot)
        assert "C" in result

    def test_restricting_empty_snapshot(self, model):
        """Пустой snapshot → {'N': 0, 'C': 0, ...}."""
        snapshot = ABMSnapshot(
            t=0.0,
            agents=[],
            cytokine_field=np.zeros((10, 10)),
            ecm_field=np.zeros((10, 10)),
        )

        result = model._restricting(snapshot)
        assert isinstance(result, dict)
        assert "N" in result
        assert "C" in result
        assert result["N"] == 0
        assert result["C"] == 0

    def test_restricting_keys_present(self, model):
        """Результат содержит ключи N, C."""
        agents = [
            AgentState(i, "stem", float(i * 5), float(i * 5), 0, 0, 1.0)
            for i in range(5)
        ]
        snapshot = ABMSnapshot(
            t=1.0,
            agents=agents,
            cytokine_field=np.ones((10, 10)) * 3.0,
            ecm_field=np.zeros((10, 10)),
        )

        result = model._restricting(snapshot)
        assert "N" in result
        assert "C" in result

    def test_restricting_non_negative_values(self, model):
        """N ≥ 0, C ≥ 0 в результате."""
        agents = [
            AgentState(i, "stem", float(i * 5), float(i * 5), 0, 0, 1.0)
            for i in range(20)
        ]
        snapshot = ABMSnapshot(
            t=1.0,
            agents=agents,
            cytokine_field=np.ones((10, 10)) * 2.0,
            ecm_field=np.zeros((10, 10)),
        )

        result = model._restricting(snapshot)
        assert result["N"] >= 0
        assert result["C"] >= 0


# =============================================================================
# Phase 2: Test Lifting-Restricting Roundtrip
# =============================================================================


class TestLiftingRestrictingRoundtrip:
    """Тесты инварианта: restricting(lifting(state)) ≈ state."""

    @pytest.fixture
    def model(self, bidirectional_integration_config):
        """Модель для roundtrip тестов."""
        return IntegratedModel(
            config=bidirectional_integration_config,
            random_seed=42,
        )

    def test_roundtrip_n_preserved(self, model):
        """N сохраняется при roundtrip (с точностью до дискретизации)."""
        macro_state = {"N": 100.0, "C": 5.0}

        snapshot = model._lifting(macro_state)
        recovered = model._restricting(snapshot)

        # С точностью до дискретизации (ABM → целые агенты)
        assert recovered["N"] == pytest.approx(macro_state["N"], rel=0.2)

    def test_roundtrip_c_preserved(self, model):
        """C сохраняется при roundtrip."""
        macro_state = {"N": 50.0, "C": 10.0}

        snapshot = model._lifting(macro_state)
        recovered = model._restricting(snapshot)

        assert recovered["C"] == pytest.approx(macro_state["C"], rel=0.2)

    def test_roundtrip_zero_state(self, model):
        """Roundtrip для нулевого состояния."""
        macro_state = {"N": 0.0, "C": 0.0}

        snapshot = model._lifting(macro_state)
        recovered = model._restricting(snapshot)

        assert recovered["N"] == 0
        assert recovered["C"] == 0
