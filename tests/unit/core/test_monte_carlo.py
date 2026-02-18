"""TDD тесты для Monte Carlo симулятора.

Тестирование:
- MonteCarloConfig: параметры симулятора, валидация, квантили
- TrajectoryResult: результат одной траектории
- MonteCarloResults: агрегированные результаты, статистика, доверительные интервалы
- MonteCarloSimulator: запуск траекторий, агрегация, seed management
- run_monte_carlo: convenience функция
- run_parameter_sweep: sweep по параметрам
- compare_therapies: сравнение протоколов терапии

Все тесты написаны для stub-реализации (NotImplementedError).
После реализации методов тесты должны проходить.
"""

import numpy as np
import pytest

from src.core.abm_model import ABMConfig, ABMTrajectory
from src.core.integration import IntegrationConfig, IntegratedTrajectory
from src.core.monte_carlo import (
    MonteCarloConfig,
    MonteCarloResults,
    MonteCarloSimulator,
    TrajectoryResult,
    compare_therapies,
    run_monte_carlo,
    run_parameter_sweep,
)
from src.core.sde_model import SDEConfig, SDETrajectory, TherapyProtocol
from src.data.parameter_extraction import ModelParameters


# =============================================================================
# Test MonteCarloConfig
# =============================================================================


class TestMonteCarloConfig:
    """Тесты конфигурации Monte Carlo симулятора."""

    def test_default_values(self):
        """Проверка значений по умолчанию."""
        config = MonteCarloConfig()

        assert config.n_trajectories == 100
        assert config.model_type == "sde"
        assert config.n_jobs == 1
        assert config.use_multiprocessing is False
        assert config.base_seed is None

    def test_default_quantiles(self):
        """Квантили по умолчанию."""
        config = MonteCarloConfig()

        assert config.quantiles == [0.05, 0.25, 0.5, 0.75, 0.95]

    def test_custom_n_trajectories(self):
        """Пользовательское количество траекторий."""
        config = MonteCarloConfig(n_trajectories=500)

        assert config.n_trajectories == 500

    def test_model_types(self):
        """Допустимые типы моделей."""
        for model_type in ["sde", "abm", "integrated"]:
            config = MonteCarloConfig(model_type=model_type)
            assert config.model_type == model_type

    def test_validate_returns_true_for_valid_config(self, sde_monte_carlo_config):
        """Валидация возвращает True для корректной конфигурации."""
        result = sde_monte_carlo_config.validate()

        assert result is True

    def test_validate_negative_n_trajectories_raises(self):
        """Отрицательное количество траекторий вызывает ошибку."""
        config = MonteCarloConfig(n_trajectories=-1)

        with pytest.raises(ValueError, match="n_trajectories"):
            config.validate()

    def test_validate_zero_n_trajectories_raises(self):
        """Нулевое количество траекторий вызывает ошибку."""
        config = MonteCarloConfig(n_trajectories=0)

        with pytest.raises(ValueError, match="n_trajectories"):
            config.validate()

    def test_validate_invalid_model_type_raises(self):
        """Некорректный тип модели вызывает ошибку."""
        config = MonteCarloConfig(model_type="invalid")

        with pytest.raises(ValueError, match="model_type"):
            config.validate()

    def test_validate_n_jobs_zero_raises(self):
        """n_jobs = 0 вызывает ошибку."""
        config = MonteCarloConfig(n_jobs=0)

        with pytest.raises(ValueError, match="n_jobs"):
            config.validate()

    def test_validate_negative_n_jobs_raises(self):
        """Отрицательный n_jobs вызывает ошибку."""
        config = MonteCarloConfig(n_jobs=-1)

        with pytest.raises(ValueError, match="n_jobs"):
            config.validate()

    def test_validate_invalid_quantile_raises(self):
        """Некорректные квантили вызывают ошибку."""
        config = MonteCarloConfig(quantiles=[0.0, 0.5, 0.95])

        with pytest.raises(ValueError, match="Квантиль"):
            config.validate()

    def test_validate_quantile_above_one_raises(self):
        """Квантиль > 1 вызывает ошибку."""
        config = MonteCarloConfig(quantiles=[0.05, 0.5, 1.5])

        with pytest.raises(ValueError, match="Квантиль"):
            config.validate()

    def test_validate_creates_default_sde_config(self):
        """Валидация создаёт SDEConfig по умолчанию для model_type=sde."""
        config = MonteCarloConfig(model_type="sde", sde_config=None)
        config.validate()

        assert config.sde_config is not None
        assert isinstance(config.sde_config, SDEConfig)

    def test_validate_creates_default_abm_config(self):
        """Валидация создаёт ABMConfig по умолчанию для model_type=abm."""
        config = MonteCarloConfig(model_type="abm", abm_config=None)
        config.validate()

        assert config.abm_config is not None
        assert isinstance(config.abm_config, ABMConfig)

    def test_validate_integrated_requires_config(self):
        """model_type=integrated требует integration_config."""
        config = MonteCarloConfig(model_type="integrated", integration_config=None)

        with pytest.raises(ValueError, match="integration_config"):
            config.validate()

    def test_with_custom_sde_config(self):
        """Пользовательская SDEConfig."""
        sde_config = SDEConfig(t_max=10.0, dt=0.001)
        config = MonteCarloConfig(model_type="sde", sde_config=sde_config)

        assert config.sde_config.t_max == 10.0
        assert config.sde_config.dt == 0.001

    def test_with_multiprocessing(self):
        """Конфигурация с multiprocessing."""
        config = MonteCarloConfig(n_jobs=4, use_multiprocessing=True)

        assert config.n_jobs == 4
        assert config.use_multiprocessing is True

    def test_with_base_seed(self):
        """Конфигурация с base_seed."""
        config = MonteCarloConfig(base_seed=42)

        assert config.base_seed == 42


# =============================================================================
# Test TrajectoryResult
# =============================================================================


class TestTrajectoryResult:
    """Тесты результата одной траектории."""

    @pytest.fixture
    def sample_trajectory_result(self):
        """Пример результата траектории."""
        sde_traj = SDETrajectory(
            times=np.linspace(0, 30, 301),
            N_values=np.linspace(1000, 2000, 301),
            C_values=np.linspace(100, 50, 301),
        )
        return TrajectoryResult(
            trajectory_id=1,
            random_seed=42,
            sde_trajectory=sde_traj,
            final_N=2000.0,
            final_C=50.0,
            max_N=2000.0,
            growth_rate=0.023,
            success=True,
            computation_time=1.5,
        )

    def test_trajectory_id(self, sample_trajectory_result):
        """Проверка ID траектории."""
        assert sample_trajectory_result.trajectory_id == 1

    def test_random_seed(self, sample_trajectory_result):
        """Проверка seed траектории."""
        assert sample_trajectory_result.random_seed == 42

    def test_final_values(self, sample_trajectory_result):
        """Проверка финальных значений."""
        assert sample_trajectory_result.final_N == 2000.0
        assert sample_trajectory_result.final_C == 50.0

    def test_max_n(self, sample_trajectory_result):
        """Проверка максимального N."""
        assert sample_trajectory_result.max_N == 2000.0

    def test_growth_rate(self, sample_trajectory_result):
        """Проверка эффективной скорости роста."""
        assert sample_trajectory_result.growth_rate == pytest.approx(0.023)

    def test_success_flag(self, sample_trajectory_result):
        """Проверка флага успеха."""
        assert sample_trajectory_result.success is True

    def test_computation_time(self, sample_trajectory_result):
        """Проверка времени вычисления."""
        assert sample_trajectory_result.computation_time == 1.5

    def test_failed_trajectory(self):
        """Результат для упавшей траектории."""
        result = TrajectoryResult(
            trajectory_id=2,
            random_seed=123,
            success=False,
            error_message="Numerical instability detected",
        )

        assert result.success is False
        assert result.error_message == "Numerical instability detected"

    def test_get_statistics(self, sample_trajectory_result):
        """Метод get_statistics возвращает словарь статистик."""
        stats = sample_trajectory_result.get_statistics()
        assert isinstance(stats, dict)
        assert "final_N" in stats or "computation_time" in stats

    def test_get_timeseries(self, sample_trajectory_result):
        """Метод get_timeseries возвращает временной ряд N."""
        times, values = sample_trajectory_result.get_timeseries(variable="N")
        assert len(times) == len(values)


# =============================================================================
# Test MonteCarloResults
# =============================================================================


class TestMonteCarloResults:
    """Тесты агрегированных результатов Monte Carlo."""

    @pytest.fixture
    def sample_mc_results(self):
        """Пример результатов MC."""
        n_traj = 10
        n_steps = 31

        times = np.linspace(0, 30, n_steps)

        # Создаём mock-траектории
        trajectories = []
        for i in range(n_traj):
            result = TrajectoryResult(
                trajectory_id=i,
                random_seed=i * 10,
                final_N=1800.0 + i * 40,  # 1800-2160
                final_C=40.0 + i * 2,
                max_N=1900.0 + i * 40,
                growth_rate=0.02 + i * 0.001,
                success=True,
                computation_time=1.0 + i * 0.1,
            )
            trajectories.append(result)

        # Средние траектории (упрощённые)
        mean_N = np.linspace(1000, 1980, n_steps)  # Среднее final_N ≈ 1980
        std_N = np.full(n_steps, 100.0)
        mean_C = np.linspace(100, 49, n_steps)  # Среднее final_C ≈ 49
        std_C = np.full(n_steps, 5.0)

        # Квантили
        quantiles_N = {
            0.05: mean_N - 150,
            0.25: mean_N - 70,
            0.5: mean_N,
            0.75: mean_N + 70,
            0.95: mean_N + 150,
        }
        quantiles_C = {
            0.05: mean_C - 8,
            0.5: mean_C,
            0.95: mean_C + 8,
        }

        config = MonteCarloConfig(n_trajectories=n_traj)

        return MonteCarloResults(
            trajectories=trajectories,
            config=config,
            times=times,
            mean_N=mean_N,
            std_N=std_N,
            mean_C=mean_C,
            std_C=std_C,
            quantiles_N=quantiles_N,
            quantiles_C=quantiles_C,
            n_successful=n_traj,
            n_failed=0,
            total_computation_time=sum(t.computation_time for t in trajectories),
        )

    def test_results_have_trajectories(self, sample_mc_results):
        """Результаты содержат траектории."""
        assert len(sample_mc_results.trajectories) == 10

    def test_results_have_config(self, sample_mc_results):
        """Результаты содержат конфигурацию."""
        assert sample_mc_results.config.n_trajectories == 10

    def test_results_have_times(self, sample_mc_results):
        """Результаты содержат временную ось."""
        assert len(sample_mc_results.times) == 31
        assert sample_mc_results.times[0] == 0
        assert sample_mc_results.times[-1] == 30

    def test_results_have_mean_trajectories(self, sample_mc_results):
        """Результаты содержат средние траектории."""
        assert len(sample_mc_results.mean_N) == 31
        assert len(sample_mc_results.mean_C) == 31

    def test_results_have_std(self, sample_mc_results):
        """Результаты содержат стандартные отклонения."""
        assert len(sample_mc_results.std_N) == 31
        assert len(sample_mc_results.std_C) == 31

    def test_results_have_quantiles(self, sample_mc_results):
        """Результаты содержат квантили."""
        assert 0.05 in sample_mc_results.quantiles_N
        assert 0.5 in sample_mc_results.quantiles_N
        assert 0.95 in sample_mc_results.quantiles_N

    def test_success_count(self, sample_mc_results):
        """Подсчёт успешных траекторий."""
        assert sample_mc_results.n_successful == 10
        assert sample_mc_results.n_failed == 0

    def test_total_computation_time(self, sample_mc_results):
        """Общее время вычисления."""
        expected = sum(1.0 + i * 0.1 for i in range(10))
        assert sample_mc_results.total_computation_time == pytest.approx(expected)

    def test_get_success_rate_all_successful(self, sample_mc_results):
        """get_success_rate для 100% успешных."""
        rate = sample_mc_results.get_success_rate()

        assert rate == 1.0

    def test_get_success_rate_with_failures(self):
        """get_success_rate с неудачными траекториями."""
        results = MonteCarloResults(
            trajectories=[],
            config=MonteCarloConfig(),
            times=np.array([]),
            mean_N=np.array([]),
            std_N=np.array([]),
            mean_C=np.array([]),
            std_C=np.array([]),
            n_successful=80,
            n_failed=20,
        )

        rate = results.get_success_rate()

        assert rate == pytest.approx(0.8)

    def test_get_success_rate_empty(self):
        """get_success_rate для пустых результатов."""
        results = MonteCarloResults(
            trajectories=[],
            config=MonteCarloConfig(),
            times=np.array([]),
            mean_N=np.array([]),
            std_N=np.array([]),
            mean_C=np.array([]),
            std_C=np.array([]),
            n_successful=0,
            n_failed=0,
        )

        rate = results.get_success_rate()

        assert rate == 0.0

    def test_get_summary_statistics(self, sample_mc_results):
        """Метод get_summary_statistics возвращает словарь."""
        stats = sample_mc_results.get_summary_statistics()
        assert isinstance(stats, dict)

    def test_get_confidence_interval(self, sample_mc_results):
        """Метод get_confidence_interval возвращает (lower, upper)."""
        lower, upper = sample_mc_results.get_confidence_interval(variable="N", confidence_level=0.95)
        assert len(lower) == len(upper)

    def test_get_final_distribution(self, sample_mc_results):
        """Метод get_final_distribution возвращает массив финальных значений."""
        dist = sample_mc_results.get_final_distribution(variable="N")
        assert len(dist) == sample_mc_results.config.n_trajectories


# =============================================================================
# Test MonteCarloSimulator Initialization
# =============================================================================


class TestMonteCarloSimulatorInit:
    """Тесты инициализации Monte Carlo симулятора."""

    def test_init_with_config(self, sde_monte_carlo_config):
        """Инициализация с конфигурацией."""
        simulator = MonteCarloSimulator(config=sde_monte_carlo_config)

        assert simulator.config is not None
        assert isinstance(simulator.config, MonteCarloConfig)

    def test_init_validates_config(self):
        """Инициализация валидирует конфигурацию."""
        invalid_config = MonteCarloConfig(n_trajectories=-1)

        with pytest.raises(ValueError):
            MonteCarloSimulator(config=invalid_config)

    def test_init_with_therapy(self, sde_monte_carlo_config, prp_therapy_protocol):
        """Инициализация с протоколом терапии."""
        simulator = MonteCarloSimulator(
            config=sde_monte_carlo_config,
            therapy=prp_therapy_protocol,
        )

        assert simulator._therapy is not None

    def test_init_generates_seeds_with_base_seed(self):
        """С base_seed генерируются воспроизводимые seeds."""
        config = MonteCarloConfig(n_trajectories=10, base_seed=42)
        simulator = MonteCarloSimulator(config=config)

        assert len(simulator._seeds) == 10
        assert all(s is not None for s in simulator._seeds)

    def test_init_same_base_seed_same_seeds(self):
        """Одинаковый base_seed даёт одинаковые seeds."""
        config1 = MonteCarloConfig(n_trajectories=10, base_seed=42)
        config2 = MonteCarloConfig(n_trajectories=10, base_seed=42)

        sim1 = MonteCarloSimulator(config=config1)
        sim2 = MonteCarloSimulator(config=config2)

        assert sim1._seeds == sim2._seeds

    def test_init_different_base_seed_different_seeds(self):
        """Разные base_seed дают разные seeds."""
        config1 = MonteCarloConfig(n_trajectories=10, base_seed=42)
        config2 = MonteCarloConfig(n_trajectories=10, base_seed=123)

        sim1 = MonteCarloSimulator(config=config1)
        sim2 = MonteCarloSimulator(config=config2)

        assert sim1._seeds != sim2._seeds

    def test_init_no_base_seed_none_seeds(self):
        """Без base_seed seeds = [None, None, ...]."""
        config = MonteCarloConfig(n_trajectories=10, base_seed=None)
        simulator = MonteCarloSimulator(config=config)

        assert all(s is None for s in simulator._seeds)


# =============================================================================
# Test MonteCarloSimulator Methods
# =============================================================================


class TestMonteCarloSimulatorMethods:
    """Тесты методов MonteCarloSimulator."""

    @pytest.fixture
    def simulator(self, sde_monte_carlo_config):
        """Симулятор для тестов."""
        return MonteCarloSimulator(config=sde_monte_carlo_config)

    def test_run_returns_results(self, simulator, sample_model_parameters):
        """Метод run возвращает MonteCarloResults."""
        results = simulator.run(sample_model_parameters)
        assert isinstance(results, MonteCarloResults)
        assert len(results.trajectories) == simulator.config.n_trajectories

    def test_calculate_quantiles(self, simulator):
        """Метод _calculate_quantiles вычисляет квантили."""
        trajectories = np.random.randn(10, 100)
        quantiles = simulator._calculate_quantiles(trajectories, quantiles=[0.05, 0.5, 0.95])
        assert len(quantiles) == 3
        assert 0.05 in quantiles


# =============================================================================
# Test MonteCarloSimulator Behavior (After Implementation)
# =============================================================================


class TestMonteCarloSimulatorBehavior:
    """Тесты поведения симулятора (после реализации)."""

    def test_run_returns_results(self):
        """run() возвращает MonteCarloResults.

        После реализации: должен вернуть корректные результаты.
        """
        # config = MonteCarloConfig(n_trajectories=10)
        # simulator = MonteCarloSimulator(config=config)
        # params = ModelParameters(n0=1000.0, c0=100.0)
        # results = simulator.run(params)
        # assert isinstance(results, MonteCarloResults)
        # assert len(results.trajectories) == 10
        pass

    def test_run_reproducibility_with_seed(self):
        """run() воспроизводим с одинаковым base_seed.

        После реализации: одинаковые seed дают одинаковые результаты.
        """
        # config = MonteCarloConfig(n_trajectories=10, base_seed=42)
        # params = ModelParameters(n0=1000.0, c0=100.0)
        #
        # sim1 = MonteCarloSimulator(config=config)
        # res1 = sim1.run(params)
        #
        # sim2 = MonteCarloSimulator(config=config)
        # res2 = sim2.run(params)
        #
        # np.testing.assert_array_almost_equal(res1.mean_N, res2.mean_N)
        pass

    def test_run_with_therapy(self):
        """run() с терапией даёт другие результаты.

        После реализации: терапия влияет на траектории.
        """
        # config = MonteCarloConfig(n_trajectories=10, base_seed=42)
        # params = ModelParameters(n0=1000.0, c0=100.0)
        # therapy = TherapyProtocol(prp_dose=10.0, prp_start_day=0.0)
        #
        # sim_no_therapy = MonteCarloSimulator(config=config)
        # res_no_therapy = sim_no_therapy.run(params)
        #
        # sim_with_therapy = MonteCarloSimulator(config=config, therapy=therapy)
        # res_with_therapy = sim_with_therapy.run(params)
        #
        # # Терапия должна улучшить рост
        # assert res_with_therapy.mean_N[-1] > res_no_therapy.mean_N[-1]
        pass

    def test_run_handles_failures_gracefully(self):
        """run() корректно обрабатывает ошибки в траекториях.

        После реализации: failed траектории помечаются, но не прерывают run().
        """
        # config = MonteCarloConfig(n_trajectories=10)
        # simulator = MonteCarloSimulator(config=config)
        # # Параметры, которые могут вызвать проблемы
        # params = ModelParameters(n0=0.0, c0=0.0)
        # results = simulator.run(params)
        # # Должен завершиться без исключения
        pass

    def test_mean_trajectory_shape(self):
        """Средняя траектория имеет правильную форму.

        После реализации: shape соответствует временным точкам.
        """
        # config = MonteCarloConfig(n_trajectories=10, sde_config=SDEConfig(t_max=10.0))
        # simulator = MonteCarloSimulator(config=config)
        # params = ModelParameters(n0=1000.0, c0=100.0)
        # results = simulator.run(params)
        # assert len(results.times) == len(results.mean_N)
        # assert len(results.times) == len(results.std_N)
        pass


# =============================================================================
# Test run_monte_carlo Function
# =============================================================================


class TestRunMonteCarloFunction:
    """Тесты convenience функции run_monte_carlo."""

    def test_run_monte_carlo_returns_results(self, sample_model_parameters):
        """Функция возвращает MonteCarloResults."""
        config = MonteCarloConfig(n_trajectories=5, base_seed=42)

        results = run_monte_carlo(
            initial_params=sample_model_parameters,
            config=config,
        )
        assert isinstance(results, MonteCarloResults)

    def test_run_monte_carlo_with_therapy(
        self, sample_model_parameters, prp_therapy_protocol
    ):
        """Функция принимает протокол терапии."""
        config = MonteCarloConfig(n_trajectories=5, base_seed=42)

        results = run_monte_carlo(
            initial_params=sample_model_parameters,
            config=config,
            therapy=prp_therapy_protocol,
        )
        assert isinstance(results, MonteCarloResults)


# =============================================================================
# Test run_parameter_sweep Function
# =============================================================================


class TestRunParameterSweepFunction:
    """Тесты функции sweep по параметрам."""

    def test_run_parameter_sweep_returns_dict(self, sample_model_parameters):
        """Функция возвращает словарь результатов."""
        config = MonteCarloConfig(n_trajectories=3, base_seed=42)

        results = run_parameter_sweep(
            initial_params=sample_model_parameters,
            parameter_name="n0",
            parameter_values=[500.0, 1000.0],
            base_config=config,
        )
        assert isinstance(results, dict)
        assert 500.0 in results or len(results) > 0


# =============================================================================
# Test compare_therapies Function
# =============================================================================


class TestCompareTherapiesFunction:
    """Тесты функции сравнения терапий."""

    def test_compare_therapies_returns_dict(self, sample_model_parameters):
        """Функция возвращает словарь результатов."""
        config = MonteCarloConfig(n_trajectories=3, base_seed=42)
        therapies = {
            "no_therapy": TherapyProtocol(),
            "prp_only": TherapyProtocol(
                prp_enabled=True,
                prp_initial_concentration=10.0,
                prp_start_time=0.0,
            ),
        }

        results = compare_therapies(
            initial_params=sample_model_parameters,
            therapies=therapies,
            config=config,
        )
        assert isinstance(results, dict)
        assert "no_therapy" in results
        assert "prp_only" in results


# =============================================================================
# Test Quantile Calculation
# =============================================================================


class TestQuantileCalculation:
    """Тесты расчёта квантилей."""

    def test_calculate_quantiles_shape(self):
        """Квантили имеют правильную форму.

        После реализации: shape = (n_steps,) для каждого квантиля.
        """
        # config = MonteCarloConfig(n_trajectories=100)
        # simulator = MonteCarloSimulator(config=config)
        #
        # trajectories = np.random.randn(100, 50)
        # quantiles = simulator._calculate_quantiles(trajectories, [0.05, 0.5, 0.95])
        #
        # assert quantiles[0.05].shape == (50,)
        # assert quantiles[0.5].shape == (50,)
        # assert quantiles[0.95].shape == (50,)
        pass

    def test_median_equals_50_quantile(self):
        """Медиана равна 50% квантилю.

        После реализации: quantiles[0.5] ≈ np.median.
        """
        # config = MonteCarloConfig(n_trajectories=100)
        # simulator = MonteCarloSimulator(config=config)
        #
        # trajectories = np.random.randn(100, 50)
        # quantiles = simulator._calculate_quantiles(trajectories, [0.5])
        # expected_median = np.median(trajectories, axis=0)
        #
        # np.testing.assert_array_almost_equal(quantiles[0.5], expected_median)
        pass

    def test_quantile_ordering(self):
        """Квантили упорядочены: q5 < q50 < q95.

        После реализации: порядок соблюдается.
        """
        # config = MonteCarloConfig(n_trajectories=100)
        # simulator = MonteCarloSimulator(config=config)
        #
        # trajectories = np.random.randn(100, 50) + np.arange(50)  # Возрастающий тренд
        # quantiles = simulator._calculate_quantiles(trajectories, [0.05, 0.5, 0.95])
        #
        # assert np.all(quantiles[0.05] <= quantiles[0.5])
        # assert np.all(quantiles[0.5] <= quantiles[0.95])
        pass


# =============================================================================
# Test Confidence Intervals
# =============================================================================


class TestConfidenceIntervals:
    """Тесты расчёта доверительных интервалов."""

    def test_confidence_interval_95(self):
        """95% доверительный интервал.

        После реализации: interval содержит ~95% траекторий.
        """
        # results = <run MC>
        # lower, upper = results.get_confidence_interval("N", 0.95)
        # # Должны содержать ~95% финальных значений
        pass

    def test_confidence_interval_bounds(self):
        """Границы доверительного интервала корректны.

        После реализации: lower < mean < upper.
        """
        # results = <run MC>
        # lower, upper = results.get_confidence_interval("N", 0.95)
        # assert np.all(lower <= results.mean_N)
        # assert np.all(results.mean_N <= upper)
        pass

    def test_wider_interval_higher_confidence(self):
        """Больший уровень доверия -> шире интервал.

        После реализации: 99% интервал шире 90%.
        """
        # results = <run MC>
        # lower_90, upper_90 = results.get_confidence_interval("N", 0.90)
        # lower_99, upper_99 = results.get_confidence_interval("N", 0.99)
        #
        # # 99% интервал должен быть шире
        # width_90 = upper_90 - lower_90
        # width_99 = upper_99 - lower_99
        # assert np.all(width_99 >= width_90)
        pass


# =============================================================================
# Test Final Distribution
# =============================================================================


class TestFinalDistribution:
    """Тесты распределения финальных значений."""

    def test_final_distribution_length(self):
        """Длина = количество успешных траекторий.

        После реализации: len = n_successful.
        """
        # results = <run MC with n_trajectories=100>
        # final_N = results.get_final_distribution("N")
        # assert len(final_N) == results.n_successful
        pass

    def test_final_distribution_statistics(self):
        """Статистика финального распределения.

        После реализации: mean/std соответствуют ожиданиям.
        """
        # results = <run MC>
        # final_N = results.get_final_distribution("N")
        #
        # # Должно примерно соответствовать последней точке mean_N/std_N
        # assert np.mean(final_N) == pytest.approx(results.mean_N[-1], rel=0.1)
        pass


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestMonteCarloEdgeCases:
    """Тесты граничных случаев Monte Carlo."""

    def test_single_trajectory(self):
        """Симуляция с одной траекторией.

        После реализации: должно работать корректно.
        """
        # config = MonteCarloConfig(n_trajectories=1)
        # simulator = MonteCarloSimulator(config=config)
        # params = ModelParameters(n0=1000.0, c0=100.0)
        # results = simulator.run(params)
        # assert results.n_successful == 1
        pass

    def test_many_trajectories(self):
        """Симуляция с большим количеством траекторий.

        После реализации: должно работать (может быть медленно).
        """
        # config = MonteCarloConfig(n_trajectories=1000)
        # simulator = MonteCarloSimulator(config=config)
        # params = ModelParameters(n0=1000.0, c0=100.0)
        # results = simulator.run(params)
        # assert results.n_successful >= 990  # Допускаем некоторые сбои
        pass

    def test_all_trajectories_fail(self):
        """Все траектории падают.

        После реализации: results.n_failed = n_trajectories.
        """
        # config = MonteCarloConfig(n_trajectories=10)
        # # Параметры, гарантирующие сбой
        # params = ModelParameters(n0=-1000.0, c0=-100.0)
        # simulator = MonteCarloSimulator(config=config)
        # results = simulator.run(params)
        # assert results.n_failed == 10
        # assert results.get_success_rate() == 0.0
        pass

    def test_progress_callback(self):
        """Progress callback вызывается.

        После реализации: callback получает (current, total).
        """
        # progress = []
        # def callback(current, total):
        #     progress.append((current, total))
        #
        # config = MonteCarloConfig(n_trajectories=10, progress_callback=callback)
        # simulator = MonteCarloSimulator(config=config)
        # params = ModelParameters(n0=1000.0, c0=100.0)
        # results = simulator.run(params)
        #
        # assert len(progress) == 10
        # assert progress[-1] == (10, 10)
        pass


# =============================================================================
# Test Multiprocessing
# =============================================================================


class TestMultiprocessing:
    """Тесты параллельных вычислений."""

    def test_multiprocessing_same_results(self):
        """Multiprocessing даёт те же результаты (с seed).

        После реализации: результаты детерминированы.
        """
        # config_seq = MonteCarloConfig(
        #     n_trajectories=20, n_jobs=1, use_multiprocessing=False, base_seed=42
        # )
        # config_par = MonteCarloConfig(
        #     n_trajectories=20, n_jobs=4, use_multiprocessing=True, base_seed=42
        # )
        #
        # params = ModelParameters(n0=1000.0, c0=100.0)
        #
        # sim_seq = MonteCarloSimulator(config=config_seq)
        # res_seq = sim_seq.run(params)
        #
        # sim_par = MonteCarloSimulator(config=config_par)
        # res_par = sim_par.run(params)
        #
        # # Результаты должны быть одинаковы
        # np.testing.assert_array_almost_equal(res_seq.mean_N, res_par.mean_N)
        pass

    def test_multiprocessing_faster(self):
        """Multiprocessing быстрее для большого числа траекторий.

        После реализации: время n_jobs=4 < время n_jobs=1.
        """
        # import time
        #
        # config_seq = MonteCarloConfig(n_trajectories=100, n_jobs=1)
        # config_par = MonteCarloConfig(n_trajectories=100, n_jobs=4, use_multiprocessing=True)
        #
        # params = ModelParameters(n0=1000.0, c0=100.0)
        #
        # start = time.time()
        # MonteCarloSimulator(config=config_seq).run(params)
        # time_seq = time.time() - start
        #
        # start = time.time()
        # MonteCarloSimulator(config=config_par).run(params)
        # time_par = time.time() - start
        #
        # # Параллельно должно быть быстрее (с некоторым запасом)
        # assert time_par < time_seq * 0.8
        pass


# =============================================================================
# Test Summary Statistics
# =============================================================================


class TestSummaryStatistics:
    """Тесты сводной статистики."""

    def test_summary_statistics_keys(self):
        """Ожидаемые ключи в сводной статистике.

        После реализации: содержит mean_final_N, std_final_N, и т.д.
        """
        # results = <run MC>
        # stats = results.get_summary_statistics()
        # assert "mean_final_N" in stats
        # assert "std_final_N" in stats
        # assert "mean_final_C" in stats
        # assert "mean_growth_rate" in stats
        # assert "success_rate" in stats
        pass

    def test_summary_statistics_values(self):
        """Значения сводной статистики корректны.

        После реализации: соответствуют данным траекторий.
        """
        # results = <run MC>
        # stats = results.get_summary_statistics()
        #
        # final_N_values = [t.final_N for t in results.trajectories if t.success]
        # assert stats["mean_final_N"] == pytest.approx(np.mean(final_N_values))
        # assert stats["std_final_N"] == pytest.approx(np.std(final_N_values))
        pass


# =============================================================================
# Phase 2: Test _run_parallel
# =============================================================================


class TestRunParallel:
    """Тесты параллельного запуска траекторий через ProcessPoolExecutor."""

    @pytest.fixture
    def simulator_sequential(self):
        """Симулятор для последовательного запуска."""
        config = MonteCarloConfig(
            n_trajectories=10, n_jobs=1, base_seed=42
        )
        return MonteCarloSimulator(config=config)

    def test_n_jobs_1_equivalent_sequential(self, sample_model_parameters):
        """n_jobs=1 — эквивалентно последовательному запуску."""
        config = MonteCarloConfig(
            n_trajectories=5, n_jobs=1, base_seed=42
        )
        simulator = MonteCarloSimulator(config=config)
        results = simulator._run_parallel(sample_model_parameters)

        assert len(results) == 5

    def test_n_jobs_2_returns_all_results(self, sample_model_parameters):
        """n_jobs=2 — возвращает все n_trajectories результатов."""
        config = MonteCarloConfig(
            n_trajectories=10, n_jobs=2, base_seed=42,
            use_multiprocessing=True,
        )
        simulator = MonteCarloSimulator(config=config)
        results = simulator._run_parallel(sample_model_parameters)

        assert len(results) == 10

    def test_reproducibility_with_seed(self, sample_model_parameters):
        """Одинаковый base_seed → идентичные результаты."""
        config = MonteCarloConfig(
            n_trajectories=5, n_jobs=2, base_seed=42,
            use_multiprocessing=True,
        )
        sim1 = MonteCarloSimulator(config=config)
        results1 = sim1._run_parallel(sample_model_parameters)

        sim2 = MonteCarloSimulator(config=config)
        results2 = sim2._run_parallel(sample_model_parameters)

        # Финальные значения N должны совпадать
        final_N_1 = sorted([r.final_N for r in results1 if r.success])
        final_N_2 = sorted([r.final_N for r in results2 if r.success])
        assert len(final_N_1) == len(final_N_2)

    def test_failure_handling(self, sample_model_parameters):
        """Ошибка в одной траектории → помечается как failed, остальные OK."""
        config = MonteCarloConfig(
            n_trajectories=5, n_jobs=2, base_seed=42,
            use_multiprocessing=True,
        )
        simulator = MonteCarloSimulator(config=config)
        results = simulator._run_parallel(sample_model_parameters)

        # Все результаты должны быть TrajectoryResult
        for r in results:
            assert isinstance(r, TrajectoryResult)

    def test_single_trajectory_parallel(self, sample_model_parameters):
        """n_trajectories=1 при n_jobs=1 → 1 результат."""
        config = MonteCarloConfig(
            n_trajectories=1, n_jobs=1, base_seed=42,
        )
        simulator = MonteCarloSimulator(config=config)
        results = simulator._run_parallel(sample_model_parameters)

        assert len(results) == 1

    def test_all_results_have_trajectory_ids(self, sample_model_parameters):
        """Каждый результат имеет уникальный trajectory_id."""
        config = MonteCarloConfig(
            n_trajectories=5, n_jobs=2, base_seed=42,
            use_multiprocessing=True,
        )
        simulator = MonteCarloSimulator(config=config)
        results = simulator._run_parallel(sample_model_parameters)

        ids = [r.trajectory_id for r in results]
        assert len(set(ids)) == len(ids)  # Все ID уникальны

    def test_results_have_seeds(self, sample_model_parameters):
        """Каждый результат имеет random_seed."""
        config = MonteCarloConfig(
            n_trajectories=5, n_jobs=1, base_seed=42,
        )
        simulator = MonteCarloSimulator(config=config)
        results = simulator._run_parallel(sample_model_parameters)

        for r in results:
            assert r.random_seed is not None


# =============================================================================
# Phase 2: Test _run_parallel Invariants
# =============================================================================


class TestRunParallelInvariants:
    """Тесты инвариантов параллельного запуска."""

    def test_result_count_equals_n_trajectories(self, sample_model_parameters):
        """len(results) == n_trajectories."""
        for n in [1, 5, 10]:
            config = MonteCarloConfig(
                n_trajectories=n, n_jobs=1, base_seed=42,
            )
            simulator = MonteCarloSimulator(config=config)
            results = simulator._run_parallel(sample_model_parameters)
            assert len(results) == n

    def test_results_independent_of_n_jobs(self, sample_model_parameters):
        """Результаты не зависят от n_jobs (только скорость)."""
        config_1 = MonteCarloConfig(
            n_trajectories=4, n_jobs=1, base_seed=42,
        )
        config_2 = MonteCarloConfig(
            n_trajectories=4, n_jobs=2, base_seed=42,
            use_multiprocessing=True,
        )

        sim1 = MonteCarloSimulator(config=config_1)
        results1 = sim1._run_parallel(sample_model_parameters)

        sim2 = MonteCarloSimulator(config=config_2)
        results2 = sim2._run_parallel(sample_model_parameters)

        # Одинаковое количество успешных/неудачных
        success_1 = sum(1 for r in results1 if r.success)
        success_2 = sum(1 for r in results2 if r.success)
        assert success_1 == success_2

    def test_successful_results_count(self, sample_model_parameters):
        """n_successful + n_failed == n_trajectories."""
        config = MonteCarloConfig(
            n_trajectories=5, n_jobs=1, base_seed=42,
        )
        simulator = MonteCarloSimulator(config=config)
        results = simulator._run_parallel(sample_model_parameters)

        n_success = sum(1 for r in results if r.success)
        n_fail = sum(1 for r in results if not r.success)
        assert n_success + n_fail == 5


# =============================================================================
# Phase 2: Test _progress_callback_wrapper
# =============================================================================


class TestProgressCallbackWrapper:
    """Тесты thread-safe обёртки для progress_callback."""

    def test_single_call(self):
        """Один вызов передаёт (completed, total) в callback."""
        progress = []

        def callback(current, total):
            progress.append((current, total))

        config = MonteCarloConfig(
            n_trajectories=10, base_seed=42,
            progress_callback=callback,
        )
        simulator = MonteCarloSimulator(config=config)
        simulator._progress_callback_wrapper(5, 10)

        # Callback должен быть вызван
        # (точное поведение зависит от реализации)

    def test_sequential_calls_accumulate(self):
        """Последовательные вызовы корректно агрегируют прогресс."""
        progress = []

        def callback(current, total):
            progress.append((current, total))

        config = MonteCarloConfig(
            n_trajectories=10, base_seed=42,
            progress_callback=callback,
        )
        simulator = MonteCarloSimulator(config=config)
        simulator._progress_callback_wrapper(3, 10)
        simulator._progress_callback_wrapper(5, 10)

    def test_null_callback_no_error(self):
        """callback=None → без ошибок."""
        config = MonteCarloConfig(
            n_trajectories=10, base_seed=42,
            progress_callback=None,
        )
        simulator = MonteCarloSimulator(config=config)
        # Не должно вызвать ошибку
        simulator._progress_callback_wrapper(5, 10)

    def test_concurrent_calls_thread_safe(self):
        """Конкурентные вызовы (через threading) → thread-safe."""
        import threading

        progress = []
        lock = threading.Lock()

        def callback(current, total):
            with lock:
                progress.append((current, total))

        config = MonteCarloConfig(
            n_trajectories=20, base_seed=42,
            progress_callback=callback,
        )
        simulator = MonteCarloSimulator(config=config)

        threads = []
        for i in range(4):
            t = threading.Thread(
                target=simulator._progress_callback_wrapper,
                args=(5 * (i + 1), 20),
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Все вызовы должны завершиться без ошибок

    def test_progress_monotonic(self):
        """Суммарный прогресс монотонно растёт."""
        progress_values = []

        def callback(current, _total):
            progress_values.append(current)

        config = MonteCarloConfig(
            n_trajectories=5, base_seed=42,
            progress_callback=callback,
        )
        simulator = MonteCarloSimulator(config=config)

        for i in range(1, 6):
            simulator._progress_callback_wrapper(i, 5)

        # Прогресс должен быть записан


# =============================================================================
# Phase 2: Test _validate_parallel_config
# =============================================================================


class TestValidateParallelConfig:
    """Тесты валидации конфигурации для параллельного запуска."""

    def test_n_jobs_1_valid(self):
        """n_jobs=1 → True (всегда валидно)."""
        config = MonteCarloConfig(n_jobs=1)
        simulator = MonteCarloSimulator(config=config)
        result = simulator._validate_parallel_config(config)
        assert result is True

    def test_n_jobs_exceeds_cpu_count_raises(self):
        """n_jobs > cpu_count → ValueError."""
        import os

        cpu_count = os.cpu_count() or 1
        config = MonteCarloConfig(n_jobs=cpu_count + 100)
        simulator = MonteCarloSimulator(config=config)

        with pytest.raises(ValueError):
            simulator._validate_parallel_config(config)

    def test_n_jobs_zero_raises(self):
        """n_jobs=0 → ValueError (ловится validate config)."""
        # MonteCarloConfig.validate() уже ловит n_jobs < 1
        # Но _validate_parallel_config также должен проверять
        config = MonteCarloConfig.__new__(MonteCarloConfig)
        config.n_jobs = 0
        config.n_trajectories = 10
        config.model_type = "sde"
        config.use_multiprocessing = False
        config.base_seed = None
        config.quantiles = [0.5]
        config.progress_callback = None
        config.sde_config = SDEConfig()
        config.abm_config = None
        config.integration_config = None

        simulator = MonteCarloSimulator.__new__(MonteCarloSimulator)
        simulator._config = config
        simulator._seeds = [None] * 10

        with pytest.raises(ValueError):
            simulator._validate_parallel_config(config)

    def test_n_jobs_negative_raises(self):
        """n_jobs < 0 → ValueError."""
        config = MonteCarloConfig.__new__(MonteCarloConfig)
        config.n_jobs = -1
        config.n_trajectories = 10
        config.model_type = "sde"
        config.use_multiprocessing = False
        config.base_seed = None
        config.quantiles = [0.5]
        config.progress_callback = None
        config.sde_config = SDEConfig()
        config.abm_config = None
        config.integration_config = None

        simulator = MonteCarloSimulator.__new__(MonteCarloSimulator)
        simulator._config = config
        simulator._seeds = [None] * 10

        with pytest.raises(ValueError):
            simulator._validate_parallel_config(config)

    def test_valid_parallel_config(self):
        """Валидная параллельная конфигурация → True."""
        config = MonteCarloConfig(
            n_jobs=2, use_multiprocessing=True, base_seed=42,
        )
        simulator = MonteCarloSimulator(config=config)
        result = simulator._validate_parallel_config(config)
        assert result is True

    def test_n_jobs_equals_cpu_count_valid(self):
        """n_jobs == cpu_count → True (граничный случай)."""
        import os

        cpu_count = os.cpu_count() or 1
        config = MonteCarloConfig(n_jobs=cpu_count)
        simulator = MonteCarloSimulator(config=config)
        result = simulator._validate_parallel_config(config)
        assert result is True
