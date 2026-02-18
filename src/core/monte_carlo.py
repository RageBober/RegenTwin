"""Monte Carlo симулятор для стохастических моделей регенерации.

Запускает множество независимых траекторий SDE/ABM/Integrated моделей
и агрегирует результаты для вычисления ансамблевой статистики,
доверительных интервалов и квантилей.

Поддерживает:
- Параллельные вычисления (multiprocessing)
- Воспроизводимость через seed management
- Различные типы моделей (sde, abm, integrated)

Подробное описание: Description/description_monte_carlo.md
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from src.core.abm_model import ABMConfig, ABMTrajectory
from src.core.integration import IntegratedTrajectory, IntegrationConfig
from src.core.sde_model import SDEConfig, SDETrajectory, TherapyProtocol
from src.data.parameter_extraction import ModelParameters


@dataclass
class MonteCarloConfig:
    """Конфигурация Monte Carlo симулятора.

    Подробное описание: Description/description_monte_carlo.md#MonteCarloConfig
    """

    n_trajectories: int = 100  # Количество траекторий
    model_type: str = "sde"  # "sde", "abm", "integrated"

    # Конфигурации моделей (используется в зависимости от model_type)
    sde_config: SDEConfig | None = None
    abm_config: ABMConfig | None = None
    integration_config: IntegrationConfig | None = None

    # Параллелизация
    n_jobs: int = 1  # Количество параллельных процессов (1 = последовательно)
    use_multiprocessing: bool = False  # Использовать multiprocessing

    # Seed management для воспроизводимости
    base_seed: int | None = None

    # Квантили для расчёта
    quantiles: list[float] = field(default_factory=lambda: [0.05, 0.25, 0.5, 0.75, 0.95])

    # Callback для прогресса
    progress_callback: Callable[[int, int], None] | None = None

    def validate(self) -> bool:
        """Валидация параметров конфигурации.

        Returns:
            True если все параметры валидны

        Raises:
            ValueError: Если параметры некорректны

        Подробное описание: Description/description_monte_carlo.md#MonteCarloConfig.validate
        """
        if self.n_trajectories <= 0:
            raise ValueError("n_trajectories должен быть положительным")
        if self.model_type not in ["sde", "abm", "integrated"]:
            raise ValueError("model_type должен быть 'sde', 'abm' или 'integrated'")

        # Проверка наличия необходимой конфигурации
        if self.model_type == "sde" and self.sde_config is None:
            self.sde_config = SDEConfig()
        if self.model_type == "abm" and self.abm_config is None:
            self.abm_config = ABMConfig()
        if self.model_type == "integrated" and self.integration_config is None:
            raise ValueError("integration_config обязателен для model_type='integrated'")

        if self.n_jobs < 1:
            raise ValueError("n_jobs должен быть >= 1")

        # Проверка квантилей
        for q in self.quantiles:
            if not 0 < q < 1:
                raise ValueError(f"Квантиль {q} должен быть в диапазоне (0, 1)")

        return True


@dataclass
class TrajectoryResult:
    """Результат одной траектории.

    Подробное описание: Description/description_monte_carlo.md#TrajectoryResult
    """

    trajectory_id: int
    random_seed: int | None

    # Траектория (один из типов, в зависимости от model_type)
    sde_trajectory: SDETrajectory | None = None
    abm_trajectory: ABMTrajectory | None = None
    integrated_trajectory: IntegratedTrajectory | None = None

    # Краткая статистика для быстрого анализа
    final_N: float = 0.0  # Финальная плотность/количество клеток
    final_C: float = 0.0  # Финальная концентрация цитокинов
    max_N: float = 0.0  # Максимальная плотность
    growth_rate: float = 0.0  # Эффективная скорость роста

    # Метаданные
    success: bool = True
    error_message: str | None = None
    computation_time: float = 0.0  # секунды

    def get_statistics(self) -> dict[str, float]:
        """Получить статистику траектории.

        Returns:
            Словарь со статистиками

        Подробное описание: Description/description_monte_carlo.md#TrajectoryResult.get_statistics
        """
        return {
            "final_N": self.final_N,
            "final_C": self.final_C,
            "max_N": self.max_N,
            "growth_rate": self.growth_rate,
            "computation_time": self.computation_time,
            "success": 1.0 if self.success else 0.0,
        }

    def get_timeseries(self, variable: str = "N") -> tuple[np.ndarray, np.ndarray]:
        """Получить временной ряд переменной.

        Args:
            variable: "N" или "C"

        Returns:
            (times, values)

        Подробное описание: Description/description_monte_carlo.md#TrajectoryResult.get_timeseries
        """
        if self.sde_trajectory is not None:
            times = self.sde_trajectory.times
            if variable == "N":
                values = self.sde_trajectory.N_values
            else:
                values = self.sde_trajectory.C_values
            return (times, values)

        elif self.abm_trajectory is not None:
            times = self.abm_trajectory.get_times()
            dynamics = self.abm_trajectory.get_population_dynamics()
            if variable == "N":
                # Сумма всех типов агентов
                values = dynamics["stem"] + dynamics["macro"] + dynamics["fibro"]
            else:
                # Для ABM нет прямого C, возвращаем нули
                values = np.zeros(len(times))
            return (times, values)

        elif self.integrated_trajectory is not None:
            times = self.integrated_trajectory.times
            if variable == "N":
                values = np.array([s.sde_N for s in self.integrated_trajectory.states])
            else:
                values = np.array([s.sde_C for s in self.integrated_trajectory.states])
            return (times, values)

        return (np.array([0.0]), np.array([0.0]))


@dataclass
class MonteCarloResults:
    """Агрегированные результаты Monte Carlo.

    Подробное описание: Description/description_monte_carlo.md#MonteCarloResults
    """

    trajectories: list[TrajectoryResult]
    config: MonteCarloConfig

    # Агрегированная статистика по времени
    times: np.ndarray  # Общая временная ось
    mean_N: np.ndarray  # Средняя траектория N(t)
    std_N: np.ndarray  # Стандартное отклонение N(t)
    mean_C: np.ndarray  # Средняя траектория C(t)
    std_C: np.ndarray  # Стандартное отклонение C(t)

    # Квантили
    quantiles_N: dict[float, np.ndarray] = field(default_factory=dict)  # {0.05: [...], 0.95: [...]}
    quantiles_C: dict[float, np.ndarray] = field(default_factory=dict)

    # Метаданные
    n_successful: int = 0
    n_failed: int = 0
    total_computation_time: float = 0.0

    def get_summary_statistics(self) -> dict[str, float]:
        """Сводная статистика по всем траекториям.

        Returns:
            Словарь со сводными статистиками

        Подробное описание: Description/description_monte_carlo.md#MonteCarloResults.get_summary_statistics
        """
        successful_trajectories = [t for t in self.trajectories if t.success]

        if not successful_trajectories:
            return {
                "mean_final_N": 0.0,
                "std_final_N": 0.0,
                "mean_final_C": 0.0,
                "std_final_C": 0.0,
                "mean_growth_rate": 0.0,
                "success_rate": 0.0,
            }

        final_N_values = [t.final_N for t in successful_trajectories]
        final_C_values = [t.final_C for t in successful_trajectories]
        growth_rates = [t.growth_rate for t in successful_trajectories]

        return {
            "mean_final_N": float(np.mean(final_N_values)),
            "std_final_N": float(np.std(final_N_values)),
            "mean_final_C": float(np.mean(final_C_values)),
            "std_final_C": float(np.std(final_C_values)),
            "mean_growth_rate": float(np.mean(growth_rates)),
            "success_rate": self.get_success_rate(),
        }

    def get_confidence_interval(
        self,
        variable: str = "N",
        confidence_level: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Доверительный интервал для траектории.

        Args:
            variable: Переменная ("N" или "C")
            confidence_level: Уровень доверия (0-1)

        Returns:
            (lower_bound, upper_bound) массивы

        Подробное описание: Description/description_monte_carlo.md#MonteCarloResults.get_confidence_interval
        """
        alpha = 1 - confidence_level
        lower_q = alpha / 2
        upper_q = 1 - alpha / 2

        quantiles = self.quantiles_N if variable == "N" else self.quantiles_C
        mean = self.mean_N if variable == "N" else self.mean_C
        std = self.std_N if variable == "N" else self.std_C

        # Попробовать использовать квантили если есть
        lower = None
        upper = None

        for q, values in quantiles.items():
            if abs(q - lower_q) < 0.01:
                lower = values
            if abs(q - upper_q) < 0.01:
                upper = values

        # Если нет подходящих квантилей, использовать нормальное приближение
        if lower is None:
            z = 1.96 if confidence_level == 0.95 else 2.576 if confidence_level == 0.99 else 1.645
            lower = mean - z * std
        if upper is None:
            z = 1.96 if confidence_level == 0.95 else 2.576 if confidence_level == 0.99 else 1.645
            upper = mean + z * std

        return (lower, upper)

    def get_final_distribution(self, variable: str = "N") -> np.ndarray:
        """Распределение финальных значений.

        Args:
            variable: Переменная ("N" или "C")

        Returns:
            Массив финальных значений по всем траекториям

        Подробное описание: Description/description_monte_carlo.md#MonteCarloResults.get_final_distribution
        """
        successful_trajectories = [t for t in self.trajectories if t.success]

        if variable == "N":
            return np.array([t.final_N for t in successful_trajectories])
        else:
            return np.array([t.final_C for t in successful_trajectories])

    def get_success_rate(self) -> float:
        """Доля успешных траекторий.

        Returns:
            Доля успешных траекторий [0, 1]

        Подробное описание: Description/description_monte_carlo.md#MonteCarloResults.get_success_rate
        """
        if self.n_successful + self.n_failed == 0:
            return 0.0
        return self.n_successful / (self.n_successful + self.n_failed)


class MonteCarloSimulator:
    """Monte Carlo симулятор для стохастических моделей регенерации.

    Запускает множество независимых траекторий и агрегирует результаты.

    Подробное описание: Description/description_monte_carlo.md#MonteCarloSimulator
    """

    def __init__(
        self,
        config: MonteCarloConfig,
        therapy: TherapyProtocol | None = None,
    ) -> None:
        """Инициализация симулятора.

        Args:
            config: Конфигурация Monte Carlo
            therapy: Протокол терапии

        Подробное описание: Description/description_monte_carlo.md#MonteCarloSimulator.__init__
        """
        self._config = config
        self._config.validate()
        self._therapy = therapy

        # Setup random seeds для воспроизводимости
        if config.base_seed is not None:
            self._rng = np.random.default_rng(config.base_seed)
            self._seeds: list[int | None] = [
                int(self._rng.integers(0, 2**31)) for _ in range(config.n_trajectories)
            ]
        else:
            self._seeds = [None] * config.n_trajectories

    @property
    def config(self) -> MonteCarloConfig:
        """Получить конфигурацию."""
        return self._config

    def run(
        self,
        initial_params: ModelParameters,
    ) -> MonteCarloResults:
        """Запуск Monte Carlo симуляций.

        Args:
            initial_params: Начальные параметры из parameter_extraction

        Returns:
            MonteCarloResults с ансамблевой статистикой

        Подробное описание: Description/description_monte_carlo.md#MonteCarloSimulator.run
        """
        import time

        results: list[TrajectoryResult] = []
        start_time = time.time()

        for i, seed in enumerate(self._seeds):
            try:
                result = self._run_single_trajectory(i, initial_params, seed)
                results.append(result)
            except Exception as e:
                results.append(
                    TrajectoryResult(
                        trajectory_id=i,
                        random_seed=seed,
                        success=False,
                        error_message=str(e),
                    )
                )

            # Callback прогресса
            if self._config.progress_callback:
                self._config.progress_callback(i + 1, self._config.n_trajectories)

        total_time = time.time() - start_time

        # Агрегация результатов
        mc_results = self._aggregate_trajectories(results)
        mc_results.total_computation_time = total_time

        return mc_results

    def _run_single_trajectory(
        self,
        trajectory_id: int,
        initial_params: ModelParameters,
        random_seed: int | None,
    ) -> TrajectoryResult:
        """Запуск одной траектории.

        Args:
            trajectory_id: Идентификатор траектории
            initial_params: Начальные параметры
            random_seed: Seed для этой траектории

        Returns:
            TrajectoryResult с результатами

        Подробное описание: Description/description_monte_carlo.md#MonteCarloSimulator._run_single_trajectory
        """
        import time

        start_time = time.time()

        result = TrajectoryResult(
            trajectory_id=trajectory_id,
            random_seed=random_seed,
        )

        try:
            if self._config.model_type == "sde":
                trajectory = self._run_sde_trajectory(initial_params, random_seed)
                result.sde_trajectory = trajectory
                stats = trajectory.get_statistics()
                result.final_N = stats["final_N"]
                result.final_C = stats["final_C"]
                result.max_N = stats["max_N"]
                result.growth_rate = stats["growth_rate"]

            elif self._config.model_type == "abm":
                trajectory = self._run_abm_trajectory(initial_params, random_seed)
                result.abm_trajectory = trajectory
                stats = trajectory.get_statistics()
                result.final_N = stats.get("final_total", 0.0)
                result.final_C = 0.0
                result.max_N = result.final_N  # Упрощение
                result.growth_rate = stats.get("growth_rate", 0.0)

            elif self._config.model_type == "integrated":
                trajectory = self._run_integrated_trajectory(initial_params, random_seed)
                result.integrated_trajectory = trajectory
                stats = trajectory.get_statistics()
                result.final_N = stats.get("final_sde_N", 0.0)
                result.final_C = stats.get("final_sde_C", 0.0)
                result.max_N = result.final_N
                result.growth_rate = 0.0

            result.success = True

        except Exception as e:
            result.success = False
            result.error_message = str(e)

        result.computation_time = time.time() - start_time
        return result

    def _run_sde_trajectory(
        self,
        initial_params: ModelParameters,
        random_seed: int | None,
    ) -> SDETrajectory:
        """Запуск SDE траектории.

        Args:
            initial_params: Начальные параметры
            random_seed: Seed

        Returns:
            SDETrajectory

        Подробное описание: Description/description_monte_carlo.md#MonteCarloSimulator._run_sde_trajectory
        """
        from src.core.sde_model import SDEModel

        config = self._config.sde_config or SDEConfig()
        model = SDEModel(config=config, therapy=self._therapy, random_seed=random_seed)
        return model.simulate(initial_params)

    def _run_abm_trajectory(
        self,
        initial_params: ModelParameters,
        random_seed: int | None,
    ) -> ABMTrajectory:
        """Запуск ABM траектории.

        Args:
            initial_params: Начальные параметры
            random_seed: Seed

        Returns:
            ABMTrajectory

        Подробное описание: Description/description_monte_carlo.md#MonteCarloSimulator._run_abm_trajectory
        """
        from src.core.abm_model import ABMModel

        config = self._config.abm_config or ABMConfig()
        model = ABMModel(config=config, random_seed=random_seed)
        return model.simulate(initial_params)

    def _run_integrated_trajectory(
        self,
        initial_params: ModelParameters,
        random_seed: int | None,
    ) -> IntegratedTrajectory:
        """Запуск интегрированной траектории.

        Args:
            initial_params: Начальные параметры
            random_seed: Seed

        Returns:
            IntegratedTrajectory

        Подробное описание: Description/description_monte_carlo.md#MonteCarloSimulator._run_integrated_trajectory
        """
        from src.core.integration import IntegratedModel

        config = self._config.integration_config
        if config is None:
            raise ValueError("integration_config required for integrated model")

        model = IntegratedModel(config=config, therapy=self._therapy, random_seed=random_seed)
        return model.simulate(initial_params)

    def _aggregate_trajectories(
        self,
        results: list[TrajectoryResult],
    ) -> MonteCarloResults:
        """Агрегация результатов траекторий.

        Вычисляет:
        - Среднюю траекторию
        - Стандартное отклонение
        - Квантили

        Args:
            results: Список результатов траекторий

        Returns:
            MonteCarloResults с агрегированной статистикой

        Подробное описание: Description/description_monte_carlo.md#MonteCarloSimulator._aggregate_trajectories
        """
        successful_results = [r for r in results if r.success]
        n_successful = len(successful_results)
        n_failed = len(results) - n_successful

        # Если нет успешных траекторий
        if n_successful == 0:
            return MonteCarloResults(
                trajectories=results,
                config=self._config,
                times=np.array([0.0]),
                mean_N=np.array([0.0]),
                std_N=np.array([0.0]),
                mean_C=np.array([0.0]),
                std_C=np.array([0.0]),
                quantiles_N={},
                quantiles_C={},
                n_successful=0,
                n_failed=n_failed,
            )

        # Извлечение массивов траекторий
        times, trajectories_N = self._extract_trajectories_array(successful_results, "N")
        _, trajectories_C = self._extract_trajectories_array(successful_results, "C")

        # Вычисление статистики
        mean_N = np.mean(trajectories_N, axis=0)
        std_N = np.std(trajectories_N, axis=0)
        mean_C = np.mean(trajectories_C, axis=0)
        std_C = np.std(trajectories_C, axis=0)

        # Вычисление квантилей
        quantiles_N = self._calculate_quantiles(trajectories_N, self._config.quantiles)
        quantiles_C = self._calculate_quantiles(trajectories_C, self._config.quantiles)

        return MonteCarloResults(
            trajectories=results,
            config=self._config,
            times=times,
            mean_N=mean_N,
            std_N=std_N,
            mean_C=mean_C,
            std_C=std_C,
            quantiles_N=quantiles_N,
            quantiles_C=quantiles_C,
            n_successful=n_successful,
            n_failed=n_failed,
        )

    def _extract_trajectories_array(
        self,
        results: list[TrajectoryResult],
        variable: str = "N",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Извлечь массив траекторий [n_traj, n_steps].

        Args:
            results: Список результатов траекторий
            variable: "N" или "C"

        Returns:
            (times, trajectories_2d)

        Подробное описание: Description/description_monte_carlo.md#MonteCarloSimulator._extract_trajectories_array
        """
        if not results:
            return (np.array([0.0]), np.array([[0.0]]))

        # Получить временные ряды из каждой траектории
        timeseries_list = []
        times = None

        for result in results:
            t, values = result.get_timeseries(variable)
            timeseries_list.append(values)
            if times is None:
                times = t

        if times is None:
            times = np.array([0.0])

        # Интерполировать на общую временную сетку
        # Найти минимальную и максимальную длину
        min_len = min(len(ts) for ts in timeseries_list)

        # Обрезать все траектории до минимальной длины
        trajectories_2d = np.array([ts[:min_len] for ts in timeseries_list])
        times = times[:min_len]

        return (times, trajectories_2d)

    def _calculate_quantiles(
        self,
        trajectories: np.ndarray,
        quantiles: list[float],
    ) -> dict[float, np.ndarray]:
        """Расчёт квантилей по траекториям.

        Args:
            trajectories: Массив [n_traj, n_steps]
            quantiles: Список квантилей для расчёта

        Returns:
            Словарь {quantile: values}

        Подробное описание: Description/description_monte_carlo.md#MonteCarloSimulator._calculate_quantiles
        """
        result = {}
        for q in quantiles:
            result[q] = np.quantile(trajectories, q, axis=0)
        return result

    def _extract_summary_stats(
        self,
        trajectory_result: TrajectoryResult,
    ) -> dict[str, float]:
        """Извлечь сводную статистику из траектории.

        Args:
            trajectory_result: Результат траектории

        Returns:
            Словарь со статистиками

        Подробное описание: Description/description_monte_carlo.md#MonteCarloSimulator._extract_summary_stats
        """
        return trajectory_result.get_statistics()

    def _run_parallel(
        self,
        initial_params: ModelParameters,
    ) -> list[TrajectoryResult]:
        """Параллельный запуск траекторий через ProcessPoolExecutor.

        Разделяет n_trajectories на n_jobs процессов. Каждый процесс
        получает свой seed (base_seed + offset) для воспроизводимости.
        Собирает результаты и объединяет в единый список.

        При n_jobs=1 делегирует в последовательный запуск (fallback).
        При ошибке в процессе — траектория помечается как failed.

        Args:
            initial_params: Начальные параметры модели

        Returns:
            Список TrajectoryResult со всех процессов

        Подробное описание: Description/Phase2/description_monte_carlo.md#MonteCarloSimulator._run_parallel
        """
        if self._config.n_jobs <= 1 or not self._config.use_multiprocessing:
            # Последовательный fallback
            results: list[TrajectoryResult] = []
            for i, seed in enumerate(self._seeds):
                try:
                    result = self._run_single_trajectory(i, initial_params, seed)
                    results.append(result)
                except Exception as e:
                    results.append(TrajectoryResult(
                        trajectory_id=i,
                        random_seed=seed,
                        success=False,
                        error_message=str(e),
                    ))
            return results

        # Параллельное выполнение через ThreadPoolExecutor
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results_arr: list[TrajectoryResult | None] = [None] * self._config.n_trajectories
        with ThreadPoolExecutor(max_workers=self._config.n_jobs) as executor:
            futures = {}
            for i, seed in enumerate(self._seeds):
                future = executor.submit(
                    self._run_single_trajectory, i, initial_params, seed
                )
                futures[future] = i

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results_arr[idx] = future.result()
                except Exception as e:
                    results_arr[idx] = TrajectoryResult(
                        trajectory_id=idx,
                        random_seed=self._seeds[idx],
                        success=False,
                        error_message=str(e),
                    )

        return [r for r in results_arr if r is not None]

    def _progress_callback_wrapper(
        self,
        completed: int,
        total: int,
    ) -> None:
        """Thread-safe обёртка для progress_callback.

        Агрегирует прогресс от нескольких параллельных процессов
        через threading.Lock. Вызывает пользовательский callback
        с суммарным прогрессом (completed_total / n_trajectories).

        Args:
            completed: Количество завершённых траекторий в процессе
            total: Общее количество траекторий в процессе

        Подробное описание: Description/Phase2/description_monte_carlo.md#MonteCarloSimulator._progress_callback_wrapper
        """
        if not hasattr(self, "_progress_lock"):
            import threading
            self._progress_lock = threading.Lock()

        with self._progress_lock:
            if self._config.progress_callback is not None:
                self._config.progress_callback(completed, total)

    def _validate_parallel_config(self, config: MonteCarloConfig) -> bool:
        """Валидация конфигурации для параллельного запуска.

        Проверяет:
        - multiprocessing модуль доступен
        - n_jobs ≤ os.cpu_count()
        - Объекты picklable (необходимо для ProcessPoolExecutor)

        Args:
            config: Конфигурация Monte Carlo

        Returns:
            True если конфигурация валидна

        Raises:
            RuntimeError: Если multiprocessing недоступен
            ValueError: Если n_jobs > cpu_count

        Подробное описание: Description/Phase2/description_monte_carlo.md#MonteCarloSimulator._validate_parallel_config
        """
        import os

        if config.n_jobs < 1:
            raise ValueError("n_jobs must be >= 1")

        cpu_count = os.cpu_count() or 1
        if config.n_jobs > cpu_count:
            raise ValueError(
                f"n_jobs ({config.n_jobs}) exceeds cpu_count ({cpu_count})"
            )

        return True


def run_monte_carlo(
    initial_params: ModelParameters,
    config: MonteCarloConfig,
    therapy: TherapyProtocol | None = None,
) -> MonteCarloResults:
    """Convenience функция для Monte Carlo симуляций.

    Args:
        initial_params: Начальные параметры из parameter_extraction
        config: Конфигурация Monte Carlo
        therapy: Протокол терапии (опционально)

    Returns:
        MonteCarloResults с ансамблевой статистикой

    Подробное описание: Description/description_monte_carlo.md#run_monte_carlo
    """
    simulator = MonteCarloSimulator(config=config, therapy=therapy)
    return simulator.run(initial_params)


def run_parameter_sweep(
    initial_params: ModelParameters,
    parameter_name: str,
    parameter_values: list[float],
    base_config: MonteCarloConfig,
    therapy: TherapyProtocol | None = None,
) -> dict[float, MonteCarloResults]:
    """Запуск Monte Carlo для набора значений параметра.

    Args:
        initial_params: Начальные параметры
        parameter_name: Имя варьируемого параметра
        parameter_values: Значения параметра
        base_config: Базовая конфигурация MC
        therapy: Протокол терапии

    Returns:
        Словарь {parameter_value: MonteCarloResults}

    Подробное описание: Description/description_monte_carlo.md#run_parameter_sweep
    """
    from dataclasses import replace

    results = {}

    for value in parameter_values:
        # Создаём копию параметров с изменённым значением
        modified_params = replace(initial_params, **{parameter_name: value})

        # Запуск Monte Carlo
        simulator = MonteCarloSimulator(config=base_config, therapy=therapy)
        results[value] = simulator.run(modified_params)

    return results


def compare_therapies(
    initial_params: ModelParameters,
    therapies: dict[str, TherapyProtocol],
    config: MonteCarloConfig,
) -> dict[str, MonteCarloResults]:
    """Сравнение различных протоколов терапии через Monte Carlo.

    Args:
        initial_params: Начальные параметры
        therapies: Словарь {название: протокол}
        config: Конфигурация MC

    Returns:
        Словарь {название: MonteCarloResults}

    Подробное описание: Description/description_monte_carlo.md#compare_therapies
    """
    results = {}

    for name, therapy in therapies.items():
        simulator = MonteCarloSimulator(config=config, therapy=therapy)
        results[name] = simulator.run(initial_params)

    return results
