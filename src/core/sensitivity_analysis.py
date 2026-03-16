"""Анализ чувствительности 20-переменной SDE системы регенерации тканей.

Четыре метода анализа за единым интерфейсом SensitivityAnalyzer:
- Sobol indices: глобальная чувствительность (SALib, first-order + total)
- Morris screening: скрининг 40+ параметров для отбора ключевых (SALib)
- Local sensitivity: частные производные вблизи номинальных значений (SciPy)
- Tornado diagrams: визуализация ранжированной чувствительности (Matplotlib)

Подробное описание: Description/Phase3/description_sensitivity_analysis.md
"""

from __future__ import annotations

import dataclasses
import time  # noqa: F401 — используется в реализации run_sobol/run_morris/run_local (Этап 3)
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from src.core.extended_sde import ExtendedSDEModel, ExtendedSDEState
from src.core.parameters import ParameterSet

try:
    from loguru import logger
except ImportError:
    import logging as _logging

    logger = _logging.getLogger(__name__)  # type: ignore[assignment]


# =====================================================================
# Enums
# =====================================================================


# Description: Description/Phase3/description_sensitivity_analysis.md#SensitivityMethod
class SensitivityMethod(Enum):
    """Метод анализа чувствительности.

    Определяет алгоритм и соответствующую библиотеку:
    - SOBOL: глобальный, variance-based (SALib)
    - MORRIS: скрининг, OAT-based (SALib)
    - LOCAL: локальный, finite differences (SciPy/NumPy)

    Подробное описание:
        Description/Phase3/description_sensitivity_analysis.md#SensitivityMethod
    """

    SOBOL = "sobol"
    MORRIS = "morris"
    LOCAL = "local"


# =====================================================================
# Dataclasses
# =====================================================================


# Description: Description/Phase3/description_sensitivity_analysis.md#ParameterBounds
@dataclass
class ParameterBounds:
    """Границы одного параметра для анализа чувствительности.

    Задаёт диапазон варьирования параметра при сэмплировании.
    Номинальное значение используется для локальной чувствительности.

    Подробное описание:
        Description/Phase3/description_sensitivity_analysis.md#ParameterBounds
    """

    name: str  # Имя параметра (поле ParameterSet)
    lower: float  # Нижняя граница
    upper: float  # Верхняя граница
    nominal: float | None = None  # Номинальное значение (None → из ParameterSet)


# Description: Description/Phase3/description_sensitivity_analysis.md#SensitivityConfig
@dataclass
class SensitivityConfig:
    """Единая конфигурация для всех методов анализа чувствительности.

    Содержит границы параметров, выходные переменные, агрегацию,
    настройки forward model и seed воспроизводимости.

    Подробное описание:
        Description/Phase3/description_sensitivity_analysis.md#SensitivityConfig
    """

    # Метод
    method: SensitivityMethod = SensitivityMethod.SOBOL

    # Границы параметров
    parameter_bounds: list[ParameterBounds] = field(default_factory=list)

    # Выходные переменные SDE
    output_variables: list[str] = field(default_factory=lambda: ["F"])

    # Агрегация выхода
    output_time_index: int = -1  # Индекс временного шага (-1 = финальное)
    output_aggregation: str = "final"  # "final", "mean", "max", "auc"

    # Forward model
    t_span: tuple[float, float] = (0.0, 720.0)  # Временной диапазон (часы)
    dt: float = 0.01  # Шаг SDE солвера

    # Воспроизводимость
    rng_seed: int | None = None

    # Description: Description/Phase3/description_sensitivity_analysis.md#SensitivityConfig.validate
    def validate(self) -> bool:
        """Валидация конфигурации.

        Проверяет корректность всех полей: bounds непусты, dt > 0,
        output_aggregation допустим, t_span корректен.

        Returns:
            True если конфигурация валидна

        Raises:
            ValueError: Если поле невалидно

        Подробное описание:
            Description/Phase3/description_sensitivity_analysis.md#SensitivityConfig.validate
        """
        if not self.parameter_bounds:
            raise ValueError("parameter_bounds must not be empty")
        if self.dt <= 0:
            raise ValueError("dt must be > 0")
        if self.t_span[1] <= self.t_span[0]:
            raise ValueError("t_span[1] must be > t_span[0]")
        if not self.output_variables:
            raise ValueError("output_variables must not be empty")
        if self.output_aggregation not in {"final", "mean", "max", "auc"}:
            raise ValueError(
                f"output_aggregation must be 'final', 'mean', 'max', or 'auc', "
                f"got '{self.output_aggregation}'"
            )
        for b in self.parameter_bounds:
            if b.lower >= b.upper:
                raise ValueError(
                    f"Parameter '{b.name}': lower ({b.lower}) must be < upper ({b.upper})"
                )
            if b.nominal is not None and not (b.lower <= b.nominal <= b.upper):
                raise ValueError(
                    f"Parameter '{b.name}': nominal ({b.nominal}) "
                    f"must be in [{b.lower}, {b.upper}]"
                )
        # Validate parameter names against ParameterSet
        valid_names = {f.name for f in dataclasses.fields(ParameterSet)}
        for b in self.parameter_bounds:
            if b.name not in valid_names:
                raise ValueError(f"Parameter '{b.name}' not found in ParameterSet")
        return True


# Description: Description/Phase3/description_sensitivity_analysis.md#SobolResult
@dataclass
class SobolResult:
    """Результат анализа чувствительности методом Sobol.

    Содержит first-order (S1) и total-effect (ST) индексы,
    опционально second-order (S2), доверительные интервалы.

    Подробное описание:
        Description/Phase3/description_sensitivity_analysis.md#SobolResult
    """

    parameter_names: list[str] = field(default_factory=list)
    S1: np.ndarray = field(default_factory=lambda: np.array([]))  # First-order, (n_params,)
    ST: np.ndarray = field(default_factory=lambda: np.array([]))  # Total-effect, (n_params,)
    S2: np.ndarray | None = None  # Second-order, (n_params, n_params)
    S1_conf: np.ndarray = field(default_factory=lambda: np.array([]))  # CI для S1
    ST_conf: np.ndarray = field(default_factory=lambda: np.array([]))  # CI для ST
    output_variable: str = ""  # Имя выходной переменной
    n_samples: int = 0  # Число базовых сэмплов (N)
    n_model_runs: int = 0  # Фактическое число запусков модели
    elapsed_seconds: float = 0.0  # Время выполнения

    # Description: Description/Phase3/description_sensitivity_analysis.md#SobolResult.get_ranking
    def get_ranking(self) -> list[tuple[str, float, float]]:
        """Ранжирование параметров по убыванию total-effect индекса ST.

        Returns:
            Список (name, S1, ST), отсортированный по ST убыванию

        Подробное описание:
            Description/Phase3/description_sensitivity_analysis.md#SobolResult.get_ranking
        """
        if len(self.parameter_names) == 0:
            return []
        indices = np.argsort(-self.ST)
        return [(self.parameter_names[i], float(self.S1[i]), float(self.ST[i])) for i in indices]


# Description: Description/Phase3/description_sensitivity_analysis.md#MorrisResult
@dataclass
class MorrisResult:
    """Результат анализа чувствительности методом Morris.

    Содержит mu (среднее элементарных эффектов), mu_star (среднее
    абсолютных эффектов — главная метрика скрининга), sigma (СКО).

    Подробное описание:
        Description/Phase3/description_sensitivity_analysis.md#MorrisResult
    """

    parameter_names: list[str] = field(default_factory=list)
    mu: np.ndarray = field(default_factory=lambda: np.array([]))  # Среднее эффектов, (n_params,)
    mu_star: np.ndarray = field(default_factory=lambda: np.array([]))  # |mu|, (n_params,)
    sigma: np.ndarray = field(default_factory=lambda: np.array([]))  # СКО эффектов, (n_params,)
    mu_star_conf: np.ndarray = field(default_factory=lambda: np.array([]))  # CI для mu_star
    output_variable: str = ""  # Имя выходной переменной
    n_trajectories: int = 0  # Число Morris траекторий
    n_levels: int = 4  # Число уровней сетки
    n_model_runs: int = 0  # Фактическое число запусков
    elapsed_seconds: float = 0.0  # Время выполнения

    # Description: Description/Phase3/description_sensitivity_analysis.md#MorrisResult.get_influential
    def get_influential(self, threshold_ratio: float = 0.1) -> list[str]:
        """Отбор влиятельных параметров по порогу mu_star.

        Возвращает параметры, у которых mu_star > threshold_ratio * max(mu_star).
        Это ключевой метод для скрининга из 40+ параметров.

        Args:
            threshold_ratio: Доля от максимального mu_star (0.0–1.0)

        Returns:
            Список имён влиятельных параметров

        Raises:
            ValueError: Если threshold_ratio вне (0, 1]

        Подробное описание:
            Description/Phase3/description_sensitivity_analysis.md#MorrisResult.get_influential
        """
        if not (0 < threshold_ratio <= 1.0):
            raise ValueError("threshold_ratio must be in (0, 1]")
        if len(self.mu_star) == 0:
            return []
        threshold = threshold_ratio * float(np.max(self.mu_star))
        return [
            name
            for name, ms in zip(self.parameter_names, self.mu_star, strict=True)
            if float(ms) >= threshold
        ]


# Description: Description/Phase3/description_sensitivity_analysis.md#LocalSensitivityResult
@dataclass
class LocalSensitivityResult:
    """Результат локального анализа чувствительности.

    Содержит частные производные dY/dp_i и безразмерные индексы
    эластичности (p_i/Y)*(dY/dp_i) вблизи номинальных значений.

    Подробное описание:
        Description/Phase3/description_sensitivity_analysis.md#LocalSensitivityResult
    """

    parameter_names: list[str] = field(default_factory=list)
    partial_derivatives: np.ndarray = field(  # dY/dp_i, (n_params,)
        default_factory=lambda: np.array([]),
    )
    elasticity_indices: np.ndarray = field(  # (p_i/Y)*(dY/dp_i), (n_params,)
        default_factory=lambda: np.array([]),
    )
    nominal_output: float = 0.0  # Y при номинальных параметрах
    nominal_params: dict[str, float] = field(default_factory=dict)
    delta: float = 0.01  # Относительное возмущение
    output_variable: str = ""  # Имя выходной переменной
    elapsed_seconds: float = 0.0  # Время выполнения

    # Description: Description/Phase3/description_sensitivity_analysis.md#LocalSensitivityResult.get_ranking
    def get_ranking(self) -> list[tuple[str, float, float]]:
        """Ранжирование параметров по убыванию |elasticity|.

        Returns:
            Список (name, partial_derivative, elasticity), отсортированный
            по |elasticity| убыванию

        Подробное описание:
            Description/Phase3/description_sensitivity_analysis.md#LocalSensitivityResult.get_ranking
        """
        if len(self.parameter_names) == 0:
            return []
        indices = np.argsort(-np.abs(self.elasticity_indices))
        return [
            (
                self.parameter_names[i],
                float(self.partial_derivatives[i]),
                float(self.elasticity_indices[i]),
            )
            for i in indices
        ]


# Description: Description/Phase3/description_sensitivity_analysis.md#TornadoData
@dataclass
class TornadoData:
    """Данные для построения tornado diagram.

    Унифицированный контейнер, создаваемый из результатов любого метода
    через classmethods TornadoPlotter.from_sobol/from_morris/from_local.

    Подробное описание:
        Description/Phase3/description_sensitivity_analysis.md#TornadoData
    """

    parameter_names: list[str] = field(default_factory=list)  # Отсортированы по важности
    values: np.ndarray = field(default_factory=lambda: np.array([]))  # Метрика, (n_params,)
    lower_values: np.ndarray | None = None  # Нижние CI (для error bars)
    upper_values: np.ndarray | None = None  # Верхние CI
    metric_name: str = ""  # "S1", "ST", "mu_star", "elasticity"
    title: str = ""  # Заголовок диаграммы
    source_method: SensitivityMethod | None = None  # Метод-источник
    top_n: int | None = None  # Показать только top N параметров


# =====================================================================
# SensitivityAnalyzer
# =====================================================================


# Description: Description/Phase3/description_sensitivity_analysis.md#SensitivityAnalyzer
class SensitivityAnalyzer:
    """Главный оркестратор анализа чувствительности.

    Управляет тремя вычислительными методами (Sobol, Morris, Local)
    за единым интерфейсом. Принимает модель, параметры и конфигурацию,
    возвращает типизированные результаты.

    Подробное описание:
        Description/Phase3/description_sensitivity_analysis.md#SensitivityAnalyzer
    """

    def __init__(
        self,
        model: ExtendedSDEModel,
        params: ParameterSet,
        config: SensitivityConfig,
    ) -> None:
        """Инициализация анализатора.

        Args:
            model: Экземпляр 20-переменной SDE модели
            params: Номинальный набор параметров (литературные defaults)
            config: Конфигурация анализа чувствительности

        Raises:
            ValueError: Если config не проходит валидацию

        Подробное описание:
            Description/Phase3/description_sensitivity_analysis.md#SensitivityAnalyzer.__init__
        """
        self.model = model
        self.params = params
        self.config = config

        # Автогенерация bounds, если не заданы
        if not config.parameter_bounds:
            config.parameter_bounds = self._auto_bounds()

        config.validate()

    # Description: Description/Phase3/description_sensitivity_analysis.md#run_sobol
    def run_sobol(
        self,
        output_variables: list[str] | None = None,
        n_samples: int = 1024,
    ) -> SobolResult:
        """Глобальный анализ чувствительности методом Sobol.

        Использует SALib: Saltelli sampling → model evaluation → Sobol analyze.
        Вычисляет first-order (S1) и total-effect (ST) индексы.

        Args:
            output_variables: Переменные для анализа (None → из config)
            n_samples: Число базовых сэмплов N (итого N*(2D+2) запусков модели)

        Returns:
            SobolResult с S1, ST, S2, CI

        Raises:
            ImportError: Если SALib не установлен
            ValueError: Если n_samples < 16

        Подробное описание:
            Description/Phase3/description_sensitivity_analysis.md#run_sobol
        """
        if n_samples < 16:
            raise ValueError("n_samples must be >= 16")

        try:
            from SALib.analyze import sobol as sobol_analyze
            from SALib.sample import saltelli as saltelli_sample
        except ImportError:
            raise ImportError("SALib is required for Sobol analysis. " "Install: pip install SALib")

        output_vars = output_variables or self.config.output_variables
        output_var = output_vars[0]
        problem = self._build_salib_problem()
        t_start = time.time()

        param_values = saltelli_sample.sample(problem, n_samples)
        n_runs = len(param_values)
        Y = self._evaluate_model(param_values, output_var)
        Y = np.where(np.isfinite(Y), Y, 0.0)

        n_params = problem["num_vars"]
        if Y.std() == 0:
            logger.warning(
                "All model outputs identical — " "Sobol indices undefined, returning zeros"
            )
            return SobolResult(
                parameter_names=problem["names"],
                S1=np.zeros(n_params),
                ST=np.zeros(n_params),
                S1_conf=np.zeros(n_params),
                ST_conf=np.zeros(n_params),
                output_variable=output_var,
                n_samples=n_samples,
                n_model_runs=n_runs,
                elapsed_seconds=time.time() - t_start,
            )

        si = sobol_analyze.analyze(problem, Y)

        def _nan_to_zero(arr: Any) -> np.ndarray:
            a = np.asarray(arr, dtype=np.float64)
            return np.where(np.isfinite(a), a, 0.0)

        return SobolResult(
            parameter_names=problem["names"],
            S1=_nan_to_zero(si["S1"]),
            ST=_nan_to_zero(si["ST"]),
            S2=_nan_to_zero(si["S2"]) if "S2" in si else None,
            S1_conf=_nan_to_zero(si["S1_conf"]),
            ST_conf=_nan_to_zero(si["ST_conf"]),
            output_variable=output_var,
            n_samples=n_samples,
            n_model_runs=n_runs,
            elapsed_seconds=time.time() - t_start,
        )

    # Description: Description/Phase3/description_sensitivity_analysis.md#run_morris
    def run_morris(
        self,
        output_variables: list[str] | None = None,
        n_trajectories: int = 10,
        n_levels: int = 4,
    ) -> MorrisResult:
        """Скрининг параметров методом Morris (Elementary Effects).

        Использует SALib: Morris sampling → model evaluation → Morris analyze.
        Эффективен для скрининга 40+ параметров: n_trajectories*(D+1) запусков.

        Args:
            output_variables: Переменные для анализа (None → из config)
            n_trajectories: Число Morris траекторий (рек. 10–20)
            n_levels: Число уровней сетки (рек. 4–8)

        Returns:
            MorrisResult с mu, mu_star, sigma

        Raises:
            ImportError: Если SALib не установлен
            ValueError: Если n_trajectories < 2 или n_levels < 2

        Подробное описание:
            Description/Phase3/description_sensitivity_analysis.md#run_morris
        """
        if n_trajectories < 2:
            raise ValueError("n_trajectories must be >= 2")
        if n_levels < 2:
            raise ValueError("n_levels must be >= 2")

        try:
            from SALib.analyze import morris as morris_analyze
            from SALib.sample import morris as morris_sample
        except ImportError:
            raise ImportError(
                "SALib is required for Morris analysis. " "Install: pip install SALib"
            )

        output_vars = output_variables or self.config.output_variables
        output_var = output_vars[0]
        problem = self._build_salib_problem()
        t_start = time.time()

        param_values = morris_sample.sample(problem, N=n_trajectories, num_levels=n_levels)
        n_runs = len(param_values)
        Y = self._evaluate_model(param_values, output_var)
        Y = np.where(np.isfinite(Y), Y, 0.0)

        si = morris_analyze.analyze(problem, param_values, Y, num_levels=n_levels)

        return MorrisResult(
            parameter_names=problem["names"],
            mu=np.asarray(si["mu"], dtype=np.float64),
            mu_star=np.asarray(si["mu_star"], dtype=np.float64),
            sigma=np.asarray(si["sigma"], dtype=np.float64),
            mu_star_conf=np.asarray(si["mu_star_conf"], dtype=np.float64),
            output_variable=output_var,
            n_trajectories=n_trajectories,
            n_levels=n_levels,
            n_model_runs=n_runs,
            elapsed_seconds=time.time() - t_start,
        )

    # Description: Description/Phase3/description_sensitivity_analysis.md#run_local
    def run_local(
        self,
        output_variables: list[str] | None = None,
        delta: float = 0.01,
    ) -> LocalSensitivityResult:
        """Локальный анализ чувствительности методом конечных разностей.

        Центральные конечные разности: dY/dp_i ≈ (Y(p+δ) - Y(p-δ)) / (2δp).
        Безразмерные индексы эластичности: (p_i/Y) * (dY/dp_i).

        Args:
            output_variables: Переменные для анализа (None → из config)
            delta: Относительное возмущение параметра (0 < delta < 1)

        Returns:
            LocalSensitivityResult с partial_derivatives, elasticity_indices

        Raises:
            ValueError: Если delta <= 0 или delta >= 1

        Подробное описание:
            Description/Phase3/description_sensitivity_analysis.md#run_local
        """
        if delta <= 0 or delta >= 1:
            raise ValueError("delta must be in (0, 1)")

        output_vars = output_variables or self.config.output_variables
        output_var = output_vars[0]
        t_start = time.time()

        param_names = [b.name for b in self.config.parameter_bounds]
        base_dict = self.params.to_dict()
        nominal_dict: dict[str, float] = {}
        for b in self.config.parameter_bounds:
            nominal_dict[b.name] = b.nominal if b.nominal is not None else float(base_dict[b.name])

        Y_nom = self._evaluate_model_single(nominal_dict, output_var)

        n_params = len(param_names)
        partial_derivatives = np.zeros(n_params)
        elasticity_indices = np.zeros(n_params)

        for i, name in enumerate(param_names):
            p0 = nominal_dict[name]

            if p0 == 0:
                logger.warning(f"Parameter '{name}' has nominal value 0, skipping")
                continue

            dp = p0 * delta

            params_plus = dict(nominal_dict)
            params_plus[name] = p0 + dp
            Y_plus = self._evaluate_model_single(params_plus, output_var)

            if p0 - dp > 0:
                params_minus = dict(nominal_dict)
                params_minus[name] = p0 - dp
                Y_minus = self._evaluate_model_single(params_minus, output_var)
                partial_derivatives[i] = (Y_plus - Y_minus) / (2 * dp)
            else:
                partial_derivatives[i] = (Y_plus - Y_nom) / dp

            if Y_nom != 0:
                elasticity_indices[i] = (p0 / Y_nom) * partial_derivatives[i]
            else:
                logger.warning(f"Nominal output is 0, elasticity for '{name}' is inf")
                elasticity_indices[i] = float("inf") if partial_derivatives[i] != 0 else 0.0

        return LocalSensitivityResult(
            parameter_names=param_names,
            partial_derivatives=partial_derivatives,
            elasticity_indices=elasticity_indices,
            nominal_output=Y_nom,
            nominal_params=nominal_dict,
            delta=delta,
            output_variable=output_var,
            elapsed_seconds=time.time() - t_start,
        )

    # Description: Description/Phase3/description_sensitivity_analysis.md#_evaluate_model
    def _evaluate_model(
        self,
        param_values: np.ndarray,
        output_variable: str | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        """Запуск модели для массива параметрических сэмплов.

        Для каждой строки param_values подставляет значения в ParameterSet,
        запускает SDE симуляцию и агрегирует выход.

        Args:
            param_values: Матрица сэмплов, shape (n_runs, n_params)
            output_variable: Имя выходной переменной (None → первая из config)
            progress_callback: Функция (current, total) для прогресса

        Returns:
            Массив скалярных выходов, shape (n_runs,)

        Подробное описание:
            Description/Phase3/description_sensitivity_analysis.md#_evaluate_model
        """
        output_var = output_variable or self.config.output_variables[0]
        param_names = [b.name for b in self.config.parameter_bounds]
        n_runs = len(param_values)
        if n_runs == 0:
            return np.array([])
        Y = np.empty(n_runs)
        for i in range(n_runs):
            param_dict = {name: float(param_values[i, j]) for j, name in enumerate(param_names)}
            Y[i] = self._evaluate_model_single(param_dict, output_var)
            if progress_callback is not None:
                progress_callback(i + 1, n_runs)
        return Y

    # Description: Description/Phase3/description_sensitivity_analysis.md#_evaluate_model_single
    def _evaluate_model_single(
        self,
        param_dict: dict[str, float],
        output_variable: str | None = None,
    ) -> float:
        """Запуск модели для одного набора параметров.

        Args:
            param_dict: {имя_параметра: значение}
            output_variable: Имя выходной переменной (None → первая из config)

        Returns:
            Скалярный агрегированный выход модели

        Подробное описание:
            Description/Phase3/description_sensitivity_analysis.md#_evaluate_model_single
        """
        output_var = output_variable or self.config.output_variables[0]
        try:
            d = self.params.to_dict()
            d.update(param_dict)
            modified_params = ParameterSet.from_dict(d)
            model = ExtendedSDEModel(params=modified_params, rng_seed=self.config.rng_seed)
            traj = model.simulate(ExtendedSDEState(), t_span=self.config.t_span)
            values = traj.get_variable(output_var)
            if len(values) == 0:
                return 0.0
            agg = self.config.output_aggregation
            if agg == "final":
                result = float(values[self.config.output_time_index])
            elif agg == "mean":
                result = float(np.mean(values))
            elif agg == "max":
                result = float(np.max(values))
            elif agg == "auc":
                result = float(np.trapezoid(values, traj.times))
            else:
                result = float(values[-1])
            if not np.isfinite(result):
                return 0.0
            return result
        except Exception:
            logger.warning("Model evaluation failed, returning 0.0")
            return 0.0

    # Description: Description/Phase3/description_sensitivity_analysis.md#_build_salib_problem
    def _build_salib_problem(self) -> dict[str, Any]:
        """Построение SALib problem definition из config.parameter_bounds.

        Returns:
            dict с ключами 'num_vars', 'names', 'bounds'

        Подробное описание:
            Description/Phase3/description_sensitivity_analysis.md#_build_salib_problem
        """
        bounds = self.config.parameter_bounds
        return {
            "num_vars": len(bounds),
            "names": [b.name for b in bounds],
            "bounds": [[b.lower, b.upper] for b in bounds],
        }

    # Description: Description/Phase3/description_sensitivity_analysis.md#_auto_bounds
    def _auto_bounds(self) -> list[ParameterBounds]:
        """Автоматическая генерация bounds из ParameterSet (±50% от номинала).

        Returns:
            Список ParameterBounds для всех числовых параметров ParameterSet

        Подробное описание:
            Description/Phase3/description_sensitivity_analysis.md#_auto_bounds
        """
        bounds: list[ParameterBounds] = []
        param_dict = self.params.to_dict()
        for name, value in param_dict.items():
            if isinstance(value, (int, float)) and value > 0:
                bounds.append(
                    ParameterBounds(
                        name=name,
                        lower=value * 0.5,
                        upper=value * 2.0,
                        nominal=value,
                    )
                )
        return bounds


# =====================================================================
# TornadoPlotter
# =====================================================================


# Description: Description/Phase3/description_sensitivity_analysis.md#TornadoPlotter
class TornadoPlotter:
    """Визуализация ранжированной чувствительности (tornado diagram).

    Предоставляет classmethods для конвертации результатов любого метода
    в TornadoData, и метод plot() для построения горизонтальной столбчатой
    диаграммы через matplotlib.

    Подробное описание:
        Description/Phase3/description_sensitivity_analysis.md#TornadoPlotter
    """

    # Description: Description/Phase3/description_sensitivity_analysis.md#TornadoPlotter.from_sobol
    @classmethod
    def from_sobol(
        cls,
        result: SobolResult,
        metric: str = "ST",
        top_n: int | None = 15,
    ) -> TornadoData:
        """Конвертация SobolResult в TornadoData.

        Args:
            result: Результат Sobol анализа
            metric: Метрика для визуализации: "S1" или "ST"
            top_n: Показать только top N параметров (None → все)

        Returns:
            TornadoData отсортированный по убыванию метрики

        Raises:
            ValueError: Если metric не "S1" и не "ST"

        Подробное описание:
            Description/Phase3/description_sensitivity_analysis.md#TornadoPlotter.from_sobol
        """
        if metric not in {"S1", "ST"}:
            raise ValueError(f"metric must be 'S1' or 'ST', got '{metric}'")

        values = result.S1 if metric == "S1" else result.ST
        conf = result.S1_conf if metric == "S1" else result.ST_conf
        indices = np.argsort(-values)

        if top_n is not None:
            indices = indices[:top_n]

        return TornadoData(
            parameter_names=[result.parameter_names[i] for i in indices],
            values=values[indices],
            lower_values=values[indices] - conf[indices] if len(conf) > 0 else None,
            upper_values=values[indices] + conf[indices] if len(conf) > 0 else None,
            metric_name=metric,
            title=f"Sobol {metric} — {result.output_variable}",
            source_method=SensitivityMethod.SOBOL,
            top_n=top_n,
        )

    # Description: Description/Phase3/description_sensitivity_analysis.md#TornadoPlotter.from_morris
    @classmethod
    def from_morris(
        cls,
        result: MorrisResult,
        top_n: int | None = 15,
    ) -> TornadoData:
        """Конвертация MorrisResult в TornadoData.

        Args:
            result: Результат Morris скрининга
            top_n: Показать только top N параметров (None → все)

        Returns:
            TornadoData отсортированный по убыванию mu_star

        Подробное описание:
            Description/Phase3/description_sensitivity_analysis.md#TornadoPlotter.from_morris
        """
        indices = np.argsort(-result.mu_star)

        if top_n is not None:
            indices = indices[:top_n]

        return TornadoData(
            parameter_names=[result.parameter_names[i] for i in indices],
            values=result.mu_star[indices],
            lower_values=(
                result.mu_star[indices] - result.mu_star_conf[indices]
                if len(result.mu_star_conf) > 0
                else None
            ),
            upper_values=(
                result.mu_star[indices] + result.mu_star_conf[indices]
                if len(result.mu_star_conf) > 0
                else None
            ),
            metric_name="mu_star",
            title=f"Morris μ* — {result.output_variable}",
            source_method=SensitivityMethod.MORRIS,
            top_n=top_n,
        )

    # Description: Description/Phase3/description_sensitivity_analysis.md#TornadoPlotter.from_local
    @classmethod
    def from_local(
        cls,
        result: LocalSensitivityResult,
        top_n: int | None = 15,
    ) -> TornadoData:
        """Конвертация LocalSensitivityResult в TornadoData.

        Args:
            result: Результат локальной чувствительности
            top_n: Показать только top N параметров (None → все)

        Returns:
            TornadoData отсортированный по убыванию |elasticity|

        Подробное описание:
            Description/Phase3/description_sensitivity_analysis.md#TornadoPlotter.from_local
        """
        indices = np.argsort(-np.abs(result.elasticity_indices))

        if top_n is not None:
            indices = indices[:top_n]

        return TornadoData(
            parameter_names=[result.parameter_names[i] for i in indices],
            values=result.elasticity_indices[indices],
            metric_name="elasticity",
            title=f"Local Elasticity — {result.output_variable}",
            source_method=SensitivityMethod.LOCAL,
            top_n=top_n,
        )

    # Description: Description/Phase3/description_sensitivity_analysis.md#TornadoPlotter.plot
    def plot(
        self,
        data: TornadoData,
        output_path: str | None = None,
    ) -> Any:
        """Построение tornado diagram (горизонтальная столбчатая диаграмма).

        Визуализирует ранжированную чувствительность: параметры отсортированы
        по убыванию значения метрики, опционально с error bars.

        Args:
            data: Подготовленные данные для визуализации
            output_path: Путь для сохранения PNG (None → не сохранять)

        Returns:
            matplotlib.figure.Figure

        Raises:
            ImportError: Если matplotlib не установлен

        Подробное описание:
            Description/Phase3/description_sensitivity_analysis.md#TornadoPlotter.plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. " "Install: pip install matplotlib"
            )

        if len(data.parameter_names) == 0:
            fig, ax = plt.subplots()
            ax.set_title(data.title or "Sensitivity Analysis (no data)")
            if output_path:
                fig.savefig(output_path, dpi=150, bbox_inches="tight")
            return fig

        n = len(data.parameter_names)
        fig, ax = plt.subplots(figsize=(10, max(6, n * 0.4)))
        y_positions = np.arange(n)

        xerr = None
        if data.lower_values is not None and data.upper_values is not None:
            xerr = [
                data.values - data.lower_values,
                data.upper_values - data.values,
            ]

        ax.barh(
            y_positions,
            data.values,
            xerr=xerr,
            align="center",
            color="steelblue",
            edgecolor="black",
            linewidth=0.5,
            capsize=3,
        )
        ax.set_yticks(y_positions)
        ax.set_yticklabels(data.parameter_names)
        ax.set_xlabel(data.metric_name)
        ax.set_title(data.title)
        ax.invert_yaxis()
        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")

        return fig


# =====================================================================
# Convenience function
# =====================================================================


# Description: Description/Phase3/description_sensitivity_analysis.md#run_sensitivity_analysis
def run_sensitivity_analysis(
    method: str | SensitivityMethod = "sobol",
    params: ParameterSet | None = None,
    parameter_names: list[str] | None = None,
    output_variables: list[str] | None = None,
    n_samples: int = 1024,
    config: SensitivityConfig | None = None,
) -> SobolResult | MorrisResult | LocalSensitivityResult:
    """Convenience-функция для запуска анализа чувствительности.

    Создаёт все необходимые объекты (модель, конфигурацию, анализатор)
    и запускает анализ выбранным методом. Аналог estimate_parameters()
    из parameter_estimation.py.

    Args:
        method: Метод анализа: "sobol", "morris", "local" или SensitivityMethod
        params: Набор параметров (None → ParameterSet с defaults)
        parameter_names: Имена параметров для анализа (None → все)
        output_variables: Выходные переменные (None → ["F"])
        n_samples: Число сэмплов (для Sobol — N, для Morris — n_trajectories)
        config: Готовая конфигурация (перекрывает остальные аргументы)

    Returns:
        SobolResult | MorrisResult | LocalSensitivityResult

    Raises:
        ValueError: Если method неизвестен

    Подробное описание:
        Description/Phase3/description_sensitivity_analysis.md#run_sensitivity_analysis
    """
    params = params or ParameterSet()
    model = ExtendedSDEModel(params=params)

    method_enum = SensitivityMethod(method) if isinstance(method, str) else method

    if config is None:
        bounds: list[ParameterBounds] = []
        if parameter_names:
            param_dict = params.to_dict()
            for name in parameter_names:
                val = param_dict.get(name)
                if val is not None and isinstance(val, (int, float)) and val > 0:
                    bounds.append(
                        ParameterBounds(
                            name=name,
                            lower=val * 0.5,
                            upper=val * 2.0,
                            nominal=float(val),
                        )
                    )
        config = SensitivityConfig(
            method=method_enum,
            parameter_bounds=bounds,
            output_variables=output_variables or ["F"],
        )

    analyzer = SensitivityAnalyzer(model, params, config)

    if method_enum == SensitivityMethod.SOBOL:
        return analyzer.run_sobol(output_variables=output_variables, n_samples=n_samples)
    elif method_enum == SensitivityMethod.MORRIS:
        return analyzer.run_morris(output_variables=output_variables, n_trajectories=n_samples)
    elif method_enum == SensitivityMethod.LOCAL:
        return analyzer.run_local(output_variables=output_variables)
    else:
        raise ValueError(f"Unknown method: {method}")
