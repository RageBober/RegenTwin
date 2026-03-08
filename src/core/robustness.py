"""Верификация робастности численных методов для SDE/ABM моделей.

Обеспечивает high-level проверки:
- Позитивность (физичность) переменных с накоплением статистики
- Обнаружение и восстановление после NaN/Inf
- Проверка законов сохранения (масса, цитокины, ECM)
- Верификация порядка сходимости (Method of Manufactured Solutions)
- Сравнение SDE vs ABM при большом N (Закон больших чисел)

Отличие от numerical_utils.py: данный модуль — high-level верификация,
а numerical_utils — low-level утилиты (clip, detect NaN, adaptive dt).

Математическое обоснование: Doks/RegenTwin_Mathematical_Framework.md §2

Подробное описание: Description/Phase2/description_robustness.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

try:
    from loguru import logger
except ImportError:
    import logging as _logging

    logger = _logging.getLogger(__name__)  # type: ignore[assignment]

if TYPE_CHECKING:
    from src.core.sde_numerics import SDESolver


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ViolationStats:
    """Статистика нарушений позитивности за симуляцию.

    Накапливает информацию о том, сколько раз и какие переменные
    стали отрицательными (нефизичными) в процессе интегрирования.

    Подробное описание: Description/Phase2/description_robustness.md#ViolationStats
    """

    count: int = 0  # Общее число нарушений
    variables: dict[str, int] = field(  # {имя_перем: число_нарушений}
        default_factory=dict,
    )
    timestamps: list[float] = field(  # Моменты времени нарушений
        default_factory=list,
    )
    total_clipped: float = 0.0  # Суммарная величина отсечения


@dataclass
class ConservationReport:
    """Отчёт о проверке законов сохранения.

    Содержит ошибки баланса массы, цитокинов и ECM.
    is_conserved = True если все ошибки в пределах допуска.

    Подробное описание: Description/Phase2/description_robustness.md#ConservationReport
    """

    mass_error: float = 0.0  # Относительная ошибка баланса клеток
    cytokine_error: float = 0.0  # Относительная ошибка баланса цитокинов
    ecm_error: float = 0.0  # Относительная ошибка баланса ECM
    is_conserved: bool = True  # Все ошибки в допуске
    tolerance: float = 0.05  # Допуск (5% по умолчанию)
    details: str = ""  # Текстовая диагностика


@dataclass
class ConvergenceResult:
    """Результат верификации порядка сходимости.

    Содержит оценённый порядок, последовательность ошибок
    и шагов dt для log-log анализа.

    Подробное описание: Description/Phase2/description_robustness.md#ConvergenceResult
    """

    estimated_order: float = 0.0  # Оценённый порядок сходимости
    errors: list[float] = field(  # Ошибки для каждого dt
        default_factory=list,
    )
    dt_sequence: list[float] = field(  # Последовательность dt
        default_factory=list,
    )
    reference_order: float = 0.0  # Теоретический порядок
    is_valid: bool = False  # Оценка ≈ reference ± 0.2


@dataclass
class ComparisonMetrics:
    """Метрики сравнения SDE и ABM траекторий.

    Для оценки согласия при большом числе агентов (ЗБЧ).

    Подробное описание: Description/Phase2/description_robustness.md#ComparisonMetrics
    """

    wasserstein_distance: float = 0.0  # W1 расстояние
    mean_diff: float = 0.0  # |mean_SDE - mean_ABM|
    std_diff: float = 0.0  # |std_SDE - std_ABM|
    ks_statistic: float = 0.0  # Kolmogorov-Smirnov статистика
    ks_pvalue: float = 0.0  # KS p-value
    is_consistent: bool = False  # True если p > 0.05


# ---------------------------------------------------------------------------
# PositivityEnforcer
# ---------------------------------------------------------------------------


class PositivityEnforcer:
    """Контроль позитивности с накоплением статистики нарушений.

    В отличие от clip_negative_concentrations() из numerical_utils,
    данный класс сохраняет историю нарушений (какие переменные, когда,
    суммарная величина) для диагностики качества интегрирования.

    Подробное описание: Description/Phase2/description_robustness.md#PositivityEnforcer
    """

    def __init__(
        self,
        variable_names: list[str] | None = None,
        min_value: float = 0.0,
    ) -> None:
        """Инициализация с набором контролируемых переменных.

        Args:
            variable_names: Имена переменных для контроля (None = все 20)
            min_value: Минимальное допустимое значение

        Подробное описание:
            Description/Phase2/description_robustness.md#PositivityEnforcer.__init__
        """
        self._variable_names = variable_names
        self._min_value = min_value
        self._stats = ViolationStats()

    def enforce(
        self,
        state: np.ndarray,
        t: float = 0.0,
        variable_names: list[str] | None = None,
    ) -> np.ndarray:
        """Отсечение отрицательных значений + обновление статистики.

        Args:
            state: Массив состояния shape (20,)
            t: Текущее время (для логирования)
            variable_names: Имена переменных (для статистики)

        Returns:
            Скорректированный массив (новая копия)

        Подробное описание:
            Description/Phase2/description_robustness.md#PositivityEnforcer.enforce
        """
        result = state.copy()
        names = variable_names or self._variable_names
        violations_in_call = 0

        for i in range(len(result)):
            if np.isnan(result[i]):
                continue
            if result[i] < self._min_value:
                clipped_amount = abs(result[i] - self._min_value)
                self._stats.count += 1
                self._stats.total_clipped += clipped_amount
                violations_in_call += 1

                if names and i < len(names):
                    name = names[i]
                    self._stats.variables[name] = self._stats.variables.get(name, 0) + 1

                result[i] = self._min_value

        if violations_in_call > 0:
            self._stats.timestamps.append(t)

        return result

    def get_violation_stats(self) -> ViolationStats:
        """Получить накопленную статистику нарушений.

        Returns:
            Копия ViolationStats

        Подробное описание:
            Description/Phase2/description_robustness.md#get_violation_stats
        """
        return self._stats

    def reset_stats(self) -> None:
        """Сбросить статистику нарушений.

        Подробное описание:
            Description/Phase2/description_robustness.md#reset_stats
        """
        self._stats = ViolationStats()


# ---------------------------------------------------------------------------
# NaNHandler
# ---------------------------------------------------------------------------


class NaNHandler:
    """Обнаружение NaN/Inf и стратегия восстановления.

    High-level обёртка: проверка → логирование → восстановление
    (откат к last_valid + уменьшение dt). Счётчик восстановлений
    для мониторинга стабильности.

    Подробное описание: Description/Phase2/description_robustness.md#NaNHandler
    """

    def __init__(
        self,
        max_recoveries: int = 10,
        dt_reduction_factor: float = 0.5,
    ) -> None:
        """Инициализация NaN-обработчика.

        Args:
            max_recoveries: Максимум восстановлений до остановки
            dt_reduction_factor: Множитель уменьшения dt при восстановлении

        Подробное описание:
            Description/Phase2/description_robustness.md#NaNHandler.__init__
        """
        self._max_recoveries = max_recoveries
        self._dt_reduction_factor = dt_reduction_factor
        self._recovery_count: int = 0

    def check(self, state: np.ndarray) -> bool:
        """Проверить наличие NaN/Inf в состоянии.

        Args:
            state: Массив состояния shape (20,)

        Returns:
            True если обнаружены NaN или Inf

        Подробное описание:
            Description/Phase2/description_robustness.md#NaNHandler.check
        """
        return bool(not np.all(np.isfinite(state)))

    def recover(
        self,
        state: np.ndarray,  # noqa: ARG002
        last_valid_state: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, float, bool]:
        """Восстановление после NaN/Inf.

        Стратегия: откат к last_valid_state + dt *= reduction_factor.
        Если recovery_count >= max_recoveries → should_stop=True.

        Args:
            state: Текущее (повреждённое) состояние
            last_valid_state: Последнее валидное состояние
            dt: Текущий шаг времени

        Returns:
            (recovered_state, new_dt, should_stop)

        Подробное описание:
            Description/Phase2/description_robustness.md#NaNHandler.recover
        """
        self._recovery_count += 1
        new_dt = dt * self._dt_reduction_factor
        should_stop = self._recovery_count >= self._max_recoveries
        return last_valid_state.copy(), new_dt, should_stop

    def get_recovery_count(self) -> int:
        """Получить число выполненных восстановлений.

        Returns:
            Счётчик восстановлений

        Подробное описание:
            Description/Phase2/description_robustness.md#get_recovery_count
        """
        return self._recovery_count

    def reset(self) -> None:
        """Сбросить счётчик восстановлений.

        Подробное описание:
            Description/Phase2/description_robustness.md#NaNHandler.reset
        """
        self._recovery_count = 0


# ---------------------------------------------------------------------------
# ConservationChecker
# ---------------------------------------------------------------------------


class ConservationChecker:
    """Проверка законов сохранения для биологической модели.

    Проверяет баланс:
    - Клеточных популяций (births - deaths ≈ ΔN)
    - Цитокинов (production - degradation ≈ ΔC)
    - ECM (synthesis - degradation ≈ Δρ)

    Допуск: 1–5% относительной ошибки (биологические модели неточны).

    Подробное описание: Description/Phase2/description_robustness.md#ConservationChecker
    """

    def __init__(self, tolerance: float = 0.05) -> None:
        """Инициализация с допуском.

        Args:
            tolerance: Допустимая относительная ошибка (default 5%)

        Подробное описание:
            Description/Phase2/description_robustness.md#ConservationChecker.__init__
        """
        self._tolerance = tolerance
        self._reports: list[ConservationReport] = []

    def check_mass_balance(
        self,
        births: np.ndarray,
        deaths: np.ndarray,
        population_current: np.ndarray,
        population_previous: np.ndarray,
        dt: float,
    ) -> ConservationReport:
        """Проверка баланса клеточных популяций.

        ΔN ≈ (births - deaths) · dt
        error = |ΔN_actual - ΔN_expected| / max(|N|, ε)

        Args:
            births: Скорости рождения по популяциям, shape (8,)
            deaths: Скорости гибели по популяциям, shape (8,)
            population_current: Текущие популяции, shape (8,)
            population_previous: Предыдущие популяции, shape (8,)
            dt: Шаг времени

        Returns:
            ConservationReport с mass_error

        Подробное описание:
            Description/Phase2/description_robustness.md#check_mass_balance
        """
        delta_expected = (births - deaths) * dt
        delta_actual = population_current - population_previous
        denominator = np.maximum(np.abs(population_previous), 1e-10)
        error = float(np.linalg.norm((delta_actual - delta_expected) / denominator))

        is_conserved = error <= self._tolerance
        report = ConservationReport(
            mass_error=error,
            is_conserved=is_conserved,
            tolerance=self._tolerance,
        )
        self._reports.append(report)
        return report

    def check_cytokine_balance(
        self,
        production: np.ndarray,
        degradation: np.ndarray,
        concentration_current: np.ndarray,
        concentration_previous: np.ndarray,
        dt: float,
    ) -> ConservationReport:
        """Проверка баланса цитокинов.

        ΔC ≈ (production - degradation) · dt
        error = |ΔC_actual - ΔC_expected| / max(|C|, ε)

        Args:
            production: Скорости продукции, shape (7,)
            degradation: Скорости деградации, shape (7,)
            concentration_current: Текущие концентрации, shape (7,)
            concentration_previous: Предыдущие концентрации, shape (7,)
            dt: Шаг времени

        Returns:
            ConservationReport с cytokine_error

        Подробное описание:
            Description/Phase2/description_robustness.md#check_cytokine_balance
        """
        delta_expected = (production - degradation) * dt
        delta_actual = concentration_current - concentration_previous
        denominator = np.maximum(np.abs(concentration_previous), 1e-10)
        error = float(np.linalg.norm((delta_actual - delta_expected) / denominator))

        is_conserved = error <= self._tolerance
        report = ConservationReport(
            cytokine_error=error,
            is_conserved=is_conserved,
            tolerance=self._tolerance,
        )
        self._reports.append(report)
        return report

    def report(self) -> list[ConservationReport]:
        """Получить все накопленные отчёты.

        Returns:
            Список ConservationReport

        Подробное описание:
            Description/Phase2/description_robustness.md#ConservationChecker.report
        """
        return list(self._reports)

    def reset(self) -> None:
        """Сбросить накопленные отчёты.

        Подробное описание:
            Description/Phase2/description_robustness.md#ConservationChecker.reset
        """
        self._reports = []


# ---------------------------------------------------------------------------
# ConvergenceVerifier
# ---------------------------------------------------------------------------


class ConvergenceVerifier:
    """Верификация порядка сходимости методом Method of Manufactured Solutions.

    Алгоритм:
    1. Создать тестовую задачу с известным аналитическим решением
    2. Запустить солвер с последовательностью dt: dt, dt/2, dt/4, dt/8
    3. Вычислить strong error E(dt) = E[||X_num - X_exact||]
    4. Оценить порядок p из log-log регрессии: log(E) ≈ p·log(dt) + C
    5. Сравнить с теоретическим порядком

    Подробное описание: Description/Phase2/description_robustness.md#ConvergenceVerifier
    """

    def __init__(self, n_realizations: int = 100) -> None:
        """Инициализация верификатора.

        Args:
            n_realizations: Число реализаций для Monte Carlo оценки strong error

        Подробное описание:
            Description/Phase2/description_robustness.md#ConvergenceVerifier.__init__
        """
        self._n_realizations = n_realizations

    def compute_order(
        self,
        errors: list[float],
        dt_sequence: list[float],
    ) -> float:
        """Оценка порядка сходимости из log-log регрессии.

        p = slope of linear fit: log(errors) vs log(dt_sequence)

        Args:
            errors: Strong errors для каждого dt
            dt_sequence: Последовательность шагов времени

        Returns:
            Оценённый порядок сходимости p

        Подробное описание:
            Description/Phase2/description_robustness.md#compute_order
        """
        log_errors = np.log(errors)
        log_dt = np.log(dt_sequence)
        coeffs = np.polyfit(log_dt, log_errors, 1)
        return float(coeffs[0])

    def verify_solver(
        self,
        solver: SDESolver,
        reference_order: float,
        dt_base: float = 0.01,
        n_refinements: int = 4,
    ) -> ConvergenceResult:
        """Полная верификация солвера с MMS.

        Запускает тестовую задачу (geometric Brownian motion) с
        последовательностью dt, вычисляет strong errors, оценивает порядок.

        Args:
            solver: Солвер для верификации
            reference_order: Ожидаемый порядок (0.5 для EM, 1.0 для Milstein)
            dt_base: Начальный шаг
            n_refinements: Число делений dt (4 → dt, dt/2, dt/4, dt/8)

        Returns:
            ConvergenceResult с estimated_order и is_valid

        Подробное описание:
            Description/Phase2/description_robustness.md#verify_solver
        """
        raise NotImplementedError("Stub: требуется реализация в Этап 3")

    def manufactured_solution(
        self,
        t: float,
        x0: float = 1.0,
        mu: float = 0.05,
        sigma: float = 0.2,
    ) -> float:
        """Аналитическое решение geometric Brownian motion (тестовая задача).

        dX = μ·X·dt + σ·X·dW
        X(t) = x0 · exp((μ - σ²/2)·t + σ·W(t))

        Args:
            t: Время
            x0: Начальное значение
            mu: Коэффициент дрифта
            sigma: Коэффициент диффузии

        Returns:
            Детерминированная часть решения (без W(t))

        Подробное описание:
            Description/Phase2/description_robustness.md#manufactured_solution
        """
        return float(x0 * np.exp((mu - sigma**2 / 2) * t))


# ---------------------------------------------------------------------------
# SDEvsABMComparator
# ---------------------------------------------------------------------------


class SDEvsABMComparator:
    """Сравнение SDE и ABM траекторий для верификации согласованности.

    При большом числе агентов N → ∞ ABM должна сходиться к SDE
    (Закон больших чисел). Данный класс вычисляет метрики
    расхождения: Wasserstein distance, KS-тест, разность средних.

    Подробное описание: Description/Phase2/description_robustness.md#SDEvsABMComparator
    """

    def __init__(self, significance_level: float = 0.05) -> None:
        """Инициализация компаратора.

        Args:
            significance_level: Уровень значимости KS-теста

        Подробное описание:
            Description/Phase2/description_robustness.md#SDEvsABMComparator.__init__
        """
        self._significance_level = significance_level

    def compare(
        self,
        sde_values: np.ndarray,
        abm_values: np.ndarray,
    ) -> ComparisonMetrics:
        """Сравнение выборок SDE и ABM.

        Вычисляет Wasserstein distance, KS-тест, разность средних и std.

        Args:
            sde_values: Значения из SDE симуляций, shape (n_sde,)
            abm_values: Значения из ABM симуляций, shape (n_abm,)

        Returns:
            ComparisonMetrics со всеми метриками

        Подробное описание:
            Description/Phase2/description_robustness.md#SDEvsABMComparator.compare
        """
        if len(sde_values) == 0 or len(abm_values) == 0:
            msg = "Массивы не могут быть пустыми"
            raise ValueError(msg)

        from scipy.stats import ks_2samp
        from scipy.stats import wasserstein_distance as _wasserstein

        w1 = float(_wasserstein(sde_values, abm_values))
        ks_stat, ks_p = ks_2samp(sde_values, abm_values)
        mean_diff = float(abs(np.mean(sde_values) - np.mean(abm_values)))
        std_diff = float(abs(np.std(sde_values) - np.std(abm_values)))
        is_consistent = float(ks_p) >= self._significance_level

        return ComparisonMetrics(
            wasserstein_distance=w1,
            mean_diff=mean_diff,
            std_diff=std_diff,
            ks_statistic=float(ks_stat),
            ks_pvalue=float(ks_p),
            is_consistent=is_consistent,
        )

    def wasserstein_distance(
        self,
        sde_values: np.ndarray,
        abm_values: np.ndarray,
    ) -> float:
        """Wasserstein-1 (Earth Mover's) расстояние между распределениями.

        Args:
            sde_values: Выборка SDE, shape (n,)
            abm_values: Выборка ABM, shape (m,)

        Returns:
            W1 расстояние (скаляр ≥ 0)

        Подробное описание:
            Description/Phase2/description_robustness.md#wasserstein_distance
        """
        from scipy.stats import wasserstein_distance as _wasserstein

        return float(_wasserstein(sde_values, abm_values))

    def summary(
        self,
        metrics: ComparisonMetrics,
    ) -> str:
        """Текстовый отчёт о сравнении.

        Args:
            metrics: Результат compare()

        Returns:
            Многострочная строка с диагностикой

        Подробное описание:
            Description/Phase2/description_robustness.md#SDEvsABMComparator.summary
        """
        lines = [
            "SDE vs ABM Comparison Report",
            f"  Wasserstein distance: {metrics.wasserstein_distance:.6f}",
            f"  Mean difference: {metrics.mean_diff:.6f}",
            f"  Std difference: {metrics.std_diff:.6f}",
            f"  KS statistic: {metrics.ks_statistic:.6f}",
            f"  KS p-value: {metrics.ks_pvalue:.6f}",
            f"  Consistent: {metrics.is_consistent}",
        ]
        return "\n".join(lines)
