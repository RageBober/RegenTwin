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

    count: int = 0                                   # Общее число нарушений
    variables: dict[str, int] = field(               # {имя_перем: число_нарушений}
        default_factory=dict,
    )
    timestamps: list[float] = field(                 # Моменты времени нарушений
        default_factory=list,
    )
    total_clipped: float = 0.0                       # Суммарная величина отсечения


@dataclass
class ConservationReport:
    """Отчёт о проверке законов сохранения.

    Содержит ошибки баланса массы, цитокинов и ECM.
    is_conserved = True если все ошибки в пределах допуска.

    Подробное описание: Description/Phase2/description_robustness.md#ConservationReport
    """

    mass_error: float = 0.0          # Относительная ошибка баланса клеток
    cytokine_error: float = 0.0      # Относительная ошибка баланса цитокинов
    ecm_error: float = 0.0           # Относительная ошибка баланса ECM
    is_conserved: bool = True         # Все ошибки в допуске
    tolerance: float = 0.05           # Допуск (5% по умолчанию)
    details: str = ""                 # Текстовая диагностика


@dataclass
class ConvergenceResult:
    """Результат верификации порядка сходимости.

    Содержит оценённый порядок, последовательность ошибок
    и шагов dt для log-log анализа.

    Подробное описание: Description/Phase2/description_robustness.md#ConvergenceResult
    """

    estimated_order: float = 0.0                 # Оценённый порядок сходимости
    errors: list[float] = field(                 # Ошибки для каждого dt
        default_factory=list,
    )
    dt_sequence: list[float] = field(            # Последовательность dt
        default_factory=list,
    )
    reference_order: float = 0.0                 # Теоретический порядок
    is_valid: bool = False                       # Оценка ≈ reference ± 0.2


@dataclass
class ComparisonMetrics:
    """Метрики сравнения SDE и ABM траекторий.

    Для оценки согласия при большом числе агентов (ЗБЧ).

    Подробное описание: Description/Phase2/description_robustness.md#ComparisonMetrics
    """

    wasserstein_distance: float = 0.0    # W1 расстояние
    mean_diff: float = 0.0               # |mean_SDE - mean_ABM|
    std_diff: float = 0.0                # |std_SDE - std_ABM|
    ks_statistic: float = 0.0            # Kolmogorov-Smirnov статистика
    ks_pvalue: float = 0.0               # KS p-value
    is_consistent: bool = False          # True если p > 0.05


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
        raise NotImplementedError("Stub: требуется реализация в Этап 3")

    def get_violation_stats(self) -> ViolationStats:
        """Получить накопленную статистику нарушений.

        Returns:
            Копия ViolationStats

        Подробное описание:
            Description/Phase2/description_robustness.md#get_violation_stats
        """
        raise NotImplementedError("Stub: требуется реализация в Этап 3")

    def reset_stats(self) -> None:
        """Сбросить статистику нарушений.

        Подробное описание:
            Description/Phase2/description_robustness.md#reset_stats
        """
        raise NotImplementedError("Stub: требуется реализация в Этап 3")


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
        raise NotImplementedError("Stub: требуется реализация в Этап 3")

    def recover(
        self,
        state: np.ndarray,
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
        raise NotImplementedError("Stub: требуется реализация в Этап 3")

    def get_recovery_count(self) -> int:
        """Получить число выполненных восстановлений.

        Returns:
            Счётчик восстановлений

        Подробное описание:
            Description/Phase2/description_robustness.md#get_recovery_count
        """
        raise NotImplementedError("Stub: требуется реализация в Этап 3")

    def reset(self) -> None:
        """Сбросить счётчик восстановлений.

        Подробное описание:
            Description/Phase2/description_robustness.md#NaNHandler.reset
        """
        raise NotImplementedError("Stub: требуется реализация в Этап 3")


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
        raise NotImplementedError("Stub: требуется реализация в Этап 3")

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
        raise NotImplementedError("Stub: требуется реализация в Этап 3")

    def report(self) -> list[ConservationReport]:
        """Получить все накопленные отчёты.

        Returns:
            Список ConservationReport

        Подробное описание:
            Description/Phase2/description_robustness.md#ConservationChecker.report
        """
        raise NotImplementedError("Stub: требуется реализация в Этап 3")

    def reset(self) -> None:
        """Сбросить накопленные отчёты.

        Подробное описание:
            Description/Phase2/description_robustness.md#ConservationChecker.reset
        """
        raise NotImplementedError("Stub: требуется реализация в Этап 3")


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
        raise NotImplementedError("Stub: требуется реализация в Этап 3")

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
        raise NotImplementedError("Stub: требуется реализация в Этап 3")


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
        raise NotImplementedError("Stub: требуется реализация в Этап 3")

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
        raise NotImplementedError("Stub: требуется реализация в Этап 3")

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
        raise NotImplementedError("Stub: требуется реализация в Этап 3")
