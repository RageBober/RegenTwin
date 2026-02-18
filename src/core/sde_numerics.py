"""Численные методы для стохастических дифференциальных уравнений.

Реализует продвинутые солверы для 20-переменной SDE системы регенерации:
- Euler-Maruyama (strong order 0.5) — базовый метод
- Milstein (strong order 1.0) — учитывает производную диффузии
- IMEX splitting — для стиффных систем (быстрые цитокины + медленный ECM)
- Адаптивный контроль шага с PI-контроллером
- Stochastic Runge-Kutta SRI2W1 (strong order 1.0) — мультимерный шум

Архитектура: Strategy pattern — каждый солвер реализует протокол SDESolver.
Математическое обоснование: Kloeden & Platen (1992), Doks/RegenTwin_Mathematical_Framework.md §2

Подробное описание: Description/Phase2/description_sde_numerics.md
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

try:
    from loguru import logger
except ImportError:
    import logging as _logging

    logger = _logging.getLogger(__name__)  # type: ignore[assignment]

if TYPE_CHECKING:
    from src.core.extended_sde import (
        ExtendedSDEModel,
        ExtendedSDEState,
        ExtendedSDETrajectory,
    )
    from src.core.parameters import ParameterSet


# ---------------------------------------------------------------------------
# Enums & Config
# ---------------------------------------------------------------------------


class SolverType(Enum):
    """Тип численного солвера SDE.

    Подробное описание: Description/Phase2/description_sde_numerics.md#SolverType
    """

    EM = "euler_maruyama"          # Strong order 0.5
    MILSTEIN = "milstein"          # Strong order 1.0
    IMEX = "imex"                  # Implicit-Explicit splitting
    SRK = "stochastic_rk"         # SRI2W1
    ADAPTIVE = "adaptive"         # Адаптивный шаг (обёртка)


@dataclass
class SolverConfig:
    """Конфигурация численного солвера.

    Управляет параметрами интегрирования: шаг времени, допуски,
    выбор метода. Используется всеми солверами.

    Подробное описание: Description/Phase2/description_sde_numerics.md#SolverConfig
    """

    solver_type: SolverType = SolverType.EM  # Тип солвера
    dt: float = 0.01                          # Базовый шаг времени (ч)
    dt_min: float = 1e-6                      # Минимальный шаг
    dt_max: float = 1.0                       # Максимальный шаг
    tolerance: float = 1e-3                   # Допуск ошибки (для адаптивного)
    max_steps: int = 100_000                  # Максимум шагов
    safety_factor: float = 0.9                # Запас прочности PI-контроллера
    fd_epsilon: float = 1e-6                  # Epsilon для конечных разностей (Milstein)


@dataclass
class StepResult:
    """Результат одного шага интегрирования.

    Содержит новое состояние, использованный dt, оценку ошибки
    и количество вызовов drift/diffusion.

    Подробное описание: Description/Phase2/description_sde_numerics.md#StepResult
    """

    new_state: np.ndarray                    # Массив состояния shape (20,)
    dt_used: float = 0.0                     # Фактически использованный шаг
    error_estimate: float = 0.0              # Оценка локальной ошибки
    n_function_evals: int = 0                # Число вызовов drift/diffusion
    rejected: bool = False                   # Шаг был отклонён (adaptive)


# ---------------------------------------------------------------------------
# Протокол SDESolver
# ---------------------------------------------------------------------------


@runtime_checkable
class SDESolver(Protocol):
    """Протокол для всех SDE солверов.

    Каждый солвер должен реализовать step() для одного шага
    и simulate() для полной траектории. Совместим с ExtendedSDEModel.

    Подробное описание: Description/Phase2/description_sde_numerics.md#SDESolver
    """

    def step(
        self,
        state: np.ndarray,
        drift: np.ndarray,
        diffusion: np.ndarray,
        dt: float,
        dW: np.ndarray,
    ) -> StepResult:
        """Один шаг интегрирования SDE.

        Вычисляет X_{n+1} из текущего X_n с данными drift, diffusion, dW.

        Args:
            state: Текущее состояние X_n, shape (20,)
            drift: Вектор дрифта μ(X_n), shape (20,)
            diffusion: Вектор диффузии σ(X_n), shape (20,)
            dt: Шаг времени
            dW: Винеровские приращения, shape (20,)

        Returns:
            StepResult с новым состоянием X_{n+1}

        Подробное описание:
            Description/Phase2/description_sde_numerics.md#SDESolver.step
        """
        ...

    def simulate(
        self,
        model: ExtendedSDEModel,
        initial_state: ExtendedSDEState,
        params: ParameterSet,
    ) -> ExtendedSDETrajectory:
        """Полная симуляция SDE от t=0 до t=t_max.

        Args:
            model: Модель с _compute_drift() и _compute_diffusion()
            initial_state: Начальное состояние системы
            params: Набор параметров с dt, t_max

        Returns:
            Траектория с историей всех состояний

        Подробное описание:
            Description/Phase2/description_sde_numerics.md#SDESolver.simulate
        """
        ...


# ---------------------------------------------------------------------------
# Euler-Maruyama
# ---------------------------------------------------------------------------


class EulerMaruyamaSolver:
    """Солвер Эйлера-Маруямы (strong order 0.5).

    X_{n+1} = X_n + μ(X_n)·dt + σ(X_n)·dW_n

    Базовый метод, совместимый с текущей реализацией в ExtendedSDEModel.
    Рефакторинг для Strategy pattern.

    Подробное описание: Description/Phase2/description_sde_numerics.md#EulerMaruyamaSolver
    """

    def __init__(self, config: SolverConfig | None = None) -> None:
        """Инициализация солвера EM.

        Args:
            config: Конфигурация (None = дефолт)

        Подробное описание:
            Description/Phase2/description_sde_numerics.md#EulerMaruyamaSolver.__init__
        """
        self._config = config or SolverConfig(solver_type=SolverType.EM)

    def step(
        self,
        state: np.ndarray,
        drift: np.ndarray,
        diffusion: np.ndarray,
        dt: float,
        dW: np.ndarray,
    ) -> StepResult:
        """Один шаг Эйлера-Маруямы: X += μ·dt + σ·dW.

        Args:
            state: X_n, shape (20,)
            drift: μ(X_n), shape (20,)
            diffusion: σ(X_n), shape (20,)
            dt: Шаг времени
            dW: Винеровские приращения, shape (20,)

        Returns:
            StepResult с X_{n+1}

        Подробное описание:
            Description/Phase2/description_sde_numerics.md#EulerMaruyamaSolver.step
        """
        raise NotImplementedError("Stub: требуется реализация в Этап 3")

    def simulate(
        self,
        model: ExtendedSDEModel,
        initial_state: ExtendedSDEState,
        params: ParameterSet,
    ) -> ExtendedSDETrajectory:
        """Полная симуляция методом EM.

        Args:
            model: Расширенная SDE модель
            initial_state: Начальное состояние
            params: Параметры (dt, t_max)

        Returns:
            Траектория симуляции

        Подробное описание:
            Description/Phase2/description_sde_numerics.md#EulerMaruyamaSolver.simulate
        """
        raise NotImplementedError("Stub: требуется реализация в Этап 3")


# ---------------------------------------------------------------------------
# Milstein
# ---------------------------------------------------------------------------


class MilsteinSolver:
    """Солвер Милштейна (strong order 1.0).

    X_{n+1} = X_n + μ·dt + σ·dW + 0.5·σ·σ'·(dW² - dt)

    Поправка Милштейна: 0.5·σ(X_n)·σ'(X_n)·(ΔW² - Δt)
    учитывает производную диффузии (Itô-Taylor expansion).
    σ' вычисляется численно (конечные разности).

    Подробное описание: Description/Phase2/description_sde_numerics.md#MilsteinSolver
    """

    def __init__(self, config: SolverConfig | None = None) -> None:
        """Инициализация солвера Милштейна.

        Args:
            config: Конфигурация (fd_epsilon для конечных разностей)

        Подробное описание:
            Description/Phase2/description_sde_numerics.md#MilsteinSolver.__init__
        """
        self._config = config or SolverConfig(solver_type=SolverType.MILSTEIN)

    def step(
        self,
        state: np.ndarray,
        drift: np.ndarray,
        diffusion: np.ndarray,
        dt: float,
        dW: np.ndarray,
        diffusion_derivative: np.ndarray | None = None,
    ) -> StepResult:
        """Один шаг Милштейна: X += μdt + σdW + 0.5σσ'(dW²-dt).

        Args:
            state: X_n, shape (20,)
            drift: μ(X_n), shape (20,)
            diffusion: σ(X_n), shape (20,)
            dt: Шаг времени
            dW: Винеровские приращения, shape (20,)
            diffusion_derivative: σ'(X_n), shape (20,); если None — вычисляется
                через _compute_diffusion_derivative

        Returns:
            StepResult с X_{n+1}

        Подробное описание:
            Description/Phase2/description_sde_numerics.md#MilsteinSolver.step
        """
        raise NotImplementedError("Stub: требуется реализация в Этап 3")

    def simulate(
        self,
        model: ExtendedSDEModel,
        initial_state: ExtendedSDEState,
        params: ParameterSet,
    ) -> ExtendedSDETrajectory:
        """Полная симуляция методом Милштейна.

        Args:
            model: Расширенная SDE модель
            initial_state: Начальное состояние
            params: Параметры (dt, t_max)

        Returns:
            Траектория симуляции

        Подробное описание:
            Description/Phase2/description_sde_numerics.md#MilsteinSolver.simulate
        """
        raise NotImplementedError("Stub: требуется реализация в Этап 3")

    def _compute_diffusion_derivative(
        self,
        model: ExtendedSDEModel,
        state: ExtendedSDEState,
        eps: float | None = None,
    ) -> np.ndarray:
        """Численная производная диффузии σ'(X) методом конечных разностей.

        Для каждой компоненты i:
        σ'_i ≈ (σ_i(X + ε·e_i) - σ_i(X)) / ε

        Args:
            model: Модель с _compute_diffusion()
            state: Текущее состояние
            eps: Шаг конечных разностей (None → config.fd_epsilon)

        Returns:
            np.ndarray shape (20,) — покомпонентные производные σ'

        Подробное описание:
            Description/Phase2/description_sde_numerics.md#_compute_diffusion_derivative
        """
        raise NotImplementedError("Stub: требуется реализация в Этап 3")


# ---------------------------------------------------------------------------
# IMEX Splitting
# ---------------------------------------------------------------------------


# Индексы быстрых (стиффных) переменных: цитокины
FAST_INDICES: list[int] = list(range(8, 15))  # C_TNF..C_IL8 (StateIndex 8..14)

# Индексы медленных переменных: клетки + ECM + вспомогательные
SLOW_INDICES: list[int] = list(range(0, 8)) + list(range(15, 20))


class IMEXSplitter:
    """IMEX (Implicit-Explicit) splitting для стиффных SDE систем.

    Цитокины (индексы 8–14) имеют быструю деградацию (γ ≈ 0.1–0.5 ч⁻¹),
    что делает систему стиффной. IMEX разделяет:
    - Fast (implicit, backward Euler): цитокины — стиффная часть
    - Slow (explicit, EM): клетки + ECM — нестиффная часть

    X_fast^{n+1} = X_fast^n + μ_fast(X^{n+1})·dt  (implicit)
    X_slow^{n+1} = X_slow^n + μ_slow(X^n)·dt + σ_slow(X^n)·dW  (explicit)

    Подробное описание: Description/Phase2/description_sde_numerics.md#IMEXSplitter
    """

    def __init__(
        self,
        config: SolverConfig | None = None,
        fast_indices: list[int] | None = None,
        slow_indices: list[int] | None = None,
    ) -> None:
        """Инициализация IMEX солвера.

        Args:
            config: Конфигурация солвера
            fast_indices: Индексы стиффных переменных (None → FAST_INDICES)
            slow_indices: Индексы нестиффных переменных (None → SLOW_INDICES)

        Подробное описание:
            Description/Phase2/description_sde_numerics.md#IMEXSplitter.__init__
        """
        self._config = config or SolverConfig(solver_type=SolverType.IMEX)
        self._fast_indices = fast_indices if fast_indices is not None else FAST_INDICES
        self._slow_indices = slow_indices if slow_indices is not None else SLOW_INDICES

    def step(
        self,
        state: np.ndarray,
        drift: np.ndarray,
        diffusion: np.ndarray,
        dt: float,
        dW: np.ndarray,
    ) -> StepResult:
        """Один комбинированный IMEX шаг.

        1. Explicit шаг для медленных компонент (cells + ECM)
        2. Implicit шаг для быстрых компонент (цитокины)
        3. Объединение в полное состояние

        Args:
            state: X_n, shape (20,)
            drift: μ(X_n), shape (20,)
            diffusion: σ(X_n), shape (20,)
            dt: Шаг времени
            dW: Винеровские приращения, shape (20,)

        Returns:
            StepResult с X_{n+1}

        Подробное описание:
            Description/Phase2/description_sde_numerics.md#IMEXSplitter.step
        """
        raise NotImplementedError("Stub: требуется реализация в Этап 3")

    def simulate(
        self,
        model: ExtendedSDEModel,
        initial_state: ExtendedSDEState,
        params: ParameterSet,
    ) -> ExtendedSDETrajectory:
        """Полная симуляция методом IMEX splitting.

        Args:
            model: Расширенная SDE модель
            initial_state: Начальное состояние
            params: Параметры (dt, t_max)

        Returns:
            Траектория симуляции

        Подробное описание:
            Description/Phase2/description_sde_numerics.md#IMEXSplitter.simulate
        """
        raise NotImplementedError("Stub: требуется реализация в Этап 3")

    def _implicit_step(
        self,
        state_fast: np.ndarray,
        drift_fast: np.ndarray,
        dt: float,
        max_iter: int = 10,
        tol: float = 1e-8,
    ) -> np.ndarray:
        """Implicit шаг (backward Euler) для стиффных переменных.

        Решает X^{n+1} = X^n + μ(X^{n+1})·dt
        методом неподвижной точки (fixed-point iteration).

        Args:
            state_fast: Быстрые компоненты X_fast^n
            drift_fast: Дрифт быстрых компонент μ_fast(X^n)
            dt: Шаг времени
            max_iter: Максимум итераций fixed-point
            tol: Допуск сходимости

        Returns:
            X_fast^{n+1} после implicit шага

        Подробное описание:
            Description/Phase2/description_sde_numerics.md#_implicit_step
        """
        raise NotImplementedError("Stub: требуется реализация в Этап 3")

    def _explicit_step(
        self,
        state_slow: np.ndarray,
        drift_slow: np.ndarray,
        diffusion_slow: np.ndarray,
        dt: float,
        dW_slow: np.ndarray,
    ) -> np.ndarray:
        """Explicit шаг (Euler-Maruyama) для нестиффных переменных.

        X_slow^{n+1} = X_slow^n + μ_slow·dt + σ_slow·dW

        Args:
            state_slow: Медленные компоненты X_slow^n
            drift_slow: Дрифт медленных μ_slow(X^n)
            diffusion_slow: Диффузия медленных σ_slow(X^n)
            dt: Шаг времени
            dW_slow: Винеровские приращения медленных компонент

        Returns:
            X_slow^{n+1} после explicit шага

        Подробное описание:
            Description/Phase2/description_sde_numerics.md#_explicit_step
        """
        raise NotImplementedError("Stub: требуется реализация в Этап 3")

    def _split_state(
        self,
        state: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Разделение полного состояния на быстрые и медленные компоненты.

        Args:
            state: Полное состояние shape (20,)

        Returns:
            (state_fast, state_slow) — подмассивы по fast/slow индексам

        Подробное описание:
            Description/Phase2/description_sde_numerics.md#_split_state
        """
        raise NotImplementedError("Stub: требуется реализация в Этап 3")

    def _merge_state(
        self,
        state_fast: np.ndarray,
        state_slow: np.ndarray,
    ) -> np.ndarray:
        """Объединение быстрых и медленных компонент в полное состояние.

        Args:
            state_fast: Быстрые компоненты (цитокины)
            state_slow: Медленные компоненты (клетки + ECM)

        Returns:
            Полное состояние shape (20,)

        Подробное описание:
            Description/Phase2/description_sde_numerics.md#_merge_state
        """
        raise NotImplementedError("Stub: требуется реализация в Этап 3")


# ---------------------------------------------------------------------------
# Adaptive Timestepper
# ---------------------------------------------------------------------------


class AdaptiveTimestepper:
    """Адаптивный контроль шага времени с PI-контроллером.

    Обёртка над произвольным base_solver. Выполняет двойной шаг
    (full step dt + два half steps dt/2) для оценки ошибки,
    затем адаптирует dt через PI-контроллер:

    dt_new = dt · safety · (tol / error)^(k_I / p) · (error_prev / error)^(k_P / p)

    где p = порядок метода, k_I = 0.3, k_P = 0.4 (PID параметры).

    Подробное описание: Description/Phase2/description_sde_numerics.md#AdaptiveTimestepper
    """

    def __init__(
        self,
        base_solver: SDESolver,
        config: SolverConfig | None = None,
    ) -> None:
        """Инициализация адаптивного солвера.

        Args:
            base_solver: Базовый солвер (EM, Milstein, SRK)
            config: Конфигурация (tolerance, dt_min, dt_max, safety_factor)

        Подробное описание:
            Description/Phase2/description_sde_numerics.md#AdaptiveTimestepper.__init__
        """
        self._base_solver = base_solver
        self._config = config or SolverConfig(solver_type=SolverType.ADAPTIVE)
        self._error_prev: float = 1.0

    def step(
        self,
        state: np.ndarray,
        drift: np.ndarray,
        diffusion: np.ndarray,
        dt: float,
        dW: np.ndarray,
    ) -> StepResult:
        """Один шаг с адаптацией dt.

        1. Выполнить full step: X_full = base_solver.step(state, dt)
        2. Выполнить два half steps: X_half = step(step(state, dt/2), dt/2)
        3. error = ||X_full - X_half|| / (2^p - 1)
        4. Если error < tol: принять X_half, вычислить dt_new
        5. Если error > tol: отклонить шаг, уменьшить dt

        Args:
            state: X_n, shape (20,)
            drift: μ(X_n), shape (20,)
            diffusion: σ(X_n), shape (20,)
            dt: Предлагаемый шаг
            dW: Винеровские приращения, shape (20,)

        Returns:
            StepResult с X_{n+1} и обновлённым dt_used

        Подробное описание:
            Description/Phase2/description_sde_numerics.md#AdaptiveTimestepper.step
        """
        raise NotImplementedError("Stub: требуется реализация в Этап 3")

    def simulate(
        self,
        model: ExtendedSDEModel,
        initial_state: ExtendedSDEState,
        params: ParameterSet,
    ) -> ExtendedSDETrajectory:
        """Полная симуляция с адаптивным шагом.

        Args:
            model: Расширенная SDE модель
            initial_state: Начальное состояние
            params: Параметры

        Returns:
            Траектория с переменным шагом

        Подробное описание:
            Description/Phase2/description_sde_numerics.md#AdaptiveTimestepper.simulate
        """
        raise NotImplementedError("Stub: требуется реализация в Этап 3")

    def _estimate_error(
        self,
        state_full: np.ndarray,
        state_half: np.ndarray,
        order: int = 1,
    ) -> float:
        """Оценка локальной ошибки методом Richardson extrapolation.

        error ≈ ||X_full - X_half|| / (2^p - 1)

        Args:
            state_full: Результат одного шага dt
            state_half: Результат двух шагов dt/2
            order: Порядок метода p

        Returns:
            Скалярная оценка ошибки (L2-норма)

        Подробное описание:
            Description/Phase2/description_sde_numerics.md#_estimate_error
        """
        raise NotImplementedError("Stub: требуется реализация в Этап 3")

    def _pi_controller(
        self,
        error: float,
        tolerance: float,
        dt_current: float,
        order: int = 1,
    ) -> float:
        """PI-контроллер для адаптации шага времени.

        dt_new = dt · safety · (tol/error)^(k_I/p) · (error_prev/error)^(k_P/p)
        k_I = 0.3, k_P = 0.4 (Gustafsson)

        Args:
            error: Текущая оценка ошибки
            tolerance: Целевой допуск
            dt_current: Текущий шаг
            order: Порядок метода p

        Returns:
            Новый шаг dt_new в диапазоне [dt_min, dt_max]

        Подробное описание:
            Description/Phase2/description_sde_numerics.md#_pi_controller
        """
        raise NotImplementedError("Stub: требуется реализация в Этап 3")


# ---------------------------------------------------------------------------
# Stochastic Runge-Kutta (SRI2W1)
# ---------------------------------------------------------------------------


class StochasticRungeKutta:
    """Стохастический метод Рунге-Кутты SRI2W1 (strong order 1.0).

    Опциональный метод для мультимерных перекрёстных шумовых термов.
    Обеспечивает strong order 1.0 без вычисления σ'(X) (в отличие от Milstein).
    Использует tableau Rößler (2010).

    Подробное описание: Description/Phase2/description_sde_numerics.md#StochasticRungeKutta
    """

    def __init__(self, config: SolverConfig | None = None) -> None:
        """Инициализация SRK солвера.

        Args:
            config: Конфигурация солвера

        Подробное описание:
            Description/Phase2/description_sde_numerics.md#StochasticRungeKutta.__init__
        """
        self._config = config or SolverConfig(solver_type=SolverType.SRK)

    def step(
        self,
        state: np.ndarray,
        drift: np.ndarray,
        diffusion: np.ndarray,
        dt: float,
        dW: np.ndarray,
    ) -> StepResult:
        """Один шаг SRI2W1.

        Использует стадии Рунге-Кутты с детерминированным и стохастическим
        tableau для достижения strong order 1.0.

        Args:
            state: X_n, shape (20,)
            drift: μ(X_n), shape (20,)
            diffusion: σ(X_n), shape (20,)
            dt: Шаг времени
            dW: Винеровские приращения, shape (20,)

        Returns:
            StepResult с X_{n+1}

        Подробное описание:
            Description/Phase2/description_sde_numerics.md#StochasticRungeKutta.step
        """
        raise NotImplementedError("Stub: требуется реализация в Этап 3")

    def simulate(
        self,
        model: ExtendedSDEModel,
        initial_state: ExtendedSDEState,
        params: ParameterSet,
    ) -> ExtendedSDETrajectory:
        """Полная симуляция методом SRI2W1.

        Args:
            model: Расширенная SDE модель
            initial_state: Начальное состояние
            params: Параметры

        Returns:
            Траектория симуляции

        Подробное описание:
            Description/Phase2/description_sde_numerics.md#StochasticRungeKutta.simulate
        """
        raise NotImplementedError("Stub: требуется реализация в Этап 3")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_solver(config: SolverConfig) -> SDESolver:
    """Фабрика для создания солвера по конфигурации.

    Args:
        config: Конфигурация с solver_type

    Returns:
        Экземпляр соответствующего солвера

    Raises:
        ValueError: Неизвестный solver_type

    Подробное описание:
        Description/Phase2/description_sde_numerics.md#create_solver
    """
    raise NotImplementedError("Stub: требуется реализация в Этап 3")
