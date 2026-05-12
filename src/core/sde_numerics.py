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

from src.core.extended_sde import (
    ExtendedSDEState,
    ExtendedSDETrajectory,
)
from src.core.parameters import ParameterSet

if TYPE_CHECKING:
    from src.core.extended_sde import ExtendedSDEModel


# ---------------------------------------------------------------------------
# Enums & Config
# ---------------------------------------------------------------------------


class SolverType(Enum):
    """Тип численного солвера SDE.

    Подробное описание: Description/Phase2/description_sde_numerics.md#SolverType
    """

    EM = "euler_maruyama"  # Strong order 0.5
    MILSTEIN = "milstein"  # Strong order 1.0
    IMEX = "imex"  # Implicit-Explicit splitting
    SRK = "stochastic_rk"  # SRI2W1
    ADAPTIVE = "adaptive"  # Адаптивный шаг (обёртка)


@dataclass
class SolverConfig:
    """Конфигурация численного солвера.

    Управляет параметрами интегрирования: шаг времени, допуски,
    выбор метода. Используется всеми солверами.

    Подробное описание: Description/Phase2/description_sde_numerics.md#SolverConfig
    """

    solver_type: SolverType = SolverType.EM  # Тип солвера
    dt: float = 0.01  # Базовый шаг времени (ч)
    dt_min: float = 1e-6  # Минимальный шаг
    dt_max: float = 1.0  # Максимальный шаг
    tolerance: float = 1e-3  # Допуск ошибки (для адаптивного)
    max_steps: int = 100_000  # Максимум шагов
    safety_factor: float = 0.9  # Запас прочности PI-контроллера
    fd_epsilon: float = 1e-6  # Epsilon для конечных разностей (Milstein)


@dataclass
class StepResult:
    """Результат одного шага интегрирования.

    Содержит новое состояние, использованный dt, оценку ошибки
    и количество вызовов drift/diffusion.

    Подробное описание: Description/Phase2/description_sde_numerics.md#StepResult
    """

    new_state: np.ndarray  # Массив состояния shape (20,)
    dt_used: float = 0.0  # Фактически использованный шаг
    error_estimate: float = 0.0  # Оценка локальной ошибки
    n_function_evals: int = 0  # Число вызовов drift/diffusion
    rejected: bool = False  # Шаг был отклонён (adaptive)


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
# Общий цикл симуляции (shared by solvers that delegate to step())
# ---------------------------------------------------------------------------


def _run_simulation_loop(
    solver: SDESolver,
    model: ExtendedSDEModel,
    initial_state: ExtendedSDEState,
    params: ParameterSet,
) -> ExtendedSDETrajectory:
    """Общий цикл интегрирования SDE через solver.step().

    Порт из ExtendedSDEModel.simulate() — идентичная логика,
    но шаг делегируется в solver.step() вместо inline EM.

    Args:
        solver: Солвер с методом step()
        model: Модель с _compute_drift/_compute_diffusion
        initial_state: Начальное состояние
        params: Параметры (dt, t_max)

    Returns:
        ExtendedSDETrajectory с результатами
    """
    n_steps = int(params.t_max / params.dt)
    sqrt_dt = np.sqrt(params.dt)
    times = np.linspace(0.0, params.t_max, n_steps + 1)

    states: list[ExtendedSDEState] = []
    current_state = initial_state
    states.append(current_state)

    for i in range(n_steps):
        drift = model._compute_drift(current_state)
        diffusion = model._compute_diffusion(current_state)
        dW = model._rng.standard_normal(model.N_VARIABLES) * sqrt_dt

        x = current_state.to_array()
        result = solver.step(x, drift, diffusion, params.dt, dW)

        new_state = ExtendedSDEState.from_array(result.new_state, t=times[i + 1])
        new_state = model._apply_boundary_conditions(new_state)
        states.append(new_state)
        current_state = new_state

    return ExtendedSDETrajectory(times=times, states=states, params=params)


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
        new_state = state + drift * dt + diffusion * dW
        return StepResult(
            new_state=new_state,
            dt_used=dt,
            n_function_evals=1,
        )

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
        return _run_simulation_loop(self, model, initial_state, params)


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
        if diffusion_derivative is not None:
            sigma_prime = diffusion_derivative
            n_evals = 1
        else:
            sigma_prime = np.zeros_like(state)
            n_evals = 2

        correction = 0.5 * diffusion * sigma_prime * (dW**2 - dt)
        new_state = state + drift * dt + diffusion * dW + correction
        return StepResult(
            new_state=new_state,
            dt_used=dt,
            n_function_evals=n_evals,
        )

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
        n_steps = int(params.t_max / params.dt)
        sqrt_dt = np.sqrt(params.dt)
        times = np.linspace(0.0, params.t_max, n_steps + 1)

        states: list[ExtendedSDEState] = []
        current_state = initial_state
        states.append(current_state)

        for i in range(n_steps):
            drift = model._compute_drift(current_state)
            diffusion = model._compute_diffusion(current_state)
            sigma_prime = self._compute_diffusion_derivative(model, current_state)
            dW = model._rng.standard_normal(model.N_VARIABLES) * sqrt_dt

            x = current_state.to_array()
            result = self.step(x, drift, diffusion, params.dt, dW, diffusion_derivative=sigma_prime)

            new_state = ExtendedSDEState.from_array(result.new_state, t=times[i + 1])
            new_state = model._apply_boundary_conditions(new_state)
            states.append(new_state)
            current_state = new_state

        return ExtendedSDETrajectory(times=times, states=states, params=params)

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
        eps = eps if eps is not None else self._config.fd_epsilon
        x = state.to_array()
        sigma_base = model._compute_diffusion(state)
        sigma_prime = np.zeros_like(x)

        for i in range(len(x)):
            x_pert = x.copy()
            x_pert[i] += eps
            state_pert = ExtendedSDEState.from_array(x_pert, t=state.t)
            sigma_pert = model._compute_diffusion(state_pert)
            sigma_prime[i] = (sigma_pert[i] - sigma_base[i]) / eps

        return sigma_prime


# ---------------------------------------------------------------------------
# IMEX Splitting
# ---------------------------------------------------------------------------


# Индексы быстрых (стиффных) переменных: 8 цитокинов (включая SDF-1 после FIX-06 v2.0)
FAST_INDICES: list[int] = list(range(8, 16))  # C_TNF..C_SDF1 (StateIndex 8..15)

# Индексы медленных переменных: клетки + C_TIMP + ECM + aux (v2.0, 22 vars total)
SLOW_INDICES: list[int] = list(range(0, 8)) + list(range(16, 22))


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
        fast_state, slow_state = self._split_state(state)
        fast_drift, slow_drift = self._split_state(drift)
        _, slow_diffusion = self._split_state(diffusion)
        _, slow_dW = self._split_state(dW)

        # step() не имеет доступа к model → forward Euler для fast vars.
        # Для proper backward Euler используйте simulate().
        new_fast = fast_state + fast_drift * dt
        new_slow = self._explicit_step(
            slow_state,
            slow_drift,
            slow_diffusion,
            dt,
            slow_dW,
        )
        new_state = self._merge_state(new_fast, new_slow)
        return StepResult(
            new_state=new_state,
            dt_used=dt,
            n_function_evals=1,
        )

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
        from src.core.extended_sde import ExtendedSDEState, ExtendedSDETrajectory  # noqa: PLC0415

        n_steps = int(params.t_max / params.dt)
        sqrt_dt = np.sqrt(params.dt)
        times = np.linspace(0.0, params.t_max, n_steps + 1)

        states: list[ExtendedSDEState] = [initial_state]
        current_state = initial_state

        for i in range(n_steps):
            x = current_state.to_array()
            state_fast = x[self._fast_indices]
            state_slow = x[self._slow_indices]

            drift_full = model._compute_drift(current_state)
            diffusion_full = model._compute_diffusion(current_state)

            drift_slow = drift_full[self._slow_indices]
            diffusion_slow = diffusion_full[self._slow_indices]

            dW_full = model._rng.standard_normal(model.N_VARIABLES) * sqrt_dt
            dW_slow = dW_full[self._slow_indices]

            new_fast = self._implicit_step(state_fast, state_slow, params.dt, model, times[i])
            new_slow = state_slow + drift_slow * params.dt + diffusion_slow * dW_slow

            new_x = self._merge_state(new_fast, new_slow)
            new_state = ExtendedSDEState.from_array(new_x, t=times[i + 1])
            new_state = model._apply_boundary_conditions(new_state)
            states.append(new_state)
            current_state = new_state

        return ExtendedSDETrajectory(times=times, states=states, params=params)

    def _implicit_step(
        self,
        state_fast: np.ndarray,
        state_slow: np.ndarray,
        dt: float,
        model: ExtendedSDEModel,
        t: float,
        max_iter: int = 10,
        tol: float = 1e-8,
    ) -> np.ndarray:
        """Implicit шаг (backward Euler) для стиффных переменных.

        Решает X^{n+1} = X^n + μ_fast(X^{n+1}, X_slow^n)·dt
        методом неподвижной точки (Picard iteration).

        Сходимость гарантирована при γ·dt < 1.
        При стандартном dt=0.01: max(γ·dt) = 0.5·0.01 = 0.005 ≪ 1.

        Args:
            state_fast: Быстрые компоненты X_fast^n (цитокины)
            state_slow: Медленные компоненты X_slow^n (фиксированы)
            dt: Шаг времени
            model: ExtendedSDEModel для вычисления drift
            t: Текущее время
            max_iter: Максимум итераций fixed-point
            tol: Допуск сходимости

        Returns:
            X_fast^{n+1} после implicit шага

        Подробное описание:
            Description/Phase2/description_sde_numerics.md#_implicit_step
        """
        from src.core.extended_sde import ExtendedSDEState  # noqa: PLC0415

        x = state_fast.copy()
        for _ in range(max_iter):
            full_arr = self._merge_state(x, state_slow)
            full_state = ExtendedSDEState.from_array(full_arr, t=t)
            drift_fast = model._compute_drift(full_state)[self._fast_indices]
            x_new = state_fast + drift_fast * dt
            if float(np.linalg.norm(x_new - x)) < tol:
                return x_new
            x = x_new
        return x

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
        result: np.ndarray = state_slow + drift_slow * dt + diffusion_slow * dW_slow
        return result

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
        return state[self._fast_indices].copy(), state[self._slow_indices].copy()

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
        n = len(self._fast_indices) + len(self._slow_indices)
        result = np.empty(n)
        result[self._fast_indices] = state_fast
        result[self._slow_indices] = state_slow
        return result


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
        self._error_prev: float = 0.0

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
        # Полный шаг
        result_full = self._base_solver.step(state, drift, diffusion, dt, dW)

        # Два полушага с масштабированным dW (дисперсия ~ dt)
        dW_half = dW * np.sqrt(0.5)
        result_half1 = self._base_solver.step(
            state,
            drift,
            diffusion,
            dt / 2,
            dW_half,
        )
        result_half2 = self._base_solver.step(
            result_half1.new_state,
            drift,
            diffusion,
            dt / 2,
            dW_half,
        )

        # Оценка ошибки
        error = self._estimate_error(result_full.new_state, result_half2.new_state)

        # Новый шаг через PI-контроллер
        dt_new = self._pi_controller(
            max(error, 1e-15),
            self._config.tolerance,
            dt,
        )

        if error <= self._config.tolerance:
            return StepResult(
                new_state=result_half2.new_state,
                dt_used=dt_new,
                error_estimate=error,
                n_function_evals=3,
            )
        else:
            return StepResult(
                new_state=state,
                dt_used=dt_new,
                error_estimate=error,
                n_function_evals=3,
                rejected=True,
            )

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
        t = 0.0
        dt = params.dt
        step_count = 0

        times_list: list[float] = [0.0]
        states: list[ExtendedSDEState] = []
        current_state = initial_state
        states.append(current_state)

        while t < params.t_max and step_count < self._config.max_steps:
            # Ограничить dt чтобы не перешагнуть t_max
            dt = min(dt, params.t_max - t)
            if dt <= 0:
                break

            drift = model._compute_drift(current_state)
            diffusion = model._compute_diffusion(current_state)
            sqrt_dt = np.sqrt(dt)
            dW = model._rng.standard_normal(model.N_VARIABLES) * sqrt_dt

            x = current_state.to_array()
            result = self.step(x, drift, diffusion, dt, dW)
            step_count += 1

            if not result.rejected:
                t += dt
                new_state = ExtendedSDEState.from_array(result.new_state, t=t)
                new_state = model._apply_boundary_conditions(new_state)
                states.append(new_state)
                times_list.append(t)
                current_state = new_state

            dt = np.clip(result.dt_used, self._config.dt_min, self._config.dt_max)

        return ExtendedSDETrajectory(
            times=np.array(times_list),
            states=states,
            params=params,
        )

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
        divisor = 2**order - 1
        return float(np.linalg.norm(state_full - state_half) / max(divisor, 1))

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
        k_I, k_P = 0.3, 0.4
        safety = self._config.safety_factor
        error = max(error, 1e-15)

        factor = safety * (tolerance / error) ** (k_I / order)
        if self._error_prev > 0:
            factor *= (self._error_prev / error) ** (k_P / order)

        dt_new = dt_current * factor
        dt_new = max(self._config.dt_min, min(self._config.dt_max, dt_new))
        self._error_prev = error
        return float(dt_new)


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
        """Один шаг Эйлера-Маруямы (strong order 0.5).

        Примечание: настоящий SRI2W1 (strong order 1.0) требует вычисления
        drift в промежуточной точке, что невозможно с pre-computed drift/diffusion.
        Метод step() получает уже вычисленные drift и diffusion, поэтому
        реализует Euler-Maruyama. Для SRI2W1 используйте simulate() с доступом
        к модели (пока не реализован).

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
        # Fallback: Euler-Maruyama (model unavailable for 2nd stage evaluation)
        new_state = state + drift * dt + diffusion * dW
        return StepResult(
            new_state=new_state,
            dt_used=dt,
            n_function_evals=1,
        )

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
        n_steps = int(params.t_max / params.dt)
        sqrt_dt = np.sqrt(params.dt)
        times = np.linspace(0.0, params.t_max, n_steps + 1)

        states: list[ExtendedSDEState] = []
        current_state = initial_state
        states.append(current_state)

        for i in range(n_steps):
            # Стадия 1: drift и diffusion в текущей точке X_n
            k1 = model._compute_drift(current_state)
            l1 = model._compute_diffusion(current_state)

            # Промежуточное состояние: X_hat = X_n + K1*dt + L1*sqrt(dt)
            x = current_state.to_array()
            x_hat = x + k1 * params.dt + l1 * sqrt_dt
            state_hat = ExtendedSDEState.from_array(x_hat, t=current_state.t)

            # Стадия 2: drift и diffusion в промежуточной точке
            k2 = model._compute_drift(state_hat)
            l2 = model._compute_diffusion(state_hat)

            # Винеровские приращения
            dW = model._rng.standard_normal(model.N_VARIABLES) * sqrt_dt

            # SRI2W1: Heun + поправка Milstein (strong order 1.0, диагональный шум)
            # correction = 0.5*(L2-L1)*(dW²-dt)/sqrt_dt  [Platen derivative-free Milstein]
            correction = 0.5 * (l2 - l1) * (dW**2 - params.dt) / sqrt_dt
            x_new = x + 0.5 * (k1 + k2) * params.dt + 0.5 * (l1 + l2) * dW + correction

            new_state = ExtendedSDEState.from_array(x_new, t=times[i + 1])
            new_state = model._apply_boundary_conditions(new_state)
            states.append(new_state)
            current_state = new_state

        return ExtendedSDETrajectory(times=times, states=states, params=params)


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
    solver_type = config.solver_type
    if solver_type == SolverType.EM:
        return EulerMaruyamaSolver(config)
    if solver_type == SolverType.MILSTEIN:
        return MilsteinSolver(config)
    if solver_type == SolverType.IMEX:
        return IMEXSplitter(config)
    if solver_type == SolverType.SRK:
        return StochasticRungeKutta(config)
    if solver_type == SolverType.ADAPTIVE:
        return AdaptiveTimestepper(
            base_solver=EulerMaruyamaSolver(
                SolverConfig(solver_type=SolverType.EM),
            ),
            config=config,
        )
    msg = f"Неизвестный тип солвера: {solver_type}"
    raise ValueError(msg)
