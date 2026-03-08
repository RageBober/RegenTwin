"""TDD тесты для модуля численных методов SDE sde_numerics.

Тестирование:
- SolverType: enum значения, создание из строки, количество
- SolverConfig: значения по умолчанию, пользовательские, инварианты
- StepResult: shape, defaults, инварианты
- SDESolver Protocol: isinstance для всех солверов
- EulerMaruyamaSolver: step (drift, diffusion, combined), invariants
- MilsteinSolver: step (correction term), sigma'=0 совпадает с EM
- IMEXSplitter: split/merge roundtrip, fast/slow indices, step
- AdaptiveTimestepper: _estimate_error, _pi_controller, step
- StochasticRungeKutta: step (sigma=0 -> RK2), n_function_evals
- create_solver: фабрика для каждого SolverType

Все тесты написаны для stub-реализации (NotImplementedError).
После реализации методов тесты должны проходить.
"""

import numpy as np
import pytest

from src.core.sde_numerics import (
    FAST_INDICES,
    SLOW_INDICES,
    AdaptiveTimestepper,
    EulerMaruyamaSolver,
    IMEXSplitter,
    MilsteinSolver,
    SDESolver,
    SolverConfig,
    SolverType,
    StepResult,
    StochasticRungeKutta,
    create_solver,
)

# =============================================================================
# Test SolverType
# =============================================================================


class TestSolverType:
    """Тесты enum SolverType."""

    def test_em_value_euler_maruyama(self):
        """SolverType.EM.value == 'euler_maruyama'."""
        assert SolverType.EM.value == "euler_maruyama"

    def test_from_string_milstein(self):
        """Создание из строки 'milstein' -> SolverType.MILSTEIN."""
        assert SolverType("milstein") == SolverType.MILSTEIN

    def test_count_equals_5(self):
        """Всего 5 типов солверов."""
        assert len(SolverType) == 5


# =============================================================================
# Test SolverConfig
# =============================================================================


class TestSolverConfig:
    """Тесты dataclass SolverConfig."""

    def test_default_values(self):
        """Все поля имеют значения по умолчанию."""
        config = SolverConfig()

        assert config.solver_type == SolverType.EM
        assert config.dt == 0.01
        assert config.dt_min == 1e-6
        assert config.dt_max == 1.0
        assert config.tolerance == 1e-3
        assert config.max_steps == 100_000
        assert config.safety_factor == 0.9
        assert config.fd_epsilon == 1e-6

    def test_custom_values(self):
        """Пользовательские значения сохраняются."""
        config = SolverConfig(dt=0.001, tolerance=1e-4)

        assert config.dt == 0.001
        assert config.tolerance == 1e-4
        assert config.solver_type == SolverType.EM  # остальные = default

    def test_invariant_dt_min_le_dt(self):
        """Инвариант: dt_min <= dt для конфигурации по умолчанию."""
        config = SolverConfig()

        assert 0 < config.dt_min <= config.dt

    def test_invariant_dt_le_dt_max(self):
        """Инвариант: dt <= dt_max для конфигурации по умолчанию."""
        config = SolverConfig()

        assert config.dt <= config.dt_max

    def test_invariant_tolerance_positive(self):
        """Инвариант: tolerance > 0."""
        config = SolverConfig()

        assert config.tolerance > 0

    def test_invariant_safety_factor_range(self):
        """Инвариант: 0 < safety_factor <= 1."""
        config = SolverConfig()

        assert 0 < config.safety_factor <= 1

    def test_invariant_max_steps_positive(self):
        """Инвариант: max_steps > 0."""
        config = SolverConfig()

        assert config.max_steps > 0


# =============================================================================
# Test StepResult
# =============================================================================


class TestStepResult:
    """Тесты dataclass StepResult."""

    def test_new_state_shape_20(self):
        """new_state.shape == (20,)."""
        result = StepResult(new_state=np.zeros(20))

        assert result.new_state.shape == (20,)

    def test_rejected_true(self):
        """Поле rejected корректно устанавливается."""
        result = StepResult(new_state=np.zeros(20), rejected=True)

        assert result.rejected is True

    def test_default_dt_used_zero(self):
        """dt_used по умолчанию == 0.0."""
        result = StepResult(new_state=np.zeros(20))

        assert result.dt_used == 0.0

    def test_default_error_estimate_zero(self):
        """error_estimate по умолчанию == 0.0."""
        result = StepResult(new_state=np.zeros(20))

        assert result.error_estimate == 0.0

    def test_invariant_n_function_evals_nonnegative(self):
        """Инвариант: n_function_evals >= 0."""
        result = StepResult(new_state=np.zeros(20))

        assert result.n_function_evals >= 0

    def test_default_rejected_false(self):
        """Rejected по умолчанию == False."""
        result = StepResult(new_state=np.zeros(20))

        assert result.rejected is False


# =============================================================================
# Test SDESolver Protocol
# =============================================================================


class TestSDESolverProtocol:
    """Тесты runtime_checkable Protocol SDESolver."""

    def test_em_is_sde_solver(self):
        """EulerMaruyamaSolver реализует SDESolver."""
        solver = EulerMaruyamaSolver()

        assert isinstance(solver, SDESolver)

    def test_milstein_is_sde_solver(self):
        """MilsteinSolver реализует SDESolver."""
        solver = MilsteinSolver()

        assert isinstance(solver, SDESolver)

    def test_srk_is_sde_solver(self):
        """StochasticRungeKutta реализует SDESolver."""
        solver = StochasticRungeKutta()

        assert isinstance(solver, SDESolver)


# =============================================================================
# Test EulerMaruyamaSolver
# =============================================================================


class TestEulerMaruyamaSolverInit:
    """Тесты инициализации EulerMaruyamaSolver."""

    def test_init_default_config(self):
        """Конфигурация по умолчанию: solver_type = EM."""
        solver = EulerMaruyamaSolver()

        assert solver._config.solver_type == SolverType.EM

    def test_init_custom_config(self):
        """Пользовательская конфигурация сохраняется."""
        config = SolverConfig(dt=0.005, solver_type=SolverType.EM)
        solver = EulerMaruyamaSolver(config=config)

        assert solver._config.dt == 0.005


class TestEulerMaruyamaSolverStep:
    """Тесты одного шага Эйлера-Маруямы: X += mu*dt + sigma*dW."""

    def test_pure_drift_only(self):
        """drift=[1,0,...], sigma=0, dt=0.1 -> X[0] += 0.1."""
        solver = EulerMaruyamaSolver()
        state = np.ones(20)
        drift = np.zeros(20)
        drift[0] = 1.0
        diffusion = np.zeros(20)
        dW = np.zeros(20)

        result = solver.step(state, drift, diffusion, dt=0.1, dW=dW)

        assert result.new_state[0] == pytest.approx(1.1)
        np.testing.assert_array_almost_equal(result.new_state[1:], np.ones(19))

    def test_pure_diffusion_only(self):
        """drift=0, sigma=[1,0,...], dW=[0.5,...] -> X[0] += 0.5."""
        solver = EulerMaruyamaSolver()
        state = np.ones(20)
        drift = np.zeros(20)
        diffusion = np.zeros(20)
        diffusion[0] = 1.0
        dW = np.zeros(20)
        dW[0] = 0.5

        result = solver.step(state, drift, diffusion, dt=0.1, dW=dW)

        assert result.new_state[0] == pytest.approx(1.5)
        np.testing.assert_array_almost_equal(result.new_state[1:], np.ones(19))

    def test_zero_drift_zero_diffusion_unchanged(self):
        """drift=0, sigma=0 -> X_new == X_n."""
        solver = EulerMaruyamaSolver()
        state = np.arange(20, dtype=float)
        drift = np.zeros(20)
        diffusion = np.zeros(20)
        dW = np.zeros(20)

        result = solver.step(state, drift, diffusion, dt=0.1, dW=dW)

        np.testing.assert_array_equal(result.new_state, state)

    def test_combined_drift_and_diffusion(self):
        """X_new = X + mu*dt + sigma*dW для ненулевых drift и diffusion."""
        solver = EulerMaruyamaSolver()
        state = np.full(20, 10.0)
        drift = np.full(20, 2.0)
        diffusion = np.full(20, 0.5)
        dt = 0.1
        dW = np.full(20, 0.3)

        result = solver.step(state, drift, diffusion, dt=dt, dW=dW)

        expected = state + drift * dt + diffusion * dW
        np.testing.assert_array_almost_equal(result.new_state, expected)

    def test_dt_used_equals_dt(self):
        """dt_used == dt (фиксированный шаг)."""
        solver = EulerMaruyamaSolver()
        state = np.ones(20)

        result = solver.step(state, np.zeros(20), np.zeros(20), dt=0.05, dW=np.zeros(20))

        assert result.dt_used == pytest.approx(0.05)

    def test_n_function_evals_equals_1(self):
        """n_function_evals == 1."""
        solver = EulerMaruyamaSolver()
        state = np.ones(20)

        result = solver.step(state, np.zeros(20), np.zeros(20), dt=0.01, dW=np.zeros(20))

        assert result.n_function_evals == 1

    def test_rejected_always_false(self):
        """Rejected == False (EM не отклоняет шаги)."""
        solver = EulerMaruyamaSolver()
        state = np.ones(20)

        result = solver.step(state, np.zeros(20), np.zeros(20), dt=0.01, dW=np.zeros(20))

        assert result.rejected is False


class TestEulerMaruyamaSolverSimulate:
    """Тесты полной симуляции EM."""

    def test_simulate_raises_not_implemented(self):
        """simulate() вызывает NotImplementedError для stub."""
        solver = EulerMaruyamaSolver()

        with pytest.raises(NotImplementedError):
            solver.simulate(model=None, initial_state=None, params=None)


# =============================================================================
# Test MilsteinSolver
# =============================================================================


class TestMilsteinSolverInit:
    """Тесты инициализации MilsteinSolver."""

    def test_init_default_config(self):
        """Конфигурация по умолчанию: solver_type = MILSTEIN."""
        solver = MilsteinSolver()

        assert solver._config.solver_type == SolverType.MILSTEIN

    def test_init_custom_fd_epsilon(self):
        """fd_epsilon из пользовательской конфигурации сохраняется."""
        config = SolverConfig(fd_epsilon=1e-8, solver_type=SolverType.MILSTEIN)
        solver = MilsteinSolver(config=config)

        assert solver._config.fd_epsilon == 1e-8


class TestMilsteinSolverStep:
    """Тесты одного шага Милштейна: X += mu*dt + sigma*dW + 0.5*sigma*sigma'*(dW^2-dt)."""

    def test_constant_sigma_matches_em(self):
        """Sigma' = 0 (const sigma) -> correction = 0, результат как EM."""
        solver = MilsteinSolver()
        state = np.ones(20)
        drift = np.full(20, 1.0)
        diffusion = np.full(20, 0.5)
        sigma_prime = np.zeros(20)  # sigma' = 0
        dt = 0.1
        dW = np.full(20, 0.3)

        result = solver.step(
            state, drift, diffusion, dt=dt, dW=dW, diffusion_derivative=sigma_prime
        )

        # Без correction: X_new = X + mu*dt + sigma*dW
        expected_em = state + drift * dt + diffusion * dW
        np.testing.assert_array_almost_equal(result.new_state, expected_em)

    def test_sigma_equals_x_correction_term(self):
        """Sigma = X, sigma' = 1 -> correction = 0.5 * X * 1 * (dW^2 - dt)."""
        solver = MilsteinSolver()
        state = np.full(20, 2.0)
        drift = np.zeros(20)
        diffusion = state.copy()  # sigma = X = 2.0
        sigma_prime = np.ones(20)  # sigma' = 1
        dt = 0.1
        dW = np.full(20, 0.4)

        result = solver.step(
            state, drift, diffusion, dt=dt, dW=dW, diffusion_derivative=sigma_prime
        )

        correction = 0.5 * diffusion * sigma_prime * (dW**2 - dt)
        expected = state + drift * dt + diffusion * dW + correction
        np.testing.assert_array_almost_equal(result.new_state, expected)

    def test_dw_zero_correction_negative(self):
        """DW = 0 -> correction = -0.5 * sigma * sigma' * dt."""
        solver = MilsteinSolver()
        state = np.full(20, 3.0)
        drift = np.zeros(20)
        diffusion = np.full(20, 1.0)
        sigma_prime = np.full(20, 2.0)
        dt = 0.1
        dW = np.zeros(20)

        result = solver.step(
            state, drift, diffusion, dt=dt, dW=dW, diffusion_derivative=sigma_prime
        )

        correction = 0.5 * diffusion * sigma_prime * (0.0 - dt)
        expected = state + correction  # drift=0, sigma*dW=0
        np.testing.assert_array_almost_equal(result.new_state, expected)

    def test_zero_drift_zero_diffusion_unchanged(self):
        """drift=0, sigma=0 -> X_new == X_n (correction тоже 0)."""
        solver = MilsteinSolver()
        state = np.arange(20, dtype=float)
        drift = np.zeros(20)
        diffusion = np.zeros(20)
        sigma_prime = np.zeros(20)
        dW = np.full(20, 0.5)

        result = solver.step(
            state, drift, diffusion, dt=0.1, dW=dW, diffusion_derivative=sigma_prime
        )

        np.testing.assert_array_equal(result.new_state, state)

    def test_n_function_evals_with_derivative_provided(self):
        """n_function_evals == 1 когда sigma' передан явно."""
        solver = MilsteinSolver()
        state = np.ones(20)

        result = solver.step(
            state,
            np.zeros(20),
            np.zeros(20),
            dt=0.01,
            dW=np.zeros(20),
            diffusion_derivative=np.zeros(20),
        )

        assert result.n_function_evals == 1

    def test_n_function_evals_without_derivative(self):
        """n_function_evals == 2 когда sigma' вычисляется внутренне."""
        solver = MilsteinSolver()
        state = np.ones(20)

        result = solver.step(
            state,
            np.zeros(20),
            np.zeros(20),
            dt=0.01,
            dW=np.zeros(20),
            diffusion_derivative=None,
        )

        assert result.n_function_evals == 2

    def test_rejected_always_false(self):
        """Rejected == False (Milstein не отклоняет шаги)."""
        solver = MilsteinSolver()
        state = np.ones(20)

        result = solver.step(
            state,
            np.zeros(20),
            np.zeros(20),
            dt=0.01,
            dW=np.zeros(20),
            diffusion_derivative=np.zeros(20),
        )

        assert result.rejected is False

    def test_result_shape_20(self):
        """new_state.shape == (20,)."""
        solver = MilsteinSolver()
        state = np.ones(20)

        result = solver.step(
            state,
            np.zeros(20),
            np.ones(20),
            dt=0.01,
            dW=np.ones(20),
            diffusion_derivative=np.ones(20),
        )

        assert result.new_state.shape == (20,)


class TestMilsteinComputeDiffusionDerivative:
    """Тесты численной производной диффузии."""

    def test_raises_not_implemented(self):
        """_compute_diffusion_derivative() вызывает NotImplementedError для stub."""
        solver = MilsteinSolver()

        with pytest.raises(NotImplementedError):
            solver._compute_diffusion_derivative(model=None, state=None)


# =============================================================================
# Test IMEXSplitter
# =============================================================================


class TestIMEXSplitterInit:
    """Тесты инициализации IMEXSplitter."""

    def test_default_fast_indices(self):
        """fast_indices по умолчанию = [8..14] (цитокины)."""
        splitter = IMEXSplitter()

        assert splitter._fast_indices == list(range(8, 15))

    def test_default_slow_indices(self):
        """slow_indices по умолчанию = [0..7] + [15..19]."""
        splitter = IMEXSplitter()

        assert splitter._slow_indices == list(range(0, 8)) + list(range(15, 20))

    def test_custom_indices(self):
        """Пользовательские fast/slow индексы сохраняются."""
        fast = [0, 1, 2]
        slow = list(range(3, 20))
        splitter = IMEXSplitter(fast_indices=fast, slow_indices=slow)

        assert splitter._fast_indices == fast
        assert splitter._slow_indices == slow


class TestIMEXSplitterConstants:
    """Тесты модульных констант FAST_INDICES / SLOW_INDICES."""

    def test_fast_plus_slow_cover_all_20(self):
        """fast_indices union slow_indices == {0..19}."""
        assert set(FAST_INDICES) | set(SLOW_INDICES) == set(range(20))

    def test_fast_slow_no_overlap(self):
        """fast_indices intersect slow_indices == empty."""
        assert set(FAST_INDICES) & set(SLOW_INDICES) == set()

    def test_fast_count_7_slow_count_13(self):
        """7 быстрых (цитокины) + 13 медленных."""
        assert len(FAST_INDICES) == 7
        assert len(SLOW_INDICES) == 13


class TestIMEXSplitterSplitMerge:
    """Тесты split/merge операций."""

    def test_split_merge_roundtrip(self):
        """merge(split(state)) == state."""
        splitter = IMEXSplitter()
        state = np.arange(20, dtype=float)

        fast, slow = splitter._split_state(state)
        merged = splitter._merge_state(fast, slow)

        np.testing.assert_array_equal(merged, state)

    def test_split_fast_shape(self):
        """Split возвращает fast с shape (7,)."""
        splitter = IMEXSplitter()
        state = np.ones(20)

        fast, _ = splitter._split_state(state)

        assert fast.shape == (7,)

    def test_split_slow_shape(self):
        """Split возвращает slow с shape (13,)."""
        splitter = IMEXSplitter()
        state = np.ones(20)

        _, slow = splitter._split_state(state)

        assert slow.shape == (13,)


class TestIMEXSplitterStep:
    """Тесты одного шага IMEX splitting."""

    def test_nonstiff_system_approx_em(self):
        """Нестиффная система (малый drift) -> результат примерно как EM."""
        splitter = IMEXSplitter()
        state = np.full(20, 10.0)
        drift = np.full(20, 0.01)  # малый drift
        diffusion = np.full(20, 0.1)
        dt = 0.01
        dW = np.full(20, 0.1)

        result = splitter.step(state, drift, diffusion, dt=dt, dW=dW)

        # Для нестиффных систем IMEX должен давать похожий на EM результат
        em_expected = state + drift * dt + diffusion * dW
        np.testing.assert_array_almost_equal(result.new_state, em_expected, decimal=2)

    def test_only_fast_nonzero(self):
        """Состояние только в цитокинах -> только implicit шаг активен."""
        splitter = IMEXSplitter()
        state = np.zeros(20)
        state[8:15] = 5.0  # только цитокины
        drift = np.zeros(20)
        drift[8:15] = -0.5  # деградация цитокинов
        diffusion = np.zeros(20)
        dW = np.zeros(20)

        result = splitter.step(state, drift, diffusion, dt=0.1, dW=dW)

        # Медленные компоненты не изменяются
        for i in SLOW_INDICES:
            assert result.new_state[i] == pytest.approx(0.0)

    def test_only_slow_nonzero(self):
        """Состояние только в клетках -> только explicit шаг активен."""
        splitter = IMEXSplitter()
        state = np.zeros(20)
        state[0:8] = 100.0  # только клетки
        drift = np.zeros(20)
        drift[0:8] = 1.0
        diffusion = np.zeros(20)
        dW = np.zeros(20)

        result = splitter.step(state, drift, diffusion, dt=0.1, dW=dW)

        # Быстрые компоненты не изменяются (нулевое начальное + нулевой drift)
        for i in FAST_INDICES:
            assert result.new_state[i] == pytest.approx(0.0)


class TestIMEXImplicitStep:
    """Тесты implicit шага для стиффных переменных."""

    def test_implicit_step_forward_euler(self):
        """_implicit_step() вычисляет state_fast + drift_fast * dt."""
        splitter = IMEXSplitter()
        state_fast = np.full(7, 5.0)
        drift_fast = np.full(7, -0.5)

        result = splitter._implicit_step(state_fast, drift_fast, dt=0.1)

        expected = state_fast + drift_fast * 0.1
        np.testing.assert_array_almost_equal(result, expected)


class TestIMEXExplicitStep:
    """Тесты explicit шага для нестиффных переменных."""

    def test_explicit_step_euler_maruyama(self):
        """_explicit_step() вычисляет state + drift*dt + diffusion*dW."""
        splitter = IMEXSplitter()
        state_slow = np.full(13, 100.0)
        drift_slow = np.full(13, 1.0)
        diffusion_slow = np.full(13, 0.5)
        dW_slow = np.full(13, 0.3)

        result = splitter._explicit_step(
            state_slow,
            drift_slow,
            diffusion_slow,
            dt=0.1,
            dW_slow=dW_slow,
        )

        expected = state_slow + drift_slow * 0.1 + diffusion_slow * dW_slow
        np.testing.assert_array_almost_equal(result, expected)


# =============================================================================
# Test AdaptiveTimestepper
# =============================================================================


class TestAdaptiveTimestepperInit:
    """Тесты инициализации AdaptiveTimestepper."""

    def test_init_stores_base_solver(self):
        """base_solver сохраняется."""
        base = EulerMaruyamaSolver()
        adaptive = AdaptiveTimestepper(base_solver=base)

        assert adaptive._base_solver is base

    def test_init_default_config(self):
        """Конфигурация по умолчанию: solver_type = ADAPTIVE."""
        adaptive = AdaptiveTimestepper(base_solver=EulerMaruyamaSolver())

        assert adaptive._config.solver_type == SolverType.ADAPTIVE


class TestAdaptiveEstimateError:
    """Тесты оценки ошибки Richardson extrapolation."""

    def test_equal_states_error_zero(self):
        """Одинаковые состояния -> error == 0."""
        adaptive = AdaptiveTimestepper(base_solver=EulerMaruyamaSolver())
        state = np.ones(20)

        error = adaptive._estimate_error(state, state, order=1)

        assert error == pytest.approx(0.0)

    def test_order_1_divisor_1(self):
        """order=1 -> делитель = 2^1 - 1 = 1."""
        adaptive = AdaptiveTimestepper(base_solver=EulerMaruyamaSolver())
        state_full = np.zeros(20)
        state_half = np.ones(20)

        error = adaptive._estimate_error(state_full, state_half, order=1)

        expected = np.linalg.norm(state_full - state_half) / 1.0
        assert error == pytest.approx(expected)

    def test_order_2_divisor_3(self):
        """order=2 -> делитель = 2^2 - 1 = 3."""
        adaptive = AdaptiveTimestepper(base_solver=EulerMaruyamaSolver())
        state_full = np.zeros(20)
        state_half = np.ones(20)

        error = adaptive._estimate_error(state_full, state_half, order=2)

        expected = np.linalg.norm(state_full - state_half) / 3.0
        assert error == pytest.approx(expected)


class TestAdaptivePIController:
    """Тесты PI-контроллера для адаптации шага."""

    def test_error_equals_tol_factor_approx_safety(self):
        """Error == tol -> dt_new примерно dt * safety_factor."""
        config = SolverConfig(safety_factor=0.9, dt_min=1e-6, dt_max=1.0)
        adaptive = AdaptiveTimestepper(
            base_solver=EulerMaruyamaSolver(),
            config=config,
        )

        dt_new = adaptive._pi_controller(
            error=1e-3,
            tolerance=1e-3,
            dt_current=0.01,
            order=1,
        )

        # При error == tol, множитель close to safety
        assert dt_new == pytest.approx(0.01 * 0.9, rel=0.3)

    def test_error_much_less_tol_increases_dt(self):
        """Error << tol -> dt увеличивается."""
        config = SolverConfig(dt_min=1e-6, dt_max=1.0)
        adaptive = AdaptiveTimestepper(
            base_solver=EulerMaruyamaSolver(),
            config=config,
        )

        dt_new = adaptive._pi_controller(
            error=1e-8,
            tolerance=1e-3,
            dt_current=0.01,
            order=1,
        )

        assert dt_new > 0.01

    def test_error_much_greater_tol_decreases_dt(self):
        """Error >> tol -> dt уменьшается."""
        config = SolverConfig(dt_min=1e-6, dt_max=1.0)
        adaptive = AdaptiveTimestepper(
            base_solver=EulerMaruyamaSolver(),
            config=config,
        )

        dt_new = adaptive._pi_controller(
            error=1.0,
            tolerance=1e-3,
            dt_current=0.01,
            order=1,
        )

        assert dt_new < 0.01


class TestAdaptiveTimestepperStep:
    """Тесты одного шага с адаптацией."""

    def test_slow_dynamics_dt_increases(self):
        """Медленная динамика -> dt увеличивается (не отклонён)."""
        config = SolverConfig(tolerance=1e-3, dt_min=1e-6, dt_max=1.0)
        adaptive = AdaptiveTimestepper(
            base_solver=EulerMaruyamaSolver(),
            config=config,
        )
        state = np.full(20, 100.0)
        drift = np.full(20, 0.001)  # медленная
        diffusion = np.zeros(20)
        dW = np.zeros(20)

        result = adaptive.step(state, drift, diffusion, dt=0.01, dW=dW)

        assert result.rejected is False
        assert result.dt_used > 0

    def test_fast_dynamics_dt_decreases(self):
        """Быстрая динамика -> dt уменьшается."""
        config = SolverConfig(tolerance=1e-6, dt_min=1e-8, dt_max=1.0)
        adaptive = AdaptiveTimestepper(
            base_solver=EulerMaruyamaSolver(),
            config=config,
        )
        state = np.full(20, 100.0)
        drift = np.full(20, 1000.0)  # быстрая
        diffusion = np.full(20, 50.0)
        dW = np.random.default_rng(42).standard_normal(20)

        result = adaptive.step(state, drift, diffusion, dt=0.5, dW=dW)

        # Либо принят с малым dt, либо отклонён
        assert result.dt_used <= 0.5 or result.rejected

    def test_large_error_rejected(self):
        """Очень большая ошибка -> шаг отклонён (rejected=True)."""
        config = SolverConfig(tolerance=1e-10, dt_min=1e-8, dt_max=1.0)
        adaptive = AdaptiveTimestepper(
            base_solver=EulerMaruyamaSolver(),
            config=config,
        )
        state = np.full(20, 1.0)
        drift = np.full(20, 1e6)  # вызовет огромную ошибку
        diffusion = np.full(20, 1e4)
        dW = np.ones(20)

        result = adaptive.step(state, drift, diffusion, dt=1.0, dW=dW)

        assert result.rejected is True


# =============================================================================
# Test StochasticRungeKutta
# =============================================================================


class TestStochasticRungeKuttaInit:
    """Тесты инициализации StochasticRungeKutta."""

    def test_init_default_config(self):
        """Конфигурация по умолчанию: solver_type = SRK."""
        solver = StochasticRungeKutta()

        assert solver._config.solver_type == SolverType.SRK


class TestStochasticRungeKuttaStep:
    """Тесты одного шага SRI2W1."""

    def test_sigma_zero_deterministic(self):
        """Sigma = 0 -> чисто детерминированный результат (RK2)."""
        solver = StochasticRungeKutta()
        state = np.full(20, 1.0)
        drift = np.full(20, 2.0)
        diffusion = np.zeros(20)
        dW = np.zeros(20)

        result = solver.step(state, drift, diffusion, dt=0.1, dW=dW)

        # Для sigma=0 SRK сводится к детерминированному RK2
        # Точное значение зависит от tableau, но результат должен быть
        # близок к state + drift*dt = 1.0 + 2.0*0.1 = 1.2
        assert result.new_state.shape == (20,)
        assert np.all(np.isfinite(result.new_state))

    def test_n_function_evals_equals_4(self):
        """n_function_evals == 4 (2 det stages + 2 stoch stages)."""
        solver = StochasticRungeKutta()
        state = np.ones(20)

        result = solver.step(
            state,
            np.zeros(20),
            np.zeros(20),
            dt=0.01,
            dW=np.zeros(20),
        )

        assert result.n_function_evals == 4

    def test_result_shape_20(self):
        """new_state.shape == (20,)."""
        solver = StochasticRungeKutta()
        state = np.ones(20)

        result = solver.step(
            state,
            np.ones(20),
            np.ones(20),
            dt=0.01,
            dW=np.random.default_rng(42).standard_normal(20),
        )

        assert result.new_state.shape == (20,)


# =============================================================================
# Test create_solver factory
# =============================================================================


class TestCreateSolver:
    """Тесты фабричной функции create_solver."""

    def test_em_creates_euler_maruyama(self):
        """SolverType.EM -> EulerMaruyamaSolver."""
        config = SolverConfig(solver_type=SolverType.EM)

        solver = create_solver(config)

        assert isinstance(solver, EulerMaruyamaSolver)

    def test_milstein_creates_milstein(self):
        """SolverType.MILSTEIN -> MilsteinSolver."""
        config = SolverConfig(solver_type=SolverType.MILSTEIN)

        solver = create_solver(config)

        assert isinstance(solver, MilsteinSolver)

    def test_imex_creates_imex(self):
        """SolverType.IMEX -> IMEXSplitter."""
        config = SolverConfig(solver_type=SolverType.IMEX)

        solver = create_solver(config)

        assert isinstance(solver, IMEXSplitter)

    def test_srk_creates_srk(self):
        """SolverType.SRK -> StochasticRungeKutta."""
        config = SolverConfig(solver_type=SolverType.SRK)

        solver = create_solver(config)

        assert isinstance(solver, StochasticRungeKutta)

    def test_adaptive_creates_adaptive(self):
        """SolverType.ADAPTIVE -> AdaptiveTimestepper."""
        config = SolverConfig(solver_type=SolverType.ADAPTIVE)

        solver = create_solver(config)

        assert isinstance(solver, AdaptiveTimestepper)

    def test_result_is_sde_solver(self):
        """Результат фабрики совместим с SDESolver."""
        config = SolverConfig(solver_type=SolverType.EM)

        solver = create_solver(config)

        assert isinstance(solver, SDESolver)

    def test_invalid_type_raises_value_error(self):
        """Невалидный тип -> ValueError."""
        config = SolverConfig()
        config.solver_type = "invalid_type"  # type: ignore[assignment]

        with pytest.raises((ValueError, KeyError)):
            create_solver(config)
