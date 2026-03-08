"""TDD тесты для модуля верификации робастности robustness.

Тестирование:
- ViolationStats: dataclass, defaults, инварианты
- ConservationReport: dataclass, is_conserved, tolerance
- ConvergenceResult: dataclass, errors/dt_sequence, is_valid
- ComparisonMetrics: dataclass, wasserstein, ks_statistic, is_consistent
- PositivityEnforcer: enforce (клиппинг, статистика), get/reset stats
- NaNHandler: check (NaN/Inf), recover (откат + dt reduction), reset
- ConservationChecker: mass/cytokine balance, report, reset
- ConvergenceVerifier: compute_order (log-log), manufactured_solution (GBM)
- SDEvsABMComparator: compare (W1, KS), wasserstein_distance, summary

Все тесты написаны для stub-реализации (NotImplementedError).
После реализации методов тесты должны проходить.
"""

import numpy as np
import pytest

from src.core.robustness import (
    ComparisonMetrics,
    ConservationChecker,
    ConservationReport,
    ConvergenceResult,
    ConvergenceVerifier,
    NaNHandler,
    PositivityEnforcer,
    SDEvsABMComparator,
    ViolationStats,
)

# =============================================================================
# Test ViolationStats
# =============================================================================


class TestViolationStats:
    """Тесты dataclass ViolationStats."""

    def test_default_count_zero(self):
        """Count по умолчанию == 0."""
        stats = ViolationStats()

        assert stats.count == 0

    def test_default_variables_empty(self):
        """Variables по умолчанию == {}."""
        stats = ViolationStats()

        assert stats.variables == {}

    def test_default_timestamps_empty(self):
        """Timestamps по умолчанию == []."""
        stats = ViolationStats()

        assert stats.timestamps == []

    def test_default_total_clipped_zero(self):
        """total_clipped по умолчанию == 0.0."""
        stats = ViolationStats()

        assert stats.total_clipped == 0.0

    def test_invariant_total_clipped_nonneg(self):
        """Инвариант: total_clipped >= 0."""
        stats = ViolationStats()

        assert stats.total_clipped >= 0


# =============================================================================
# Test ConservationReport
# =============================================================================


class TestConservationReport:
    """Тесты dataclass ConservationReport."""

    def test_exact_balance_is_conserved(self):
        """Точный баланс (все ошибки 0) -> is_conserved == True."""
        report = ConservationReport()

        assert report.is_conserved is True

    def test_mass_error_above_tol_not_conserved(self):
        """mass_error > tolerance -> is_conserved == False."""
        report = ConservationReport(mass_error=0.1, is_conserved=False)

        assert report.is_conserved is False

    def test_default_tolerance(self):
        """Tolerance по умолчанию == 0.05 (5%)."""
        report = ConservationReport()

        assert report.tolerance == 0.05

    def test_default_errors_zero(self):
        """Все ошибки по умолчанию == 0.0."""
        report = ConservationReport()

        assert report.mass_error == 0.0
        assert report.cytokine_error == 0.0
        assert report.ecm_error == 0.0

    def test_details_default_empty(self):
        """Details по умолчанию — пустая строка."""
        report = ConservationReport()

        assert report.details == ""


# =============================================================================
# Test ConvergenceResult
# =============================================================================


class TestConvergenceResult:
    """Тесты dataclass ConvergenceResult."""

    def test_default_estimated_order_zero(self):
        """estimated_order по умолчанию == 0.0."""
        result = ConvergenceResult()

        assert result.estimated_order == 0.0

    def test_errors_dt_same_length(self):
        """len(errors) == len(dt_sequence) для заданных списков."""
        result = ConvergenceResult(
            errors=[0.1, 0.05, 0.025],
            dt_sequence=[0.1, 0.05, 0.025],
        )

        assert len(result.errors) == len(result.dt_sequence)

    def test_is_valid_close_to_reference(self):
        """is_valid == True когда |estimated - reference| < 0.2."""
        result = ConvergenceResult(
            estimated_order=0.9,
            reference_order=1.0,
            is_valid=True,
        )

        assert result.is_valid is True

    def test_is_valid_false_when_far(self):
        """is_valid == False когда |estimated - reference| >= 0.2."""
        result = ConvergenceResult(
            estimated_order=0.5,
            reference_order=1.0,
            is_valid=False,
        )

        assert result.is_valid is False

    def test_default_is_valid_false(self):
        """is_valid по умолчанию == False."""
        result = ConvergenceResult()

        assert result.is_valid is False


# =============================================================================
# Test ComparisonMetrics
# =============================================================================


class TestComparisonMetrics:
    """Тесты dataclass ComparisonMetrics."""

    def test_default_wasserstein_zero(self):
        """wasserstein_distance по умолчанию == 0.0."""
        metrics = ComparisonMetrics()

        assert metrics.wasserstein_distance == 0.0

    def test_invariant_wasserstein_nonneg(self):
        """Инвариант: wasserstein_distance >= 0."""
        metrics = ComparisonMetrics()

        assert metrics.wasserstein_distance >= 0

    def test_invariant_ks_statistic_range(self):
        """Инвариант: 0 <= ks_statistic <= 1."""
        metrics = ComparisonMetrics(ks_statistic=0.5)

        assert 0 <= metrics.ks_statistic <= 1

    def test_invariant_ks_pvalue_range(self):
        """Инвариант: 0 <= ks_pvalue <= 1."""
        metrics = ComparisonMetrics(ks_pvalue=0.95)

        assert 0 <= metrics.ks_pvalue <= 1

    def test_default_not_consistent(self):
        """is_consistent по умолчанию == False."""
        metrics = ComparisonMetrics()

        assert metrics.is_consistent is False


# =============================================================================
# Test PositivityEnforcer
# =============================================================================


class TestPositivityEnforcerInit:
    """Тесты инициализации PositivityEnforcer."""

    def test_init_default_min_value(self):
        """min_value по умолчанию == 0.0."""
        enforcer = PositivityEnforcer()

        assert enforcer._min_value == 0.0

    def test_init_custom_min_value(self):
        """Пользовательский min_value сохраняется."""
        enforcer = PositivityEnforcer(min_value=-1.0)

        assert enforcer._min_value == -1.0

    def test_init_stats_zeroed(self):
        """Статистика инициализирована нулями."""
        enforcer = PositivityEnforcer()

        assert enforcer._stats.count == 0
        assert enforcer._stats.variables == {}
        assert enforcer._stats.timestamps == []
        assert enforcer._stats.total_clipped == 0.0


class TestPositivityEnforcerEnforce:
    """Тесты клиппинга отрицательных значений с накоплением статистики."""

    def test_all_positive_unchanged(self):
        """Все положительные -> без изменений, count == 0."""
        enforcer = PositivityEnforcer()
        state = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = enforcer.enforce(state)

        np.testing.assert_array_equal(result, state)

    def test_single_negative_clipped(self):
        """state[3] = -5 -> state[3] = 0, count = 1, total_clipped = 5."""
        enforcer = PositivityEnforcer()
        state = np.array([1.0, 2.0, 3.0, -5.0, 5.0])

        result = enforcer.enforce(state)

        assert result[3] == pytest.approx(0.0)
        assert result[0] == pytest.approx(1.0)
        stats = enforcer.get_violation_stats()
        assert stats.count == 1
        assert stats.total_clipped == pytest.approx(5.0)

    def test_multiple_negatives_count_matches(self):
        """3 отрицательных -> count == 3."""
        enforcer = PositivityEnforcer()
        state = np.array([-1.0, -2.0, -3.0, 4.0, 5.0])

        enforcer.enforce(state)

        stats = enforcer.get_violation_stats()
        assert stats.count == 3

    def test_cumulative_count_across_calls(self):
        """Повторные вызовы enforce() накапливают count."""
        enforcer = PositivityEnforcer()
        state1 = np.array([-1.0, 2.0, 3.0])
        state2 = np.array([1.0, -2.0, -3.0])

        enforcer.enforce(state1)
        enforcer.enforce(state2)

        stats = enforcer.get_violation_stats()
        assert stats.count == 3  # 1 + 2

    def test_returns_new_array_immutability(self):
        """Возвращает НОВЫЙ массив, не мутирует оригинал."""
        enforcer = PositivityEnforcer()
        state = np.array([-5.0, 10.0, -3.0])
        original = state.copy()

        result = enforcer.enforce(state)

        # Оригинал не изменён
        np.testing.assert_array_equal(state, original)
        # Результат — другой объект
        assert result is not state

    def test_result_all_ge_min_value(self):
        """result[i] >= min_value для всех i (кроме NaN)."""
        enforcer = PositivityEnforcer(min_value=0.0)
        state = np.array([-10.0, -5.0, 0.0, 3.0, -0.001])

        result = enforcer.enforce(state)

        assert np.all(result >= 0.0)

    def test_nan_not_clipped(self):
        """NaN не клипуется, остаётся NaN."""
        enforcer = PositivityEnforcer()
        state = np.array([1.0, np.nan, 3.0])

        result = enforcer.enforce(state)

        assert np.isnan(result[1])
        assert result[0] == pytest.approx(1.0)
        assert result[2] == pytest.approx(3.0)

    def test_total_clipped_sum_of_abs_diffs(self):
        """total_clipped == сумма |original - min_value| для отрицательных."""
        enforcer = PositivityEnforcer(min_value=0.0)
        state = np.array([-3.0, -7.0, 5.0])

        enforcer.enforce(state)

        stats = enforcer.get_violation_stats()
        assert stats.total_clipped == pytest.approx(10.0)  # 3 + 7

    def test_timestamps_recorded(self):
        """Timestamps пополняются при нарушениях."""
        enforcer = PositivityEnforcer()
        state = np.array([-1.0, 2.0])

        enforcer.enforce(state, t=1.5)

        stats = enforcer.get_violation_stats()
        assert 1.5 in stats.timestamps


class TestPositivityEnforcerStats:
    """Тесты get_violation_stats и reset_stats."""

    def test_get_violation_stats_returns_violation_stats(self):
        """get_violation_stats() возвращает экземпляр ViolationStats."""
        enforcer = PositivityEnforcer()

        stats = enforcer.get_violation_stats()

        assert isinstance(stats, ViolationStats)

    def test_reset_stats_zeroes_count(self):
        """После reset_stats() count == 0."""
        enforcer = PositivityEnforcer()
        enforcer.enforce(np.array([-1.0, -2.0]))

        enforcer.reset_stats()

        stats = enforcer.get_violation_stats()
        assert stats.count == 0

    def test_reset_stats_clears_variables(self):
        """После reset_stats() variables == {}."""
        enforcer = PositivityEnforcer()
        enforcer.enforce(np.array([-1.0]))

        enforcer.reset_stats()

        stats = enforcer.get_violation_stats()
        assert stats.variables == {}


# =============================================================================
# Test NaNHandler
# =============================================================================


class TestNaNHandlerInit:
    """Тесты инициализации NaNHandler."""

    def test_init_default_max_recoveries(self):
        """max_recoveries по умолчанию == 10."""
        handler = NaNHandler()

        assert handler._max_recoveries == 10

    def test_init_default_dt_reduction(self):
        """dt_reduction_factor по умолчанию == 0.5."""
        handler = NaNHandler()

        assert handler._dt_reduction_factor == 0.5


class TestNaNHandlerCheck:
    """Тесты проверки NaN/Inf в состоянии."""

    def test_all_finite_returns_false(self):
        """Все конечные -> False."""
        handler = NaNHandler()
        state = np.array([1.0, 2.0, 3.0])

        assert handler.check(state) is False

    def test_one_nan_returns_true(self):
        """Один NaN -> True."""
        handler = NaNHandler()
        state = np.array([1.0, np.nan, 3.0])

        assert handler.check(state) is True

    def test_one_inf_returns_true(self):
        """Один Inf -> True."""
        handler = NaNHandler()
        state = np.array([1.0, np.inf, 3.0])

        assert handler.check(state) is True

    def test_all_nan_returns_true(self):
        """Все NaN -> True."""
        handler = NaNHandler()
        state = np.full(20, np.nan)

        assert handler.check(state) is True


class TestNaNHandlerRecover:
    """Тесты восстановления после NaN/Inf."""

    def test_first_recovery_not_stopped(self):
        """Первое восстановление: should_stop == False."""
        handler = NaNHandler(max_recoveries=10)
        state = np.full(20, np.nan)
        last_valid = np.ones(20)

        _, _, should_stop = handler.recover(state, last_valid, dt=0.1)

        assert should_stop is False

    def test_dt_reduced_by_factor(self):
        """new_dt == dt * dt_reduction_factor."""
        handler = NaNHandler(dt_reduction_factor=0.5)
        state = np.full(20, np.nan)
        last_valid = np.ones(20)

        _, new_dt, _ = handler.recover(state, last_valid, dt=0.1)

        assert new_dt == pytest.approx(0.05)

    def test_recovered_state_equals_last_valid(self):
        """recovered_state == last_valid_state."""
        handler = NaNHandler()
        state = np.full(20, np.nan)
        last_valid = np.arange(20, dtype=float)

        recovered, _, _ = handler.recover(state, last_valid, dt=0.1)

        np.testing.assert_array_equal(recovered, last_valid)

    def test_max_recoveries_reached_should_stop(self):
        """После max_recoveries восстановлений should_stop == True."""
        handler = NaNHandler(max_recoveries=3)
        state = np.full(20, np.nan)
        last_valid = np.ones(20)

        for _ in range(2):
            handler.recover(state, last_valid, dt=0.1)

        _, _, should_stop = handler.recover(state, last_valid, dt=0.1)

        assert should_stop is True

    def test_recovery_count_monotonically_increases(self):
        """get_recovery_count() увеличивается на 1 после каждого recover()."""
        handler = NaNHandler()
        state = np.full(20, np.nan)
        last_valid = np.ones(20)

        handler.recover(state, last_valid, dt=0.1)
        assert handler.get_recovery_count() == 1

        handler.recover(state, last_valid, dt=0.05)
        assert handler.get_recovery_count() == 2


class TestNaNHandlerReset:
    """Тесты сброса счётчика восстановлений."""

    def test_get_recovery_count_after_reset(self):
        """После reset() get_recovery_count() == 0."""
        handler = NaNHandler()
        state = np.full(20, np.nan)
        last_valid = np.ones(20)
        handler.recover(state, last_valid, dt=0.1)

        handler.reset()

        assert handler.get_recovery_count() == 0

    def test_reset_allows_new_recoveries(self):
        """После reset() можно восстанавливаться заново."""
        handler = NaNHandler(max_recoveries=2)
        state = np.full(20, np.nan)
        last_valid = np.ones(20)

        # Исчерпать лимит
        handler.recover(state, last_valid, dt=0.1)
        handler.recover(state, last_valid, dt=0.05)

        handler.reset()

        # Снова можно
        _, _, should_stop = handler.recover(state, last_valid, dt=0.1)
        assert should_stop is False


# =============================================================================
# Test ConservationChecker
# =============================================================================


class TestConservationCheckerInit:
    """Тесты инициализации ConservationChecker."""

    def test_init_default_tolerance(self):
        """Tolerance по умолчанию == 0.05."""
        checker = ConservationChecker()

        assert checker._tolerance == 0.05

    def test_init_empty_reports(self):
        """Reports инициализирован пустым списком."""
        checker = ConservationChecker()

        assert checker._reports == []


class TestConservationCheckerMassBalance:
    """Тесты проверки баланса клеточных популяций."""

    def test_exact_euler_mass_error_approx_zero(self):
        """Точный Euler (delta = (births-deaths)*dt) -> mass_error ~ 0."""
        checker = ConservationChecker()
        births = np.full(8, 10.0)
        deaths = np.full(8, 5.0)
        dt = 0.01
        population_prev = np.full(8, 1000.0)
        # Точный Euler: pop_current = pop_prev + (births - deaths) * dt
        population_curr = population_prev + (births - deaths) * dt

        report = checker.check_mass_balance(
            births,
            deaths,
            population_curr,
            population_prev,
            dt,
        )

        assert report.mass_error == pytest.approx(0.0, abs=1e-10)

    def test_zero_births_zero_deaths_zero_delta(self):
        """births=0, deaths=0, population unchanged -> mass_error == 0."""
        checker = ConservationChecker()
        zeros = np.zeros(8)
        population = np.full(8, 500.0)

        report = checker.check_mass_balance(
            zeros,
            zeros,
            population,
            population,
            dt=0.01,
        )

        assert report.mass_error == pytest.approx(0.0, abs=1e-10)

    def test_returns_conservation_report(self):
        """Возвращает экземпляр ConservationReport."""
        checker = ConservationChecker()

        report = checker.check_mass_balance(
            np.zeros(8),
            np.zeros(8),
            np.ones(8),
            np.ones(8),
            dt=0.01,
        )

        assert isinstance(report, ConservationReport)


class TestConservationCheckerCytokineBalance:
    """Тесты проверки баланса цитокинов."""

    def test_steady_state_error_approx_zero(self):
        """Стационарное состояние (production == degradation) -> error ~ 0."""
        checker = ConservationChecker()
        rates = np.full(7, 2.0)
        concentration = np.full(7, 10.0)

        report = checker.check_cytokine_balance(
            rates,
            rates,
            concentration,
            concentration,
            dt=0.01,
        )

        assert report.cytokine_error == pytest.approx(0.0, abs=1e-10)

    def test_returns_conservation_report(self):
        """Возвращает экземпляр ConservationReport."""
        checker = ConservationChecker()

        report = checker.check_cytokine_balance(
            np.zeros(7),
            np.zeros(7),
            np.ones(7),
            np.ones(7),
            dt=0.01,
        )

        assert isinstance(report, ConservationReport)


class TestConservationCheckerReportReset:
    """Тесты report() и reset()."""

    def test_report_returns_list(self):
        """report() возвращает list[ConservationReport]."""
        checker = ConservationChecker()

        reports = checker.report()

        assert isinstance(reports, list)

    def test_reset_clears_reports(self):
        """После reset() report() возвращает пустой список."""
        checker = ConservationChecker()
        # Добавить отчёт
        checker.check_mass_balance(
            np.zeros(8),
            np.zeros(8),
            np.ones(8),
            np.ones(8),
            dt=0.01,
        )

        checker.reset()

        assert checker.report() == []


# =============================================================================
# Test ConvergenceVerifier
# =============================================================================


class TestConvergenceVerifierInit:
    """Тесты инициализации ConvergenceVerifier."""

    def test_init_default_n_realizations(self):
        """n_realizations по умолчанию == 100."""
        verifier = ConvergenceVerifier()

        assert verifier._n_realizations == 100


class TestConvergenceVerifierComputeOrder:
    """Тесты оценки порядка сходимости через log-log регрессию."""

    def test_linear_errors_order_1(self):
        """Errors = [1, 0.5, 0.25], dt = [1, 0.5, 0.25] -> order ~ 1.0."""
        verifier = ConvergenceVerifier()

        order = verifier.compute_order(
            errors=[1.0, 0.5, 0.25],
            dt_sequence=[1.0, 0.5, 0.25],
        )

        assert order == pytest.approx(1.0, abs=0.1)

    def test_sqrt_errors_order_05(self):
        """Errors ~ sqrt(dt) -> order ~ 0.5."""
        verifier = ConvergenceVerifier()

        order = verifier.compute_order(
            errors=[1.0, 1.0 / np.sqrt(2), 0.5],
            dt_sequence=[1.0, 0.5, 0.25],
        )

        assert order == pytest.approx(0.5, abs=0.1)

    def test_constant_errors_order_0(self):
        """Константные errors -> order ~ 0."""
        verifier = ConvergenceVerifier()

        order = verifier.compute_order(
            errors=[1.0, 1.0, 1.0],
            dt_sequence=[1.0, 0.5, 0.25],
        )

        assert order == pytest.approx(0.0, abs=0.1)


class TestConvergenceVerifierManufacturedSolution:
    """Тесты аналитического решения GBM для MMS."""

    def test_t_zero_returns_x0(self):
        """T = 0 -> x0."""
        verifier = ConvergenceVerifier()

        result = verifier.manufactured_solution(t=0.0, x0=1.0)

        assert result == pytest.approx(1.0)

    def test_mu_zero_sigma_zero_returns_x0(self):
        """mu=0, sigma=0, любое t -> x0."""
        verifier = ConvergenceVerifier()

        result = verifier.manufactured_solution(t=5.0, x0=2.0, mu=0.0, sigma=0.0)

        assert result == pytest.approx(2.0)

    def test_positive_mu_grows(self):
        """T > 0, mu > 0, sigma^2/2 < mu -> результат > x0."""
        verifier = ConvergenceVerifier()

        result = verifier.manufactured_solution(t=1.0, x0=1.0, mu=0.1, sigma=0.05)

        # exp((0.1 - 0.05^2/2) * 1) = exp(0.09875) > 1.0
        assert result > 1.0


class TestConvergenceVerifierVerifySolver:
    """Тесты полной верификации солвера."""

    def test_verify_solver_raises_not_implemented(self):
        """verify_solver() вызывает NotImplementedError для stub."""
        verifier = ConvergenceVerifier()

        with pytest.raises(NotImplementedError):
            verifier.verify_solver(solver=None, reference_order=0.5)


# =============================================================================
# Test SDEvsABMComparator
# =============================================================================


class TestSDEvsABMComparatorInit:
    """Тесты инициализации SDEvsABMComparator."""

    def test_init_default_significance(self):
        """significance_level по умолчанию == 0.05."""
        comparator = SDEvsABMComparator()

        assert comparator._significance_level == 0.05


class TestSDEvsABMComparatorCompare:
    """Тесты сравнения SDE vs ABM выборок."""

    def test_same_arrays_consistent(self):
        """Идентичные массивы -> W1=0, ks_pvalue ~ 1.0, is_consistent=True."""
        comparator = SDEvsABMComparator()
        data = np.random.default_rng(42).standard_normal(1000)

        metrics = comparator.compare(data, data)

        assert metrics.wasserstein_distance == pytest.approx(0.0, abs=1e-10)
        assert metrics.is_consistent is True

    def test_shifted_arrays_inconsistent(self):
        """Сдвиг на 100 -> W1 > 0, ks_pvalue < 0.05, is_consistent=False."""
        comparator = SDEvsABMComparator()
        rng = np.random.default_rng(42)
        sde = rng.standard_normal(1000)
        abm = sde + 100.0  # большой сдвиг

        metrics = comparator.compare(sde, abm)

        assert metrics.wasserstein_distance > 0
        assert metrics.ks_pvalue < 0.05
        assert metrics.is_consistent is False

    def test_empty_arrays_raises_value_error(self):
        """Пустые массивы -> ValueError."""
        comparator = SDEvsABMComparator()

        with pytest.raises(ValueError):
            comparator.compare(np.array([]), np.array([]))


class TestSDEvsABMComparatorWasserstein:
    """Тесты Wasserstein-1 расстояния."""

    def test_same_arrays_distance_zero(self):
        """Идентичные массивы -> W1 == 0."""
        comparator = SDEvsABMComparator()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        w1 = comparator.wasserstein_distance(data, data)

        assert w1 == pytest.approx(0.0)

    def test_different_arrays_distance_positive(self):
        """Различные массивы -> W1 > 0."""
        comparator = SDEvsABMComparator()
        sde = np.array([1.0, 2.0, 3.0])
        abm = np.array([10.0, 20.0, 30.0])

        w1 = comparator.wasserstein_distance(sde, abm)

        assert w1 > 0


class TestSDEvsABMComparatorSummary:
    """Тесты текстового отчёта."""

    def test_summary_returns_string(self):
        """summary() возвращает непустую строку."""
        comparator = SDEvsABMComparator()
        metrics = ComparisonMetrics(
            wasserstein_distance=0.5,
            mean_diff=0.1,
            std_diff=0.05,
            ks_statistic=0.3,
            ks_pvalue=0.8,
            is_consistent=True,
        )

        result = comparator.summary(metrics)

        assert isinstance(result, str)
        assert len(result) > 0
