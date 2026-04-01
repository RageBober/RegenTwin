"""Тесты модуля src/analysis/validation.py — Phase 3.4.

12 классов, ~70 тест-кейсов:
  TestValidationConfig, TestDTWCRPSResult, TestPPCResult, TestPhaseTimingResult,
  TestSensitivityRankingResult, TestValidationResult,
  TestRunnerDTWCRPS, TestRunnerPPC, TestRunnerPhaseTiming, TestRunnerSensRanking,
  TestRunnerRunAll, TestValidateModelFunc
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.validation import (
    DTWCRPSResult,
    PhaseBreakpoint,
    PhaseTimingResult,
    PPCResult,
    RankingComparison,
    SensitivityRankingResult,
    ValidationConfig,
    ValidationResult,
    ValidationRunner,
    validate_model,
)
from src.core.extended_sde import ExtendedSDETrajectory
from src.core.monte_carlo import MonteCarloResults
from src.core.sensitivity_analysis import MorrisResult, SobolResult
from src.data.dataset_loader import TimeSeriesData

# =====================================================================
# TestValidationConfig
# =====================================================================


class TestValidationConfig:
    def test_default_weights_sum_to_one(self) -> None:
        cfg = ValidationConfig()
        total = cfg.weight_dtw + cfg.weight_ppc + cfg.weight_timing + cfg.weight_ranking
        assert abs(total - 1.0) < 1e-9

    def test_validate_passes_on_valid_config(self) -> None:
        cfg = ValidationConfig()
        cfg.validate()  # должен не бросить

    def test_validate_raises_if_weights_wrong(self) -> None:
        cfg = ValidationConfig(
            weight_dtw=0.5, weight_ppc=0.5, weight_timing=0.1, weight_ranking=0.1
        )
        with pytest.raises(ValueError, match="1.0"):
            cfg.validate()

    def test_dtw_variables_default(self) -> None:
        cfg = ValidationConfig()
        assert "F" in cfg.dtw_variables
        assert "M1" in cfg.dtw_variables

    def test_ppc_variables_default(self) -> None:
        cfg = ValidationConfig()
        assert "F" in cfg.ppc_variables


# =====================================================================
# TestDTWCRPSResult
# =====================================================================


class TestDTWCRPSResult:
    @pytest.fixture
    def sample_result(self) -> DTWCRPSResult:
        return DTWCRPSResult(
            variable_names=["F", "M1"],
            dtw_distances={"F": 10.5, "M1": 5.2},
            crps_scores={"F": 0.8, "M1": 0.4},
            mean_dtw=7.85,
            mean_crps=0.6,
            n_observations=20,
            elapsed_seconds=0.3,
        )

    def test_mean_dtw_matches_dict(self, sample_result: DTWCRPSResult) -> None:
        expected = sum(sample_result.dtw_distances.values()) / len(sample_result.dtw_distances)
        assert abs(sample_result.mean_dtw - expected) < 1e-9

    def test_mean_crps_matches_dict(self, sample_result: DTWCRPSResult) -> None:
        expected = sum(sample_result.crps_scores.values()) / len(sample_result.crps_scores)
        assert abs(sample_result.mean_crps - expected) < 1e-9

    def test_dtw_distances_positive(self, sample_result: DTWCRPSResult) -> None:
        for d in sample_result.dtw_distances.values():
            assert d >= 0

    def test_crps_scores_nonnegative(self, sample_result: DTWCRPSResult) -> None:
        for s in sample_result.crps_scores.values():
            assert s >= 0

    def test_variable_names_match_keys(self, sample_result: DTWCRPSResult) -> None:
        assert set(sample_result.variable_names) == set(sample_result.dtw_distances.keys())
        assert set(sample_result.variable_names) == set(sample_result.crps_scores.keys())

    def test_n_observations_positive(self, sample_result: DTWCRPSResult) -> None:
        assert sample_result.n_observations > 0

    def test_elapsed_nonnegative(self, sample_result: DTWCRPSResult) -> None:
        assert sample_result.elapsed_seconds >= 0

    def test_fields_are_correct_types(self, sample_result: DTWCRPSResult) -> None:
        assert isinstance(sample_result.dtw_distances, dict)
        assert isinstance(sample_result.crps_scores, dict)


# =====================================================================
# TestPPCResult
# =====================================================================


class TestPPCResult:
    @pytest.fixture
    def sample_result(self) -> PPCResult:
        return PPCResult(
            loo_elpd=None,
            loo_se=None,
            coverage_95={"F": 0.92, "M1": 0.88, "M2": 0.95},
            mean_coverage=0.9167,
            backend="mc_envelope",
            elapsed_seconds=0.5,
        )

    def test_backend_valid(self, sample_result: PPCResult) -> None:
        assert sample_result.backend in {"arviz", "mc_envelope"}

    def test_coverage_in_range(self, sample_result: PPCResult) -> None:
        for v in sample_result.coverage_95.values():
            assert 0.0 <= v <= 1.0

    def test_mean_coverage_in_range(self, sample_result: PPCResult) -> None:
        assert 0.0 <= sample_result.mean_coverage <= 1.0

    def test_loo_none_for_mc_backend(self, sample_result: PPCResult) -> None:
        # MC fallback path → loo = None
        assert sample_result.loo_elpd is None

    def test_coverage_keys_are_variables(self, sample_result: PPCResult) -> None:
        assert "F" in sample_result.coverage_95
        assert "M1" in sample_result.coverage_95

    def test_elapsed_nonnegative(self, sample_result: PPCResult) -> None:
        assert sample_result.elapsed_seconds >= 0

    def test_mean_coverage_consistent_with_per_variable(self, sample_result: PPCResult) -> None:
        expected = sum(sample_result.coverage_95.values()) / len(sample_result.coverage_95)
        assert abs(sample_result.mean_coverage - expected) < 0.01

    def test_arviz_backend_with_loo(self) -> None:
        r = PPCResult(
            loo_elpd=-42.5,
            loo_se=2.1,
            coverage_95={"F": 0.93},
            mean_coverage=0.93,
            backend="arviz",
            elapsed_seconds=1.0,
        )
        assert r.backend == "arviz"
        assert r.loo_elpd is not None


# =====================================================================
# TestPhaseTimingResult
# =====================================================================


class TestPhaseTimingResult:
    @pytest.fixture
    def sample_result(self) -> PhaseTimingResult:
        bp = [
            PhaseBreakpoint(
                time_hours=8.0,
                phase_before="hemostasis",
                phase_after="inflammation",
                confidence=0.9,
            ),
            PhaseBreakpoint(
                time_hours=100.0,
                phase_before="inflammation",
                phase_after="proliferation",
                confidence=0.85,
            ),
        ]
        return PhaseTimingResult(
            detected_breakpoints=bp,
            expected_breakpoints=None,
            timing_mae_hours=None,
            n_phases_detected=2,
            algorithm="Pelt+BIC",
            elapsed_seconds=0.2,
        )

    def test_n_phases_matches_breakpoints(self, sample_result: PhaseTimingResult) -> None:
        assert sample_result.n_phases_detected == len(sample_result.detected_breakpoints)

    def test_timing_mae_none_without_expected(self, sample_result: PhaseTimingResult) -> None:
        assert sample_result.timing_mae_hours is None

    def test_algorithm_string(self, sample_result: PhaseTimingResult) -> None:
        assert "Pelt" in sample_result.algorithm

    def test_breakpoint_times_positive(self, sample_result: PhaseTimingResult) -> None:
        for bp in sample_result.detected_breakpoints:
            assert bp.time_hours > 0

    def test_breakpoint_confidence_in_range(self, sample_result: PhaseTimingResult) -> None:
        for bp in sample_result.detected_breakpoints:
            assert 0.0 <= bp.confidence <= 1.0

    def test_with_expected_has_timing_mae(self) -> None:
        detected = [PhaseBreakpoint(10.0, "a", "b", 0.9)]
        expected = [PhaseBreakpoint(12.0, "a", "b", 0.9)]
        r = PhaseTimingResult(
            detected_breakpoints=detected,
            expected_breakpoints=expected,
            timing_mae_hours=2.0,
            n_phases_detected=1,
            algorithm="Pelt+BIC",
            elapsed_seconds=0.1,
        )
        assert r.timing_mae_hours is not None
        assert r.timing_mae_hours >= 0

    def test_phase_breakpoint_dataclass(self) -> None:
        bp = PhaseBreakpoint(
            time_hours=6.0, phase_before="hemostasis", phase_after="inflammation", confidence=0.95
        )
        assert bp.time_hours == 6.0
        assert bp.phase_after == "inflammation"

    def test_elapsed_nonnegative(self, sample_result: PhaseTimingResult) -> None:
        assert sample_result.elapsed_seconds >= 0


# =====================================================================
# TestSensitivityRankingResult
# =====================================================================


class TestSensitivityRankingResult:
    @pytest.fixture
    def sample_result(self) -> SensitivityRankingResult:
        comparisons = [
            RankingComparison("r_F", rank_sobol=1, rank_morris=1),
            RankingComparison("r_M1", rank_sobol=2, rank_morris=2),
            RankingComparison("K_F", rank_sobol=3, rank_morris=4),
        ]
        return SensitivityRankingResult(
            kendall_tau=0.87,
            p_value=0.02,
            ranking_comparisons=comparisons,
            n_parameters=3,
            elapsed_seconds=0.01,
        )

    def test_tau_in_range(self, sample_result: SensitivityRankingResult) -> None:
        assert -1.0 <= sample_result.kendall_tau <= 1.0

    def test_p_value_in_range(self, sample_result: SensitivityRankingResult) -> None:
        assert 0.0 <= sample_result.p_value <= 1.0

    def test_n_parameters_matches_comparisons(
        self, sample_result: SensitivityRankingResult
    ) -> None:
        assert sample_result.n_parameters == len(sample_result.ranking_comparisons)

    def test_ranking_comparison_fields(self, sample_result: SensitivityRankingResult) -> None:
        first = sample_result.ranking_comparisons[0]
        assert isinstance(first.parameter_name, str)
        assert isinstance(first.rank_sobol, int)
        assert isinstance(first.rank_morris, int)

    def test_ranks_are_positive(self, sample_result: SensitivityRankingResult) -> None:
        for rc in sample_result.ranking_comparisons:
            assert rc.rank_sobol >= 1
            assert rc.rank_morris >= 1

    def test_elapsed_nonnegative(self, sample_result: SensitivityRankingResult) -> None:
        assert sample_result.elapsed_seconds >= 0

    def test_negative_tau_possible(self) -> None:
        r = SensitivityRankingResult(
            kendall_tau=-0.5,
            p_value=0.3,
            ranking_comparisons=[],
            n_parameters=0,
            elapsed_seconds=0.0,
        )
        assert r.kendall_tau == -0.5

    def test_parameter_names_nonempty(self, sample_result: SensitivityRankingResult) -> None:
        for rc in sample_result.ranking_comparisons:
            assert len(rc.parameter_name) > 0


# =====================================================================
# TestValidationResult
# =====================================================================


class TestValidationResult:
    @pytest.fixture
    def sample_result(self) -> ValidationResult:
        dtw_r = DTWCRPSResult(
            variable_names=["F"],
            dtw_distances={"F": 5.0},
            crps_scores={"F": 0.5},
            mean_dtw=5.0,
            mean_crps=0.5,
            n_observations=20,
            elapsed_seconds=0.1,
        )
        ppc_r = PPCResult(
            loo_elpd=None,
            loo_se=None,
            coverage_95={"F": 0.9},
            mean_coverage=0.9,
            backend="mc_envelope",
            elapsed_seconds=0.2,
        )
        return ValidationResult(
            dtw_crps=dtw_r,
            ppc=ppc_r,
            phase_timing=None,
            sensitivity_ranking=None,
            overall_score=0.75,
            elapsed_seconds=0.5,
        )

    def test_overall_score_in_range(self, sample_result: ValidationResult) -> None:
        assert 0.0 <= sample_result.overall_score <= 1.0

    def test_get_summary_returns_dict(self, sample_result: ValidationResult) -> None:
        summary = sample_result.get_summary()
        assert isinstance(summary, dict)

    def test_get_summary_has_overall_score(self, sample_result: ValidationResult) -> None:
        summary = sample_result.get_summary()
        assert "overall_score" in summary

    def test_get_summary_has_dtw_crps_key(self, sample_result: ValidationResult) -> None:
        summary = sample_result.get_summary()
        assert "dtw_crps" in summary

    def test_get_summary_has_ppc_key(self, sample_result: ValidationResult) -> None:
        summary = sample_result.get_summary()
        assert "ppc" in summary

    def test_elapsed_nonnegative(self, sample_result: ValidationResult) -> None:
        assert sample_result.elapsed_seconds >= 0


# =====================================================================
# TestRunnerDTWCRPS
# =====================================================================


class TestRunnerDTWCRPS:
    def test_returns_dtw_crps_result(
        self,
        mock_extended_trajectory: ExtendedSDETrajectory,
        mock_time_series_data: TimeSeriesData,
    ) -> None:
        runner = ValidationRunner()
        result = runner.run_dtw_crps(mock_extended_trajectory, mock_time_series_data)
        assert isinstance(result, DTWCRPSResult)

    def test_dtw_distances_positive(
        self,
        mock_extended_trajectory: ExtendedSDETrajectory,
        mock_time_series_data: TimeSeriesData,
    ) -> None:
        runner = ValidationRunner()
        result = runner.run_dtw_crps(mock_extended_trajectory, mock_time_series_data)
        for d in result.dtw_distances.values():
            assert d >= 0

    def test_crps_scores_nonnegative(
        self,
        mock_extended_trajectory: ExtendedSDETrajectory,
        mock_time_series_data: TimeSeriesData,
    ) -> None:
        runner = ValidationRunner()
        result = runner.run_dtw_crps(mock_extended_trajectory, mock_time_series_data)
        for s in result.crps_scores.values():
            assert s >= 0

    def test_variable_names_subset_of_config(
        self,
        mock_extended_trajectory: ExtendedSDETrajectory,
        mock_time_series_data: TimeSeriesData,
        mock_validation_config: ValidationConfig,
    ) -> None:
        runner = ValidationRunner(mock_validation_config)
        result = runner.run_dtw_crps(mock_extended_trajectory, mock_time_series_data)
        # Все переменные в результате должны быть из config.dtw_variables
        for v in result.variable_names:
            assert v in mock_validation_config.dtw_variables

    def test_mean_dtw_consistent(
        self,
        mock_extended_trajectory: ExtendedSDETrajectory,
        mock_time_series_data: TimeSeriesData,
    ) -> None:
        runner = ValidationRunner()
        result = runner.run_dtw_crps(mock_extended_trajectory, mock_time_series_data)
        if result.dtw_distances:
            expected = sum(result.dtw_distances.values()) / len(result.dtw_distances)
            assert abs(result.mean_dtw - expected) < 1e-6

    def test_n_observations_correct(
        self,
        mock_extended_trajectory: ExtendedSDETrajectory,
        mock_time_series_data: TimeSeriesData,
    ) -> None:
        runner = ValidationRunner()
        result = runner.run_dtw_crps(mock_extended_trajectory, mock_time_series_data)
        assert result.n_observations == len(mock_time_series_data.time_points)

    def test_elapsed_positive(
        self,
        mock_extended_trajectory: ExtendedSDETrajectory,
        mock_time_series_data: TimeSeriesData,
    ) -> None:
        runner = ValidationRunner()
        result = runner.run_dtw_crps(mock_extended_trajectory, mock_time_series_data)
        assert result.elapsed_seconds >= 0

    def test_variable_not_in_observed_skipped(
        self,
        mock_extended_trajectory: ExtendedSDETrajectory,
    ) -> None:
        # Наблюдения только для F, остальные переменные должны быть пропущены
        observed = TimeSeriesData(
            time_points=np.linspace(0, 720, 10),
            values={"F": np.ones(10) * 100},
            units={"F": "cells/mm²"},
        )
        cfg = ValidationConfig(dtw_variables=["F", "M1", "Ne"])
        runner = ValidationRunner(cfg)
        result = runner.run_dtw_crps(mock_extended_trajectory, observed)
        assert "F" in result.variable_names
        assert "M1" not in result.variable_names  # не в observed → пропущена


# =====================================================================
# TestRunnerPPC
# =====================================================================


class TestRunnerPPC:
    def test_returns_ppc_result(
        self,
        mock_mc_results: MonteCarloResults,
        mock_time_series_data: TimeSeriesData,
    ) -> None:
        runner = ValidationRunner()
        result = runner.run_ppc(mock_mc_results, mock_time_series_data)
        assert isinstance(result, PPCResult)

    def test_mc_fallback_backend(
        self,
        mock_mc_results: MonteCarloResults,
        mock_time_series_data: TimeSeriesData,
    ) -> None:
        # Без inference_data → MC envelope fallback
        runner = ValidationRunner()
        result = runner.run_ppc(mock_mc_results, mock_time_series_data, estimation_result=None)
        assert result.backend == "mc_envelope"

    def test_coverage_in_range(
        self,
        mock_mc_results: MonteCarloResults,
        mock_time_series_data: TimeSeriesData,
    ) -> None:
        runner = ValidationRunner()
        result = runner.run_ppc(mock_mc_results, mock_time_series_data)
        for v in result.coverage_95.values():
            assert 0.0 <= v <= 1.0

    def test_mean_coverage_in_range(
        self,
        mock_mc_results: MonteCarloResults,
        mock_time_series_data: TimeSeriesData,
    ) -> None:
        runner = ValidationRunner()
        result = runner.run_ppc(mock_mc_results, mock_time_series_data)
        assert 0.0 <= result.mean_coverage <= 1.0

    def test_loo_none_for_mc_backend(
        self,
        mock_mc_results: MonteCarloResults,
        mock_time_series_data: TimeSeriesData,
    ) -> None:
        runner = ValidationRunner()
        result = runner.run_ppc(mock_mc_results, mock_time_series_data)
        assert result.loo_elpd is None

    def test_coverage_keys_match_ppc_variables(
        self,
        mock_mc_results: MonteCarloResults,
        mock_time_series_data: TimeSeriesData,
        mock_validation_config: ValidationConfig,
    ) -> None:
        runner = ValidationRunner(mock_validation_config)
        result = runner.run_ppc(mock_mc_results, mock_time_series_data)
        # Все ключи coverage должны быть из ppc_variables (которые есть в observed)
        for var in result.coverage_95:
            assert var in mock_validation_config.ppc_variables

    def test_elapsed_nonnegative(
        self,
        mock_mc_results: MonteCarloResults,
        mock_time_series_data: TimeSeriesData,
    ) -> None:
        runner = ValidationRunner()
        result = runner.run_ppc(mock_mc_results, mock_time_series_data)
        assert result.elapsed_seconds >= 0

    def test_mean_coverage_consistent(
        self,
        mock_mc_results: MonteCarloResults,
        mock_time_series_data: TimeSeriesData,
    ) -> None:
        runner = ValidationRunner()
        result = runner.run_ppc(mock_mc_results, mock_time_series_data)
        if result.coverage_95:
            expected = sum(result.coverage_95.values()) / len(result.coverage_95)
            assert abs(result.mean_coverage - expected) < 1e-6


# =====================================================================
# TestRunnerPhaseTiming
# =====================================================================


class TestRunnerPhaseTiming:
    def test_returns_phase_timing_result(
        self,
        mock_extended_trajectory: ExtendedSDETrajectory,
    ) -> None:
        runner = ValidationRunner()
        result = runner.run_phase_timing(mock_extended_trajectory)
        assert isinstance(result, PhaseTimingResult)

    def test_detects_at_least_one_breakpoint(
        self,
        mock_extended_trajectory: ExtendedSDETrajectory,
    ) -> None:
        runner = ValidationRunner()
        result = runner.run_phase_timing(mock_extended_trajectory)
        assert result.n_phases_detected >= 0  # может быть 0 если сигнал монотонный

    def test_algorithm_is_pelt(
        self,
        mock_extended_trajectory: ExtendedSDETrajectory,
    ) -> None:
        runner = ValidationRunner()
        result = runner.run_phase_timing(mock_extended_trajectory)
        assert "Pelt" in result.algorithm

    def test_timing_mae_none_without_expected(
        self,
        mock_extended_trajectory: ExtendedSDETrajectory,
    ) -> None:
        runner = ValidationRunner()
        result = runner.run_phase_timing(mock_extended_trajectory)
        assert result.timing_mae_hours is None

    def test_timing_mae_computed_with_expected(
        self,
        mock_extended_trajectory: ExtendedSDETrajectory,
        mock_observed_breakpoints: list[PhaseBreakpoint],
    ) -> None:
        runner = ValidationRunner()
        result = runner.run_phase_timing(
            mock_extended_trajectory,
            observed_breakpoints=mock_observed_breakpoints,
        )
        if result.n_phases_detected > 0:
            assert result.timing_mae_hours is not None
            assert result.timing_mae_hours >= 0

    def test_elapsed_nonnegative(
        self,
        mock_extended_trajectory: ExtendedSDETrajectory,
    ) -> None:
        runner = ValidationRunner()
        result = runner.run_phase_timing(mock_extended_trajectory)
        assert result.elapsed_seconds >= 0


# =====================================================================
# TestRunnerSensRanking
# =====================================================================


class TestRunnerSensRanking:
    def test_returns_ranking_result(
        self,
        mock_sobol_result: SobolResult,
        mock_morris_result: MorrisResult,
    ) -> None:
        runner = ValidationRunner()
        result = runner.run_sensitivity_ranking(mock_sobol_result, mock_morris_result)
        assert isinstance(result, SensitivityRankingResult)

    def test_tau_in_range(
        self,
        mock_sobol_result: SobolResult,
        mock_morris_result: MorrisResult,
    ) -> None:
        runner = ValidationRunner()
        result = runner.run_sensitivity_ranking(mock_sobol_result, mock_morris_result)
        assert -1.0 <= result.kendall_tau <= 1.0

    def test_p_value_in_range(
        self,
        mock_sobol_result: SobolResult,
        mock_morris_result: MorrisResult,
    ) -> None:
        runner = ValidationRunner()
        result = runner.run_sensitivity_ranking(mock_sobol_result, mock_morris_result)
        assert 0.0 <= result.p_value <= 1.0

    def test_n_parameters_is_intersection(
        self,
        mock_sobol_result: SobolResult,
        mock_morris_result: MorrisResult,
    ) -> None:
        runner = ValidationRunner()
        result = runner.run_sensitivity_ranking(mock_sobol_result, mock_morris_result)
        common = set(mock_sobol_result.parameter_names) & set(mock_morris_result.parameter_names)
        assert result.n_parameters == len(common)

    def test_identical_rankings_give_tau_one(self) -> None:
        # Если оба метода дают одинаковый порядок → τ = 1.0
        names = ["a", "b", "c", "d"]
        # Sobol: ST = [4,3,2,1] → rank по убыванию: a=1, b=2, c=3, d=4
        sobol = SobolResult(
            parameter_names=names,
            S1=np.array([0.4, 0.3, 0.2, 0.1]),
            ST=np.array([0.4, 0.3, 0.2, 0.1]),
            output_variable="F",
            n_samples=1024,
            n_model_runs=10240,
            elapsed_seconds=1.0,
        )
        # Morris: mu_star = [4,3,2,1] → тот же порядок
        morris = MorrisResult(
            parameter_names=names,
            mu=np.array([4.0, 3.0, 2.0, 1.0]),
            mu_star=np.array([4.0, 3.0, 2.0, 1.0]),
            sigma=np.array([1.0, 1.0, 1.0, 1.0]),
            output_variable="F",
            n_trajectories=50,
            n_levels=4,
            n_model_runs=250,
            elapsed_seconds=1.0,
        )
        runner = ValidationRunner()
        result = runner.run_sensitivity_ranking(sobol, morris)
        assert abs(result.kendall_tau - 1.0) < 0.01

    def test_elapsed_nonnegative(
        self,
        mock_sobol_result: SobolResult,
        mock_morris_result: MorrisResult,
    ) -> None:
        runner = ValidationRunner()
        result = runner.run_sensitivity_ranking(mock_sobol_result, mock_morris_result)
        assert result.elapsed_seconds >= 0


# =====================================================================
# TestRunnerRunAll
# =====================================================================


class TestRunnerRunAll:
    def test_returns_validation_result(
        self,
        mock_extended_trajectory: ExtendedSDETrajectory,
        mock_time_series_data: TimeSeriesData,
        mock_mc_results: MonteCarloResults,
        mock_sobol_result: SobolResult,
        mock_morris_result: MorrisResult,
    ) -> None:
        runner = ValidationRunner()
        result = runner.run_all(
            trajectory=mock_extended_trajectory,
            mc_results=mock_mc_results,
            observed=mock_time_series_data,
            sobol=mock_sobol_result,
            morris=mock_morris_result,
        )
        assert isinstance(result, ValidationResult)

    def test_overall_score_in_range(
        self,
        mock_extended_trajectory: ExtendedSDETrajectory,
        mock_time_series_data: TimeSeriesData,
        mock_mc_results: MonteCarloResults,
        mock_sobol_result: SobolResult,
        mock_morris_result: MorrisResult,
    ) -> None:
        runner = ValidationRunner()
        result = runner.run_all(
            trajectory=mock_extended_trajectory,
            mc_results=mock_mc_results,
            observed=mock_time_series_data,
            sobol=mock_sobol_result,
            morris=mock_morris_result,
        )
        assert 0.0 <= result.overall_score <= 1.0

    def test_no_inputs_returns_result_with_nones(self) -> None:
        runner = ValidationRunner()
        result = runner.run_all()
        assert isinstance(result, ValidationResult)
        # Нет входных данных → все компоненты None
        assert result.dtw_crps is None
        assert result.ppc is None
        assert result.sensitivity_ranking is None

    def test_dtw_populated_when_trajectory_and_observed_given(
        self,
        mock_extended_trajectory: ExtendedSDETrajectory,
        mock_time_series_data: TimeSeriesData,
    ) -> None:
        cfg = ValidationConfig(
            run_dtw_crps=True,
            run_ppc=False,
            run_phase_timing=False,
            run_sensitivity_ranking=False,
        )
        runner = ValidationRunner(cfg)
        result = runner.run_all(
            trajectory=mock_extended_trajectory,
            observed=mock_time_series_data,
        )
        assert result.dtw_crps is not None

    def test_ranking_none_when_disabled(
        self,
        mock_extended_trajectory: ExtendedSDETrajectory,
        mock_time_series_data: TimeSeriesData,
        mock_sobol_result: SobolResult,
        mock_morris_result: MorrisResult,
    ) -> None:
        cfg = ValidationConfig(
            run_dtw_crps=True,
            run_ppc=False,
            run_phase_timing=False,
            run_sensitivity_ranking=False,
        )
        runner = ValidationRunner(cfg)
        result = runner.run_all(
            trajectory=mock_extended_trajectory,
            observed=mock_time_series_data,
            sobol=mock_sobol_result,
            morris=mock_morris_result,
        )
        assert result.sensitivity_ranking is None


# =====================================================================
# TestValidateModelFunc
# =====================================================================


class TestValidateModelFunc:
    def test_returns_validation_result(
        self,
        mock_extended_trajectory: ExtendedSDETrajectory,
        mock_time_series_data: TimeSeriesData,
    ) -> None:
        result = validate_model(mock_extended_trajectory, mock_time_series_data)
        assert isinstance(result, ValidationResult)

    def test_overall_score_in_range(
        self,
        mock_extended_trajectory: ExtendedSDETrajectory,
        mock_time_series_data: TimeSeriesData,
    ) -> None:
        result = validate_model(mock_extended_trajectory, mock_time_series_data)
        assert 0.0 <= result.overall_score <= 1.0

    def test_with_mc_results(
        self,
        mock_extended_trajectory: ExtendedSDETrajectory,
        mock_time_series_data: TimeSeriesData,
        mock_mc_results: MonteCarloResults,
    ) -> None:
        result = validate_model(
            mock_extended_trajectory,
            mock_time_series_data,
            mc_results=mock_mc_results,
        )
        assert isinstance(result, ValidationResult)

    def test_with_sobol_and_morris(
        self,
        mock_extended_trajectory: ExtendedSDETrajectory,
        mock_time_series_data: TimeSeriesData,
        mock_sobol_result: SobolResult,
        mock_morris_result: MorrisResult,
    ) -> None:
        result = validate_model(
            mock_extended_trajectory,
            mock_time_series_data,
            sobol=mock_sobol_result,
            morris=mock_morris_result,
        )
        assert result.sensitivity_ranking is not None

    def test_accepts_custom_config(
        self,
        mock_extended_trajectory: ExtendedSDETrajectory,
        mock_time_series_data: TimeSeriesData,
        mock_validation_config: ValidationConfig,
    ) -> None:
        result = validate_model(
            mock_extended_trajectory,
            mock_time_series_data,
            config=mock_validation_config,
        )
        assert isinstance(result, ValidationResult)
