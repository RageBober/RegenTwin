"""TDD тесты для sensitivity_analysis.py — Phase 3.2 Анализ чувствительности.

Тестирование:
- SensitivityMethod: enum выбора метода
- ParameterBounds: границы варьирования параметра
- SensitivityConfig: конфигурация и валидация
- SobolResult: контейнер результатов Sobol + get_ranking
- MorrisResult: контейнер результатов Morris + get_influential
- LocalSensitivityResult: контейнер локальных результатов + get_ranking
- TornadoData: данные для tornado diagram
- SensitivityAnalyzer: оркестратор (init, _build_salib_problem, _auto_bounds, стабы)
- TornadoPlotter: from_sobol, from_morris, from_local, plot (stub)
- run_sensitivity_analysis: convenience-функция (stub)

Все тесты для stub-методов проверяют NotImplementedError.
xfail тесты фиксируют контракт валидации аргументов для Phase 3 реализации.

Основано на спецификации: Description/Phase3/description_sensitivity_analysis.md
"""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.core.extended_sde import ExtendedSDEModel
from src.core.parameters import ParameterSet
from src.core.sensitivity_analysis import (
    LocalSensitivityResult,
    MorrisResult,
    ParameterBounds,
    SensitivityAnalyzer,
    SensitivityConfig,
    SensitivityMethod,
    SobolResult,
    TornadoData,
    TornadoPlotter,
    run_sensitivity_analysis,
)

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

SAMPLE_PARAM_NAMES = ["r_F", "delta_Ne", "gamma_TNF"]
SAMPLE_BOUNDS = [
    ParameterBounds("r_F", 0.01, 0.06, 0.03),
    ParameterBounds("delta_Ne", 0.02, 0.10, 0.05),
    ParameterBounds("gamma_TNF", 0.2, 1.0, 0.5),
]
DEFAULT_R_F = 0.03
NUM_PARAMETER_SET_FIELDS = len(dataclasses.fields(ParameterSet))


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def base_params() -> ParameterSet:
    """ParameterSet с литературными defaults."""
    return ParameterSet()


@pytest.fixture
def mock_sde_model() -> MagicMock:
    """MagicMock ExtendedSDEModel — хранится, не вызывается при init."""
    return MagicMock(spec=ExtendedSDEModel)


@pytest.fixture
def sample_bounds() -> list[ParameterBounds]:
    """3 валидных ParameterBounds для r_F, delta_Ne, gamma_TNF."""
    return [
        ParameterBounds("r_F", 0.01, 0.06, 0.03),
        ParameterBounds("delta_Ne", 0.02, 0.10, 0.05),
        ParameterBounds("gamma_TNF", 0.2, 1.0, 0.5),
    ]


@pytest.fixture
def valid_config(sample_bounds: list[ParameterBounds]) -> SensitivityConfig:
    """SensitivityConfig, проходящий validate()."""
    return SensitivityConfig(
        parameter_bounds=sample_bounds,
        output_variables=["F"],
    )


@pytest.fixture
def analyzer(
    mock_sde_model: MagicMock,
    base_params: ParameterSet,
    valid_config: SensitivityConfig,
) -> SensitivityAnalyzer:
    """SensitivityAnalyzer с валидной конфигурацией."""
    return SensitivityAnalyzer(mock_sde_model, base_params, valid_config)


@pytest.fixture
def sobol_result_3params() -> SobolResult:
    """SobolResult с 3 параметрами и known индексами."""
    return SobolResult(
        parameter_names=["p1", "p2", "p3"],
        S1=np.array([0.1, 0.3, 0.2]),
        ST=np.array([0.15, 0.5, 0.35]),
        S1_conf=np.array([0.02, 0.05, 0.03]),
        ST_conf=np.array([0.03, 0.08, 0.05]),
        output_variable="F",
        n_samples=1024,
        n_model_runs=8192,
    )


@pytest.fixture
def morris_result_5params() -> MorrisResult:
    """MorrisResult с 5 параметрами для тестирования get_influential."""
    return MorrisResult(
        parameter_names=["p1", "p2", "p3", "p4", "p5"],
        mu=np.array([90.0, 70.0, 3.0, -2.0, 0.5]),
        mu_star=np.array([100.0, 80.0, 5.0, 3.0, 1.0]),
        sigma=np.array([30.0, 25.0, 2.0, 1.0, 0.3]),
        mu_star_conf=np.array([10.0, 8.0, 1.0, 0.5, 0.2]),
        output_variable="F",
        n_trajectories=10,
        n_levels=4,
    )


@pytest.fixture
def local_result_3params() -> LocalSensitivityResult:
    """LocalSensitivityResult с 3 параметрами для get_ranking."""
    return LocalSensitivityResult(
        parameter_names=["p1", "p2", "p3"],
        partial_derivatives=np.array([1.0, -5.0, 3.0]),
        elasticity_indices=np.array([0.5, -2.0, 1.0]),
        nominal_output=100.0,
        delta=0.01,
        output_variable="F",
    )


# =============================================================================
# Тесты по: Description/Phase3/description_sensitivity_analysis.md#SensitivityMethod
# =============================================================================


class TestSensitivityMethod:
    """Тесты для SensitivityMethod enum."""

    def test_creation_from_string_sobol(self) -> None:
        """Тест создания SOBOL из строки 'sobol'."""
        assert SensitivityMethod("sobol") is SensitivityMethod.SOBOL

    def test_creation_from_string_morris(self) -> None:
        """Тест создания MORRIS из строки 'morris'."""
        assert SensitivityMethod("morris") is SensitivityMethod.MORRIS

    def test_creation_from_string_local(self) -> None:
        """Тест создания LOCAL из строки 'local'."""
        assert SensitivityMethod("local") is SensitivityMethod.LOCAL

    def test_all_values_count(self) -> None:
        """Тест что enum содержит ровно 3 члена."""
        assert len(SensitivityMethod) == 3

    def test_invalid_string_raises(self) -> None:
        """Тест что невалидная строка вызывает ValueError."""
        with pytest.raises(ValueError):
            SensitivityMethod("invalid")


# =============================================================================
# Тесты по: Description/Phase3/description_sensitivity_analysis.md#ParameterBounds
# =============================================================================


class TestParameterBounds:
    """Тесты для ParameterBounds dataclass.

    ParameterBounds — plain dataclass без валидации в __init__.
    Валидация происходит в SensitivityConfig.validate().
    """

    def test_valid_creation_with_nominal(self) -> None:
        """Тест создания с корректными значениями и номиналом."""
        b = ParameterBounds("r_F", 0.01, 0.06, 0.03)
        assert b.name == "r_F"
        assert b.lower == 0.01
        assert b.upper == 0.06
        assert b.nominal == 0.03

    def test_creation_without_nominal(self) -> None:
        """Тест создания без номинала — nominal=None по умолчанию."""
        b = ParameterBounds("r_F", 0.01, 0.06)
        assert b.nominal is None

    def test_lower_ge_upper_detected_by_config_validate(self) -> None:
        """Тест что lower >= upper обнаруживается при config.validate()."""
        bounds = [ParameterBounds("r_F", 0.06, 0.01)]
        config = SensitivityConfig(parameter_bounds=bounds)
        with pytest.raises(ValueError, match="lower"):
            config.validate()

    def test_nominal_out_of_range_detected_by_config_validate(self) -> None:
        """Тест что nominal вне [lower, upper] обнаруживается при validate()."""
        bounds = [ParameterBounds("r_F", 0.01, 0.06, 0.1)]
        config = SensitivityConfig(parameter_bounds=bounds)
        with pytest.raises(ValueError, match="nominal"):
            config.validate()

    def test_nonexistent_name_detected_by_config_validate(self) -> None:
        """Тест что несуществующее имя параметра обнаруживается при validate()."""
        bounds = [ParameterBounds("nonexistent", 0.0, 1.0)]
        config = SensitivityConfig(parameter_bounds=bounds)
        with pytest.raises(ValueError, match="not found in ParameterSet"):
            config.validate()


# =============================================================================
# Тесты по: Description/Phase3/description_sensitivity_analysis.md#SensitivityConfig.validate
# =============================================================================


class TestSensitivityConfigValidate:
    """Тесты для SensitivityConfig.validate()."""

    def test_valid_config(self, valid_config: SensitivityConfig) -> None:
        """Тест что валидная конфигурация возвращает True."""
        assert valid_config.validate() is True

    def test_empty_bounds(self) -> None:
        """Тест что пустые parameter_bounds вызывают ValueError."""
        config = SensitivityConfig(parameter_bounds=[])
        with pytest.raises(ValueError, match="must not be empty"):
            config.validate()

    def test_dt_zero(self, sample_bounds: list[ParameterBounds]) -> None:
        """Тест что dt=0 вызывает ValueError."""
        config = SensitivityConfig(parameter_bounds=sample_bounds, dt=0.0)
        with pytest.raises(ValueError, match="dt"):
            config.validate()

    def test_dt_negative(self, sample_bounds: list[ParameterBounds]) -> None:
        """Тест что dt<0 вызывает ValueError."""
        config = SensitivityConfig(parameter_bounds=sample_bounds, dt=-0.01)
        with pytest.raises(ValueError, match="dt"):
            config.validate()

    def test_inverted_t_span(self, sample_bounds: list[ParameterBounds]) -> None:
        """Тест что инвертированный t_span вызывает ValueError."""
        config = SensitivityConfig(parameter_bounds=sample_bounds, t_span=(720.0, 0.0))
        with pytest.raises(ValueError, match="t_span"):
            config.validate()

    def test_equal_t_span(self, sample_bounds: list[ParameterBounds]) -> None:
        """Тест что одинаковые t_span вызывают ValueError."""
        config = SensitivityConfig(parameter_bounds=sample_bounds, t_span=(100.0, 100.0))
        with pytest.raises(ValueError, match="t_span"):
            config.validate()

    def test_invalid_aggregation(self, sample_bounds: list[ParameterBounds]) -> None:
        """Тест что невалидная агрегация вызывает ValueError."""
        config = SensitivityConfig(parameter_bounds=sample_bounds, output_aggregation="median")
        with pytest.raises(ValueError, match="output_aggregation"):
            config.validate()

    def test_empty_output_variables(self, sample_bounds: list[ParameterBounds]) -> None:
        """Тест что пустые output_variables вызывают ValueError."""
        config = SensitivityConfig(parameter_bounds=sample_bounds, output_variables=[])
        with pytest.raises(ValueError, match="output_variables"):
            config.validate()

    def test_lower_ge_upper_in_bounds(self) -> None:
        """Тест что lower >= upper в bounds вызывает ValueError."""
        bounds = [ParameterBounds("r_F", 1.0, 0.5)]
        config = SensitivityConfig(parameter_bounds=bounds)
        with pytest.raises(ValueError, match="lower"):
            config.validate()

    def test_nominal_out_of_range(self) -> None:
        """Тест что nominal вне [lower, upper] вызывает ValueError."""
        bounds = [ParameterBounds("r_F", 0.0, 1.0, 5.0)]
        config = SensitivityConfig(parameter_bounds=bounds)
        with pytest.raises(ValueError, match="nominal"):
            config.validate()

    def test_nonexistent_parameter_name(self) -> None:
        """Тест что несуществующий параметр вызывает ValueError."""
        bounds = [ParameterBounds("fake_param", 0.0, 1.0)]
        config = SensitivityConfig(parameter_bounds=bounds)
        with pytest.raises(ValueError, match="not found in ParameterSet"):
            config.validate()

    @pytest.mark.parametrize("agg", ["final", "mean", "max", "auc"])
    def test_all_valid_aggregations(self, sample_bounds: list[ParameterBounds], agg: str) -> None:
        """Тест что все допустимые агрегации проходят валидацию."""
        config = SensitivityConfig(parameter_bounds=sample_bounds, output_aggregation=agg)
        assert config.validate() is True


# =============================================================================
# Тесты по: Description/Phase3/description_sensitivity_analysis.md#SobolResult.get_ranking
# =============================================================================


class TestSobolResultGetRanking:
    """Тесты для SobolResult.get_ranking()."""

    def test_3_params_sorted_by_st_descending(self, sobol_result_3params: SobolResult) -> None:
        """Тест ранжирования 3 параметров по ST убыванию."""
        ranking = sobol_result_3params.get_ranking()
        assert len(ranking) == 3
        # p2 имеет наибольший ST=0.5
        assert ranking[0][0] == "p2"
        assert ranking[0][1] == pytest.approx(0.3)
        assert ranking[0][2] == pytest.approx(0.5)
        # p3 второй (ST=0.35)
        assert ranking[1][0] == "p3"
        # p1 последний (ST=0.15)
        assert ranking[2][0] == "p1"

    def test_empty_result(self) -> None:
        """Тест что пустой SobolResult возвращает пустой список."""
        result = SobolResult()
        assert result.get_ranking() == []

    def test_equal_st_values(self) -> None:
        """Тест что при одинаковых ST все параметры возвращаются."""
        result = SobolResult(
            parameter_names=["a", "b", "c"],
            S1=np.array([0.1, 0.1, 0.1]),
            ST=np.array([0.3, 0.3, 0.3]),
        )
        ranking = result.get_ranking()
        assert len(ranking) == 3
        names = {r[0] for r in ranking}
        assert names == {"a", "b", "c"}


# =============================================================================
# Тесты по: Description/Phase3/description_sensitivity_analysis.md#MorrisResult.get_influential
# =============================================================================


class TestMorrisResultGetInfluential:
    """Тесты для MorrisResult.get_influential()."""

    def test_5_params_threshold_0_1(self, morris_result_5params: MorrisResult) -> None:
        """Тест отбора влиятельных при threshold_ratio=0.1.

        mu_star=[100, 80, 5, 3, 1], threshold=0.1*100=10.
        Только p1 (100) и p2 (80) >= 10.
        """
        influential = morris_result_5params.get_influential(threshold_ratio=0.1)
        assert influential == ["p1", "p2"]

    def test_all_equal(self) -> None:
        """Тест что при одинаковых mu_star все параметры возвращаются."""
        result = MorrisResult(
            parameter_names=["a", "b", "c"],
            mu=np.array([1.0, 1.0, 1.0]),
            mu_star=np.array([1.0, 1.0, 1.0]),
            sigma=np.array([0.1, 0.1, 0.1]),
        )
        influential = result.get_influential(threshold_ratio=0.5)
        assert len(influential) == 3

    def test_empty_result(self) -> None:
        """Тест что пустой MorrisResult возвращает пустой список."""
        result = MorrisResult()
        assert result.get_influential() == []

    def test_threshold_ratio_zero_raises(self, morris_result_5params: MorrisResult) -> None:
        """Тест что threshold_ratio=0.0 вызывает ValueError."""
        with pytest.raises(ValueError, match="threshold_ratio"):
            morris_result_5params.get_influential(threshold_ratio=0.0)

    def test_threshold_ratio_negative_raises(self, morris_result_5params: MorrisResult) -> None:
        """Тест что отрицательный threshold_ratio вызывает ValueError."""
        with pytest.raises(ValueError, match="threshold_ratio"):
            morris_result_5params.get_influential(threshold_ratio=-0.1)

    def test_threshold_ratio_1_returns_max_only(self, morris_result_5params: MorrisResult) -> None:
        """Тест что threshold_ratio=1.0 возвращает только параметр с max mu_star."""
        influential = morris_result_5params.get_influential(threshold_ratio=1.0)
        assert influential == ["p1"]

    def test_threshold_ratio_above_1_raises(self, morris_result_5params: MorrisResult) -> None:
        """Тест что threshold_ratio > 1.0 вызывает ValueError."""
        with pytest.raises(ValueError, match="threshold_ratio"):
            morris_result_5params.get_influential(threshold_ratio=1.5)


# =============================================================================
# Тесты по: Description/Phase3/description_sensitivity_analysis.md#LocalSensitivityResult.get_ranking
# =============================================================================


class TestLocalSensitivityResultGetRanking:
    """Тесты для LocalSensitivityResult.get_ranking()."""

    def test_3_params_sorted_by_abs_elasticity(
        self, local_result_3params: LocalSensitivityResult
    ) -> None:
        """Тест ранжирования по |elasticity| убыванию.

        elasticity=[0.5, -2.0, 1.0] → |elasticity|=[0.5, 2.0, 1.0].
        Порядок: p2 (2.0), p3 (1.0), p1 (0.5).
        """
        ranking = local_result_3params.get_ranking()
        assert len(ranking) == 3
        # p2 имеет наибольший |elasticity|=2.0
        assert ranking[0][0] == "p2"
        assert ranking[0][1] == pytest.approx(-5.0)
        assert ranking[0][2] == pytest.approx(-2.0)
        # p3 второй
        assert ranking[1][0] == "p3"
        # p1 последний
        assert ranking[2][0] == "p1"

    def test_empty_result(self) -> None:
        """Тест что пустой результат возвращает пустой список."""
        result = LocalSensitivityResult()
        assert result.get_ranking() == []

    def test_zero_elasticities(self) -> None:
        """Тест что нулевые эластичности возвращают все параметры."""
        result = LocalSensitivityResult(
            parameter_names=["a", "b", "c"],
            partial_derivatives=np.array([0.0, 0.0, 0.0]),
            elasticity_indices=np.array([0.0, 0.0, 0.0]),
        )
        ranking = result.get_ranking()
        assert len(ranking) == 3


# =============================================================================
# Тесты по: Description/Phase3/description_sensitivity_analysis.md#TornadoData
# =============================================================================


class TestTornadoData:
    """Тесты для TornadoData dataclass."""

    def test_basic_creation(self) -> None:
        """Тест создания TornadoData со всеми полями."""
        td = TornadoData(
            parameter_names=["a", "b"],
            values=np.array([0.5, 0.3]),
            metric_name="ST",
            title="Test",
        )
        assert td.parameter_names == ["a", "b"]
        assert td.metric_name == "ST"
        assert td.title == "Test"
        assert len(td.values) == 2

    def test_ci_lengths_match_values(self) -> None:
        """Тест что длины CI массивов совпадают с values."""
        vals = np.array([0.5, 0.3, 0.1])
        td = TornadoData(
            parameter_names=["a", "b", "c"],
            values=vals,
            lower_values=np.array([0.4, 0.2, 0.05]),
            upper_values=np.array([0.6, 0.4, 0.15]),
        )
        assert len(td.lower_values) == len(td.values)  # type: ignore[arg-type]
        assert len(td.upper_values) == len(td.values)  # type: ignore[arg-type]

    def test_source_method_stored(self) -> None:
        """Тест что source_method сохраняется корректно."""
        td = TornadoData(source_method=SensitivityMethod.SOBOL)
        assert td.source_method is SensitivityMethod.SOBOL

    def test_default_empty(self) -> None:
        """Тест что TornadoData() по умолчанию пустой."""
        td = TornadoData()
        assert td.parameter_names == []
        assert len(td.values) == 0
        assert td.lower_values is None
        assert td.upper_values is None


# =============================================================================
# Тесты по: Description/Phase3/description_sensitivity_analysis.md#SensitivityAnalyzer.__init__
# =============================================================================


class TestSensitivityAnalyzerInit:
    """Тесты для SensitivityAnalyzer.__init__."""

    def test_valid_creation(self, analyzer: SensitivityAnalyzer, mock_sde_model: MagicMock) -> None:
        """Тест что model, params, config сохраняются корректно."""
        assert analyzer.model is mock_sde_model
        assert isinstance(analyzer.params, ParameterSet)
        assert isinstance(analyzer.config, SensitivityConfig)

    def test_auto_bounds_on_empty_parameter_bounds(
        self,
        mock_sde_model: MagicMock,
        base_params: ParameterSet,
    ) -> None:
        """Тест что пустые bounds автоматически генерируются из ParameterSet.

        ParameterSet содержит NUM_PARAMETER_SET_FIELDS полей, все положительные числа.
        auto_bounds должен сгенерировать bounds для каждого.
        """
        config = SensitivityConfig(parameter_bounds=[])
        analyzer = SensitivityAnalyzer(mock_sde_model, base_params, config)
        assert len(analyzer.config.parameter_bounds) == NUM_PARAMETER_SET_FIELDS

    def test_invalid_config_raises_valueerror(
        self,
        mock_sde_model: MagicMock,
        base_params: ParameterSet,
        sample_bounds: list[ParameterBounds],
    ) -> None:
        """Тест что невалидная конфигурация вызывает ValueError при init."""
        config = SensitivityConfig(parameter_bounds=sample_bounds, dt=-1.0)
        with pytest.raises(ValueError):
            SensitivityAnalyzer(mock_sde_model, base_params, config)


# =============================================================================
# Тесты по: Description/Phase3/description_sensitivity_analysis.md#run_sobol/run_morris/run_local
# =============================================================================


class TestSensitivityAnalyzerMethods:
    """Тесты для вычислительных методов SensitivityAnalyzer."""

    def test_evaluate_model_single_returns_float(
        self, analyzer: SensitivityAnalyzer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Тест что _evaluate_model_single возвращает float."""
        mock_traj = MagicMock()
        mock_traj.get_variable.return_value = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mock_traj.times = np.array([0.0, 180.0, 360.0, 540.0, 720.0])
        mock_model_cls = MagicMock()
        mock_model_cls.return_value.simulate.return_value = mock_traj
        monkeypatch.setattr("src.core.sensitivity_analysis.ExtendedSDEModel", mock_model_cls)
        result = analyzer._evaluate_model_single({"r_F": 0.03}, "F")
        assert isinstance(result, float)
        assert result == pytest.approx(5.0)

    def test_evaluate_model_single_mean_aggregation(
        self, analyzer: SensitivityAnalyzer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Тест агрегации mean."""
        analyzer.config.output_aggregation = "mean"
        mock_traj = MagicMock()
        mock_traj.get_variable.return_value = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mock_traj.times = np.array([0.0, 180.0, 360.0, 540.0, 720.0])
        mock_model_cls = MagicMock()
        mock_model_cls.return_value.simulate.return_value = mock_traj
        monkeypatch.setattr("src.core.sensitivity_analysis.ExtendedSDEModel", mock_model_cls)
        result = analyzer._evaluate_model_single({"r_F": 0.03}, "F")
        assert result == pytest.approx(3.0)

    def test_evaluate_model_batch(
        self, analyzer: SensitivityAnalyzer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Тест batch evaluation."""
        call_count = 0

        def mock_single(param_dict: dict[str, float], _output_variable: str | None = None) -> float:
            nonlocal call_count
            call_count += 1
            return float(sum(param_dict.values()))

        monkeypatch.setattr(analyzer, "_evaluate_model_single", mock_single)
        param_values = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        Y = analyzer._evaluate_model(param_values, "F")
        assert len(Y) == 2
        assert call_count == 2

    def test_evaluate_model_progress_callback(
        self, analyzer: SensitivityAnalyzer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Тест что progress_callback вызывается."""
        monkeypatch.setattr(
            analyzer,
            "_evaluate_model_single",
            lambda *_a, **_kw: 1.0,
        )
        calls: list[tuple[int, int]] = []
        param_values = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        analyzer._evaluate_model(
            param_values, "F", progress_callback=lambda c, t: calls.append((c, t))
        )
        assert calls == [(1, 2), (2, 2)]

    def test_run_sobol_returns_sobol_result(
        self, analyzer: SensitivityAnalyzer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Тест что run_sobol возвращает SobolResult."""
        rng = np.random.default_rng(42)
        monkeypatch.setattr(
            analyzer,
            "_evaluate_model_single",
            lambda *_a, **_kw: float(rng.random()),
        )
        result = analyzer.run_sobol(n_samples=16)
        assert isinstance(result, SobolResult)
        assert len(result.S1) == 3
        assert len(result.ST) == 3
        assert result.n_samples == 16
        assert result.output_variable == "F"

    def test_run_morris_returns_morris_result(
        self, analyzer: SensitivityAnalyzer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Тест что run_morris возвращает MorrisResult."""
        rng = np.random.default_rng(42)
        monkeypatch.setattr(
            analyzer,
            "_evaluate_model_single",
            lambda *_a, **_kw: float(rng.random()),
        )
        result = analyzer.run_morris(n_trajectories=4, n_levels=4)
        assert isinstance(result, MorrisResult)
        assert len(result.mu_star) == 3
        assert result.n_trajectories == 4

    def test_run_local_returns_local_result(
        self, analyzer: SensitivityAnalyzer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Тест что run_local возвращает LocalSensitivityResult."""
        call_idx = {"i": 0}

        def mock_single(*_args: object, **_kwargs: object) -> float:
            call_idx["i"] += 1
            return float(call_idx["i"]) * 10.0

        monkeypatch.setattr(analyzer, "_evaluate_model_single", mock_single)
        result = analyzer.run_local(delta=0.05)
        assert isinstance(result, LocalSensitivityResult)
        assert len(result.partial_derivatives) == 3
        assert len(result.elasticity_indices) == 3
        assert result.delta == 0.05

    # --- Валидация аргументов ---

    def test_run_sobol_n_samples_lt_16(self, analyzer: SensitivityAnalyzer) -> None:
        """Тест что n_samples < 16 вызывает ValueError."""
        with pytest.raises(ValueError):
            analyzer.run_sobol(n_samples=8)

    def test_run_morris_n_trajectories_lt_2(self, analyzer: SensitivityAnalyzer) -> None:
        """Тест что n_trajectories < 2 вызывает ValueError."""
        with pytest.raises(ValueError):
            analyzer.run_morris(n_trajectories=1)

    def test_run_morris_n_levels_lt_2(self, analyzer: SensitivityAnalyzer) -> None:
        """Тест что n_levels < 2 вызывает ValueError."""
        with pytest.raises(ValueError):
            analyzer.run_morris(n_levels=1)

    def test_run_local_delta_le_0(self, analyzer: SensitivityAnalyzer) -> None:
        """Тест что delta <= 0 вызывает ValueError."""
        with pytest.raises(ValueError):
            analyzer.run_local(delta=-0.01)

    def test_run_local_delta_ge_1(self, analyzer: SensitivityAnalyzer) -> None:
        """Тест что delta >= 1 вызывает ValueError."""
        with pytest.raises(ValueError):
            analyzer.run_local(delta=1.0)

    def test_run_sobol_constant_output_returns_zeros(
        self, analyzer: SensitivityAnalyzer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Тест что при Y.std()==0 возвращаются нулевые индексы."""
        monkeypatch.setattr(
            analyzer,
            "_evaluate_model_single",
            lambda *_a, **_kw: 42.0,
        )
        result = analyzer.run_sobol(n_samples=16)
        assert np.all(result.S1 == 0.0)
        assert np.all(result.ST == 0.0)


# =============================================================================
# Тесты по: Description/Phase3/description_sensitivity_analysis.md#_build_salib_problem
# =============================================================================


class TestSensitivityAnalyzerBuildSalibProblem:
    """Тесты для SensitivityAnalyzer._build_salib_problem()."""

    def test_correct_structure(self, analyzer: SensitivityAnalyzer) -> None:
        """Тест что _build_salib_problem возвращает корректный SALib dict."""
        problem = analyzer._build_salib_problem()
        assert problem["num_vars"] == 3
        assert problem["names"] == SAMPLE_PARAM_NAMES
        assert len(problem["bounds"]) == 3

    def test_names_match_bounds_order(self, analyzer: SensitivityAnalyzer) -> None:
        """Тест что порядок имён соответствует порядку bounds."""
        problem = analyzer._build_salib_problem()
        expected_names = [b.name for b in analyzer.config.parameter_bounds]
        assert problem["names"] == expected_names

    def test_bounds_are_lists_of_pairs(self, analyzer: SensitivityAnalyzer) -> None:
        """Тест что каждый элемент bounds — пара [lower, upper]."""
        problem = analyzer._build_salib_problem()
        for i, b in enumerate(analyzer.config.parameter_bounds):
            assert problem["bounds"][i] == [b.lower, b.upper]


# =============================================================================
# Тесты по: Description/Phase3/description_sensitivity_analysis.md#_auto_bounds
# =============================================================================


class TestSensitivityAnalyzerAutoBounds:
    """Тесты для SensitivityAnalyzer._auto_bounds().

    Тестируется через __init__ с пустым config.parameter_bounds.
    """

    def test_generates_bounds_for_all_positive_params(
        self,
        mock_sde_model: MagicMock,
        base_params: ParameterSet,
    ) -> None:
        """Тест что auto_bounds генерирует bounds для всех полей ParameterSet."""
        config = SensitivityConfig(parameter_bounds=[])
        analyzer = SensitivityAnalyzer(mock_sde_model, base_params, config)
        assert len(analyzer.config.parameter_bounds) == NUM_PARAMETER_SET_FIELDS

    def test_bounds_are_50pct_to_200pct(
        self,
        mock_sde_model: MagicMock,
        base_params: ParameterSet,
    ) -> None:
        """Тест что bounds = [0.5 * val, 2.0 * val] для r_F."""
        config = SensitivityConfig(parameter_bounds=[])
        analyzer = SensitivityAnalyzer(mock_sde_model, base_params, config)
        r_f_bound = next(b for b in analyzer.config.parameter_bounds if b.name == "r_F")
        assert r_f_bound.lower == pytest.approx(DEFAULT_R_F * 0.5)
        assert r_f_bound.upper == pytest.approx(DEFAULT_R_F * 2.0)

    def test_all_bounds_have_valid_names(
        self,
        mock_sde_model: MagicMock,
        base_params: ParameterSet,
    ) -> None:
        """Тест что все сгенерированные имена существуют в ParameterSet."""
        config = SensitivityConfig(parameter_bounds=[])
        analyzer = SensitivityAnalyzer(mock_sde_model, base_params, config)
        valid_names = {f.name for f in dataclasses.fields(ParameterSet)}
        for b in analyzer.config.parameter_bounds:
            assert b.name in valid_names, f"'{b.name}' not in ParameterSet"

    def test_nominal_equals_original_value(
        self,
        mock_sde_model: MagicMock,
        base_params: ParameterSet,
    ) -> None:
        """Тест что nominal совпадает с литературным значением из ParameterSet."""
        config = SensitivityConfig(parameter_bounds=[])
        analyzer = SensitivityAnalyzer(mock_sde_model, base_params, config)
        r_f_bound = next(b for b in analyzer.config.parameter_bounds if b.name == "r_F")
        assert r_f_bound.nominal == pytest.approx(DEFAULT_R_F)


# =============================================================================
# Тесты по: Description/Phase3/description_sensitivity_analysis.md#TornadoPlotter.from_sobol
# =============================================================================


class TestTornadoPlotterFromSobol:
    """Тесты для TornadoPlotter.from_sobol()."""

    def test_st_metric_default(self, sobol_result_3params: SobolResult) -> None:
        """Тест конвертации по умолчанию — метрика ST."""
        td = TornadoPlotter.from_sobol(sobol_result_3params)
        assert td.metric_name == "ST"
        # p2 имеет наибольший ST=0.5
        assert td.parameter_names[0] == "p2"

    def test_s1_metric(self, sobol_result_3params: SobolResult) -> None:
        """Тест конвертации с метрикой S1."""
        td = TornadoPlotter.from_sobol(sobol_result_3params, metric="S1")
        assert td.metric_name == "S1"
        # p2 имеет наибольший S1=0.3
        assert td.parameter_names[0] == "p2"

    def test_invalid_metric_raises(self, sobol_result_3params: SobolResult) -> None:
        """Тест что невалидная метрика вызывает ValueError."""
        with pytest.raises(ValueError, match="metric"):
            TornadoPlotter.from_sobol(sobol_result_3params, metric="S2")

    def test_top_n_limits_output(self, sobol_result_3params: SobolResult) -> None:
        """Тест что top_n ограничивает число параметров."""
        td = TornadoPlotter.from_sobol(sobol_result_3params, top_n=2)
        assert len(td.parameter_names) == 2

    def test_top_n_none_returns_all(self, sobol_result_3params: SobolResult) -> None:
        """Тест что top_n=None возвращает все параметры."""
        td = TornadoPlotter.from_sobol(sobol_result_3params, top_n=None)
        assert len(td.parameter_names) == 3

    def test_empty_result(self) -> None:
        """Тест конвертации пустого SobolResult."""
        result = SobolResult()
        td = TornadoPlotter.from_sobol(result)
        assert len(td.parameter_names) == 0

    def test_source_method_is_sobol(self, sobol_result_3params: SobolResult) -> None:
        """Тест что source_method == SOBOL."""
        td = TornadoPlotter.from_sobol(sobol_result_3params)
        assert td.source_method is SensitivityMethod.SOBOL

    def test_title_contains_metric_and_variable(self, sobol_result_3params: SobolResult) -> None:
        """Тест что title содержит метрику и имя переменной."""
        td = TornadoPlotter.from_sobol(sobol_result_3params, metric="ST")
        assert "ST" in td.title
        assert "F" in td.title

    def test_ci_values_present(self, sobol_result_3params: SobolResult) -> None:
        """Тест что lower_values и upper_values присутствуют при наличии conf."""
        td = TornadoPlotter.from_sobol(sobol_result_3params)
        assert td.lower_values is not None
        assert td.upper_values is not None


# =============================================================================
# Тесты по: Description/Phase3/description_sensitivity_analysis.md#TornadoPlotter.from_morris
# =============================================================================


class TestTornadoPlotterFromMorris:
    """Тесты для TornadoPlotter.from_morris()."""

    def test_basic_conversion(self, morris_result_5params: MorrisResult) -> None:
        """Тест базовой конвертации MorrisResult в TornadoData."""
        td = TornadoPlotter.from_morris(morris_result_5params)
        assert td.metric_name == "mu_star"
        # p1 имеет наибольший mu_star=100
        assert td.parameter_names[0] == "p1"

    def test_top_n(self, morris_result_5params: MorrisResult) -> None:
        """Тест что top_n ограничивает число параметров."""
        td = TornadoPlotter.from_morris(morris_result_5params, top_n=3)
        assert len(td.parameter_names) == 3

    def test_empty_result(self) -> None:
        """Тест конвертации пустого MorrisResult."""
        result = MorrisResult()
        td = TornadoPlotter.from_morris(result)
        assert len(td.parameter_names) == 0

    def test_source_method_is_morris(self, morris_result_5params: MorrisResult) -> None:
        """Тест что source_method == MORRIS."""
        td = TornadoPlotter.from_morris(morris_result_5params)
        assert td.source_method is SensitivityMethod.MORRIS


# =============================================================================
# Тесты по: Description/Phase3/description_sensitivity_analysis.md#TornadoPlotter.from_local
# =============================================================================


class TestTornadoPlotterFromLocal:
    """Тесты для TornadoPlotter.from_local()."""

    def test_basic_conversion(self, local_result_3params: LocalSensitivityResult) -> None:
        """Тест базовой конвертации LocalSensitivityResult в TornadoData."""
        td = TornadoPlotter.from_local(local_result_3params)
        assert td.metric_name == "elasticity"
        # p2 имеет наибольший |elasticity|=2.0
        assert td.parameter_names[0] == "p2"

    def test_top_n(self, local_result_3params: LocalSensitivityResult) -> None:
        """Тест что top_n ограничивает число параметров."""
        td = TornadoPlotter.from_local(local_result_3params, top_n=2)
        assert len(td.parameter_names) == 2

    def test_empty_result(self) -> None:
        """Тест конвертации пустого LocalSensitivityResult."""
        result = LocalSensitivityResult()
        td = TornadoPlotter.from_local(result)
        assert len(td.parameter_names) == 0

    def test_source_method_is_local(self, local_result_3params: LocalSensitivityResult) -> None:
        """Тест что source_method == LOCAL."""
        td = TornadoPlotter.from_local(local_result_3params)
        assert td.source_method is SensitivityMethod.LOCAL


# =============================================================================
# Тесты по: Description/Phase3/description_sensitivity_analysis.md#TornadoPlotter.plot
# =============================================================================


class TestTornadoPlotterPlot:
    """Тесты для TornadoPlotter.plot()."""

    def test_plot_returns_figure(self) -> None:
        """Тест что plot() возвращает matplotlib Figure."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plotter = TornadoPlotter()
        td = TornadoData(
            parameter_names=["a", "b"],
            values=np.array([0.5, 0.3]),
            metric_name="ST",
            title="Test Tornado",
        )
        fig = plotter.plot(td)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_empty_data(self) -> None:
        """Тест что пустой TornadoData создаёт пустую фигуру."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plotter = TornadoPlotter()
        fig = plotter.plot(TornadoData())
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_with_error_bars(self) -> None:
        """Тест что plot с CI создаёт фигуру с error bars."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plotter = TornadoPlotter()
        td = TornadoData(
            parameter_names=["a", "b"],
            values=np.array([0.5, 0.3]),
            lower_values=np.array([0.4, 0.2]),
            upper_values=np.array([0.6, 0.4]),
            metric_name="ST",
            title="Test Tornado CI",
        )
        fig = plotter.plot(td)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =============================================================================
# Тесты по: Description/Phase3/description_sensitivity_analysis.md#run_sensitivity_analysis
# =============================================================================


class TestRunSensitivityAnalysis:
    """Тесты для run_sensitivity_analysis() convenience-функции."""

    def test_sobol_returns_sobol_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Тест что method='sobol' возвращает SobolResult."""
        rng = np.random.default_rng(42)
        monkeypatch.setattr(
            "src.core.sensitivity_analysis.SensitivityAnalyzer._evaluate_model_single",
            lambda *_a, **_kw: float(rng.random()),
        )
        result = run_sensitivity_analysis(
            method="sobol",
            parameter_names=["r_F", "delta_Ne", "gamma_TNF"],
            n_samples=16,
        )
        assert isinstance(result, SobolResult)

    def test_morris_returns_morris_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Тест что method='morris' возвращает MorrisResult."""
        rng = np.random.default_rng(42)
        monkeypatch.setattr(
            "src.core.sensitivity_analysis.SensitivityAnalyzer._evaluate_model_single",
            lambda *_a, **_kw: float(rng.random()),
        )
        result = run_sensitivity_analysis(
            method="morris",
            parameter_names=["r_F", "delta_Ne", "gamma_TNF"],
            n_samples=4,
        )
        assert isinstance(result, MorrisResult)

    def test_local_returns_local_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Тест что method='local' возвращает LocalSensitivityResult."""
        call_idx = {"i": 0}

        def mock_single(*_args: object, **_kwargs: object) -> float:
            call_idx["i"] += 1
            return float(call_idx["i"]) * 10.0

        monkeypatch.setattr(
            "src.core.sensitivity_analysis.SensitivityAnalyzer._evaluate_model_single",
            mock_single,
        )
        result = run_sensitivity_analysis(
            method="local",
            parameter_names=["r_F", "delta_Ne", "gamma_TNF"],
        )
        assert isinstance(result, LocalSensitivityResult)

    def test_invalid_method_raises(self) -> None:
        """Тест что невалидный метод вызывает ValueError."""
        with pytest.raises(ValueError):
            run_sensitivity_analysis(method="invalid")

    def test_enum_method_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Тест что SensitivityMethod enum принимается."""
        call_idx = {"i": 0}

        def mock_single(*_args: object, **_kwargs: object) -> float:
            call_idx["i"] += 1
            return float(call_idx["i"])

        monkeypatch.setattr(
            "src.core.sensitivity_analysis.SensitivityAnalyzer._evaluate_model_single",
            mock_single,
        )
        result = run_sensitivity_analysis(
            method=SensitivityMethod.LOCAL,
            parameter_names=["r_F", "delta_Ne"],
        )
        assert isinstance(result, LocalSensitivityResult)
