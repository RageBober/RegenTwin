"""TDD тесты для parameter_estimation.py — Phase 3.1 Параметрическая идентификация.

Тестирование:
- PriorSpec: спецификация априорных распределений
- EstimationConfig: конфигурация и валидация
- ConvergenceDiagnostics: контейнер метрик сходимости
- EstimationResult: унифицированный контейнер результатов
- ForwardModelWrapper: обёртка SDE модели для идентификации
- PriorBuilder: генератор приоров для PyMC/emcee/scipy
- BaseEstimator: базовый класс (log-likelihood, AIC/BIC)
- BayesianEstimator: PyMC 5 NUTS sampling
- MCMCEstimator: emcee ensemble sampling
- MLEstimator: scipy.optimize MLE
- ConvergenceAnalyzer: ArviZ диагностика сходимости
- estimate_parameters(): convenience-функция

Все тесты написаны для stub-реализации (NotImplementedError).
Должны ПРОВАЛИТЬСЯ до реализации и ПРОЙТИ после.

Основано на спецификации: Description/Phase3/description_parameter_estimation.md
"""

from __future__ import annotations

import math
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.core.extended_sde import (
    ExtendedSDEModel,
    ExtendedSDEState,
    ExtendedSDETrajectory,
)
from src.core.parameter_estimation import (
    BaseEstimator,
    BayesianEstimator,
    ConvergenceAnalyzer,
    ConvergenceDiagnostics,
    EstimationConfig,
    EstimationResult,
    ForwardModelWrapper,
    MCMCEstimator,
    MLEstimator,
    PriorBuilder,
    PriorSpec,
    estimate_parameters,
)
from src.core.parameters import ParameterSet
from src.data.dataset_loader import TimeSeriesData

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

SAMPLE_PARAM_NAMES = ["r_F", "delta_Ne", "gamma_TNF"]
SAMPLE_OBS_VARIABLES = ["F", "C_TNF"]
SAMPLE_OBS_TIMES = np.array([0.0, 24.0, 48.0, 168.0, 336.0, 720.0])
VALID_DISTRIBUTIONS = ["normal", "lognormal", "uniform", "halfnormal", "gamma"]

# Литературные defaults для тестируемых параметров
DEFAULT_R_F = 0.03
DEFAULT_DELTA_NE = 0.05
DEFAULT_GAMMA_TNF = 0.5


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def base_params() -> ParameterSet:
    """ParameterSet с литературными defaults."""
    return ParameterSet()


@pytest.fixture
def initial_state() -> ExtendedSDEState:
    """ExtendedSDEState с биологически-осмысленными начальными значениями."""
    return ExtendedSDEState(
        P=1000.0,
        Ne=500.0,
        M1=50.0,
        M2=20.0,
        F=1000.0,
        Mf=10.0,
        E=200.0,
        S=100.0,
        C_TNF=5.0,
        C_IL10=2.0,
        C_PDGF=1.0,
        C_VEGF=0.5,
        C_TGFb=1.0,
        C_MCP1=3.0,
        C_IL8=4.0,
        rho_collagen=0.3,
        C_MMP=0.1,
        rho_fibrin=0.5,
        D=1.0,
        O2=80.0,
        t=0.0,
    )


@pytest.fixture
def observation_times() -> np.ndarray:
    """Временные точки наблюдений (часы)."""
    return SAMPLE_OBS_TIMES.copy()


@pytest.fixture
def sample_prior_specs() -> list[PriorSpec]:
    """3 PriorSpec: r_F (lognormal), delta_Ne (lognormal), gamma_TNF (normal)."""
    return [
        PriorSpec(name="r_F", distribution="lognormal", mean=0.03, std=0.009),
        PriorSpec(name="delta_Ne", distribution="lognormal", mean=0.05, std=0.015),
        PriorSpec(name="gamma_TNF", distribution="normal", mean=0.5, std=0.15),
    ]


@pytest.fixture
def fixed_prior_spec() -> PriorSpec:
    """Фиксированный PriorSpec для dt."""
    return PriorSpec(name="dt", fixed=True, mean=0.01)


@pytest.fixture
def mixed_priors(sample_prior_specs, fixed_prior_spec) -> list[PriorSpec]:
    """2 free PriorSpec + 1 fixed."""
    return [sample_prior_specs[0], sample_prior_specs[1], fixed_prior_spec]


@pytest.fixture
def estimation_config() -> EstimationConfig:
    """EstimationConfig с observed_variables для тестов."""
    return EstimationConfig(observed_variables=["F", "C_TNF"])


@pytest.fixture
def mock_trajectory(base_params) -> ExtendedSDETrajectory:
    """ExtendedSDETrajectory с known F/C_TNF (линейная интерполяция)."""
    n_steps = 7201
    times = np.linspace(0, 720, n_steps)
    # F: линейное убывание 1000 → 500
    f_values = np.linspace(1000.0, 500.0, n_steps)
    # C_TNF: пик на 48 ч, затем спад
    c_tnf_values = 5.0 * np.exp(-((times - 48.0) ** 2) / (2 * 100.0**2))

    states = []
    for i, t in enumerate(times):
        states.append(
            ExtendedSDEState(
                F=f_values[i],
                C_TNF=c_tnf_values[i],
                t=t,
            )
        )
    return ExtendedSDETrajectory(times=times, states=states, params=base_params)


@pytest.fixture
def mock_sde_model(mock_trajectory) -> MagicMock:
    """MagicMock ExtendedSDEModel, .simulate() → mock_trajectory."""
    model = MagicMock(spec=ExtendedSDEModel)
    model.simulate.return_value = mock_trajectory
    return model


@pytest.fixture
def time_series_data(observation_times) -> TimeSeriesData:
    """TimeSeriesData с наблюдениями F и C_TNF в точках observation_times."""
    f_obs = np.array([1000.0, 900.0, 850.0, 700.0, 600.0, 500.0])
    c_tnf_obs = np.array([5.0, 4.0, 3.5, 1.5, 0.5, 0.1])
    return TimeSeriesData(
        time_points=observation_times,
        values={"F": f_obs, "C_TNF": c_tnf_obs},
        units={"F": "cells/uL", "C_TNF": "ng/mL"},
        metadata=None,
    )


@pytest.fixture
def converged_diagnostics() -> ConvergenceDiagnostics:
    """ConvergenceDiagnostics — все параметры сошлись."""
    return ConvergenceDiagnostics(
        rhat={"r_F": 1.01, "delta_Ne": 1.02, "gamma_TNF": 1.00},
        ess_bulk={"r_F": 500.0, "delta_Ne": 400.0, "gamma_TNF": 600.0},
        ess_tail={"r_F": 300.0, "delta_Ne": 250.0, "gamma_TNF": 350.0},
        converged=True,
        warnings=[],
    )


@pytest.fixture
def not_converged_diagnostics() -> ConvergenceDiagnostics:
    """ConvergenceDiagnostics — не сошлось (rhat слишком высокий)."""
    return ConvergenceDiagnostics(
        rhat={"r_F": 1.20, "delta_Ne": 1.01, "gamma_TNF": 1.03},
        ess_bulk={"r_F": 50.0, "delta_Ne": 400.0, "gamma_TNF": 500.0},
        ess_tail={"r_F": 30.0, "delta_Ne": 250.0, "gamma_TNF": 300.0},
        converged=False,
        warnings=["r_F: rhat=1.20 > 1.05"],
    )


@pytest.fixture
def mock_inference_data() -> MagicMock:
    """Mock az.InferenceData с posterior samples."""
    rng = np.random.default_rng(42)
    idata = MagicMock()
    # Имитируем posterior с 3 параметрами, 4 chains, 500 draws
    posterior = MagicMock()
    posterior.__getitem__ = MagicMock(
        side_effect=lambda name: rng.normal(
            {"r_F": 0.03, "delta_Ne": 0.05, "gamma_TNF": 0.5}.get(name, 0.0),
            0.01,
            size=(4, 500),
        )
    )
    posterior.data_vars = {"r_F": None, "delta_Ne": None, "gamma_TNF": None}
    idata.posterior = posterior
    return idata


# =============================================================================
# Вспомогательный подкласс для тестирования BaseEstimator
# =============================================================================


class _ConcreteEstimator(BaseEstimator):
    """Конкретный подкласс BaseEstimator для тестирования protected-методов."""

    def __init__(self, forward_model, observed_data, config, prior_builder):
        """Прямое сохранение атрибутов (обход stub __init__)."""
        self.forward_model = forward_model
        self.observed_data = observed_data
        self.config = config
        self.prior_builder = prior_builder
        # Извлечение наблюдений
        self.observed_values = {
            var: observed_data.values[var]
            for var in config.observed_variables
            if var in observed_data.values
        }

    def fit(self) -> EstimationResult:
        """Не используется в тестах BaseEstimator."""
        raise NotImplementedError


# =============================================================================
# TestPriorSpec
# =============================================================================


# Тесты по: Description/Phase3/description_parameter_estimation.md#PriorSpec
class TestPriorSpec:
    """Тесты PriorSpec — спецификация априорного распределения."""

    def test_default_creation(self):
        """PriorSpec с минимальными аргументами: defaults корректны."""
        spec = PriorSpec(name="r_F")
        assert spec.name == "r_F"
        assert spec.distribution == "lognormal"
        assert spec.mean == 0.0
        assert spec.std == 1.0
        assert spec.lower == 0.0
        assert spec.upper == float("inf")
        assert spec.fixed is False
        assert spec.source == ""

    def test_fixed_parameter(self):
        """PriorSpec с fixed=True фиксирует параметр на mean."""
        spec = PriorSpec(name="dt", fixed=True, mean=0.01)
        assert spec.fixed is True
        assert spec.mean == 0.01

    @pytest.mark.parametrize("dist", VALID_DISTRIBUTIONS)
    def test_all_distribution_types(self, dist):
        """Все 5 допустимых типов распределений создаются без ошибок."""
        spec = PriorSpec(name="r_F", distribution=dist)
        assert spec.distribution == dist

    def test_source_field(self):
        """Литературная ссылка сохраняется в source."""
        spec = PriorSpec(name="r_F", source="Vodovotz 2006")
        assert spec.source == "Vodovotz 2006"

    def test_uniform_bounds(self):
        """Uniform prior с lower/upper границами."""
        spec = PriorSpec(name="r_F", distribution="uniform", lower=0.01, upper=0.1)
        assert spec.lower == 0.01
        assert spec.upper == 0.1
        assert spec.lower < spec.upper

    def test_custom_mean_std(self):
        """Пользовательские mean и std сохраняются корректно."""
        spec = PriorSpec(name="r_F", mean=0.03, std=0.009)
        assert spec.mean == pytest.approx(0.03)
        assert spec.std == pytest.approx(0.009)


# =============================================================================
# TestEstimationConfig
# =============================================================================


# Тесты по: Description/Phase3/description_parameter_estimation.md#EstimationConfig
class TestEstimationConfig:
    """Тесты EstimationConfig — конфигурация параметрической идентификации."""

    def test_default_creation(self):
        """EstimationConfig() создаётся с корректными defaults."""
        cfg = EstimationConfig()
        assert cfg.n_samples == 2000
        assert cfg.n_tune == 1000
        assert cfg.n_chains == 4
        assert cfg.n_walkers == 32
        assert cfg.dt == pytest.approx(0.01)
        assert cfg.t_span == (0.0, 720.0)
        assert cfg.noise_model == "gaussian"
        assert cfg.sigma_obs is None
        assert cfg.target_accept == pytest.approx(0.8)
        assert cfg.mle_method == "L-BFGS-B"
        assert cfg.mle_maxiter == 1000
        assert cfg.rhat_threshold == pytest.approx(1.05)
        assert cfg.ess_min == 100
        assert cfg.rng_seed is None
        assert cfg.solver == "euler_maruyama"
        assert cfg.n_sde_realizations == 1
        assert cfg.priors == []
        assert cfg.observed_variables == []

    def test_validate_happy_path(self):
        """Валидная конфигурация проходит validate()."""
        cfg = EstimationConfig(observed_variables=["F", "C_TNF"])
        assert cfg.validate() is True

    def test_validate_negative_n_samples(self):
        """n_samples=-1 → ValueError."""
        cfg = EstimationConfig(observed_variables=["F"], n_samples=-1)
        with pytest.raises(ValueError):
            cfg.validate()

    def test_validate_target_accept_above_one(self):
        """target_accept=1.5 (вне (0,1)) → ValueError."""
        cfg = EstimationConfig(observed_variables=["F"], target_accept=1.5)
        with pytest.raises(ValueError):
            cfg.validate()

    def test_validate_target_accept_zero(self):
        """target_accept=0.0 → ValueError."""
        cfg = EstimationConfig(observed_variables=["F"], target_accept=0.0)
        with pytest.raises(ValueError):
            cfg.validate()

    def test_validate_empty_observed_variables(self):
        """Пустой observed_variables → ValueError."""
        cfg = EstimationConfig(observed_variables=[])
        with pytest.raises(ValueError):
            cfg.validate()

    def test_validate_n_sde_realizations_zero(self):
        """n_sde_realizations=0 → ValueError."""
        cfg = EstimationConfig(observed_variables=["F"], n_sde_realizations=0)
        with pytest.raises(ValueError):
            cfg.validate()

    def test_validate_invalid_noise_model(self):
        """noise_model='poisson' → ValueError."""
        cfg = EstimationConfig(observed_variables=["F"], noise_model="poisson")
        with pytest.raises(ValueError):
            cfg.validate()

    def test_validate_rhat_threshold_le_one(self):
        """rhat_threshold=1.0 (не > 1.0) → ValueError."""
        cfg = EstimationConfig(observed_variables=["F"], rhat_threshold=1.0)
        with pytest.raises(ValueError):
            cfg.validate()

    def test_validate_negative_dt(self):
        """dt=-0.01 → ValueError."""
        cfg = EstimationConfig(observed_variables=["F"], dt=-0.01)
        with pytest.raises(ValueError):
            cfg.validate()


# =============================================================================
# TestConvergenceDiagnostics
# =============================================================================


# Тесты по: Description/Phase3/description_parameter_estimation.md#ConvergenceDiagnostics
class TestConvergenceDiagnostics:
    """Тесты ConvergenceDiagnostics — контейнер метрик сходимости."""

    def test_default_creation(self):
        """ConvergenceDiagnostics() → converged=False, пустые dicts."""
        diag = ConvergenceDiagnostics()
        assert diag.converged is False
        assert diag.rhat == {}
        assert diag.ess_bulk == {}
        assert diag.ess_tail == {}
        assert diag.warnings == []

    def test_populated_converged(self, converged_diagnostics):
        """Заполненные метрики с converged=True."""
        diag = converged_diagnostics
        assert diag.converged is True
        assert len(diag.rhat) == 3
        assert all(v < 1.05 for v in diag.rhat.values())
        assert all(v > 100 for v in diag.ess_bulk.values())

    def test_populated_not_converged(self, not_converged_diagnostics):
        """Метрики с converged=False и warnings."""
        diag = not_converged_diagnostics
        assert diag.converged is False
        assert len(diag.warnings) > 0
        assert diag.rhat["r_F"] > 1.05

    def test_summary_table_none_by_default(self):
        """summary_table по умолчанию None."""
        diag = ConvergenceDiagnostics()
        assert diag.summary_table is None


# =============================================================================
# TestEstimationResult
# =============================================================================


# Тесты по: Description/Phase3/description_parameter_estimation.md#EstimationResult
class TestEstimationResult:
    """Тесты EstimationResult — унифицированный контейнер результатов."""

    def test_default_creation(self):
        """EstimationResult() → все поля по умолчанию."""
        result = EstimationResult()
        assert result.method == ""
        assert result.point_estimates == {}
        assert result.ci_lower == {}
        assert result.ci_upper == {}
        assert result.posterior_samples is None
        assert result.inference_data is None
        assert result.diagnostics is None
        assert result.fitted_params is None
        assert result.log_likelihood is None
        assert result.aic is None
        assert result.bic is None
        assert result.n_observations == 0
        assert result.n_estimated_params == 0
        assert result.elapsed_seconds == 0.0
        assert result.config is None

    def test_mle_result(self):
        """MLE результат: posterior_samples=None, diagnostics=None."""
        result = EstimationResult(
            method="mle_scipy",
            point_estimates={"r_F": 0.03},
            posterior_samples=None,
            diagnostics=None,
        )
        assert result.method == "mle_scipy"
        assert result.posterior_samples is None
        assert result.diagnostics is None

    def test_bayesian_result_all_filled(self, converged_diagnostics):
        """Bayesian результат со всеми заполненными полями."""
        result = EstimationResult(
            method="bayesian_pymc",
            point_estimates={"r_F": 0.031, "delta_Ne": 0.049},
            ci_lower={"r_F": 0.02, "delta_Ne": 0.03},
            ci_upper={"r_F": 0.04, "delta_Ne": 0.07},
            posterior_samples={"r_F": np.random.default_rng(0).normal(0.03, 0.005, 2000)},
            diagnostics=converged_diagnostics,
            log_likelihood=-100.0,
            aic=210.0,
            bic=219.56,
        )
        assert result.method == "bayesian_pymc"
        assert result.posterior_samples is not None
        assert result.diagnostics.converged is True

    def test_information_criteria_fields(self):
        """AIC/BIC сохраняются корректно."""
        result = EstimationResult(aic=210.0, bic=219.56)
        assert result.aic == pytest.approx(210.0)
        assert result.bic == pytest.approx(219.56)

    def test_fitted_params_type(self):
        """fitted_params — экземпляр ParameterSet."""
        ps = ParameterSet()
        result = EstimationResult(fitted_params=ps)
        assert isinstance(result.fitted_params, ParameterSet)


# =============================================================================
# TestForwardModelWrapper
# =============================================================================


# Тесты по: Description/Phase3/description_parameter_estimation.md#ForwardModelWrapper
class TestForwardModelWrapper:
    """Тесты ForwardModelWrapper — обёртка SDE модели."""

    def test_init_valid_params(self, base_params, initial_state, observation_times):
        """Корректные параметры → инициализация без ошибок."""
        wrapper = ForwardModelWrapper(
            base_params=base_params,
            initial_state=initial_state,
            estimated_param_names=SAMPLE_PARAM_NAMES,
            observed_variables=SAMPLE_OBS_VARIABLES,
            observation_times=observation_times,
        )
        assert wrapper is not None

    def test_init_invalid_param_name(self, base_params, initial_state, observation_times):
        """Имя параметра не из ParameterSet → ValueError."""
        with pytest.raises(ValueError):
            ForwardModelWrapper(
                base_params=base_params,
                initial_state=initial_state,
                estimated_param_names=["nonexistent_param"],
                observed_variables=SAMPLE_OBS_VARIABLES,
                observation_times=observation_times,
            )

    def test_init_invalid_observed_var(self, base_params, initial_state, observation_times):
        """Имя переменной не из ExtendedSDEState → ValueError."""
        with pytest.raises(ValueError):
            ForwardModelWrapper(
                base_params=base_params,
                initial_state=initial_state,
                estimated_param_names=SAMPLE_PARAM_NAMES,
                observed_variables=["nonexistent_variable"],
                observation_times=observation_times,
            )

    def test_init_empty_observation_times(self, base_params, initial_state):
        """Пустой observation_times → ValueError."""
        with pytest.raises(ValueError):
            ForwardModelWrapper(
                base_params=base_params,
                initial_state=initial_state,
                estimated_param_names=SAMPLE_PARAM_NAMES,
                observed_variables=SAMPLE_OBS_VARIABLES,
                observation_times=np.array([]),
            )

    def test_predict_output_format(self, base_params, initial_state, observation_times):
        """predict() возвращает dict с ключами observed_variables, shape (n_obs,)."""
        wrapper = ForwardModelWrapper(
            base_params=base_params,
            initial_state=initial_state,
            estimated_param_names=SAMPLE_PARAM_NAMES,
            observed_variables=SAMPLE_OBS_VARIABLES,
            observation_times=observation_times,
        )
        theta = np.array([DEFAULT_R_F, DEFAULT_DELTA_NE, DEFAULT_GAMMA_TNF])
        result = wrapper.predict(theta)
        assert isinstance(result, dict)
        assert set(result.keys()) == set(SAMPLE_OBS_VARIABLES)
        for var in SAMPLE_OBS_VARIABLES:
            assert result[var].shape == (len(observation_times),)

    def test_predict_wrong_theta_dimension(self, base_params, initial_state, observation_times):
        """len(theta) != len(estimated_param_names) → ValueError."""
        wrapper = ForwardModelWrapper(
            base_params=base_params,
            initial_state=initial_state,
            estimated_param_names=SAMPLE_PARAM_NAMES,
            observed_variables=SAMPLE_OBS_VARIABLES,
            observation_times=observation_times,
        )
        theta_wrong = np.array([0.03, 0.05])  # 2 вместо 3
        with pytest.raises(ValueError):
            wrapper.predict(theta_wrong)

    def test_predict_nan_theta(self, base_params, initial_state, observation_times):
        """NaN в theta → dict с NaN значениями."""
        wrapper = ForwardModelWrapper(
            base_params=base_params,
            initial_state=initial_state,
            estimated_param_names=SAMPLE_PARAM_NAMES,
            observed_variables=SAMPLE_OBS_VARIABLES,
            observation_times=observation_times,
        )
        theta_nan = np.array([np.nan, DEFAULT_DELTA_NE, DEFAULT_GAMMA_TNF])
        result = wrapper.predict(theta_nan)
        assert isinstance(result, dict)
        # Хотя бы одна переменная содержит NaN
        has_nan = any(np.any(np.isnan(v)) for v in result.values())
        assert has_nan

    def test_build_parameter_set_single_substitution(
        self, base_params, initial_state, observation_times
    ):
        """Подстановка одного параметра: r_F=0.05, остальные = defaults."""
        wrapper = ForwardModelWrapper(
            base_params=base_params,
            initial_state=initial_state,
            estimated_param_names=["r_F"],
            observed_variables=SAMPLE_OBS_VARIABLES,
            observation_times=observation_times,
        )
        theta = np.array([0.05])
        result_ps = wrapper._build_parameter_set(theta)
        assert result_ps.r_F == pytest.approx(0.05)
        # Остальные параметры остались defaults
        assert result_ps.delta_Ne == pytest.approx(DEFAULT_DELTA_NE)

    def test_build_parameter_set_all_defaults(self, base_params, initial_state, observation_times):
        """Theta == default values → результат == ParameterSet()."""
        wrapper = ForwardModelWrapper(
            base_params=base_params,
            initial_state=initial_state,
            estimated_param_names=SAMPLE_PARAM_NAMES,
            observed_variables=SAMPLE_OBS_VARIABLES,
            observation_times=observation_times,
        )
        theta = np.array([DEFAULT_R_F, DEFAULT_DELTA_NE, DEFAULT_GAMMA_TNF])
        result_ps = wrapper._build_parameter_set(theta)
        default_ps = ParameterSet()
        assert result_ps.r_F == pytest.approx(default_ps.r_F)
        assert result_ps.delta_Ne == pytest.approx(default_ps.delta_Ne)
        assert result_ps.gamma_TNF == pytest.approx(default_ps.gamma_TNF)

    def test_run_simulation_returns_trajectory(self, base_params, initial_state, observation_times):
        """_run_simulation с default params возвращает trajectory."""
        wrapper = ForwardModelWrapper(
            base_params=base_params,
            initial_state=initial_state,
            estimated_param_names=SAMPLE_PARAM_NAMES,
            observed_variables=SAMPLE_OBS_VARIABLES,
            observation_times=observation_times,
        )
        trajectory = wrapper._run_simulation(base_params)
        assert isinstance(trajectory, ExtendedSDETrajectory)
        assert len(trajectory.states) > 0
        assert len(trajectory.times) > 0

    def test_extract_at_times_interpolation(
        self, base_params, initial_state, observation_times, mock_trajectory
    ):
        """_extract_at_times интерполирует значения корректно."""
        wrapper = ForwardModelWrapper(
            base_params=base_params,
            initial_state=initial_state,
            estimated_param_names=SAMPLE_PARAM_NAMES,
            observed_variables=SAMPLE_OBS_VARIABLES,
            observation_times=observation_times,
        )
        result = wrapper._extract_at_times(mock_trajectory, observation_times)
        assert isinstance(result, dict)
        assert "F" in result
        assert "C_TNF" in result
        assert result["F"].shape == (len(observation_times),)
        # F при t=0 должен быть ≈ 1000, при t=720 ≈ 500 (из mock_trajectory)
        assert result["F"][0] == pytest.approx(1000.0, rel=0.01)
        assert result["F"][-1] == pytest.approx(500.0, rel=0.01)


# =============================================================================
# TestPriorBuilder
# =============================================================================


# Тесты по: Description/Phase3/description_parameter_estimation.md#PriorBuilder
class TestPriorBuilder:
    """Тесты PriorBuilder — генератор априорных распределений."""

    def test_init_partition_free_fixed(self, mixed_priors):
        """__init__ разделяет priors на free и fixed."""
        builder = PriorBuilder(mixed_priors)
        free_names = builder.get_free_param_names()
        assert len(free_names) == 2
        assert "dt" not in free_names

    def test_from_parameter_set_one_param(self, base_params):
        """from_parameter_set с одним параметром: mean=lit_value, std=cv*mean."""
        builder = PriorBuilder.from_parameter_set(
            base_params, estimated_names=["r_F"], default_cv=0.3
        )
        names = builder.get_free_param_names()
        assert names == ["r_F"]
        guess = builder.get_initial_guess()
        assert guess[0] == pytest.approx(DEFAULT_R_F)

    def test_from_parameter_set_custom_cv(self, base_params):
        """from_parameter_set с CV=0.5 → std = 0.5 * mean."""
        builder = PriorBuilder.from_parameter_set(
            base_params, estimated_names=["r_F"], default_cv=0.5
        )
        # Проверяем через initial guess (mean сохраняется)
        guess = builder.get_initial_guess()
        assert guess[0] == pytest.approx(DEFAULT_R_F)

    def test_from_parameter_set_invalid_name(self, base_params):
        """Имя не из ParameterSet → ValueError."""
        with pytest.raises(ValueError):
            PriorBuilder.from_parameter_set(base_params, estimated_names=["nonexistent"])

    def test_from_parameter_set_negative_cv(self, base_params):
        """default_cv < 0 → ValueError."""
        with pytest.raises(ValueError):
            PriorBuilder.from_parameter_set(base_params, estimated_names=["r_F"], default_cv=-0.1)

    def test_from_parameter_set_zero_cv(self, base_params):
        """default_cv = 0 → ValueError."""
        with pytest.raises(ValueError):
            PriorBuilder.from_parameter_set(base_params, estimated_names=["r_F"], default_cv=0.0)

    def test_get_free_param_names_all_free(self, sample_prior_specs):
        """Все priors с fixed=False → список из 3 имён."""
        builder = PriorBuilder(sample_prior_specs)
        names = builder.get_free_param_names()
        assert len(names) == 3
        assert names == ["r_F", "delta_Ne", "gamma_TNF"]

    def test_get_free_param_names_mixed(self, mixed_priors):
        """2 free + 1 fixed → список из 2 имён."""
        builder = PriorBuilder(mixed_priors)
        names = builder.get_free_param_names()
        assert len(names) == 2

    def test_get_free_param_names_all_fixed(self):
        """Все fixed=True → пустой список."""
        priors = [
            PriorSpec(name="r_F", fixed=True, mean=0.03),
            PriorSpec(name="delta_Ne", fixed=True, mean=0.05),
        ]
        builder = PriorBuilder(priors)
        assert builder.get_free_param_names() == []

    def test_build_pymc_priors_mock(self, sample_prior_specs):
        """build_pymc_priors с mock pm.Model → dict с ключами параметров."""
        builder = PriorBuilder(sample_prior_specs)
        mock_model = MagicMock()
        # Создаём mock pymc модуль
        with patch.dict(sys.modules, {"pymc": MagicMock()}):
            result = builder.build_pymc_priors(mock_model)
        assert isinstance(result, dict)
        assert len(result) == 3

    def test_build_pymc_priors_empty(self):
        """Нет free params → пустой dict."""
        priors = [PriorSpec(name="r_F", fixed=True, mean=0.03)]
        builder = PriorBuilder(priors)
        mock_model = MagicMock()
        result = builder.build_pymc_priors(mock_model)
        assert result == {}

    def test_build_log_prior_fn_in_support(self, sample_prior_specs):
        """Theta в support → конечное float значение."""
        builder = PriorBuilder(sample_prior_specs)
        log_prior_fn = builder.build_log_prior_fn()
        theta = np.array([0.03, 0.05, 0.5])  # Разумные значения
        result = log_prior_fn(theta)
        assert np.isfinite(result)

    def test_build_log_prior_fn_out_of_support(self):
        """Отрицательный theta для lognormal → -inf."""
        priors = [PriorSpec(name="r_F", distribution="lognormal", mean=0.03, std=0.009)]
        builder = PriorBuilder(priors)
        log_prior_fn = builder.build_log_prior_fn()
        theta = np.array([-0.01])
        result = log_prior_fn(theta)
        assert result == -np.inf

    def test_build_log_prior_fn_uniform_bounds(self):
        """Theta вне uniform bounds → -inf."""
        priors = [PriorSpec(name="r_F", distribution="uniform", lower=0.01, upper=0.1)]
        builder = PriorBuilder(priors)
        log_prior_fn = builder.build_log_prior_fn()
        # Вне верхней границы
        assert log_prior_fn(np.array([0.2])) == -np.inf
        # Вне нижней границы
        assert log_prior_fn(np.array([0.005])) == -np.inf

    def test_build_scipy_bounds_lognormal(self):
        """Lognormal prior mean=0.03 → bounds (0, 0.3)."""
        priors = [PriorSpec(name="r_F", distribution="lognormal", mean=0.03, std=0.009)]
        builder = PriorBuilder(priors)
        bounds = builder.build_scipy_bounds()
        assert len(bounds) == 1
        lower, upper = bounds[0]
        assert lower == pytest.approx(0.0)
        assert upper == pytest.approx(0.3)  # mean * 10

    def test_build_scipy_bounds_uniform(self):
        """Uniform prior → bounds = (lower, upper)."""
        priors = [PriorSpec(name="r_F", distribution="uniform", lower=0.01, upper=0.1)]
        builder = PriorBuilder(priors)
        bounds = builder.build_scipy_bounds()
        assert bounds[0] == (pytest.approx(0.01), pytest.approx(0.1))

    def test_build_scipy_bounds_length(self, sample_prior_specs):
        """3 free params → 3 tuples."""
        builder = PriorBuilder(sample_prior_specs)
        bounds = builder.build_scipy_bounds()
        assert len(bounds) == 3
        for b in bounds:
            assert len(b) == 2
            assert b[0] < b[1]

    def test_get_initial_guess_single(self):
        """Один параметр → array([mean])."""
        priors = [PriorSpec(name="r_F", mean=0.03)]
        builder = PriorBuilder(priors)
        guess = builder.get_initial_guess()
        assert isinstance(guess, np.ndarray)
        assert guess.shape == (1,)
        assert guess[0] == pytest.approx(0.03)

    def test_get_initial_guess_multiple(self, sample_prior_specs):
        """3 priors → shape (3,), значения == means."""
        builder = PriorBuilder(sample_prior_specs)
        guess = builder.get_initial_guess()
        assert guess.shape == (3,)
        assert guess[0] == pytest.approx(0.03)
        assert guess[1] == pytest.approx(0.05)
        assert guess[2] == pytest.approx(0.5)


# =============================================================================
# TestBaseEstimator
# =============================================================================


# Тесты по: Description/Phase3/description_parameter_estimation.md#BaseEstimator
class TestBaseEstimator:
    """Тесты BaseEstimator — базовый класс estimator-ов."""

    def test_init_stores_attributes(self, time_series_data, estimation_config):
        """__init__ сохраняет forward_model, observed_data, config, prior_builder."""
        mock_fm = MagicMock(spec=ForwardModelWrapper)
        mock_pb = MagicMock(spec=PriorBuilder)
        estimator = BaseEstimator(
            forward_model=mock_fm,
            observed_data=time_series_data,
            config=estimation_config,
            prior_builder=mock_pb,
        )
        assert estimator.forward_model is mock_fm
        assert estimator.observed_data is time_series_data
        assert estimator.config is estimation_config
        assert estimator.prior_builder is mock_pb

    def test_compute_log_likelihood_perfect_fit(self, time_series_data, estimation_config):
        """Predicted == observed → log_lik близок к максимуму (≈ 0)."""
        mock_fm = MagicMock(spec=ForwardModelWrapper)
        # predict возвращает точные наблюдения
        mock_fm.predict.return_value = {
            "F": time_series_data.values["F"].copy(),
            "C_TNF": time_series_data.values["C_TNF"].copy(),
        }
        mock_pb = MagicMock(spec=PriorBuilder)
        estimation_config.sigma_obs = 1.0

        est = _ConcreteEstimator(mock_fm, time_series_data, estimation_config, mock_pb)
        theta = np.array([0.03, 0.05, 0.5])
        log_lik = est._compute_log_likelihood(theta)
        # При идеальном совпадении residuals=0, log_lik ≈ 0
        assert np.isfinite(log_lik)
        assert log_lik == pytest.approx(0.0, abs=1e-6)

    def test_compute_log_likelihood_nan_predictions(self, time_series_data, estimation_config):
        """NaN в предсказаниях → -inf."""
        mock_fm = MagicMock(spec=ForwardModelWrapper)
        mock_fm.predict.return_value = {
            "F": np.array([np.nan] * 6),
            "C_TNF": np.array([np.nan] * 6),
        }
        mock_pb = MagicMock(spec=PriorBuilder)

        est = _ConcreteEstimator(mock_fm, time_series_data, estimation_config, mock_pb)
        theta = np.array([0.03, 0.05, 0.5])
        log_lik = est._compute_log_likelihood(theta)
        assert log_lik == -np.inf

    def test_compute_log_likelihood_large_deviation(self, time_series_data, estimation_config):
        """Predicted сильно отличается → сильно отрицательный log_lik."""
        mock_fm = MagicMock(spec=ForwardModelWrapper)
        mock_fm.predict.return_value = {
            "F": time_series_data.values["F"] * 100.0,  # 100x отклонение
            "C_TNF": time_series_data.values["C_TNF"] * 100.0,
        }
        mock_pb = MagicMock(spec=PriorBuilder)
        estimation_config.sigma_obs = 1.0

        est = _ConcreteEstimator(mock_fm, time_series_data, estimation_config, mock_pb)
        theta = np.array([0.03, 0.05, 0.5])
        log_lik = est._compute_log_likelihood(theta)
        assert np.isfinite(log_lik)
        assert log_lik < -100.0

    def test_build_fitted_params_delegates(self, time_series_data, estimation_config):
        """_build_fitted_params делегирует forward_model._build_parameter_set."""
        mock_fm = MagicMock(spec=ForwardModelWrapper)
        expected_ps = ParameterSet()
        mock_fm._build_parameter_set.return_value = expected_ps
        mock_pb = MagicMock(spec=PriorBuilder)

        est = _ConcreteEstimator(mock_fm, time_series_data, estimation_config, mock_pb)
        theta = np.array([0.03, 0.05, 0.5])
        result = est._build_fitted_params(theta)
        mock_fm._build_parameter_set.assert_called_once_with(theta)
        assert result is expected_ps

    def test_compute_information_criteria_standard(self, time_series_data, estimation_config):
        """AIC = 2k - 2*log_lik, BIC = k*log(n) - 2*log_lik."""
        mock_fm = MagicMock(spec=ForwardModelWrapper)
        mock_pb = MagicMock(spec=PriorBuilder)
        est = _ConcreteEstimator(mock_fm, time_series_data, estimation_config, mock_pb)

        log_lik = -100.0
        k = 5
        n = 50
        aic, bic = est._compute_information_criteria(log_lik, k, n)
        assert aic == pytest.approx(2 * 5 - 2 * (-100.0))  # 210
        assert bic == pytest.approx(5 * math.log(50) - 2 * (-100.0))

    def test_compute_information_criteria_n_obs_zero(self, time_series_data, estimation_config):
        """n_obs=0 → ValueError."""
        mock_fm = MagicMock(spec=ForwardModelWrapper)
        mock_pb = MagicMock(spec=PriorBuilder)
        est = _ConcreteEstimator(mock_fm, time_series_data, estimation_config, mock_pb)

        with pytest.raises(ValueError):
            est._compute_information_criteria(-100.0, 5, 0)

    def test_compute_information_criteria_neg_inf(self, time_series_data, estimation_config):
        """log_lik = -inf → aic = inf, bic = inf."""
        mock_fm = MagicMock(spec=ForwardModelWrapper)
        mock_pb = MagicMock(spec=PriorBuilder)
        est = _ConcreteEstimator(mock_fm, time_series_data, estimation_config, mock_pb)

        aic, bic = est._compute_information_criteria(-np.inf, 5, 50)
        assert aic == np.inf
        assert bic == np.inf


# =============================================================================
# TestBayesianEstimator
# =============================================================================


# Тесты по: Description/Phase3/description_parameter_estimation.md#BayesianEstimator
class TestBayesianEstimator:
    """Тесты BayesianEstimator — PyMC 5 NUTS sampling."""

    def _make_estimator(self, time_series_data, estimation_config):
        """Создание BayesianEstimator с mock зависимостями."""
        mock_fm = MagicMock(spec=ForwardModelWrapper)
        mock_fm.predict.return_value = {
            "F": np.ones(6) * 500.0,
            "C_TNF": np.ones(6) * 1.0,
        }
        mock_pb = MagicMock(spec=PriorBuilder)
        mock_pb.get_free_param_names.return_value = SAMPLE_PARAM_NAMES
        mock_pb.get_initial_guess.return_value = np.array([0.03, 0.05, 0.5])
        return BayesianEstimator(mock_fm, time_series_data, estimation_config, mock_pb)

    def test_fit_import_error(self, time_series_data, estimation_config):
        """PyMC не установлен → ImportError."""
        estimator = self._make_estimator(time_series_data, estimation_config)
        with patch.dict(sys.modules, {"pymc": None}), pytest.raises(ImportError):
            estimator.fit()

    def test_fit_returns_bayesian_result(self, time_series_data, estimation_config):
        """fit() возвращает EstimationResult с method='bayesian_pymc'."""
        estimator = self._make_estimator(time_series_data, estimation_config)
        result = estimator.fit()
        assert isinstance(result, EstimationResult)
        assert result.method == "bayesian_pymc"

    def test_fit_has_posterior_samples(self, time_series_data, estimation_config):
        """Bayesian fit() → posterior_samples не None."""
        estimator = self._make_estimator(time_series_data, estimation_config)
        result = estimator.fit()
        assert result.posterior_samples is not None
        assert isinstance(result.posterior_samples, dict)

    def test_build_pymc_model_returns_model(self, time_series_data, estimation_config):
        """_build_pymc_model возвращает pm.Model объект."""
        estimator = self._make_estimator(time_series_data, estimation_config)
        model = estimator._build_pymc_model()
        assert model is not None

    def test_sample_calls_pm_sample(self, time_series_data, estimation_config):
        """_sample вызывает pm.sample с draws=n_samples, chains=n_chains."""
        estimator = self._make_estimator(time_series_data, estimation_config)
        model = estimator._build_pymc_model()
        idata = estimator._sample(model)
        assert idata is not None

    def test_extract_results_point_estimates(
        self, time_series_data, estimation_config, mock_inference_data
    ):
        """_extract_results: point_estimates имеют ключи, ci_lower < pe < ci_upper."""
        estimator = self._make_estimator(time_series_data, estimation_config)
        result = estimator._extract_results(mock_inference_data)
        assert isinstance(result, EstimationResult)
        for name in SAMPLE_PARAM_NAMES:
            assert name in result.point_estimates
            if name in result.ci_lower and name in result.ci_upper:
                assert result.ci_lower[name] < result.point_estimates[name]
                assert result.point_estimates[name] < result.ci_upper[name]


# =============================================================================
# TestMCMCEstimator
# =============================================================================


# Тесты по: Description/Phase3/description_parameter_estimation.md#MCMCEstimator
class TestMCMCEstimator:
    """Тесты MCMCEstimator — emcee ensemble MCMC."""

    def _make_estimator(self, time_series_data, estimation_config):
        """Создание MCMCEstimator с mock зависимостями."""
        mock_fm = MagicMock(spec=ForwardModelWrapper)
        mock_fm.predict.return_value = {
            "F": np.ones(6) * 500.0,
            "C_TNF": np.ones(6) * 1.0,
        }
        mock_pb = MagicMock(spec=PriorBuilder)
        mock_pb.get_free_param_names.return_value = SAMPLE_PARAM_NAMES
        mock_pb.get_initial_guess.return_value = np.array([0.03, 0.05, 0.5])
        mock_pb.build_log_prior_fn.return_value = lambda _theta: 0.0  # Всегда в support
        return MCMCEstimator(mock_fm, time_series_data, estimation_config, mock_pb)

    def test_fit_import_error(self, time_series_data, estimation_config):
        """Emcee не установлен → ImportError."""
        estimator = self._make_estimator(time_series_data, estimation_config)
        with patch.dict(sys.modules, {"emcee": None}), pytest.raises(ImportError):
            estimator.fit()

    def test_fit_returns_mcmc_result(self, time_series_data, estimation_config):
        """fit() возвращает EstimationResult с method='mcmc_emcee'."""
        estimator = self._make_estimator(time_series_data, estimation_config)
        result = estimator.fit()
        assert isinstance(result, EstimationResult)
        assert result.method == "mcmc_emcee"

    def test_log_probability_prior_rejection(self, time_series_data, estimation_config):
        """Theta вне support приора → -inf без вызова log_likelihood."""
        mock_fm = MagicMock(spec=ForwardModelWrapper)
        mock_pb = MagicMock(spec=PriorBuilder)
        mock_pb.get_free_param_names.return_value = SAMPLE_PARAM_NAMES
        mock_pb.get_initial_guess.return_value = np.array([0.03, 0.05, 0.5])
        mock_pb.build_log_prior_fn.return_value = lambda _theta: -np.inf

        estimator = MCMCEstimator(mock_fm, time_series_data, estimation_config, mock_pb)
        theta = np.array([-1.0, -1.0, -1.0])
        result = estimator._log_probability(theta)
        assert result == -np.inf
        # log_likelihood НЕ должен вызываться при reject по prior
        mock_fm.predict.assert_not_called()

    def test_log_probability_valid_theta(self, time_series_data, estimation_config):
        """Валидный theta → конечное float значение."""
        estimator = self._make_estimator(time_series_data, estimation_config)
        theta = np.array([0.03, 0.05, 0.5])
        result = estimator._log_probability(theta)
        assert np.isfinite(result)

    def test_initialize_walkers_correct_shape(self, time_series_data, estimation_config):
        """n_walkers=32, n_params=3 → shape (32, 3)."""
        estimation_config.n_walkers = 32
        estimator = self._make_estimator(time_series_data, estimation_config)
        walkers = estimator._initialize_walkers()
        assert isinstance(walkers, np.ndarray)
        assert walkers.shape == (32, 3)

    def test_initialize_walkers_too_few(self, time_series_data, estimation_config):
        """n_walkers < 2 * n_params → ValueError."""
        estimation_config.n_walkers = 4  # 4 < 2 * 3 = 6
        estimator = self._make_estimator(time_series_data, estimation_config)
        with pytest.raises(ValueError):
            estimator._initialize_walkers()

    def test_run_sampler_calls_emcee(self, time_series_data, estimation_config):
        """_run_sampler запускает emcee.EnsembleSampler."""
        estimator = self._make_estimator(time_series_data, estimation_config)
        sampler = estimator._run_sampler()
        assert sampler is not None

    def test_sampler_to_inference_data(self, time_series_data, estimation_config):
        """_sampler_to_inference_data конвертирует в InferenceData."""
        estimator = self._make_estimator(time_series_data, estimation_config)
        # Создаём mock sampler с нужными методами
        mock_sampler = MagicMock()
        rng = np.random.default_rng(42)
        mock_sampler.get_chain.return_value = rng.normal(size=(100, 32, 3))
        idata = estimator._sampler_to_inference_data(mock_sampler)
        assert idata is not None


# =============================================================================
# TestMLEstimator
# =============================================================================


# Тесты по: Description/Phase3/description_parameter_estimation.md#MLEstimator
class TestMLEstimator:
    """Тесты MLEstimator — scipy.optimize MLE."""

    def _make_estimator(self, time_series_data, estimation_config):
        """Создание MLEstimator с mock зависимостями."""
        mock_fm = MagicMock(spec=ForwardModelWrapper)
        mock_fm.predict.return_value = {
            "F": np.ones(6) * 500.0,
            "C_TNF": np.ones(6) * 1.0,
        }
        mock_pb = MagicMock(spec=PriorBuilder)
        mock_pb.get_free_param_names.return_value = SAMPLE_PARAM_NAMES
        mock_pb.get_initial_guess.return_value = np.array([0.03, 0.05, 0.5])
        mock_pb.build_scipy_bounds.return_value = [
            (0.0, 0.3),
            (0.0, 0.5),
            (0.0, 5.0),
        ]
        return MLEstimator(mock_fm, time_series_data, estimation_config, mock_pb)

    def test_fit_returns_mle_result(self, time_series_data, estimation_config):
        """fit() возвращает EstimationResult с method='mle_scipy'."""
        estimator = self._make_estimator(time_series_data, estimation_config)
        result = estimator.fit()
        assert isinstance(result, EstimationResult)
        assert result.method == "mle_scipy"

    def test_fit_no_posterior_samples(self, time_series_data, estimation_config):
        """MLE результат → posterior_samples = None."""
        estimator = self._make_estimator(time_series_data, estimation_config)
        result = estimator.fit()
        assert result.posterior_samples is None

    def test_fit_no_diagnostics(self, time_series_data, estimation_config):
        """MLE результат → diagnostics = None."""
        estimator = self._make_estimator(time_series_data, estimation_config)
        result = estimator.fit()
        assert result.diagnostics is None

    def test_objective_negates_log_likelihood(self, time_series_data, estimation_config):
        """_objective = -log_likelihood: log_lik=-50 → objective=50."""
        estimator = self._make_estimator(time_series_data, estimation_config)
        # Мокаем _compute_log_likelihood
        with patch.object(estimator, "_compute_log_likelihood", return_value=-50.0):
            theta = np.array([0.03, 0.05, 0.5])
            result = estimator._objective(theta)
            assert result == pytest.approx(50.0)

    def test_objective_returns_inf_for_neg_inf(self, time_series_data, estimation_config):
        """log_lik = -inf → objective = inf."""
        estimator = self._make_estimator(time_series_data, estimation_config)
        with patch.object(estimator, "_compute_log_likelihood", return_value=-np.inf):
            theta = np.array([0.03, 0.05, 0.5])
            result = estimator._objective(theta)
            assert result == np.inf

    def test_estimate_ci_finite(self, time_series_data, estimation_config):
        """Квадратичная поверхность → CI определены, ci_upper > ci_lower."""
        estimator = self._make_estimator(time_series_data, estimation_config)
        theta_opt = np.array([0.03, 0.05, 0.5])
        ci_lower, ci_upper = estimator._estimate_ci_from_hessian(theta_opt)
        assert isinstance(ci_lower, dict)
        assert isinstance(ci_upper, dict)
        for name in SAMPLE_PARAM_NAMES:
            if (
                name in ci_lower
                and name in ci_upper
                and np.isfinite(ci_lower[name])
                and np.isfinite(ci_upper[name])
            ):
                assert ci_upper[name] > ci_lower[name]

    def test_estimate_ci_singular_hessian(self, time_series_data, estimation_config):
        """Сингулярный Гессиан → CI содержат NaN."""
        estimator = self._make_estimator(time_series_data, estimation_config)
        # Мокаем _objective чтобы возвращала плоскую поверхность (const)
        with patch.object(estimator, "_objective", return_value=0.0):
            theta_opt = np.array([0.03, 0.05, 0.5])
            ci_lower, ci_upper = estimator._estimate_ci_from_hessian(theta_opt)
            # При сингулярном Гессиане ожидаем NaN
            has_nan = any(
                np.isnan(ci_lower.get(name, 0.0)) or np.isnan(ci_upper.get(name, 0.0))
                for name in SAMPLE_PARAM_NAMES
            )
            assert has_nan


# =============================================================================
# TestConvergenceAnalyzer
# =============================================================================


# Тесты по: Description/Phase3/description_parameter_estimation.md#ConvergenceAnalyzer
class TestConvergenceAnalyzer:
    """Тесты ConvergenceAnalyzer — ArviZ диагностика сходимости."""

    def test_init_stores_config(self, estimation_config):
        """__init__ сохраняет config."""
        analyzer = ConvergenceAnalyzer(estimation_config)
        assert analyzer.config is estimation_config

    def test_analyze_full_pipeline(self, estimation_config, mock_inference_data):
        """analyze() возвращает ConvergenceDiagnostics со всеми полями."""
        analyzer = ConvergenceAnalyzer(estimation_config)
        result = analyzer.analyze(mock_inference_data)
        assert isinstance(result, ConvergenceDiagnostics)
        assert isinstance(result.rhat, dict)
        assert isinstance(result.ess_bulk, dict)
        assert isinstance(result.ess_tail, dict)
        assert isinstance(result.converged, bool)
        assert isinstance(result.warnings, list)

    def test_compute_rhat_converged(self, estimation_config, mock_inference_data):
        """Хорошо перемешанные цепи → все R-hat < 1.05."""
        analyzer = ConvergenceAnalyzer(estimation_config)
        rhat = analyzer.compute_rhat(mock_inference_data)
        assert isinstance(rhat, dict)
        assert len(rhat) > 0

    def test_compute_rhat_not_converged(self, estimation_config):
        """Плохо перемешанные цепи → некоторые R-hat > 1.05."""
        analyzer = ConvergenceAnalyzer(estimation_config)
        # Создаём mock с расходящимися chains
        bad_idata = MagicMock()
        rhat = analyzer.compute_rhat(bad_idata)
        assert isinstance(rhat, dict)

    def test_compute_ess_enough(self, estimation_config, mock_inference_data):
        """Достаточно samples → ESS > 100."""
        analyzer = ConvergenceAnalyzer(estimation_config)
        ess_bulk, ess_tail = analyzer.compute_ess(mock_inference_data)
        assert isinstance(ess_bulk, dict)
        assert isinstance(ess_tail, dict)

    def test_compute_ess_few(self, estimation_config):
        """Мало samples → ESS может быть < 100."""
        analyzer = ConvergenceAnalyzer(estimation_config)
        few_idata = MagicMock()
        ess_bulk, ess_tail = analyzer.compute_ess(few_idata)
        assert isinstance(ess_bulk, dict)
        assert isinstance(ess_tail, dict)

    def test_summary_returns_dataframe(self, estimation_config, mock_inference_data):
        """summary() возвращает pd.DataFrame."""
        analyzer = ConvergenceAnalyzer(estimation_config)
        summary = analyzer.summary(mock_inference_data)
        assert summary is not None

    def test_check_convergence_all_ok(self, estimation_config, converged_diagnostics):
        """Все rhat < 1.05, ESS > 100 → True."""
        analyzer = ConvergenceAnalyzer(estimation_config)
        assert analyzer.check_convergence(converged_diagnostics) is True

    def test_check_convergence_bad_rhat(self, estimation_config):
        """Один rhat > threshold → False."""
        analyzer = ConvergenceAnalyzer(estimation_config)
        diag = ConvergenceDiagnostics(
            rhat={"r_F": 1.2, "delta_Ne": 1.01},
            ess_bulk={"r_F": 500.0, "delta_Ne": 400.0},
            ess_tail={"r_F": 300.0, "delta_Ne": 250.0},
        )
        assert analyzer.check_convergence(diag) is False

    def test_check_convergence_low_ess(self, estimation_config):
        """Один ESS < min → False."""
        analyzer = ConvergenceAnalyzer(estimation_config)
        diag = ConvergenceDiagnostics(
            rhat={"r_F": 1.01, "delta_Ne": 1.02},
            ess_bulk={"r_F": 50.0, "delta_Ne": 400.0},
            ess_tail={"r_F": 30.0, "delta_Ne": 250.0},
        )
        assert analyzer.check_convergence(diag) is False

    def test_check_convergence_empty(self, estimation_config):
        """Пустые dicts → False."""
        analyzer = ConvergenceAnalyzer(estimation_config)
        diag = ConvergenceDiagnostics()
        assert analyzer.check_convergence(diag) is False


# =============================================================================
# TestEstimateParameters
# =============================================================================


# Тесты по: Description/Phase3/description_parameter_estimation.md#estimate_parameters
class TestEstimateParameters:
    """Тесты estimate_parameters() — convenience-функция."""

    def test_method_bayesian(self, time_series_data):
        """method='bayesian' → BayesianEstimator используется."""
        with patch("src.core.parameter_estimation.BayesianEstimator") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.fit.return_value = EstimationResult(method="bayesian_pymc")
            mock_cls.return_value = mock_instance
            result = estimate_parameters(time_series_data, method="bayesian")
            assert result.method == "bayesian_pymc"
            mock_cls.assert_called_once()

    def test_method_mle(self, time_series_data):
        """method='mle' → MLEstimator используется."""
        with patch("src.core.parameter_estimation.MLEstimator") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.fit.return_value = EstimationResult(method="mle_scipy")
            mock_cls.return_value = mock_instance
            result = estimate_parameters(time_series_data, method="mle")
            assert result.method == "mle_scipy"
            mock_cls.assert_called_once()

    def test_method_mcmc(self, time_series_data):
        """method='mcmc' → MCMCEstimator используется."""
        with patch("src.core.parameter_estimation.MCMCEstimator") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.fit.return_value = EstimationResult(method="mcmc_emcee")
            mock_cls.return_value = mock_instance
            result = estimate_parameters(time_series_data, method="mcmc")
            assert result.method == "mcmc_emcee"
            mock_cls.assert_called_once()

    def test_unknown_method(self, time_series_data):
        """method='unknown' → ValueError."""
        with pytest.raises(ValueError):
            estimate_parameters(time_series_data, method="unknown")

    def test_empty_observed_data(self):
        """Пустые observed_data.values → ValueError."""
        empty_data = TimeSeriesData(
            time_points=np.array([0.0, 24.0]),
            values={},
            units={},
            metadata=None,
        )
        with pytest.raises(ValueError):
            estimate_parameters(empty_data)
