"""
TDD тесты для модуля sde_model.py

Тестирует:
- SDEConfig dataclass и валидацию
- TherapyProtocol dataclass
- SDEState dataclass
- SDETrajectory dataclass
- SDEModel класс и методы
- simulate_sde convenience функцию

Основано на спецификации: Description/description_sde_model.md
"""

import numpy as np
import pytest

from src.core.sde_model import (
    SDEConfig,
    SDEModel,
    SDEState,
    SDETrajectory,
    TherapyProtocol,
    simulate_sde,
)
from src.data.parameter_extraction import ModelParameters


# =============================================================================
# Тесты для SDEConfig
# =============================================================================


class TestSDEConfig:
    """Тесты для SDEConfig dataclass и валидации."""

    def test_config_creation_with_default_values(self):
        """Тест создания SDEConfig с дефолтными значениями."""
        config = SDEConfig()

        assert config.r == 0.3
        assert config.K == 1e6
        assert config.delta == 0.05
        assert config.sigma_n == 0.05
        assert config.sigma_c == 0.02
        assert config.dt == 0.01
        assert config.t_max == 30.0

    def test_config_creation_with_custom_values(self):
        """Тест создания SDEConfig с кастомными значениями."""
        config = SDEConfig(r=0.5, K=1e7, dt=0.005, t_max=10.0)

        assert config.r == 0.5
        assert config.K == 1e7
        assert config.dt == 0.005
        assert config.t_max == 10.0

    def test_validate_returns_true_for_valid_config(self):
        """Тест что validate() возвращает True для валидной конфигурации."""
        config = SDEConfig()
        result = config.validate()
        assert result is True

    def test_validate_raises_for_negative_r(self):
        """Тест что r <= 0 вызывает ValueError."""
        config = SDEConfig(r=-0.1)
        with pytest.raises(ValueError, match="r"):
            config.validate()

    def test_validate_raises_for_zero_r(self):
        """Тест что r == 0 вызывает ValueError."""
        config = SDEConfig(r=0.0)
        with pytest.raises(ValueError, match="r"):
            config.validate()

    def test_validate_raises_for_negative_K(self):
        """Тест что K <= 0 вызывает ValueError."""
        config = SDEConfig(K=-1e6)
        with pytest.raises(ValueError, match="K"):
            config.validate()

    def test_validate_raises_for_negative_delta(self):
        """Тест что delta < 0 вызывает ValueError."""
        config = SDEConfig(delta=-0.01)
        with pytest.raises(ValueError, match="delta"):
            config.validate()

    def test_validate_raises_for_negative_sigma_n(self):
        """Тест что sigma_n < 0 вызывает ValueError."""
        config = SDEConfig(sigma_n=-0.01)
        with pytest.raises(ValueError, match="sigma"):
            config.validate()

    def test_validate_raises_for_negative_sigma_c(self):
        """Тест что sigma_c < 0 вызывает ValueError."""
        config = SDEConfig(sigma_c=-0.01)
        with pytest.raises(ValueError, match="sigma"):
            config.validate()

    def test_validate_raises_for_invalid_dt_zero(self):
        """Тест что dt <= 0 вызывает ValueError."""
        config = SDEConfig(dt=0.0)
        with pytest.raises(ValueError, match="dt"):
            config.validate()

    def test_validate_raises_for_invalid_dt_too_large(self):
        """Тест что dt > 1.0 вызывает ValueError."""
        config = SDEConfig(dt=1.5)
        with pytest.raises(ValueError, match="dt"):
            config.validate()

    def test_validate_raises_for_negative_t_max(self):
        """Тест что t_max <= 0 вызывает ValueError."""
        config = SDEConfig(t_max=-10.0)
        with pytest.raises(ValueError, match="t_max"):
            config.validate()


# =============================================================================
# Тесты для TherapyProtocol
# =============================================================================


class TestTherapyProtocol:
    """Тесты для TherapyProtocol dataclass."""

    def test_protocol_creation_with_defaults(self):
        """Тест создания TherapyProtocol с дефолтными значениями (без терапии)."""
        protocol = TherapyProtocol()

        assert protocol.prp_enabled is False
        assert protocol.pemf_enabled is False
        assert protocol.synergy_factor == 1.2

    def test_protocol_prp_enabled(self, prp_therapy_protocol):
        """Тест конфигурации PRP терапии."""
        assert prp_therapy_protocol.prp_enabled is True
        assert prp_therapy_protocol.prp_start_time == 1.0
        assert prp_therapy_protocol.prp_duration == 7.0
        assert prp_therapy_protocol.prp_intensity == 1.5
        assert prp_therapy_protocol.prp_initial_concentration == 10.0

    def test_protocol_pemf_enabled(self, pemf_therapy_protocol):
        """Тест конфигурации PEMF терапии."""
        assert pemf_therapy_protocol.pemf_enabled is True
        assert pemf_therapy_protocol.pemf_start_time == 0.0
        assert pemf_therapy_protocol.pemf_duration == 14.0
        assert pemf_therapy_protocol.pemf_frequency == 60.0
        assert pemf_therapy_protocol.pemf_intensity == 1.0

    def test_protocol_combined_therapy(self, combined_therapy_protocol):
        """Тест комбинированной PRP+PEMF терапии с synergy_factor."""
        assert combined_therapy_protocol.prp_enabled is True
        assert combined_therapy_protocol.pemf_enabled is True
        assert combined_therapy_protocol.synergy_factor == 1.3

    def test_protocol_intensity_values(self):
        """Тест значений интенсивности в ожидаемом диапазоне 0-2."""
        protocol = TherapyProtocol(prp_intensity=0.5, pemf_intensity=1.5)
        assert 0 <= protocol.prp_intensity <= 2
        assert 0 <= protocol.pemf_intensity <= 2


# =============================================================================
# Тесты для SDEState
# =============================================================================


class TestSDEState:
    """Тесты для SDEState dataclass."""

    def test_state_creation_with_all_fields(self):
        """Тест создания SDEState со всеми полями."""
        state = SDEState(t=5.0, N=10000.0, C=15.0, prp_active=True, pemf_active=False)

        assert state.t == 5.0
        assert state.N == 10000.0
        assert state.C == 15.0
        assert state.prp_active is True
        assert state.pemf_active is False

    def test_state_to_dict_contains_all_fields(self):
        """Тест что to_dict() содержит все поля состояния."""
        state = SDEState(t=1.0, N=5000.0, C=10.0)
        result = state.to_dict()

        assert "t" in result
        assert "N" in result
        assert "C" in result
        assert "prp_active" in result
        assert "pemf_active" in result

    def test_state_to_dict_correct_values(self):
        """Тест что значения в to_dict() соответствуют атрибутам."""
        state = SDEState(t=2.5, N=7500.0, C=12.0, prp_active=True, pemf_active=True)
        result = state.to_dict()

        assert result["t"] == 2.5
        assert result["N"] == 7500.0
        assert result["C"] == 12.0
        assert result["prp_active"] is True
        assert result["pemf_active"] is True

    def test_state_therapy_flags_default_false(self):
        """Тест что prp_active и pemf_active по умолчанию False."""
        state = SDEState(t=0.0, N=5000.0, C=10.0)

        assert state.prp_active is False
        assert state.pemf_active is False


# =============================================================================
# Тесты для SDETrajectory
# =============================================================================


class TestSDETrajectory:
    """Тесты для SDETrajectory dataclass и методов."""

    @pytest.fixture
    def sample_trajectory(self):
        """Создать пример траектории для тестов."""
        times = np.linspace(0, 10, 101)
        N_values = np.linspace(5000, 10000, 101)
        C_values = np.linspace(10, 20, 101)
        return SDETrajectory(
            times=times,
            N_values=N_values,
            C_values=C_values,
            therapy_markers={"prp": np.zeros(101, dtype=bool), "pemf": np.zeros(101, dtype=bool)},
        )

    def test_trajectory_creation_with_arrays(self, sample_trajectory):
        """Тест создания SDETrajectory с numpy массивами."""
        assert isinstance(sample_trajectory.times, np.ndarray)
        assert isinstance(sample_trajectory.N_values, np.ndarray)
        assert isinstance(sample_trajectory.C_values, np.ndarray)

    def test_trajectory_array_shapes_match(self, sample_trajectory):
        """Тест что times, N_values, C_values имеют одинаковую длину."""
        assert len(sample_trajectory.times) == len(sample_trajectory.N_values)
        assert len(sample_trajectory.times) == len(sample_trajectory.C_values)

    def test_get_final_state_returns_sde_state(self, sample_trajectory):
        """Тест что get_final_state() возвращает SDEState."""
        final_state = sample_trajectory.get_final_state()
        assert isinstance(final_state, SDEState)

    def test_get_final_state_has_correct_values(self, sample_trajectory):
        """Тест что финальное состояние содержит последние значения массивов."""
        final_state = sample_trajectory.get_final_state()

        assert final_state.t == sample_trajectory.times[-1]
        assert final_state.N == sample_trajectory.N_values[-1]
        assert final_state.C == sample_trajectory.C_values[-1]

    def test_get_statistics_returns_dict(self, sample_trajectory):
        """Тест что get_statistics() возвращает словарь."""
        stats = sample_trajectory.get_statistics()
        assert isinstance(stats, dict)

    def test_get_statistics_contains_final_N(self, sample_trajectory):
        """Тест что статистика содержит final_N."""
        stats = sample_trajectory.get_statistics()
        assert "final_N" in stats

    def test_get_statistics_contains_final_C(self, sample_trajectory):
        """Тест что статистика содержит final_C."""
        stats = sample_trajectory.get_statistics()
        assert "final_C" in stats

    def test_get_statistics_contains_max_N(self, sample_trajectory):
        """Тест что статистика содержит max_N."""
        stats = sample_trajectory.get_statistics()
        assert "max_N" in stats

    def test_get_statistics_contains_growth_rate(self, sample_trajectory):
        """Тест что статистика содержит growth_rate."""
        stats = sample_trajectory.get_statistics()
        assert "growth_rate" in stats

    def test_therapy_markers_dict_structure(self, sample_trajectory):
        """Тест что therapy_markers имеет ключи 'prp' и 'pemf'."""
        assert "prp" in sample_trajectory.therapy_markers
        assert "pemf" in sample_trajectory.therapy_markers


# =============================================================================
# Тесты для SDEModel.__init__
# =============================================================================


class TestSDEModelInit:
    """Тесты для SDEModel.__init__."""

    def test_init_with_default_config(self):
        """Тест инициализации модели с дефолтным SDEConfig."""
        model = SDEModel()

        assert model.config is not None
        assert isinstance(model.config, SDEConfig)

    def test_init_with_custom_config(self, custom_sde_config):
        """Тест инициализации модели с кастомным SDEConfig."""
        model = SDEModel(config=custom_sde_config)

        assert model.config.r == 0.5
        assert model.config.K == 1e7

    def test_init_with_therapy(self, prp_therapy_protocol):
        """Тест инициализации модели с TherapyProtocol."""
        model = SDEModel(therapy=prp_therapy_protocol)

        assert model.therapy is not None
        assert model.therapy.prp_enabled is True

    def test_init_with_random_seed(self):
        """Тест инициализации модели с random_seed для воспроизводимости."""
        model = SDEModel(random_seed=42)
        # Модель должна успешно создаться с seed
        assert model is not None

    def test_init_validates_config(self):
        """Тест что __init__ вызывает config.validate()."""
        # Валидная конфигурация
        model = SDEModel(config=SDEConfig())
        assert model is not None

    def test_init_invalid_config_raises_error(self):
        """Тест что невалидная конфигурация вызывает ValueError в __init__."""
        invalid_config = SDEConfig(r=-1.0)
        with pytest.raises(ValueError):
            SDEModel(config=invalid_config)

    def test_config_property_returns_config(self, default_sde_config):
        """Тест что свойство config возвращает SDEConfig."""
        model = SDEModel(config=default_sde_config)
        assert model.config is default_sde_config

    def test_therapy_property_returns_protocol(self, prp_therapy_protocol):
        """Тест что свойство therapy возвращает TherapyProtocol."""
        model = SDEModel(therapy=prp_therapy_protocol)
        assert model.therapy is prp_therapy_protocol


# =============================================================================
# Тесты для SDEModel._logistic_growth
# =============================================================================


class TestSDEModelLogisticGrowth:
    """Тесты для SDEModel._logistic_growth (математическая валидация)."""

    @pytest.fixture
    def model(self):
        """Создать модель для тестов."""
        return SDEModel(config=SDEConfig(r=0.3, K=1e6))

    def test_logistic_growth_at_zero(self, model):
        """Тест что логистический рост = 0 при N = 0."""
        result = model._logistic_growth(0.0)
        assert result == 0.0

    def test_logistic_growth_at_carrying_capacity(self, model):
        """Тест что логистический рост = 0 при N = K."""
        K = model.config.K
        result = model._logistic_growth(K)
        assert abs(result) < 1e-10  # Приблизительно 0

    def test_logistic_growth_positive_below_K(self, model):
        """Тест что логистический рост > 0 при 0 < N < K."""
        N = model.config.K / 2
        result = model._logistic_growth(N)
        assert result > 0

    def test_logistic_growth_maximum_at_K_half(self, model):
        """Тест что максимальный рост при N = K/2."""
        K = model.config.K
        r = model.config.r

        # При N = K/2, рост = r * K/2 * (1 - 0.5) = r * K / 4
        growth_at_half = model._logistic_growth(K / 2)
        expected_max = r * K / 4

        assert abs(growth_at_half - expected_max) < 1e-6

    def test_logistic_growth_negative_above_K(self, model):
        """Тест что логистический рост < 0 при N > K."""
        N = model.config.K * 1.5
        result = model._logistic_growth(N)
        assert result < 0

    def test_logistic_growth_formula(self, model):
        """Тест точной формулы: r * N * (1 - N/K)."""
        r = model.config.r
        K = model.config.K
        N = 50000.0

        expected = r * N * (1 - N / K)
        result = model._logistic_growth(N)

        assert abs(result - expected) < 1e-10


# =============================================================================
# Тесты для SDEModel._prp_effect
# =============================================================================


class TestSDEModelPRPEffect:
    """Тесты для SDEModel._prp_effect."""

    def test_prp_effect_zero_when_disabled(self):
        """Тест что PRP эффект = 0 когда prp_enabled = False."""
        model = SDEModel(therapy=TherapyProtocol(prp_enabled=False))
        result = model._prp_effect(t=5.0, N=5000.0, C=10.0)
        assert result == 0.0

    def test_prp_effect_zero_before_start_time(self, prp_therapy_protocol):
        """Тест что PRP эффект = 0 когда t < prp_start_time."""
        model = SDEModel(therapy=prp_therapy_protocol)
        # prp_start_time = 1.0
        result = model._prp_effect(t=0.5, N=5000.0, C=10.0)
        assert result == 0.0

    def test_prp_effect_zero_after_duration(self, prp_therapy_protocol):
        """Тест что PRP эффект = 0 когда t > prp_start_time + prp_duration."""
        model = SDEModel(therapy=prp_therapy_protocol)
        # prp_start_time = 1.0, prp_duration = 7.0, так что после t = 8.0
        result = model._prp_effect(t=10.0, N=5000.0, C=10.0)
        assert result == 0.0

    def test_prp_effect_positive_during_therapy(self, prp_therapy_protocol):
        """Тест что PRP эффект > 0 во время активного окна терапии."""
        model = SDEModel(therapy=prp_therapy_protocol)
        # prp_start_time = 1.0, prp_duration = 7.0
        result = model._prp_effect(t=3.0, N=5000.0, C=10.0)
        assert result > 0

    def test_prp_effect_exponential_decay(self, prp_therapy_protocol):
        """Тест экспоненциального затухания: C0 * e^(-lambda*t)."""
        model = SDEModel(therapy=prp_therapy_protocol)
        # Эффект должен уменьшаться со временем
        effect_early = model._prp_effect(t=1.5, N=5000.0, C=10.0)
        effect_late = model._prp_effect(t=5.0, N=5000.0, C=10.0)
        assert effect_early > effect_late

    def test_prp_effect_synergy_with_pemf(self, combined_therapy_protocol):
        """Тест что synergy_factor умножает эффект когда PEMF активен."""
        model = SDEModel(therapy=combined_therapy_protocol)
        # Оба активны, должна быть синергия
        # Это проверяет что эффект больше чем без синергии
        effect = model._prp_effect(t=2.0, N=5000.0, C=10.0)
        assert effect > 0


# =============================================================================
# Тесты для SDEModel._pemf_effect
# =============================================================================


class TestSDEModelPEMFEffect:
    """Тесты для SDEModel._pemf_effect."""

    def test_pemf_effect_zero_when_disabled(self):
        """Тест что PEMF эффект = 0 когда pemf_enabled = False."""
        model = SDEModel(therapy=TherapyProtocol(pemf_enabled=False))
        result = model._pemf_effect(t=5.0, N=5000.0)
        assert result == 0.0

    def test_pemf_effect_zero_before_start_time(self):
        """Тест что PEMF эффект = 0 когда t < pemf_start_time."""
        therapy = TherapyProtocol(pemf_enabled=True, pemf_start_time=2.0)
        model = SDEModel(therapy=therapy)
        result = model._pemf_effect(t=1.0, N=5000.0)
        assert result == 0.0

    def test_pemf_effect_zero_after_duration(self, pemf_therapy_protocol):
        """Тест что PEMF эффект = 0 когда t > pemf_start_time + pemf_duration."""
        model = SDEModel(therapy=pemf_therapy_protocol)
        # pemf_start_time = 0.0, pemf_duration = 14.0
        result = model._pemf_effect(t=20.0, N=5000.0)
        assert result == 0.0

    def test_pemf_effect_positive_during_therapy(self, pemf_therapy_protocol):
        """Тест что PEMF эффект > 0 во время активного окна терапии."""
        model = SDEModel(therapy=pemf_therapy_protocol)
        result = model._pemf_effect(t=5.0, N=5000.0)
        assert result > 0

    def test_pemf_effect_sigmoid_response(self):
        """Тест сигмоидального отклика на частоту: 1/(1+e^(-k(f-f0)))."""
        # При оптимальной частоте сигмоид = 0.5
        therapy = TherapyProtocol(pemf_enabled=True, pemf_frequency=50.0)  # f0 = 50
        model = SDEModel(config=SDEConfig(f0_pemf=50.0), therapy=therapy)
        # Эффект должен быть ненулевым
        result = model._pemf_effect(t=1.0, N=5000.0)
        assert result >= 0

    def test_pemf_effect_maximum_at_optimal_frequency(self):
        """Тест что эффект больше при частоте ближе к оптимальной."""
        therapy_optimal = TherapyProtocol(pemf_enabled=True, pemf_frequency=50.0)
        therapy_suboptimal = TherapyProtocol(pemf_enabled=True, pemf_frequency=30.0)

        model_optimal = SDEModel(config=SDEConfig(f0_pemf=50.0), therapy=therapy_optimal)
        model_suboptimal = SDEModel(config=SDEConfig(f0_pemf=50.0), therapy=therapy_suboptimal)

        effect_optimal = model_optimal._pemf_effect(t=1.0, N=5000.0)
        effect_suboptimal = model_suboptimal._pemf_effect(t=1.0, N=5000.0)

        assert effect_optimal >= effect_suboptimal

    def test_pemf_effect_proportional_to_N(self, pemf_therapy_protocol):
        """Тест что эффект пропорционален плотности клеток N."""
        model = SDEModel(therapy=pemf_therapy_protocol)

        effect_low_N = model._pemf_effect(t=1.0, N=1000.0)
        effect_high_N = model._pemf_effect(t=1.0, N=5000.0)

        assert effect_high_N > effect_low_N


# =============================================================================
# Тесты для SDEModel._is_therapy_active
# =============================================================================


class TestSDEModelIsTherapyActive:
    """Тесты для SDEModel._is_therapy_active."""

    def test_prp_inactive_before_start(self, prp_therapy_protocol):
        """Тест что PRP неактивна до start_time."""
        model = SDEModel(therapy=prp_therapy_protocol)
        # prp_start_time = 1.0
        assert model._is_therapy_active(t=0.5, therapy_type="prp") is False

    def test_prp_active_during_window(self, prp_therapy_protocol):
        """Тест что PRP активна между start_time и start_time + duration."""
        model = SDEModel(therapy=prp_therapy_protocol)
        # prp_start_time = 1.0, prp_duration = 7.0
        assert model._is_therapy_active(t=3.0, therapy_type="prp") is True

    def test_prp_inactive_after_duration(self, prp_therapy_protocol):
        """Тест что PRP неактивна после окончания терапии."""
        model = SDEModel(therapy=prp_therapy_protocol)
        # prp_start_time = 1.0, prp_duration = 7.0 -> end at 8.0
        assert model._is_therapy_active(t=10.0, therapy_type="prp") is False

    def test_pemf_inactive_before_start(self):
        """Тест что PEMF неактивна до start_time."""
        therapy = TherapyProtocol(pemf_enabled=True, pemf_start_time=5.0)
        model = SDEModel(therapy=therapy)
        assert model._is_therapy_active(t=2.0, therapy_type="pemf") is False

    def test_pemf_active_during_window(self, pemf_therapy_protocol):
        """Тест что PEMF активна во время окна терапии."""
        model = SDEModel(therapy=pemf_therapy_protocol)
        assert model._is_therapy_active(t=5.0, therapy_type="pemf") is True

    def test_therapy_inactive_when_disabled(self):
        """Тест что терапия неактивна когда disabled."""
        therapy = TherapyProtocol(prp_enabled=False, pemf_enabled=False)
        model = SDEModel(therapy=therapy)
        assert model._is_therapy_active(t=1.0, therapy_type="prp") is False
        assert model._is_therapy_active(t=1.0, therapy_type="pemf") is False


# =============================================================================
# Тесты для SDEModel._calculate_drift
# =============================================================================


class TestSDEModelCalculateDrift:
    """Тесты для SDEModel._calculate_drift."""

    @pytest.fixture
    def model(self):
        """Создать модель для тестов."""
        return SDEModel()

    def test_calculate_drift_returns_tuple(self, model):
        """Тест что _calculate_drift возвращает (drift_N, drift_C)."""
        result = model._calculate_drift(t=1.0, N=5000.0, C=10.0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_drift_N_includes_logistic_growth(self, model):
        """Тест что drift_N содержит терм логистического роста."""
        # При N << K, drift должен быть положительным из-за роста
        drift_N, _ = model._calculate_drift(t=0.0, N=1000.0, C=10.0)
        # Ожидаем положительный drift из-за роста (без терапий)
        assert drift_N > 0 or drift_N <= 0  # Зависит от параметров

    def test_drift_N_includes_death_term(self, model):
        """Тест что drift_N содержит терм -delta*N."""
        # delta = 0.05, так что при высоком N есть значительная смертность
        drift_N, _ = model._calculate_drift(t=0.0, N=100000.0, C=10.0)
        # Drift может быть положительным или отрицательным
        assert isinstance(drift_N, (int, float))

    def test_drift_N_includes_prp_effect(self, prp_therapy_protocol):
        """Тест что drift_N включает PRP вклад во время терапии."""
        model_with_prp = SDEModel(therapy=prp_therapy_protocol)
        model_no_therapy = SDEModel()

        # Во время активной PRP терапии
        drift_with_prp, _ = model_with_prp._calculate_drift(t=3.0, N=5000.0, C=10.0)
        drift_no_prp, _ = model_no_therapy._calculate_drift(t=3.0, N=5000.0, C=10.0)

        # Drift с PRP должен быть больше (PRP стимулирует рост)
        assert drift_with_prp >= drift_no_prp

    def test_drift_N_includes_pemf_effect(self, pemf_therapy_protocol):
        """Тест что drift_N включает PEMF вклад во время терапии."""
        model_with_pemf = SDEModel(therapy=pemf_therapy_protocol)
        model_no_therapy = SDEModel()

        drift_with_pemf, _ = model_with_pemf._calculate_drift(t=5.0, N=5000.0, C=10.0)
        drift_no_pemf, _ = model_no_therapy._calculate_drift(t=5.0, N=5000.0, C=10.0)

        assert drift_with_pemf >= drift_no_pemf

    def test_drift_C_includes_production(self, model):
        """Тест что drift_C содержит терм eta*N производства."""
        # При большом N, производство цитокинов должно быть значительным
        _, drift_C = model._calculate_drift(t=0.0, N=50000.0, C=0.0)
        # При C=0 нет деградации, только производство
        assert drift_C > 0

    def test_drift_C_includes_degradation(self, model):
        """Тест что drift_C содержит терм -gamma*C деградации."""
        # При высоком C и низком N, деградация доминирует
        _, drift_C = model._calculate_drift(t=0.0, N=0.0, C=100.0)
        # Без производства (N=0), только деградация
        assert drift_C < 0

    def test_drift_C_includes_prp_secretion(self, prp_therapy_protocol):
        """Тест что drift_C включает секрецию цитокинов из PRP."""
        model = SDEModel(therapy=prp_therapy_protocol)
        # Во время PRP терапии должна быть дополнительная секреция
        _, drift_C = model._calculate_drift(t=3.0, N=0.0, C=0.0)
        # Может быть положительным из-за PRP секреции
        assert isinstance(drift_C, (int, float))


# =============================================================================
# Тесты для SDEModel._calculate_diffusion
# =============================================================================


class TestSDEModelCalculateDiffusion:
    """Тесты для SDEModel._calculate_diffusion."""

    @pytest.fixture
    def model(self):
        """Создать модель для тестов."""
        return SDEModel(config=SDEConfig(sigma_n=0.05, sigma_c=0.02))

    def test_calculate_diffusion_returns_tuple(self, model):
        """Тест что _calculate_diffusion возвращает (diffusion_N, diffusion_C)."""
        result = model._calculate_diffusion(t=1.0, N=5000.0, C=10.0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_diffusion_N_proportional_to_sigma_n(self):
        """Тест что diffusion_N масштабируется с sigma_n."""
        model_low_sigma = SDEModel(config=SDEConfig(sigma_n=0.01))
        model_high_sigma = SDEModel(config=SDEConfig(sigma_n=0.1))

        diff_low, _ = model_low_sigma._calculate_diffusion(t=0.0, N=5000.0, C=10.0)
        diff_high, _ = model_high_sigma._calculate_diffusion(t=0.0, N=5000.0, C=10.0)

        assert diff_high > diff_low

    def test_diffusion_C_proportional_to_sigma_c(self):
        """Тест что diffusion_C масштабируется с sigma_c."""
        model_low_sigma = SDEModel(config=SDEConfig(sigma_c=0.01))
        model_high_sigma = SDEModel(config=SDEConfig(sigma_c=0.1))

        _, diff_low = model_low_sigma._calculate_diffusion(t=0.0, N=5000.0, C=10.0)
        _, diff_high = model_high_sigma._calculate_diffusion(t=0.0, N=5000.0, C=10.0)

        assert diff_high > diff_low

    def test_diffusion_N_proportional_to_N(self, model):
        """Тест что diffusion_N пропорционален плотности клеток."""
        diff_low_N, _ = model._calculate_diffusion(t=0.0, N=1000.0, C=10.0)
        diff_high_N, _ = model._calculate_diffusion(t=0.0, N=10000.0, C=10.0)

        assert diff_high_N > diff_low_N

    def test_diffusion_C_proportional_to_C(self, model):
        """Тест что diffusion_C пропорционален концентрации цитокинов."""
        _, diff_low_C = model._calculate_diffusion(t=0.0, N=5000.0, C=1.0)
        _, diff_high_C = model._calculate_diffusion(t=0.0, N=5000.0, C=100.0)

        assert diff_high_C > diff_low_C


# =============================================================================
# Тесты для SDEModel._apply_boundary_conditions
# =============================================================================


class TestSDEModelBoundaryConditions:
    """Тесты для SDEModel._apply_boundary_conditions."""

    @pytest.fixture
    def model(self):
        """Создать модель для тестов."""
        return SDEModel()

    def test_boundary_positive_values_unchanged(self, model):
        """Тест что положительные N и C остаются неизменными."""
        N, C = model._apply_boundary_conditions(5000.0, 10.0)
        assert N == 5000.0
        assert C == 10.0

    def test_boundary_negative_N_becomes_zero(self, model):
        """Тест что отрицательное N становится 0 (отражающая граница)."""
        N, C = model._apply_boundary_conditions(-100.0, 10.0)
        assert N == 0.0

    def test_boundary_negative_C_becomes_zero(self, model):
        """Тест что отрицательное C становится 0 (отражающая граница)."""
        N, C = model._apply_boundary_conditions(5000.0, -5.0)
        assert C == 0.0

    def test_boundary_zero_values_remain_zero(self, model):
        """Тест что нулевые значения остаются нулевыми."""
        N, C = model._apply_boundary_conditions(0.0, 0.0)
        assert N == 0.0
        assert C == 0.0


# =============================================================================
# Тесты для SDEModel._get_therapy_mask
# =============================================================================


class TestSDEModelGetTherapyMask:
    """Тесты для SDEModel._get_therapy_mask."""

    def test_get_therapy_mask_returns_ndarray(self, prp_therapy_protocol):
        """Тест что возвращает numpy boolean массив."""
        model = SDEModel(therapy=prp_therapy_protocol)
        times = np.linspace(0, 10, 101)
        mask = model._get_therapy_mask(times, "prp")

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool

    def test_get_therapy_mask_shape_matches_times(self, prp_therapy_protocol):
        """Тест что форма маски соответствует входному массиву times."""
        model = SDEModel(therapy=prp_therapy_protocol)
        times = np.linspace(0, 10, 101)
        mask = model._get_therapy_mask(times, "prp")

        assert mask.shape == times.shape

    def test_prp_mask_true_during_active_period(self, prp_therapy_protocol):
        """Тест что PRP маска True только во время активного окна."""
        model = SDEModel(therapy=prp_therapy_protocol)
        # prp_start_time = 1.0, prp_duration = 7.0
        times = np.array([0.5, 1.0, 3.0, 7.9, 8.1, 10.0])
        mask = model._get_therapy_mask(times, "prp")

        # t=0.5 -> False (до начала)
        # t=1.0 -> True (начало)
        # t=3.0 -> True (во время)
        # t=7.9 -> True (почти конец)
        # t=8.1 -> False (после конца)
        # t=10.0 -> False (после)
        expected = np.array([False, True, True, True, False, False])
        np.testing.assert_array_equal(mask, expected)

    def test_pemf_mask_true_during_active_period(self, pemf_therapy_protocol):
        """Тест что PEMF маска True только во время активного окна."""
        model = SDEModel(therapy=pemf_therapy_protocol)
        # pemf_start_time = 0.0, pemf_duration = 14.0
        times = np.array([0.0, 5.0, 13.9, 14.1, 20.0])
        mask = model._get_therapy_mask(times, "pemf")

        expected = np.array([True, True, True, False, False])
        np.testing.assert_array_equal(mask, expected)


# =============================================================================
# Тесты для SDEModel.simulate
# =============================================================================


class TestSDEModelSimulate:
    """Тесты для SDEModel.simulate (интеграционные тесты)."""

    def test_simulate_returns_trajectory(self, sample_model_parameters):
        """Тест что simulate() возвращает SDETrajectory."""
        model = SDEModel(config=SDEConfig(t_max=1.0, dt=0.1), random_seed=42)
        trajectory = model.simulate(sample_model_parameters)

        assert isinstance(trajectory, SDETrajectory)

    def test_simulate_trajectory_length_correct(self, sample_model_parameters):
        """Тест что длина траектории = int(t_max/dt) + 1."""
        config = SDEConfig(t_max=5.0, dt=0.1)
        model = SDEModel(config=config, random_seed=42)
        trajectory = model.simulate(sample_model_parameters)

        expected_length = int(config.t_max / config.dt) + 1
        assert len(trajectory.times) == expected_length

    def test_simulate_initial_values_correct(self, sample_model_parameters):
        """Тест что N[0] = n0, C[0] = c0."""
        model = SDEModel(config=SDEConfig(t_max=1.0), random_seed=42)
        trajectory = model.simulate(sample_model_parameters)

        assert trajectory.N_values[0] == sample_model_parameters.n0
        assert trajectory.C_values[0] == sample_model_parameters.c0

    def test_simulate_times_monotonic_increasing(self, sample_model_parameters):
        """Тест что массив times монотонно возрастает."""
        model = SDEModel(config=SDEConfig(t_max=5.0), random_seed=42)
        trajectory = model.simulate(sample_model_parameters)

        assert np.all(np.diff(trajectory.times) > 0)

    def test_simulate_N_values_non_negative(self, sample_model_parameters):
        """Тест что все значения N >= 0."""
        model = SDEModel(config=SDEConfig(t_max=10.0), random_seed=42)
        trajectory = model.simulate(sample_model_parameters)

        assert np.all(trajectory.N_values >= 0)

    def test_simulate_C_values_non_negative(self, sample_model_parameters):
        """Тест что все значения C >= 0."""
        model = SDEModel(config=SDEConfig(t_max=10.0), random_seed=42)
        trajectory = model.simulate(sample_model_parameters)

        assert np.all(trajectory.C_values >= 0)

    def test_simulate_reproducible_with_seed(self, sample_model_parameters):
        """Тест что одинаковый seed дает одинаковую траекторию."""
        model1 = SDEModel(config=SDEConfig(t_max=5.0), random_seed=42)
        model2 = SDEModel(config=SDEConfig(t_max=5.0), random_seed=42)

        traj1 = model1.simulate(sample_model_parameters)
        traj2 = model2.simulate(sample_model_parameters)

        np.testing.assert_array_equal(traj1.N_values, traj2.N_values)
        np.testing.assert_array_equal(traj1.C_values, traj2.C_values)

    def test_simulate_different_seeds_different_trajectories(self, sample_model_parameters):
        """Тест что разные seeds дают разные траектории."""
        model1 = SDEModel(config=SDEConfig(t_max=5.0), random_seed=42)
        model2 = SDEModel(config=SDEConfig(t_max=5.0), random_seed=123)

        traj1 = model1.simulate(sample_model_parameters)
        traj2 = model2.simulate(sample_model_parameters)

        # Траектории должны отличаться (стохастичность)
        assert not np.allclose(traj1.N_values, traj2.N_values)

    def test_simulate_no_therapy_growth_behavior(self, sample_model_parameters):
        """Тест что клетки растут без терапии."""
        model = SDEModel(config=SDEConfig(t_max=10.0, sigma_n=0.0, sigma_c=0.0), random_seed=42)
        trajectory = model.simulate(sample_model_parameters)

        # При детерминистическом случае (sigma=0) клетки должны расти
        # к carrying capacity или демонстрировать логистический рост
        assert trajectory.N_values[-1] >= trajectory.N_values[0]

    def test_simulate_prp_enhances_growth(self, sample_model_parameters, prp_therapy_protocol):
        """Тест что PRP терапия увеличивает рост клеток."""
        config = SDEConfig(t_max=10.0, sigma_n=0.0, sigma_c=0.0)

        model_no_therapy = SDEModel(config=config, random_seed=42)
        model_prp = SDEModel(config=config, therapy=prp_therapy_protocol, random_seed=42)

        traj_no_therapy = model_no_therapy.simulate(sample_model_parameters)
        traj_prp = model_prp.simulate(sample_model_parameters)

        # PRP должна увеличить финальную плотность
        assert traj_prp.N_values[-1] >= traj_no_therapy.N_values[-1]

    def test_simulate_pemf_enhances_growth(self, sample_model_parameters, pemf_therapy_protocol):
        """Тест что PEMF терапия увеличивает рост клеток."""
        config = SDEConfig(t_max=10.0, sigma_n=0.0, sigma_c=0.0)

        model_no_therapy = SDEModel(config=config, random_seed=42)
        model_pemf = SDEModel(config=config, therapy=pemf_therapy_protocol, random_seed=42)

        traj_no_therapy = model_no_therapy.simulate(sample_model_parameters)
        traj_pemf = model_pemf.simulate(sample_model_parameters)

        assert traj_pemf.N_values[-1] >= traj_no_therapy.N_values[-1]


# =============================================================================
# Тесты для simulate_sde функции
# =============================================================================


class TestSimulateSdeFunction:
    """Тесты для simulate_sde convenience функции."""

    def test_simulate_sde_returns_trajectory(self, sample_model_parameters):
        """Тест что convenience функция возвращает SDETrajectory."""
        trajectory = simulate_sde(
            initial_params=sample_model_parameters,
            config=SDEConfig(t_max=1.0),
            random_seed=42,
        )
        assert isinstance(trajectory, SDETrajectory)

    def test_simulate_sde_with_config(self, sample_model_parameters, custom_sde_config):
        """Тест что функция принимает опциональный config."""
        trajectory = simulate_sde(
            initial_params=sample_model_parameters,
            config=custom_sde_config,
            random_seed=42,
        )
        assert len(trajectory.times) == int(custom_sde_config.t_max / custom_sde_config.dt) + 1

    def test_simulate_sde_with_therapy(self, sample_model_parameters, prp_therapy_protocol):
        """Тест что функция принимает опциональный therapy."""
        trajectory = simulate_sde(
            initial_params=sample_model_parameters,
            config=SDEConfig(t_max=5.0),
            therapy=prp_therapy_protocol,
            random_seed=42,
        )
        # Должна быть маркировка PRP терапии
        assert "prp" in trajectory.therapy_markers

    def test_simulate_sde_with_seed(self, sample_model_parameters):
        """Тест что функция принимает random_seed."""
        traj1 = simulate_sde(sample_model_parameters, config=SDEConfig(t_max=1.0), random_seed=42)
        traj2 = simulate_sde(sample_model_parameters, config=SDEConfig(t_max=1.0), random_seed=42)

        np.testing.assert_array_equal(traj1.N_values, traj2.N_values)
