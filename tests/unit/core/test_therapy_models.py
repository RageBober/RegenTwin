"""TDD тесты для therapy_models.py — механистические модели терапий.

Фаза 2.6, Этап 2: тесты для PRPModel, PEMFModel, SynergyModel.
Все тесты должны FAIL на стабах и PASS после реализации (Этап 3).

Биологические ограничения:
    Marx 2004 (PRP дозировки)
    Giusti 2009 (двухфазная кинетика)
    Pilla 2013 (PEMF механизмы)
    Varani 2017 (PEMF + аденозин)
    Onstenk 2015 (синергия PRP+PEMF)
"""

import dataclasses
import math

import pytest

from src.core.therapy_models import (
    PEMFConfig,
    PEMFEffects,
    PEMFModel,
    PRPConfig,
    PRPModel,
    PRPReleaseState,
    SynergyConfig,
    SynergyModel,
)

# ============================================================================
# Фикстуры
# ============================================================================


@pytest.fixture
def default_prp_config():
    """PRPConfig с параметрами по умолчанию."""
    return PRPConfig()


@pytest.fixture
def default_pemf_config():
    """PEMFConfig с параметрами по умолчанию."""
    return PEMFConfig()


@pytest.fixture
def prp_model():
    """PRPModel с параметрами по умолчанию."""
    return PRPModel()


@pytest.fixture
def pemf_model():
    """PEMFModel с параметрами по умолчанию."""
    return PEMFModel()


@pytest.fixture
def pemf_model_no_field():
    """PEMFModel без магнитного поля (B=0)."""
    return PEMFModel(PEMFConfig(B_amplitude=0.0))


@pytest.fixture
def prp_model_custom_dose():
    """PRPModel с дозой 5x."""
    return PRPModel(PRPConfig(dose=5.0))


@pytest.fixture
def synergy_model(prp_model, pemf_model):
    """SynergyModel с дефолтными моделями."""
    return SynergyModel(prp_model, pemf_model)


# ============================================================================
# 1. PRPConfig
# ============================================================================


class TestPRPConfigDefaults:
    """Значения по умолчанию PRPConfig (Marx 2004, Giusti 2009)."""

    def test_dose_default(self, default_prp_config):
        """Доза PRP по умолчанию 4x."""
        assert default_prp_config.dose == 4.0

    def test_pdgf_c0_default(self, default_prp_config):
        """Начальная концентрация PDGF-AB = 20 нг/мл."""
        assert default_prp_config.pdgf_c0 == 20.0

    def test_vegf_c0_default(self, default_prp_config):
        """Начальная концентрация VEGF = 1.0 нг/мл."""
        assert default_prp_config.vegf_c0 == 1.0

    def test_tgfb_c0_default(self, default_prp_config):
        """Начальная концентрация TGF-β1 = 30 нг/мл."""
        assert default_prp_config.tgfb_c0 == 30.0

    def test_egf_c0_default(self, default_prp_config):
        """Начальная концентрация EGF = 0.2 нг/мл."""
        assert default_prp_config.egf_c0 == 0.2

    def test_tau_burst_defaults(self, default_prp_config):
        """Временные константы burst-фазы: PDGF=1, VEGF=1, TGF-β=2, EGF=0.5."""
        cfg = default_prp_config
        assert cfg.tau_burst_pdgf == 1.0
        assert cfg.tau_burst_vegf == 1.0
        assert cfg.tau_burst_tgfb == 2.0
        assert cfg.tau_burst_egf == 0.5

    def test_tau_sustained_defaults(self, default_prp_config):
        """Временные константы sustained-фазы: PDGF=48, VEGF=24, TGF-β=72, EGF=12."""
        cfg = default_prp_config
        assert cfg.tau_sustained_pdgf == 48.0
        assert cfg.tau_sustained_vegf == 24.0
        assert cfg.tau_sustained_tgfb == 72.0
        assert cfg.tau_sustained_egf == 12.0

    def test_alpha_prp_s_default(self, default_prp_config):
        """Коэффициент рекрутирования стволовых клеток = 0.5."""
        assert default_prp_config.alpha_PRP_S == 0.5


class TestPRPConfigCustom:
    """Пользовательские значения PRPConfig."""

    def test_custom_dose(self):
        """Создание PRPConfig с дозой 5x."""
        cfg = PRPConfig(dose=5.0)
        assert cfg.dose == 5.0

    def test_custom_concentrations(self):
        """Создание PRPConfig с нестандартными концентрациями."""
        cfg = PRPConfig(pdgf_c0=25.0, vegf_c0=1.5, tgfb_c0=35.0, egf_c0=0.3)
        assert cfg.pdgf_c0 == 25.0
        assert cfg.vegf_c0 == 1.5
        assert cfg.tgfb_c0 == 35.0
        assert cfg.egf_c0 == 0.3

    def test_field_count(self):
        """PRPConfig содержит 14 полей."""
        fields = dataclasses.fields(PRPConfig)
        assert len(fields) == 14


class TestPRPConfigInvariants:
    """Инварианты PRPConfig по умолчанию."""

    def test_tau_burst_less_than_sustained(self, default_prp_config):
        """tau_burst < tau_sustained для каждого фактора."""
        cfg = default_prp_config
        assert cfg.tau_burst_pdgf < cfg.tau_sustained_pdgf
        assert cfg.tau_burst_vegf < cfg.tau_sustained_vegf
        assert cfg.tau_burst_tgfb < cfg.tau_sustained_tgfb
        assert cfg.tau_burst_egf < cfg.tau_sustained_egf

    def test_all_c0_nonnegative(self, default_prp_config):
        """Все начальные концентрации >= 0."""
        cfg = default_prp_config
        assert cfg.pdgf_c0 >= 0
        assert cfg.vegf_c0 >= 0
        assert cfg.tgfb_c0 >= 0
        assert cfg.egf_c0 >= 0

    def test_dose_positive(self, default_prp_config):
        """Доза PRP > 0."""
        assert default_prp_config.dose > 0

    def test_alpha_nonnegative(self, default_prp_config):
        """alpha_PRP_S >= 0."""
        assert default_prp_config.alpha_PRP_S >= 0


# ============================================================================
# 2. PEMFConfig
# ============================================================================


class TestPEMFConfigDefaults:
    """Значения по умолчанию PEMFConfig (Pilla 2013, Varani 2017)."""

    def test_b_amplitude_default(self, default_pemf_config):
        """Амплитуда B-поля по умолчанию 1.0 мТ."""
        assert default_pemf_config.B_amplitude == 1.0

    def test_frequency_default(self, default_pemf_config):
        """Частота PEMF по умолчанию 50 Гц."""
        assert default_pemf_config.frequency == 50.0

    def test_anti_inflam_params(self, default_pemf_config):
        """Параметры аденозинового пути: f_opt=27.12, σ=10, ε_max=0.4."""
        cfg = default_pemf_config
        assert cfg.f_opt_anti_inflam == 27.12
        assert cfg.sigma_f_anti_inflam == 10.0
        assert cfg.epsilon_max_anti_inflam == 0.4

    def test_prolif_params(self, default_pemf_config):
        """Параметры Ca²⁺-CaM пути: f_center=75, σ=25, ε_max=0.3, B_half=0.5."""
        cfg = default_pemf_config
        assert cfg.f_center_prolif == 75.0
        assert cfg.sigma_window_prolif == 25.0
        assert cfg.epsilon_prolif_max == 0.3
        assert cfg.B_half_prolif == 0.5

    def test_migration_params(self, default_pemf_config):
        """Макс. усиление миграции = 0.25."""
        assert default_pemf_config.epsilon_migration_max == 0.25

    def test_field_count(self):
        """PEMFConfig содержит 12 полей."""
        fields = dataclasses.fields(PEMFConfig)
        assert len(fields) == 12


class TestPEMFConfigInvariants:
    """Инварианты PEMFConfig по умолчанию."""

    def test_epsilon_bounds(self, default_pemf_config):
        """Все epsilon ∈ [0, 1]."""
        cfg = default_pemf_config
        for eps in [
            cfg.epsilon_max_anti_inflam,
            cfg.epsilon_prolif_max,
            cfg.epsilon_migration_max,
        ]:
            assert 0.0 <= eps <= 1.0

    def test_b_thresholds_positive(self, default_pemf_config):
        """Пороговые значения B > 0."""
        assert default_pemf_config.B0_threshold > 0
        assert default_pemf_config.B_half_prolif > 0

    def test_n_b_positive(self, default_pemf_config):
        """Коэффициент Hill n_B > 0."""
        assert default_pemf_config.n_B > 0

    def test_frequency_positive(self, default_pemf_config):
        """Частота PEMF > 0."""
        assert default_pemf_config.frequency > 0


# ============================================================================
# 3. SynergyConfig
# ============================================================================


class TestSynergyConfigDefaults:
    """Значения по умолчанию SynergyConfig."""

    def test_beta_synergy_default(self):
        """Коэффициент синергии по умолчанию 0.2."""
        cfg = SynergyConfig()
        assert cfg.beta_synergy == 0.2

    def test_beta_synergy_nonnegative(self):
        """beta_synergy >= 0 по умолчанию."""
        assert SynergyConfig().beta_synergy >= 0


# ============================================================================
# 4. PRPReleaseState
# ============================================================================


class TestPRPReleaseState:
    """Dataclass состояния PRP-релиза."""

    def test_defaults_all_zero(self):
        """Все поля по умолчанию = 0.0."""
        state = PRPReleaseState()
        assert state.theta_pdgf == 0.0
        assert state.theta_vegf == 0.0
        assert state.theta_tgfb == 0.0
        assert state.theta_egf == 0.0
        assert state.theta_total == 0.0

    def test_custom_values(self):
        """Создание с пользовательскими значениями."""
        state = PRPReleaseState(theta_pdgf=15.0, theta_total=0.5)
        assert state.theta_pdgf == 15.0
        assert state.theta_total == 0.5

    def test_field_count(self):
        """PRPReleaseState содержит 5 полей."""
        fields = dataclasses.fields(PRPReleaseState)
        assert len(fields) == 5

    def test_all_theta_nonneg_by_default(self):
        """Все theta >= 0 по умолчанию."""
        state = PRPReleaseState()
        assert state.theta_pdgf >= 0
        assert state.theta_vegf >= 0
        assert state.theta_tgfb >= 0
        assert state.theta_egf >= 0
        assert state.theta_total >= 0


# ============================================================================
# 5. PEMFEffects
# ============================================================================


class TestPEMFEffects:
    """Dataclass эффектов PEMF."""

    def test_defaults_all_zero(self):
        """Все поля по умолчанию = 0.0."""
        effects = PEMFEffects()
        assert effects.anti_inflammatory == 0.0
        assert effects.proliferation == 0.0
        assert effects.migration == 0.0

    def test_custom_values(self):
        """Создание с пользовательским значением."""
        effects = PEMFEffects(anti_inflammatory=0.35)
        assert effects.anti_inflammatory == 0.35

    def test_field_count(self):
        """PEMFEffects содержит 3 поля."""
        fields = dataclasses.fields(PEMFEffects)
        assert len(fields) == 3

    def test_all_effects_nonneg_by_default(self):
        """Все эффекты >= 0 по умолчанию."""
        effects = PEMFEffects()
        assert effects.anti_inflammatory >= 0
        assert effects.proliferation >= 0
        assert effects.migration >= 0


# ============================================================================
# 6. PRPModel.__init__
# ============================================================================


class TestPRPModelInit:
    """Инициализация PRPModel."""

    def test_default_config(self):
        """PRPModel() создаёт PRPConfig() с defaults."""
        model = PRPModel()
        assert model.config == PRPConfig()

    def test_custom_config(self):
        """PRPModel с пользовательским конфигом."""
        model = PRPModel(PRPConfig(dose=5.0))
        assert model.config.dose == 5.0

    def test_config_is_stored(self, prp_model):
        """Config сохранён как PRPConfig."""
        assert isinstance(prp_model.config, PRPConfig)


# ============================================================================
# 7. PRPModel._biphasic_release
# ============================================================================


class TestBiphasicRelease:
    """Двухфазное высвобождение фактора роста."""

    def test_t_zero_returns_zero(self, prp_model):
        """t=0 → результат ≈ 0 (burst - sustained = 1 - 1 = 0)."""
        result = prp_model._biphasic_release(t=0.0, c0=20.0, tau_burst=1.0, tau_sustained=48.0)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_t_negative_returns_zero(self, prp_model):
        """T < 0 → 0.0 (до инъекции)."""
        result = prp_model._biphasic_release(t=-5.0, c0=20.0, tau_burst=1.0, tau_sustained=48.0)
        assert result == 0.0

    def test_positive_near_peak(self, prp_model):
        """t=1ч, PDGF → положительное значение вблизи пика."""
        result = prp_model._biphasic_release(t=1.0, c0=20.0, tau_burst=1.0, tau_sustained=48.0)
        assert result > 0.0

    def test_decays_over_time(self, prp_model):
        """Значение при t=48 меньше значения при t=2 (после пика)."""
        val_peak = prp_model._biphasic_release(t=2.0, c0=20.0, tau_burst=1.0, tau_sustained=48.0)
        val_late = prp_model._biphasic_release(
            t=48.0, c0=20.0, tau_burst=1.0, tau_sustained=48.0
        )
        assert val_late < val_peak

    def test_large_t_near_zero(self, prp_model):
        """t=200ч → ≈ 0 (экспоненциальное затухание)."""
        result = prp_model._biphasic_release(
            t=200.0, c0=20.0, tau_burst=1.0, tau_sustained=48.0
        )
        assert result == pytest.approx(0.0, abs=0.1)

    def test_c0_zero_returns_zero(self, prp_model):
        """c0=0 → 0.0 (нет фактора)."""
        result = prp_model._biphasic_release(t=5.0, c0=0.0, tau_burst=1.0, tau_sustained=48.0)
        assert result == 0.0

    def test_tau_equal_lhopital(self, prp_model):
        """tau_burst ≈ tau_sustained → предельная формула, без деления на 0."""
        result = prp_model._biphasic_release(t=1.0, c0=20.0, tau_burst=10.0, tau_sustained=10.0)
        assert math.isfinite(result)
        assert result >= 0.0

    def test_result_nonnegative(self, prp_model):
        """Результат >= 0 для произвольных t >= 0."""
        for t in [0.0, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0]:
            result = prp_model._biphasic_release(
                t=t, c0=20.0, tau_burst=1.0, tau_sustained=48.0
            )
            assert result >= 0.0, f"Отрицательный результат при t={t}"

    def test_dose_scaling(self):
        """Удвоение дозы удваивает результат."""
        model1 = PRPModel(PRPConfig(dose=1.0))
        model2 = PRPModel(PRPConfig(dose=2.0))
        val1 = model1._biphasic_release(t=2.0, c0=20.0, tau_burst=1.0, tau_sustained=48.0)
        val2 = model2._biphasic_release(t=2.0, c0=20.0, tau_burst=1.0, tau_sustained=48.0)
        assert val2 == pytest.approx(2.0 * val1, rel=1e-10)

    def test_peak_exists(self, prp_model):
        """Существует пиковое значение между t=0 и t→∞."""
        values = [
            prp_model._biphasic_release(t=t, c0=20.0, tau_burst=1.0, tau_sustained=48.0)
            for t in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
        ]
        max_val = max(values)
        assert max_val > 0.0
        # Пик не на границах
        assert values[0] < max_val
        assert values[-1] < max_val

    def test_monotonic_decay_after_peak(self, prp_model):
        """После пика значение монотонно убывает."""
        # Пик PDGF ~1ч. Проверяем убывание начиная с t=5
        times = [5.0, 10.0, 20.0, 50.0, 100.0]
        values = [
            prp_model._biphasic_release(t=t, c0=20.0, tau_burst=1.0, tau_sustained=48.0)
            for t in times
        ]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1], (
                f"Не убывает: v({times[i]})={values[i]} < v({times[i+1]})={values[i+1]}"
            )

    def test_peak_time_formula(self, prp_model):
        """Пик при t_peak = τ_b·τ_s·ln(τ_s/τ_b)/(τ_s-τ_b)."""
        tau_b, tau_s = 1.0, 48.0
        t_peak = tau_b * tau_s * math.log(tau_s / tau_b) / (tau_s - tau_b)
        # Значение в пике должно быть больше значений чуть раньше и позже
        val_peak = prp_model._biphasic_release(t=t_peak, c0=20.0, tau_burst=tau_b, tau_sustained=tau_s)
        val_before = prp_model._biphasic_release(
            t=t_peak * 0.5, c0=20.0, tau_burst=tau_b, tau_sustained=tau_s
        )
        val_after = prp_model._biphasic_release(
            t=t_peak * 2.0, c0=20.0, tau_burst=tau_b, tau_sustained=tau_s
        )
        assert val_peak >= val_before
        assert val_peak >= val_after


# ============================================================================
# 8. PRPModel.compute_release
# ============================================================================


class TestComputeRelease:
    """Вычисление высвобождения всех 4 факторов роста."""

    def test_t_zero_all_zero(self, prp_model):
        """t=0, app_time=0 → все theta ≈ 0."""
        state = prp_model.compute_release(t=0.0, application_time=0.0)
        assert state.theta_pdgf == pytest.approx(0.0, abs=1e-10)
        assert state.theta_vegf == pytest.approx(0.0, abs=1e-10)
        assert state.theta_tgfb == pytest.approx(0.0, abs=1e-10)
        assert state.theta_egf == pytest.approx(0.0, abs=1e-10)

    def test_active_release(self, prp_model):
        """t=2ч → theta_pdgf > 0 (активный релиз)."""
        state = prp_model.compute_release(t=2.0, application_time=0.0)
        assert state.theta_pdgf > 0.0

    def test_large_t_small_values(self, prp_model):
        """t=100ч → все theta малые."""
        state = prp_model.compute_release(t=100.0, application_time=0.0)
        assert state.theta_pdgf < 1.0
        assert state.theta_vegf < 0.1
        assert state.theta_egf < 0.01

    def test_before_application(self, prp_model):
        """T < application_time → все theta = 0."""
        state = prp_model.compute_release(t=5.0, application_time=10.0)
        assert state.theta_pdgf == 0.0
        assert state.theta_vegf == 0.0
        assert state.theta_tgfb == 0.0
        assert state.theta_egf == 0.0

    def test_theta_total_normalized(self, prp_model):
        """theta_total ∈ [0, 1]."""
        for t in [0.0, 1.0, 2.0, 5.0, 10.0, 50.0]:
            state = prp_model.compute_release(t=t)
            assert 0.0 <= state.theta_total <= 1.0, (
                f"theta_total={state.theta_total} вне [0,1] при t={t}"
            )

    def test_all_theta_nonnegative(self, prp_model):
        """Все theta >= 0 для произвольных t."""
        for t in [0.0, 0.5, 1.0, 5.0, 20.0, 100.0]:
            state = prp_model.compute_release(t=t)
            assert state.theta_pdgf >= 0.0
            assert state.theta_vegf >= 0.0
            assert state.theta_tgfb >= 0.0
            assert state.theta_egf >= 0.0

    def test_returns_prp_release_state(self, prp_model):
        """Возвращает PRPReleaseState."""
        result = prp_model.compute_release(t=1.0)
        assert isinstance(result, PRPReleaseState)

    def test_egf_peaks_earliest(self, prp_model):
        """EGF (tau_burst=0.5) пик раньше TGF-β (tau_burst=2.0)."""
        state_early = prp_model.compute_release(t=0.3)
        state_late = prp_model.compute_release(t=3.0)
        # EGF при t=0.3 ближе к пику, TGF-β при t=3.0 ближе к пику
        # Отношение EGF/TGF-β должно быть выше в ранний момент
        if state_early.theta_tgfb > 0 and state_late.theta_tgfb > 0:
            ratio_early = state_early.theta_egf / max(state_early.theta_tgfb, 1e-30)
            ratio_late = state_late.theta_egf / max(state_late.theta_tgfb, 1e-30)
            assert ratio_early > ratio_late

    def test_tgfb_longest_sustained(self, prp_model):
        """TGF-β (tau_s=72ч) дольше всех удерживает значимый уровень."""
        state = prp_model.compute_release(t=60.0)
        # TGF-β при t=60 ч должен быть больше, чем EGF и VEGF
        assert state.theta_tgfb > state.theta_egf
        assert state.theta_tgfb > state.theta_vegf

    def test_application_time_shifts_release(self, prp_model):
        """application_time=5 сдвигает кривую на 5 ч."""
        state_no_shift = prp_model.compute_release(t=2.0, application_time=0.0)
        state_shifted = prp_model.compute_release(t=7.0, application_time=5.0)
        assert state_shifted.theta_pdgf == pytest.approx(state_no_shift.theta_pdgf, rel=1e-10)


# ============================================================================
# 9. PRPModel.compute_stem_cell_factor
# ============================================================================


class TestComputeStemCellFactor:
    """Фактор рекрутирования стволовых клеток."""

    def test_t_zero_returns_zero(self, prp_model):
        """t=0 → 0.0."""
        result = prp_model.compute_stem_cell_factor(t=0.0)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_active_time_positive(self, prp_model):
        """t=2ч → > 0 (активный PRP)."""
        result = prp_model.compute_stem_cell_factor(t=2.0)
        assert result > 0.0

    def test_alpha_zero_always_zero(self):
        """alpha_PRP_S=0 → всегда 0."""
        model = PRPModel(PRPConfig(alpha_PRP_S=0.0))
        for t in [0.0, 1.0, 5.0, 50.0]:
            assert model.compute_stem_cell_factor(t=t) == 0.0

    def test_result_nonnegative(self, prp_model):
        """Результат >= 0."""
        for t in [0.0, 1.0, 5.0, 50.0, 200.0]:
            assert prp_model.compute_stem_cell_factor(t=t) >= 0.0

    def test_bounded_by_alpha(self, prp_model):
        """Результат <= alpha_PRP_S."""
        for t in [0.0, 1.0, 2.0, 5.0, 10.0, 50.0]:
            result = prp_model.compute_stem_cell_factor(t=t)
            assert result <= prp_model.config.alpha_PRP_S + 1e-10


# ============================================================================
# 10. PRPModel.is_active
# ============================================================================


class TestPRPModelIsActive:
    """Проверка активности PRP."""

    def test_t_zero_not_active(self, prp_model):
        """t=0 → False (релиз ещё не начался)."""
        assert prp_model.is_active(t=0.0) is False

    def test_active_during_release(self, prp_model):
        """t=2ч → True (активный релиз)."""
        assert prp_model.is_active(t=2.0) is True

    def test_not_active_after_decay(self, prp_model):
        """t=500ч → False (полностью затух)."""
        assert prp_model.is_active(t=500.0) is False

    def test_before_application_not_active(self, prp_model):
        """T < application_time → False."""
        assert prp_model.is_active(t=5.0, application_time=10.0) is False


# ============================================================================
# 11. PEMFModel.__init__
# ============================================================================


class TestPEMFModelInit:
    """Инициализация PEMFModel."""

    def test_default_config(self):
        """PEMFModel() создаёт PEMFConfig() с defaults."""
        model = PEMFModel()
        assert model.config == PEMFConfig()

    def test_custom_config(self):
        """PEMFModel с пользовательской амплитудой."""
        model = PEMFModel(PEMFConfig(B_amplitude=2.0))
        assert model.config.B_amplitude == 2.0


# ============================================================================
# 12. PEMFModel.compute_anti_inflammatory
# ============================================================================


class TestComputeAntiInflammatory:
    """Противовоспалительный эффект через аденозиновый путь."""

    def test_optimal_frequency(self):
        """B=1.0, f=27.12 (оптимум) → ≈ ε_max · Hill(2) ≈ 0.32."""
        model = PEMFModel(PEMFConfig(B_amplitude=1.0, frequency=27.12))
        result = model.compute_anti_inflammatory(t=0.0)
        # Hill(B/B0=2, n=2) = 4/(1+4) = 0.8; Gauss(0) = 1.0
        # ε = 0.4 * 0.8 = 0.32
        assert result == pytest.approx(0.32, abs=0.05)

    def test_b_zero_returns_zero(self):
        """B=0 → 0.0 (нет поля)."""
        model = PEMFModel(PEMFConfig(B_amplitude=0.0))
        result = model.compute_anti_inflammatory(t=0.0)
        assert result == 0.0

    def test_high_b_saturation(self):
        """B=10.0, f=27.12 → ≈ ε_max (насыщение Hill)."""
        model = PEMFModel(PEMFConfig(B_amplitude=10.0, frequency=27.12))
        result = model.compute_anti_inflammatory(t=0.0)
        assert result == pytest.approx(0.4, abs=0.02)

    def test_far_frequency_near_zero(self):
        """f=100 (далеко от f_opt=27.12) → ≈ 0."""
        model = PEMFModel(PEMFConfig(B_amplitude=1.0, frequency=100.0))
        result = model.compute_anti_inflammatory(t=0.0)
        assert result < 0.01

    def test_intermediate_values(self):
        """B=0.5, f=50 → промежуточное значение."""
        model = PEMFModel(PEMFConfig(B_amplitude=0.5, frequency=50.0))
        result = model.compute_anti_inflammatory(t=0.0)
        assert 0.0 < result < 0.4

    def test_result_in_range(self, pemf_model):
        """Результат ∈ [0, ε_max]."""
        result = pemf_model.compute_anti_inflammatory(t=0.0)
        assert 0.0 <= result <= pemf_model.config.epsilon_max_anti_inflam

    def test_monotonic_with_b(self):
        """Растёт с B при фиксированной f."""
        results = []
        for b in [0.1, 0.5, 1.0, 2.0, 5.0]:
            model = PEMFModel(PEMFConfig(B_amplitude=b, frequency=27.12))
            results.append(model.compute_anti_inflammatory(t=0.0))
        for i in range(len(results) - 1):
            assert results[i] <= results[i + 1]

    def test_max_at_f_opt(self):
        """Максимум при f = f_opt."""
        model_opt = PEMFModel(PEMFConfig(B_amplitude=1.0, frequency=27.12))
        model_off = PEMFModel(PEMFConfig(B_amplitude=1.0, frequency=50.0))
        val_opt = model_opt.compute_anti_inflammatory(t=0.0)
        val_off = model_off.compute_anti_inflammatory(t=0.0)
        assert val_opt >= val_off

    def test_symmetric_around_f_opt(self):
        """f_opt ± δ дают одинаковый результат (Гаусс)."""
        delta = 5.0
        f_opt = 27.12
        model_plus = PEMFModel(PEMFConfig(B_amplitude=1.0, frequency=f_opt + delta))
        model_minus = PEMFModel(PEMFConfig(B_amplitude=1.0, frequency=f_opt - delta))
        val_plus = model_plus.compute_anti_inflammatory(t=0.0)
        val_minus = model_minus.compute_anti_inflammatory(t=0.0)
        assert val_plus == pytest.approx(val_minus, rel=1e-10)

    def test_b0_zero_protection(self):
        """B0_threshold=0 → не должно быть деления на 0."""
        model = PEMFModel(PEMFConfig(B0_threshold=1e-15, B_amplitude=1.0))
        result = model.compute_anti_inflammatory(t=0.0)
        assert math.isfinite(result)


# ============================================================================
# 13. PEMFModel.compute_proliferation_boost
# ============================================================================


class TestComputeProliferationBoost:
    """Усиление пролиферации через Ca²⁺-CaM путь."""

    def test_optimal_params(self):
        """B=1.0, f=75 → ≈ 0.24 (ε_max · Hill(B²))."""
        model = PEMFModel(PEMFConfig(B_amplitude=1.0, frequency=75.0))
        result = model.compute_proliferation_boost(t=0.0)
        # Hill(B²=1, B_half²=0.25) = 1/(0.25+1) = 0.8; Gauss(0)=1
        # ε = 0.3 * 0.8 = 0.24
        assert result == pytest.approx(0.24, abs=0.05)

    def test_b_zero_returns_zero(self):
        """B=0 → 0.0."""
        model = PEMFModel(PEMFConfig(B_amplitude=0.0))
        result = model.compute_proliferation_boost(t=0.0)
        assert result == 0.0

    def test_high_b_saturation(self):
        """B=5.0, f=75 → ≈ ε_prolif_max (насыщение)."""
        model = PEMFModel(PEMFConfig(B_amplitude=5.0, frequency=75.0))
        result = model.compute_proliferation_boost(t=0.0)
        assert result == pytest.approx(0.3, abs=0.02)

    def test_far_frequency_near_zero(self):
        """f=200 (далеко от f_center=75) → ≈ 0."""
        model = PEMFModel(PEMFConfig(B_amplitude=1.0, frequency=200.0))
        result = model.compute_proliferation_boost(t=0.0)
        assert result < 0.01

    def test_result_in_range(self, pemf_model):
        """Результат ∈ [0, ε_prolif_max]."""
        result = pemf_model.compute_proliferation_boost(t=0.0)
        assert 0.0 <= result <= pemf_model.config.epsilon_prolif_max

    def test_monotonic_with_b(self):
        """Растёт с B при фиксированной f."""
        results = []
        for b in [0.1, 0.5, 1.0, 2.0, 5.0]:
            model = PEMFModel(PEMFConfig(B_amplitude=b, frequency=75.0))
            results.append(model.compute_proliferation_boost(t=0.0))
        for i in range(len(results) - 1):
            assert results[i] <= results[i + 1]

    def test_max_at_f_center(self):
        """Максимум при f = f_center."""
        model_opt = PEMFModel(PEMFConfig(B_amplitude=1.0, frequency=75.0))
        model_off = PEMFModel(PEMFConfig(B_amplitude=1.0, frequency=50.0))
        assert model_opt.compute_proliferation_boost(t=0.0) >= (
            model_off.compute_proliferation_boost(t=0.0)
        )

    def test_b_half_hill_function(self):
        """При B=B_half → Hill(B²) = 0.5."""
        b_half = 0.5
        model = PEMFModel(PEMFConfig(B_amplitude=b_half, frequency=75.0))
        result = model.compute_proliferation_boost(t=0.0)
        # Hill(B_half²/(B_half² + B_half²)) = 0.5; Gauss(0) = 1
        # ε = 0.3 * 0.5 = 0.15
        assert result == pytest.approx(0.15, abs=0.02)


# ============================================================================
# 14. PEMFModel.compute_migration_boost
# ============================================================================


class TestComputeMigrationBoost:
    """Усиление миграции через MAPK/ERK путь."""

    def test_standard_b(self):
        """B=1.0 → ≈ ε_max · Hill(2) ≈ 0.2."""
        model = PEMFModel(PEMFConfig(B_amplitude=1.0))
        result = model.compute_migration_boost(t=0.0)
        # Hill(B/B0=2, n=2) = 4/5 = 0.8; ε = 0.25 * 0.8 = 0.2
        assert result == pytest.approx(0.2, abs=0.05)

    def test_b_zero_returns_zero(self):
        """B=0 → 0.0."""
        model = PEMFModel(PEMFConfig(B_amplitude=0.0))
        result = model.compute_migration_boost(t=0.0)
        assert result == 0.0

    def test_high_b_saturation(self):
        """B=10.0 → ≈ ε_migration_max (насыщение)."""
        model = PEMFModel(PEMFConfig(B_amplitude=10.0))
        result = model.compute_migration_boost(t=0.0)
        assert result == pytest.approx(0.25, abs=0.02)

    def test_result_in_range(self, pemf_model):
        """Результат ∈ [0, ε_migration_max]."""
        result = pemf_model.compute_migration_boost(t=0.0)
        assert 0.0 <= result <= pemf_model.config.epsilon_migration_max

    def test_frequency_independent(self):
        """Результат не зависит от частоты."""
        model_50 = PEMFModel(PEMFConfig(B_amplitude=1.0, frequency=50.0))
        model_100 = PEMFModel(PEMFConfig(B_amplitude=1.0, frequency=100.0))
        assert model_50.compute_migration_boost(t=0.0) == pytest.approx(
            model_100.compute_migration_boost(t=0.0), rel=1e-10
        )

    def test_monotonic_with_b(self):
        """Растёт с B."""
        results = []
        for b in [0.1, 0.5, 1.0, 2.0, 5.0]:
            model = PEMFModel(PEMFConfig(B_amplitude=b))
            results.append(model.compute_migration_boost(t=0.0))
        for i in range(len(results) - 1):
            assert results[i] <= results[i + 1]


# ============================================================================
# 15. PEMFModel.compute_effects
# ============================================================================


class TestComputeEffects:
    """Вычисление всех 3 эффектов PEMF."""

    def test_all_positive_with_field(self, pemf_model):
        """B=1.0, f=50 → все 3 эффекта > 0."""
        effects = pemf_model.compute_effects(t=0.0)
        assert effects.anti_inflammatory > 0.0
        assert effects.proliferation > 0.0
        assert effects.migration > 0.0

    def test_all_zero_no_field(self, pemf_model_no_field):
        """B=0 → все 3 эффекта = 0."""
        effects = pemf_model_no_field.compute_effects(t=0.0)
        assert effects.anti_inflammatory == 0.0
        assert effects.proliferation == 0.0
        assert effects.migration == 0.0

    def test_returns_pemf_effects(self, pemf_model):
        """Возвращает PEMFEffects."""
        result = pemf_model.compute_effects(t=0.0)
        assert isinstance(result, PEMFEffects)

    def test_consistent_with_individual(self, pemf_model):
        """Совпадает с результатами индивидуальных compute_ методов."""
        effects = pemf_model.compute_effects(t=0.0)
        assert effects.anti_inflammatory == pytest.approx(
            pemf_model.compute_anti_inflammatory(t=0.0)
        )
        assert effects.proliferation == pytest.approx(
            pemf_model.compute_proliferation_boost(t=0.0)
        )
        assert effects.migration == pytest.approx(
            pemf_model.compute_migration_boost(t=0.0)
        )


# ============================================================================
# 16. PEMFModel.is_active
# ============================================================================


class TestPEMFModelIsActive:
    """Проверка активности PEMF."""

    def test_active_with_field(self, pemf_model):
        """B_amplitude=1.0 → True."""
        assert pemf_model.is_active(t=0.0) is True

    def test_inactive_no_field(self, pemf_model_no_field):
        """B_amplitude=0.0 → False."""
        assert pemf_model_no_field.is_active(t=0.0) is False


# ============================================================================
# 17. SynergyModel.__init__
# ============================================================================


class TestSynergyModelInit:
    """Инициализация SynergyModel."""

    def test_default_config(self, synergy_model):
        """SynergyModel создаёт SynergyConfig() с defaults."""
        assert synergy_model.config == SynergyConfig()

    def test_custom_config(self, prp_model, pemf_model):
        """SynergyModel с beta_synergy=0.5."""
        model = SynergyModel(prp_model, pemf_model, SynergyConfig(beta_synergy=0.5))
        assert model.config.beta_synergy == 0.5

    def test_stores_models(self, synergy_model, prp_model, pemf_model):
        """prp_model и pemf_model сохранены."""
        assert synergy_model.prp_model is prp_model
        assert synergy_model.pemf_model is pemf_model


# ============================================================================
# 18. SynergyModel.compute_synergy_factor
# ============================================================================


class TestComputeSynergyFactor:
    """Вычисление коэффициента синергии."""

    def test_both_active_above_one(self, synergy_model):
        """PRP + PEMF → synergy > 1.0."""
        # t=2 — PRP активен, PEMF активна (B=1.0)
        result = synergy_model.compute_synergy_factor(t=2.0)
        assert result > 1.0

    def test_only_prp_returns_one(self, prp_model, pemf_model_no_field):
        """Только PRP (PEMF B=0) → 1.0."""
        model = SynergyModel(prp_model, pemf_model_no_field)
        result = model.compute_synergy_factor(t=2.0)
        assert result == 1.0

    def test_only_pemf_returns_one(self, prp_model, pemf_model):
        """До инъекции PRP → 1.0."""
        model = SynergyModel(prp_model, pemf_model)
        # t=5, app_time по умолчанию 0, но t=500 → PRP не активен
        result = model.compute_synergy_factor(t=500.0)
        assert result == 1.0

    def test_neither_active_returns_one(self, pemf_model_no_field):
        """Ни одна не активна → 1.0."""
        prp = PRPModel()
        model = SynergyModel(prp, pemf_model_no_field)
        result = model.compute_synergy_factor(t=500.0)
        assert result == 1.0

    def test_beta_zero_always_one(self, prp_model, pemf_model):
        """beta_synergy=0 → всегда 1.0."""
        model = SynergyModel(prp_model, pemf_model, SynergyConfig(beta_synergy=0.0))
        result = model.compute_synergy_factor(t=2.0)
        assert result == 1.0

    def test_synergy_geq_one(self, synergy_model):
        """Synergy >= 1.0 для всех t."""
        for t in [0.0, 1.0, 2.0, 5.0, 50.0, 500.0]:
            assert synergy_model.compute_synergy_factor(t=t) >= 1.0

    def test_synergy_in_biological_range(self, synergy_model):
        """Синергия < 1.5 (Onstenk 2015)."""
        for t in [0.0, 1.0, 2.0, 5.0, 10.0]:
            assert synergy_model.compute_synergy_factor(t=t) < 1.5


# ============================================================================
# 19. SynergyModel.apply_to_drift
# ============================================================================


class TestApplyToDrift:
    """Применение синергии к drift-модификатору."""

    def test_with_synergy(self, synergy_model):
        """При активной синергии результат > modifier."""
        # t=2 — обе терапии активны
        result = synergy_model.apply_to_drift(drift_modifier=0.5, t=2.0)
        assert result > 0.5

    def test_without_synergy(self, prp_model, pemf_model_no_field):
        """Без синергии → modifier без изменений."""
        model = SynergyModel(prp_model, pemf_model_no_field)
        result = model.apply_to_drift(drift_modifier=0.5, t=2.0)
        assert result == pytest.approx(0.5)

    def test_zero_modifier(self, synergy_model):
        """modifier=0 → 0.0 (synergy не меняет ноль)."""
        result = synergy_model.apply_to_drift(drift_modifier=0.0, t=2.0)
        assert result == 0.0

    def test_result_geq_modifier(self, synergy_model):
        """|result| >= |modifier| (синергия только усиливает)."""
        modifier = 0.5
        result = synergy_model.apply_to_drift(drift_modifier=modifier, t=2.0)
        assert abs(result) >= abs(modifier) - 1e-10

    def test_negative_modifier(self, synergy_model):
        """Отрицательный modifier корректно обрабатывается."""
        result = synergy_model.apply_to_drift(drift_modifier=-0.3, t=2.0)
        assert result <= -0.3 + 1e-10  # По модулю увеличивается


# ============================================================================
# 20. Биологические свойства (интеграционная валидация)
# ============================================================================


class TestBiologicalProperties:
    """Валидация биологических свойств моделей."""

    def test_pdgf_peak_1_2_hours(self, prp_model):
        """Пик PDGF при ~1-2 ч после инъекции (Marx 2004)."""
        values = {
            t: prp_model.compute_release(t=t).theta_pdgf
            for t in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
        }
        peak_time = max(values, key=values.get)
        assert 0.5 <= peak_time <= 3.0

    def test_vegf_sustained_24h(self, prp_model):
        """VEGF значимый до 24 ч (Everts 2006)."""
        state = prp_model.compute_release(t=20.0)
        assert state.theta_vegf > 0.01

    def test_tgfb_longest_release_72h(self, prp_model):
        """TGF-β самый долгий релиз — значимый уровень около 72 ч (Eppley 2006)."""
        state = prp_model.compute_release(t=50.0)
        assert state.theta_tgfb > state.theta_pdgf
        assert state.theta_tgfb > state.theta_vegf
        assert state.theta_tgfb > state.theta_egf

    def test_egf_fast_peak_decay_12h(self, prp_model):
        """EGF: пик < 1 ч, затухание к 12 ч (Anitua 2004)."""
        values = {
            t: prp_model.compute_release(t=t).theta_egf
            for t in [0.2, 0.5, 1.0, 2.0, 5.0, 12.0]
        }
        peak_time = max(values, key=values.get)
        assert peak_time <= 1.0
        # К 12 ч — значительное затухание
        assert values[12.0] < values[peak_time] * 0.1

    def test_pemf_tnf_reduction_30_50_percent(self):
        """PEMF anti_inflam при оптимуме 0.3-0.5 (Varani 2017)."""
        model = PEMFModel(PEMFConfig(B_amplitude=10.0, frequency=27.12))
        result = model.compute_anti_inflammatory(t=0.0)
        assert 0.3 <= result <= 0.5

    def test_pemf_prolif_frequency_window(self):
        """Пролиферация активна в окне 50-100 Гц (Pilla 2013)."""
        for f in [50.0, 75.0, 100.0]:
            model = PEMFModel(PEMFConfig(B_amplitude=1.0, frequency=f))
            result = model.compute_proliferation_boost(t=0.0)
            assert result > 0.01, f"Пролиферация ≈ 0 при f={f} Гц"

    def test_synergy_moderate_range(self):
        """Синергия ∈ (1.0, 1.5) при активных терапиях (Onstenk 2015)."""
        prp = PRPModel()
        pemf = PEMFModel()
        syn = SynergyModel(prp, pemf)
        # t=2 — обе активны
        factor = syn.compute_synergy_factor(t=2.0)
        assert 1.0 < factor < 1.5

    def test_factor_ordering_by_duration(self, prp_model):
        """Порядок по длительности: EGF < VEGF < PDGF < TGF-β."""
        # Проверяем при t=30ч — ранние факторы уже затухли
        state = prp_model.compute_release(t=30.0)
        assert state.theta_egf <= state.theta_vegf
        assert state.theta_vegf <= state.theta_pdgf
        assert state.theta_pdgf <= state.theta_tgfb
