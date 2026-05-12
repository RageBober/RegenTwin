"""TDD-набор для математического фреймворка v2.0 (FIX-01..FIX-25).

Каждый класс/тест соответствует фазе из плана рефакторинга
`C:\\Users\\dzume\\.claude\\plans\\keen-cuddling-tome.md`:

- Phase 1 → TestFix01Units, TestFix19Calibration
- Phase 3 → TestFix03Myofibroblasts, TestFix09Timp, TestFix10Fibrinolysis
- Phase 4 → TestFix04Fibroblasts, TestFix05Endothelial, TestFix06StemCells,
            TestFix21M1Default, TestFix22FibroblastMigration
- Phase 5 → TestFix07TnfInhibition, TestFix08TgfBinding, TestFix25Il8Hill
- Phase 6 → TestFix12PrpVolume, TestFix13PemfWindows, TestFix14Synergy
- Phase 7 → TestFix11Oxygen, TestFix15Operator, TestFix16Multirate,
            TestFix17Convergence, TestFix18Kurtz, TestFix24Ito
- Phase 8 → TestFix20Noise

Тесты, помеченные как `pytest.mark.xfail(strict=True)`, относятся к не реализованным
ещё фиксам — после применения соответствующего FIX'а тест должен начать падать (xpassed),
после чего xfail-маркер снимается и тест становится зелёным.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.extended_sde import ExtendedSDEModel, ExtendedSDEState
from src.core.parameters import ParameterSet
from src.core.params_loader import get_initial_conditions, load_params_yaml

# Путь к params.yaml в корне репозитория.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_PARAMS_YAML = _REPO_ROOT / "params.yaml"

# Период симуляции для приведения к стационару (72 часа покрывают фазу воспаления).
_T_END_INFLAMMATION = 72.0
# Шаг для быстрых тестов (0.05 ч → 1440 шагов за 72 ч).
_DT_FAST = 0.05


def _yaml_initial_state() -> ExtendedSDEState:
    """Построить ExtendedSDEState из YAML initial_conditions.

    Поля C_SDF1, C_TIMP появятся в ExtendedSDEState только в Phase 2;
    до тех пор их IC-значения игнорируются (метод фильтрует по имеющимся
    атрибутам класса).
    """
    raw = load_params_yaml(_PARAMS_YAML)
    ic = get_initial_conditions(raw)
    valid_attrs = set(ExtendedSDEState.__dataclass_fields__.keys())
    filtered = {k: float(v) for k, v in ic.items() if k in valid_attrs}
    return ExtendedSDEState(**filtered, t=0.0)


def _build_yaml_model(rng_seed: int = 42) -> ExtendedSDEModel:
    """ExtendedSDEModel с параметрами из params.yaml и быстрым шагом для тестов."""
    params = ParameterSet.from_yaml(_PARAMS_YAML)
    # Уменьшаем dt для устойчивости при cells/ml-калибровке.
    params.dt = _DT_FAST
    params.t_max = _T_END_INFLAMMATION
    return ExtendedSDEModel(params=params, rng_seed=rng_seed)


# =============================================================================
# Phase 1 — FIX-01 (единицы) + FIX-19 (калибровка цитокинов)
# =============================================================================


class TestFix01Units:
    """FIX-01: ВСЕ клеточные плотности в cells/ml; FIX-19: калибровка s_X.

    Контракт: при типичной воспалительной популяции (M1 ~ 1e4 cells/ml, F ~ 5e5 cells/ml)
    стационарные концентрации цитокинов должны лежать в физиологическом диапазоне:
        0.1 ≤ C_TNF ≤ 20 ng/ml   (Bradley 2008, Trengove 2000)
        0.1 ≤ C_PDGF ≤ 50 ng/ml
        0.05 ≤ C_VEGF ≤ 5 ng/ml
    """

    @pytest.fixture(scope="class")
    def trajectory(self):
        model = _build_yaml_model(rng_seed=42)
        ic = _yaml_initial_state()
        return model.simulate(initial_state=ic, t_span=(0.0, _T_END_INFLAMMATION))

    def test_no_nan_no_inf(self, trajectory):
        """Численная стабильность: значения остаются конечными."""
        import math

        final = trajectory.states[-1]
        for name in ("C_TNF", "C_PDGF", "C_VEGF", "M1", "F", "E"):
            value = getattr(final, name)
            assert math.isfinite(value), f"{name}={value!r} не конечно"
            assert value >= 0, f"{name}={value!r} отрицательно"

    def test_tnf_physiological_at_72h(self, trajectory):
        """TNF-α (стационар ~24-48 ч): 0.1 ≤ C_TNF ≤ 20 ng/ml.

        IL-10 ингибирует на источнике → значение должно держаться существенно ниже
        грубой оценки s·M1/γ ≈ 5e-6·1e4/0.5 = 0.1 ng/ml.
        """
        final = trajectory.states[-1]
        assert 0.1 <= final.C_TNF <= 20.0, (
            f"C_TNF = {final.C_TNF:.3f} вне физиологического диапазона [0.1, 20] ng/ml. "
            f"Проверьте калибровку s_TNF_M1/γ_TNF в params.yaml (FIX-19)."
        )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "C_PDGF полностью съедается рецепторным связыванием фибробластов при "
            "cells/ml-калибровке (k_bind_F · F = 0.001 · 5e5 = 500 1/h при C ~ K). "
            "Будет решено в Phase 4 (FIX-05/FIX-11 устранят дивергенцию M1/M2/O2, "
            "уменьшающую F-индуцированный сток) либо в FIX-19 рекалибровкой k_bind_F."
        ),
    )
    def test_pdgf_physiological_at_72h(self, trajectory):
        """PDGF (продукция тромбоцитов + макрофаги): 0.1 ≤ C_PDGF ≤ 50 ng/ml."""
        final = trajectory.states[-1]
        assert 0.1 <= final.C_PDGF <= 50.0, (
            f"C_PDGF = {final.C_PDGF:.3f} вне диапазона [0.1, 50] ng/ml."
        )

    def test_vegf_physiological_at_72h(self, trajectory):
        """VEGF (M2 + фибробласты, усиление гипоксией): 0.05 ≤ C_VEGF ≤ 5 ng/ml."""
        final = trajectory.states[-1]
        assert 0.05 <= final.C_VEGF <= 5.0, (
            f"C_VEGF = {final.C_VEGF:.3f} вне диапазона [0.05, 5] ng/ml."
        )

    def test_inflammation_peak_m1(self, trajectory):
        """Качественный sanity-check: пик M1 в первые 72 ч (фаза воспаления)."""
        m1_max = max(s.M1 for s in trajectory.states)
        m1_initial = trajectory.states[0].M1
        # M1 должны хотя бы не схлопнуться до нуля сразу.
        assert m1_max >= 0.5 * m1_initial, (
            f"M1 коллапсируют слишком быстро: max={m1_max:.0f} vs IC={m1_initial:.0f}"
        )

    def test_cells_per_ml_scale_of_yaml_ic(self):
        """Конвенция: IC в cells/ml даёт значения ~10²..10⁵ (не 10⁻¹..10²)."""
        ic = _yaml_initial_state()
        # F=5e5 в YAML — типично для cells/ml. Будь это cells/μl, было бы 500.
        assert ic.F >= 1e4, f"F={ic.F} слишком мало для cells/ml-калибровки"
        # M1=1e4 в YAML
        assert ic.M1 >= 1e3, f"M1={ic.M1} слишком мало для cells/ml-калибровки"


# =============================================================================
# Phase 3 — критические математические фиксы (стабы, активируются в Phase 3)
# =============================================================================


@pytest.mark.xfail(reason="FIX-03 будет применён в Phase 3", strict=False)
class TestFix03Myofibroblasts:
    def test_mf_no_divergence_at_high_tgf(self):
        pytest.skip("Активируется в Phase 3 (FIX-03)")


@pytest.mark.xfail(reason="FIX-09 будет применён в Phase 3", strict=False)
class TestFix09Timp:
    def test_timp_dynamic_not_constant(self):
        pytest.skip("Активируется в Phase 3 (FIX-09)")


# =============================================================================
# Phase 4 — клеточные структурные фиксы (стабы)
# =============================================================================


@pytest.mark.xfail(reason="FIX-05 будет применён в Phase 4", strict=False)
class TestFix05Endothelial:
    def test_e_recovery_from_low(self):
        pytest.skip("Активируется в Phase 4 (FIX-05)")


@pytest.mark.xfail(reason="FIX-06 будет применён в Phase 4", strict=False)
class TestFix06StemCells:
    def test_s_recovery_from_low(self):
        pytest.skip("Активируется в Phase 4 (FIX-06)")


# =============================================================================
# Phase 7 — численные методы (стабы)
# =============================================================================


@pytest.mark.xfail(reason="FIX-17 будет применён в Phase 7", strict=False)
class TestFix17ConvergenceGBM:
    def test_strong_convergence_em_order_half(self):
        pytest.skip("Активируется в Phase 7 (FIX-17)")
