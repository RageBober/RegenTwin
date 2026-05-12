"""Phase 0 — TDD-тесты для params_loader и ParameterSet.from_yaml.

Цели:
- Убедиться, что params.yaml (v2.0 источник правды) грузится без ошибок.
- Убедиться, что критические новые параметры (FIX-03..FIX-25) доступны через
  ParameterSet.from_yaml() и соответствуют значениям из YAML.
- Зафиксировать соглашение по единицам: cells/ml (FIX-01).

Эти тесты НЕ требуют запуска SDE/ABM. Они проверяют только инфраструктуру
загрузки конфигурации.

Plan reference: C:\\Users\\dzume\\.claude\\plans\\keen-cuddling-tome.md Phase 0.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.parameters import ParameterSet
from src.core.params_loader import (
    REQUIRED_SECTIONS,
    flatten_for_parameter_set,
    load_params_yaml,
)

# Путь к params.yaml в корне репозитория. Тесты могут запускаться из любого CWD,
# поэтому путь нормализуем относительно расположения этого файла.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_PARAMS_YAML = _REPO_ROOT / "params.yaml"


# =============================================================================
# Test 1: YAML грузится без ошибок и содержит все обязательные секции.
# =============================================================================


class TestYamlLoads:
    def test_yaml_loads_without_error(self):
        """params.yaml парсится и возвращает dict с обязательными секциями."""
        data = load_params_yaml(_PARAMS_YAML)
        assert isinstance(data, dict)
        for section in REQUIRED_SECTIONS:
            assert section in data, f"Отсутствует секция {section!r} в params.yaml"

    def test_value_extraction(self):
        """{value, units, source} узлы заменены на значение."""
        data = load_params_yaml(_PARAMS_YAML)
        # cells.r_F в YAML — словарь с метаданными; после _walk_extract должно
        # остаться чистое число.
        cells = data["cells"]
        assert isinstance(cells, dict)
        assert cells["r_F"] == pytest.approx(0.03)
        # noise.sigma_P — голый скаляр (без units), должен сохраниться как есть.
        noise = data["noise"]
        assert isinstance(noise, dict)
        assert noise["sigma_P"] == pytest.approx(0.1)

    def test_initial_conditions_section_present(self):
        """initial_conditions содержит новые переменные C_SDF1 и C_TIMP."""
        data = load_params_yaml(_PARAMS_YAML)
        ic = data["initial_conditions"]
        assert "C_SDF1" in ic
        assert "C_TIMP" in ic
        assert ic["C_SDF1"] == pytest.approx(0.01)
        assert ic["C_TIMP"] == pytest.approx(0.01)


# =============================================================================
# Test 2: Критические новые параметры присутствуют в ParameterSet.from_yaml().
# =============================================================================


class TestCriticalParamsPresent:
    """Каждый из ~35 новых параметров (FIX-03..FIX-25) загружается из YAML."""

    @pytest.fixture
    def ps(self) -> ParameterSet:
        return ParameterSet.from_yaml(_PARAMS_YAML)

    def test_fix03_myofibroblast_floor(self, ps: ParameterSet):
        """FIX-03: δ_floor для устранения дивергенции Mf."""
        assert ps.delta_floor == pytest.approx(0.1)

    def test_fix21_m1_baseline_polarization(self, ps: ParameterSet):
        """FIX-21: φ_baseline для M1-default моноцитов."""
        assert ps.phi_baseline == pytest.approx(0.1)

    def test_fix22_fibroblast_migration(self, ps: ParameterSet):
        """FIX-22: миграция фибробластов с хемотаксисом по PDGF."""
        assert ps.J_F_migration == pytest.approx(50.0)
        assert ps.K_chi == pytest.approx(1.0)

    def test_fix05_endothelial_sprouting(self, ps: ParameterSet):
        """FIX-05: sprouting tip cells от перифериальных сосудов."""
        assert ps.J_sprouting == pytest.approx(1.0)
        assert ps.K_xi == pytest.approx(0.5)

    def test_fix06_stem_cell_homing(self, ps: ParameterSet):
        """FIX-06: SDF-1/CXCR4 хоминг CD34+/MSC."""
        assert ps.J_homing == pytest.approx(5.0)
        assert pytest.approx(1.0) == ps.K_SDF1
        assert ps.alpha_PRP_homing == pytest.approx(2.0)

    def test_fix07_il10_inhibition_hill(self, ps: ParameterSet):
        """FIX-07: Hill коэффициент n_inhib для IL-10-ингибиции TNF."""
        assert ps.n_inhib == pytest.approx(1.5)

    def test_fix08_tgf_receptor_binding(self, ps: ParameterSet):
        """FIX-08: рецепторное потребление TGF-β."""
        assert ps.k_bind_TGF == pytest.approx(0.001)
        assert ps.K_TGF_bind == pytest.approx(0.5)

    def test_fix09_timp_dynamics(self, ps: ParameterSet):
        """FIX-09: уравнение TIMP с TGF-β-индуцированной продукцией."""
        assert ps.s_TIMP_F == pytest.approx(2.0e-9)
        assert ps.s_TIMP_M2 == pytest.approx(1.0e-9)
        assert ps.alpha_TGF_TIMP == pytest.approx(3.0)
        assert pytest.approx(1.0) == ps.K_TIMP
        assert ps.gamma_TIMP == pytest.approx(0.05)

    def test_sdf1_equation_params(self, ps: ParameterSet):
        """SDF-1 (новое уравнение): продукция фибробластами + гипоксия эндотелием."""
        assert ps.s_SDF1_F == pytest.approx(5.0e-7)
        assert ps.s_SDF1_E == pytest.approx(2.0e-6)
        assert ps.alpha_hypoxia_SDF == pytest.approx(5.0)
        assert ps.gamma_SDF1 == pytest.approx(0.2)

    def test_fix12_prp_v_wound(self, ps: ParameterSet):
        """FIX-12: явная нормализация PRP-кинетики на V_wound."""
        assert ps.V_wound == pytest.approx(5.0)
        assert pytest.approx(1.0) == ps.D_PRP

    def test_fix13_pemf_two_windows(self, ps: ParameterSet):
        """FIX-13: две частотных оконных функции — LF (75 Hz) и RF (27.12 MHz)."""
        assert ps.f_LF == pytest.approx(75.0)
        assert ps.sigma_LF == pytest.approx(30.0)
        assert ps.f_RF == pytest.approx(2.712e7)
        assert ps.sigma_RF_log == pytest.approx(0.3)

    def test_fix14_prp_pemf_synergy(self, ps: ParameterSet):
        """FIX-14: синергия β_synergy с явной нормировкой Theta_PRP_ref."""
        assert ps.beta_synergy == pytest.approx(1.5)
        assert ps.Theta_PRP_ref == pytest.approx(1.0)

    def test_fix11_oxygen_weighted_metabolism(self, ps: ParameterSet):
        """FIX-11: метаболические веса для взвешенной суммы потребления O₂."""
        assert ps.alpha_E == pytest.approx(1.0)
        assert ps.w_Ne == pytest.approx(100.0)
        assert ps.w_M == pytest.approx(10.0)
        assert ps.w_F == pytest.approx(1.0)
        assert ps.w_E == pytest.approx(5.0)
        assert ps.w_S == pytest.approx(0.5)

    def test_fix20_sigma_sdf1(self, ps: ParameterSet):
        """FIX-20: σ_SDF1 для нового цитокина."""
        assert ps.sigma_SDF1 == pytest.approx(0.2)

    def test_fix16_multirate_subcycling(self, ps: ParameterSet):
        """FIX-16: разные шаги для цитокинов (fast) и клеток (slow)."""
        assert ps.dt_fast == pytest.approx(0.02)
        assert ps.dt_slow == pytest.approx(1.0)
        assert ps.multirate_subcycling is True
        assert ps.X_min == pytest.approx(1.0e-10)

    def test_fix24_ito_interpretation(self, ps: ParameterSet):
        """FIX-24: явная Itô-интерпретация SDE."""
        assert ps.interpretation == "Ito"

    def test_fix20_ecm_deterministic(self, ps: ParameterSet):
        """FIX-20: ECM-уравнения без шума."""
        assert ps.ecm_deterministic is True


# =============================================================================
# Test 3: Конвенция единиц — cells/ml (FIX-01).
# =============================================================================


class TestUnitsConvention:
    def test_units_convention_cells_per_ml(self):
        """FIX-01: ВСЕ клеточные плотности теперь в cells/ml (не cells/mkl)."""
        data = load_params_yaml(_PARAMS_YAML)
        # units_convention — отдельная секция, не в REQUIRED_SECTIONS, но
        # должна присутствовать как информационная.
        assert "units_convention" in data, "params.yaml: отсутствует units_convention"
        uc = data["units_convention"]
        assert uc["cells"] == "cells/ml", (
            f"Конвенция единиц нарушена: ожидалось 'cells/ml', получено {uc.get('cells')!r}"
        )
        assert uc["cytokines"] == "ng/ml"
        assert uc["time"] == "h"
        assert uc["noise_interpretation"] == "Ito"


# =============================================================================
# Доп. контракт: fallback при отсутствии файла + flatten работает.
# =============================================================================


def test_from_yaml_missing_file_falls_back_to_defaults(tmp_path: Path):
    """Если params.yaml отсутствует — возврат hardcoded defaults с warning."""
    missing = tmp_path / "no_such.yaml"
    with pytest.warns(UserWarning, match="params.yaml не найден"):
        ps = ParameterSet.from_yaml(missing)
    # Fallback должен дать те же поля, что и обычный конструктор.
    assert ps.r_F == ParameterSet().r_F


def test_flatten_skips_prp_factors_nested():
    """prp.factors (вложенная карта факторов) не попадает в плоский словарь.

    Эти данные потребляет PRPModel напрямую, не через ParameterSet.
    """
    data = load_params_yaml(_PARAMS_YAML)
    flat = flatten_for_parameter_set(data)
    # 'factors' — не валидное имя поля, и не должно появиться как ключ.
    assert "factors" not in flat
    # Но другие prp-параметры — должны.
    assert "V_wound" in flat
    assert "D_PRP" in flat
