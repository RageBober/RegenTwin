"""
TDD тесты для модуля gating.py

Тестирует:
- GateResult dataclass
- GatingResults dataclass
- GatingStrategy класс и все методы гейтирования

Основано на спецификации: Description/description_gating.md

Ожидаемые фракции популяций:
- Debris: ~20%
- Live cells: ~70%
- CD34+: ~5%
- Macrophages: ~3%
- Apoptotic: ~2%
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from src.data.gating import GateResult, GatingResults, GatingStrategy


# =============================================================================
# Тесты для GateResult
# =============================================================================

class TestGateResult:
    """Тесты для dataclass GateResult."""

    def test_gate_result_creation_with_all_fields(self):
        """Тест создания GateResult со всеми полями."""
        mask = np.array([True, False, True, True, False])

        gate = GateResult(
            name="live_cells",
            mask=mask,
            n_events=3,
            fraction=0.6,
            parent="singlets",
            statistics={"mean_fsc": 100000}
        )

        assert gate.name == "live_cells"
        assert np.array_equal(gate.mask, mask)
        assert gate.n_events == 3
        assert gate.fraction == 0.6
        assert gate.parent == "singlets"
        assert gate.statistics == {"mean_fsc": 100000}

    def test_gate_result_statistics_default_empty_dict(self):
        """Тест что statistics по умолчанию пустой словарь."""
        gate = GateResult(
            name="test",
            mask=np.array([True]),
            n_events=1,
            fraction=1.0
        )

        assert gate.statistics == {} or gate.statistics is None

    def test_gate_result_parent_default_none(self):
        """Тест что parent по умолчанию None."""
        gate = GateResult(
            name="test",
            mask=np.array([True]),
            n_events=1,
            fraction=1.0
        )

        assert gate.parent is None

    def test_gate_result_mask_is_boolean_array(self):
        """Тест что mask - boolean массив."""
        mask = np.array([True, False, True])
        gate = GateResult(
            name="test",
            mask=mask,
            n_events=2,
            fraction=0.67
        )

        assert gate.mask.dtype == bool


# =============================================================================
# Тесты для GatingResults
# =============================================================================

class TestGatingResults:
    """Тесты для GatingResults."""

    @pytest.fixture
    def sample_gating_results(self):
        """Создает пример GatingResults для тестов."""
        n = 10

        gates = {
            "live_cells": GateResult(
                name="live_cells",
                mask=np.array([True] * 7 + [False] * 3),
                n_events=7,
                fraction=0.70
            ),
            "cd34_positive": GateResult(
                name="cd34_positive",
                mask=np.array([True] * 1 + [False] * 9),
                n_events=1,
                fraction=0.10,
                parent="live_cells"
            ),
        }

        return GatingResults(total_events=n, gates=gates)

    def test_gating_results_creation(self, sample_gating_results):
        """Тест создания GatingResults."""
        assert sample_gating_results.total_events == 10
        assert len(sample_gating_results.gates) == 2

    def test_gating_results_gates_is_dict(self, sample_gating_results):
        """Тест что gates - словарь."""
        assert isinstance(sample_gating_results.gates, dict)

    def test_get_population_returns_mask(self, sample_gating_results):
        """Тест что get_population возвращает маску."""
        mask = sample_gating_results.get_population("live_cells")

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool

    def test_get_population_correct_mask(self, sample_gating_results):
        """Тест корректности возвращаемой маски."""
        mask = sample_gating_results.get_population("live_cells")

        assert mask.sum() == 7

    def test_get_population_nonexistent_raises_error(self, sample_gating_results):
        """Тест что несуществующая популяция вызывает ошибку."""
        with pytest.raises(KeyError):
            sample_gating_results.get_population("nonexistent_population")

    def test_get_statistics_returns_dict(self, sample_gating_results):
        """Тест что get_statistics возвращает словарь."""
        stats = sample_gating_results.get_statistics()

        assert isinstance(stats, dict)

    def test_get_statistics_contains_total_events(self, sample_gating_results):
        """Тест что статистика содержит total_events."""
        stats = sample_gating_results.get_statistics()

        assert "total_events" in stats
        assert stats["total_events"] == 10


# =============================================================================
# Тесты для GatingStrategy.__init__
# =============================================================================

class TestGatingStrategyInit:
    """Тесты для GatingStrategy.__init__."""

    def test_init_without_mapping_uses_defaults(self):
        """Тест что без маппинга используются DEFAULT_CHANNELS."""
        strategy = GatingStrategy()

        assert strategy._channels["fsc_area"] == "FSC-A"
        assert strategy._channels["fsc_height"] == "FSC-H"
        assert strategy._channels["ssc_area"] == "SSC-A"

    def test_init_with_custom_mapping_overrides(self):
        """Тест что кастомный маппинг переопределяет значения."""
        custom_mapping = {
            "cd34": "CD34-BV421",  # Другой флуорохром
        }

        strategy = GatingStrategy(channel_mapping=custom_mapping)

        assert strategy._channels["cd34"] == "CD34-BV421"
        # Остальные должны быть по умолчанию
        assert strategy._channels["fsc_area"] == "FSC-A"

    def test_init_preserves_default_for_unmapped(self):
        """Тест что незатронутые каналы остаются по умолчанию."""
        custom_mapping = {"cd34": "CD34-Custom"}

        strategy = GatingStrategy(channel_mapping=custom_mapping)

        assert strategy._channels["annexin"] == "Annexin-V-Pacific Blue"

    def test_init_channels_attribute_is_dict(self):
        """Тест что _channels - это словарь."""
        strategy = GatingStrategy()

        assert isinstance(strategy._channels, dict)


# =============================================================================
# Тесты для GatingStrategy._find_channel
# =============================================================================

class TestGatingStrategyFindChannel:
    """Тесты для GatingStrategy._find_channel."""

    def test_find_channel_exact_match(self):
        """Тест точного совпадения имени канала."""
        strategy = GatingStrategy()
        columns = ["FSC-A", "SSC-A", "CD34-APC"]

        result = strategy._find_channel(columns, "CD34-APC")

        assert result == "CD34-APC"

    def test_find_channel_substring_match(self):
        """Тест совпадения по подстроке."""
        strategy = GatingStrategy()
        columns = ["FSC-A", "SSC-A", "CD34-APC"]

        result = strategy._find_channel(columns, "CD34")

        assert result == "CD34-APC"

    def test_find_channel_not_found_returns_pattern(self):
        """Тест что при отсутствии совпадения возвращается сам паттерн."""
        strategy = GatingStrategy()
        columns = ["FSC-A", "SSC-A", "CD34-APC"]

        result = strategy._find_channel(columns, "NONEXISTENT")

        assert result == "NONEXISTENT"


# =============================================================================
# Тесты для GatingStrategy.debris_gate
# =============================================================================

class TestGatingStrategyDebrisGate:
    """Тесты для GatingStrategy.debris_gate."""

    @pytest.fixture
    def fsc_ssc_data(self):
        """Данные FSC/SSC с явным debris."""
        rng = np.random.default_rng(42)

        # Debris (низкие значения) - 20%
        fsc_debris = rng.uniform(1000, 25000, size=200)
        ssc_debris = rng.uniform(1000, 15000, size=200)

        # Нормальные клетки - 80%
        fsc_normal = rng.normal(100000, 20000, size=800)
        ssc_normal = rng.normal(50000, 15000, size=800)

        fsc = np.concatenate([fsc_debris, fsc_normal])
        ssc = np.concatenate([ssc_debris, ssc_normal])

        return fsc, ssc

    def test_debris_gate_returns_boolean_mask(self, fsc_ssc_data):
        """Тест что возвращается boolean маска."""
        fsc, ssc = fsc_ssc_data
        strategy = GatingStrategy()

        mask = strategy.debris_gate(fsc, ssc)

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert len(mask) == len(fsc)

    def test_debris_gate_excludes_low_fsc_ssc(self, fsc_ssc_data):
        """Тест что debris (низкий FSC/SSC) исключается."""
        fsc, ssc = fsc_ssc_data
        strategy = GatingStrategy()

        mask = strategy.debris_gate(fsc, ssc)

        # Маска True = НЕ debris
        # Первые 200 событий - debris
        debris_kept = mask[:200].sum()
        normal_kept = mask[200:].sum()

        # Большинство debris должно быть исключено
        assert debris_kept < 200 * 0.5
        # Большинство нормальных должно остаться
        assert normal_kept > 800 * 0.7

    def test_debris_gate_with_explicit_thresholds(self):
        """Тест с явно заданными порогами."""
        fsc = np.array([10000, 50000, 100000, 150000])
        ssc = np.array([5000, 30000, 50000, 80000])

        strategy = GatingStrategy()
        mask = strategy.debris_gate(fsc, ssc, fsc_threshold=40000, ssc_threshold=20000)

        # Только события выше обоих порогов
        expected = np.array([False, True, True, True])
        assert np.array_equal(mask, expected)

    def test_debris_gate_auto_threshold(self, fsc_ssc_data):
        """Тест автоматического определения порогов."""
        fsc, ssc = fsc_ssc_data
        strategy = GatingStrategy()

        # Без явных порогов должен использовать автопороги
        mask = strategy.debris_gate(fsc, ssc)

        # Должен исключить примерно 15-30% событий (debris)
        debris_fraction = (~mask).sum() / len(mask)
        assert 0.10 < debris_fraction < 0.40

    def test_debris_gate_expected_fraction(self, mock_fcs_data_normal):
        """Тест что debris fraction близка к ожидаемой (~20%)."""
        strategy = GatingStrategy()
        fsc = mock_fcs_data_normal["FSC-A"].values
        ssc = mock_fcs_data_normal["SSC-A"].values

        mask = strategy.debris_gate(fsc, ssc)
        debris_fraction = (~mask).sum() / len(mask)

        # ~20% +/- 10%
        assert 0.10 < debris_fraction < 0.35


# =============================================================================
# Тесты для GatingStrategy.singlets_gate
# =============================================================================

class TestGatingStrategySingletsGate:
    """Тесты для GatingStrategy.singlets_gate."""

    @pytest.fixture
    def fsc_ah_data(self):
        """Данные FSC-A/FSC-H с синглетами и дублетами."""
        rng = np.random.default_rng(42)

        # Синглеты (FSC-A ~ FSC-H) - 90%
        fsc_h_singlets = rng.normal(80000, 15000, size=900)
        fsc_a_singlets = fsc_h_singlets * rng.normal(1.05, 0.03, size=900)

        # Дублеты (FSC-A >> FSC-H) - 10%
        fsc_h_doublets = rng.normal(80000, 15000, size=100)
        fsc_a_doublets = fsc_h_doublets * rng.normal(1.8, 0.1, size=100)

        fsc_a = np.concatenate([fsc_a_singlets, fsc_a_doublets])
        fsc_h = np.concatenate([fsc_h_singlets, fsc_h_doublets])

        return fsc_a, fsc_h

    def test_singlets_gate_returns_boolean_mask(self, fsc_ah_data):
        """Тест что возвращается boolean маска."""
        fsc_a, fsc_h = fsc_ah_data
        strategy = GatingStrategy()

        mask = strategy.singlets_gate(fsc_a, fsc_h)

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool

    def test_singlets_gate_excludes_doublets(self, fsc_ah_data):
        """Тест что дублеты исключаются."""
        fsc_a, fsc_h = fsc_ah_data
        strategy = GatingStrategy()

        mask = strategy.singlets_gate(fsc_a, fsc_h)

        # Последние 100 - дублеты
        doublets_kept = mask[900:].sum()
        singlets_kept = mask[:900].sum()

        # Большинство дублетов исключено
        assert doublets_kept < 100 * 0.5
        # Большинство синглетов остается
        assert singlets_kept > 900 * 0.7

    def test_singlets_gate_tolerance_parameter(self):
        """Тест параметра tolerance."""
        fsc_h = np.array([100.0, 100.0, 100.0, 100.0])
        fsc_a = np.array([105.0, 115.0, 130.0, 180.0])

        strategy = GatingStrategy()

        # С tolerance=0.1, ratio=1.05 должен пройти
        mask_default = strategy.singlets_gate(fsc_a, fsc_h, tolerance=0.1)

        # С меньшим tolerance меньше событий пройдёт
        mask_strict = strategy.singlets_gate(fsc_a, fsc_h, tolerance=0.02)

        assert mask_default.sum() >= mask_strict.sum()

    def test_singlets_gate_mask_length_matches_input(self, fsc_ah_data):
        """Тест что длина маски соответствует входным данным."""
        fsc_a, fsc_h = fsc_ah_data
        strategy = GatingStrategy()

        mask = strategy.singlets_gate(fsc_a, fsc_h)

        assert len(mask) == len(fsc_a)


# =============================================================================
# Тесты для GatingStrategy.live_cells_gate
# =============================================================================

class TestGatingStrategyLiveCellsGate:
    """Тесты для GatingStrategy.live_cells_gate."""

    @pytest.fixture
    def annexin_data(self):
        """Данные Annexin-V с живыми и мертвыми клетками."""
        rng = np.random.default_rng(42)

        # Живые клетки (низкий Annexin-V) - 85%
        live = rng.exponential(3000, size=850)

        # Апоптотические/мертвые (высокий Annexin-V) - 15%
        dead = rng.normal(120000, 20000, size=150)

        return np.concatenate([live, dead])

    def test_live_cells_gate_returns_boolean_mask(self, annexin_data):
        """Тест что возвращается boolean маска."""
        strategy = GatingStrategy()

        mask = strategy.live_cells_gate(annexin_data)

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool

    def test_live_cells_gate_selects_low_annexin(self, annexin_data):
        """Тест что выбираются клетки с низким Annexin-V."""
        strategy = GatingStrategy()

        mask = strategy.live_cells_gate(annexin_data)

        # Первые 850 - живые
        live_selected = mask[:850].sum()
        dead_selected = mask[850:].sum()

        # Большинство живых выбрано
        assert live_selected > 850 * 0.7
        # Большинство мертвых исключено
        assert dead_selected < 150 * 0.4

    def test_live_cells_gate_with_explicit_threshold(self):
        """Тест с явным порогом."""
        annexin = np.array([1000, 5000, 10000, 100000, 150000])

        strategy = GatingStrategy()
        mask = strategy.live_cells_gate(annexin, threshold=50000)

        expected = np.array([True, True, True, False, False])
        assert np.array_equal(mask, expected)

    def test_live_cells_gate_expected_fraction(self, mock_fcs_data_normal):
        """Тест ожидаемой фракции живых клеток (~70%)."""
        strategy = GatingStrategy()
        annexin = mock_fcs_data_normal["Annexin-V-Pacific Blue"].values

        mask = strategy.live_cells_gate(annexin)
        live_fraction = mask.sum() / len(mask)

        # Ожидаем 60-85% живых
        assert 0.55 < live_fraction < 0.90


# =============================================================================
# Тесты для GatingStrategy.cd34_gate
# =============================================================================

class TestGatingStrategyCd34Gate:
    """Тесты для GatingStrategy.cd34_gate."""

    @pytest.fixture
    def cd34_data(self):
        """Данные CD34 с позитивными и негативными клетками."""
        rng = np.random.default_rng(42)

        # CD34- негативные (95%)
        negative = rng.exponential(5000, size=950)

        # CD34+ позитивные (5%)
        positive = rng.normal(150000, 30000, size=50)

        return np.concatenate([negative, positive])

    def test_cd34_gate_returns_boolean_mask(self, cd34_data):
        """Тест что возвращается boolean маска."""
        strategy = GatingStrategy()

        mask = strategy.cd34_gate(cd34_data)

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool

    def test_cd34_gate_selects_high_cd34(self, cd34_data):
        """Тест что выбираются клетки с высоким CD34."""
        strategy = GatingStrategy()

        mask = strategy.cd34_gate(cd34_data)

        # Последние 50 - CD34+
        positive_selected = mask[950:].sum()

        # Большинство позитивных выбрано
        assert positive_selected > 50 * 0.6

    def test_cd34_gate_percentile_parameter(self, cd34_data):
        """Тест параметра percentile."""
        strategy = GatingStrategy()

        # Высокий перцентиль = меньше клеток выбрано
        high_mask = strategy.cd34_gate(cd34_data, percentile=99.0)
        # Низкий перцентиль = больше клеток выбрано
        low_mask = strategy.cd34_gate(cd34_data, percentile=90.0)

        assert high_mask.sum() < low_mask.sum()

    def test_cd34_gate_with_explicit_threshold(self):
        """Тест с явным порогом."""
        cd34 = np.array([1000, 10000, 50000, 100000, 200000])

        strategy = GatingStrategy()
        mask = strategy.cd34_gate(cd34, threshold=80000)

        expected = np.array([False, False, False, True, True])
        assert np.array_equal(mask, expected)

    def test_cd34_gate_expected_fraction(self, mock_fcs_data_normal):
        """Тест ожидаемой фракции CD34+ (~5%)."""
        strategy = GatingStrategy()
        cd34 = mock_fcs_data_normal["CD34-APC"].values

        mask = strategy.cd34_gate(cd34)
        cd34_fraction = mask.sum() / len(mask)

        # Ожидаем 2-12%
        assert 0.02 < cd34_fraction < 0.15


# =============================================================================
# Тесты для GatingStrategy.macrophage_gate
# =============================================================================

class TestGatingStrategyMacrophageGate:
    """Тесты для GatingStrategy.macrophage_gate."""

    @pytest.fixture
    def cd14_cd68_data(self):
        """Данные CD14/CD68 с макрофагами и другими клетками."""
        rng = np.random.default_rng(42)

        # Негативные по обоим (97%)
        cd14_neg = rng.exponential(5000, size=970)
        cd68_neg = rng.exponential(3000, size=970)

        # Макрофаги CD14+/CD68+ (3%)
        cd14_pos = rng.normal(100000, 20000, size=30)
        cd68_pos = rng.normal(80000, 15000, size=30)

        cd14 = np.concatenate([cd14_neg, cd14_pos])
        cd68 = np.concatenate([cd68_neg, cd68_pos])

        return cd14, cd68

    def test_macrophage_gate_returns_boolean_mask(self, cd14_cd68_data):
        """Тест что возвращается boolean маска."""
        cd14, cd68 = cd14_cd68_data
        strategy = GatingStrategy()

        mask = strategy.macrophage_gate(cd14, cd68)

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool

    def test_macrophage_gate_selects_positive_cells(self, cd14_cd68_data):
        """Тест что выбираются CD14+/CD68+ клетки."""
        cd14, cd68 = cd14_cd68_data
        strategy = GatingStrategy()

        mask = strategy.macrophage_gate(cd14, cd68)

        # Последние 30 - макрофаги
        macrophages_selected = mask[970:].sum()

        # Большинство макрофагов выбрано
        assert macrophages_selected > 30 * 0.5

    def test_macrophage_gate_with_explicit_thresholds(self):
        """Тест с явными порогами."""
        cd14 = np.array([5000, 50000, 100000, 5000, 100000])
        cd68 = np.array([3000, 3000, 3000, 80000, 80000])

        strategy = GatingStrategy()
        mask = strategy.macrophage_gate(
            cd14, cd68,
            cd14_threshold=50000,
            cd68_threshold=50000
        )

        # OR логика: CD14+ OR CD68+
        # [0]: оба низкие -> False
        # [1]: CD14+ -> True
        # [2]: CD14+ -> True
        # [3]: CD68+ -> True
        # [4]: оба высокие -> True
        assert mask.sum() >= 3  # Минимум 3 позитивных

    def test_macrophage_gate_expected_fraction(self, mock_fcs_data_normal):
        """Тест ожидаемой фракции макрофагов (~3%)."""
        strategy = GatingStrategy()
        cd14 = mock_fcs_data_normal["CD14-PE"].values
        cd68 = mock_fcs_data_normal["CD68-FITC"].values

        mask = strategy.macrophage_gate(cd14, cd68)
        macro_fraction = mask.sum() / len(mask)

        # Ожидаем 1-10%
        assert 0.01 < macro_fraction < 0.12


# =============================================================================
# Тесты для GatingStrategy.apoptotic_gate
# =============================================================================

class TestGatingStrategyApoptoticGate:
    """Тесты для GatingStrategy.apoptotic_gate."""

    def test_apoptotic_gate_returns_boolean_mask(self):
        """Тест что возвращается boolean маска."""
        rng = np.random.default_rng(42)
        annexin = rng.exponential(10000, size=1000)

        strategy = GatingStrategy()
        mask = strategy.apoptotic_gate(annexin)

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool

    def test_apoptotic_gate_selects_high_annexin(self):
        """Тест что выбираются клетки с высоким Annexin-V."""
        annexin = np.array([1000, 5000, 10000, 100000, 150000])

        strategy = GatingStrategy()
        mask = strategy.apoptotic_gate(annexin, threshold=50000)

        expected = np.array([False, False, False, True, True])
        assert np.array_equal(mask, expected)

    def test_apoptotic_gate_inverse_of_live_cells(self):
        """Тест что апоптотический гейт - инверсия живых клеток для общего порога."""
        rng = np.random.default_rng(42)
        annexin = np.concatenate([
            rng.exponential(3000, size=800),
            rng.normal(120000, 20000, size=200)
        ])

        strategy = GatingStrategy()
        threshold = 50000

        live_mask = strategy.live_cells_gate(annexin, threshold=threshold)
        apopt_mask = strategy.apoptotic_gate(annexin, threshold=threshold)

        # Маски должны быть взаимоисключающими
        assert not np.any(live_mask & apopt_mask)

    def test_apoptotic_gate_expected_fraction(self, mock_fcs_data_normal):
        """Тест ожидаемой фракции апоптотических (~2%)."""
        strategy = GatingStrategy()
        annexin = mock_fcs_data_normal["Annexin-V-Pacific Blue"].values

        mask = strategy.apoptotic_gate(annexin)
        apopt_fraction = mask.sum() / len(mask)

        # Ожидаем 1-10%
        assert 0.01 < apopt_fraction < 0.12


# =============================================================================
# Тесты для GatingStrategy.apply
# =============================================================================

class TestGatingStrategyApply:
    """Тесты для GatingStrategy.apply."""

    def test_apply_returns_gating_results(self, mock_fcs_data_normal):
        """Тест что apply возвращает GatingResults."""
        strategy = GatingStrategy()

        results = strategy.apply(mock_fcs_data_normal)

        assert isinstance(results, GatingResults)

    def test_apply_contains_expected_gates(self, mock_fcs_data_normal):
        """Тест что результат содержит ожидаемые гейты."""
        strategy = GatingStrategy()

        results = strategy.apply(mock_fcs_data_normal)

        expected_gates = [
            "non_debris", "singlets", "live_cells",
            "cd34_positive", "macrophages", "apoptotic"
        ]
        for gate_name in expected_gates:
            assert gate_name in results.gates, f"Missing gate: {gate_name}"

    def test_apply_total_events_correct(self, mock_fcs_data_normal):
        """Тест корректности total_events."""
        strategy = GatingStrategy()

        results = strategy.apply(mock_fcs_data_normal)

        assert results.total_events == len(mock_fcs_data_normal)

    def test_apply_hierarchical_gating_cd34_subset_of_live(self, mock_fcs_data_normal):
        """Тест что CD34+ подмножество live_cells."""
        strategy = GatingStrategy()

        results = strategy.apply(mock_fcs_data_normal)

        cd34_events = results.gates["cd34_positive"].n_events
        live_events = results.gates["live_cells"].n_events

        assert cd34_events <= live_events

    def test_apply_hierarchical_gating_macrophages_subset_of_live(self, mock_fcs_data_normal):
        """Тест что macrophages подмножество live_cells."""
        strategy = GatingStrategy()

        results = strategy.apply(mock_fcs_data_normal)

        macro_events = results.gates["macrophages"].n_events
        live_events = results.gates["live_cells"].n_events

        assert macro_events <= live_events

    def test_apply_with_ndarray_input(self, mock_fcs_data_normal):
        """Тест что apply работает с numpy array."""
        strategy = GatingStrategy()

        # Передаем как ndarray - использует стандартный порядок колонок
        ndarray_data = mock_fcs_data_normal.values
        results = strategy.apply(ndarray_data)

        assert isinstance(results, GatingResults)

    def test_apply_gate_fractions_sum_reasonable(self, mock_fcs_data_normal):
        """Тест что сумма непересекающихся фракций разумна."""
        strategy = GatingStrategy()

        results = strategy.apply(mock_fcs_data_normal)

        # Debris + non_debris = 100%
        debris_fraction = 1.0 - results.gates["non_debris"].fraction
        non_debris_fraction = results.gates["non_debris"].fraction

        assert abs((debris_fraction + non_debris_fraction) - 1.0) < 0.01

    def test_apply_with_custom_channel_mapping(self, mock_fcs_data_normal):
        """Тест apply с кастомным маппингом каналов."""
        # Используем стандартные названия из mock данных
        custom_mapping = {
            "fsc_area": "FSC-A",
            "fsc_height": "FSC-H",
            "ssc_area": "SSC-A",
            "cd34": "CD34-APC",
            "cd14": "CD14-PE",
            "cd68": "CD68-FITC",
            "annexin": "Annexin-V-Pacific Blue",
        }

        strategy = GatingStrategy(channel_mapping=custom_mapping)
        results = strategy.apply(mock_fcs_data_normal)

        assert isinstance(results, GatingResults)


# =============================================================================
# Тесты для GatingStrategy._auto_threshold
# =============================================================================

class TestGatingStrategyAutoThreshold:
    """Тесты для GatingStrategy._auto_threshold."""

    def test_auto_threshold_otsu_returns_float(self):
        """Тест что метод Оцу возвращает float."""
        rng = np.random.default_rng(42)
        # Бимодальное распределение
        low = rng.normal(1000, 200, size=700)
        high = rng.normal(5000, 500, size=300)
        data = np.concatenate([low, high])

        strategy = GatingStrategy()
        threshold = strategy._auto_threshold(data, method="otsu")

        assert isinstance(threshold, (int, float, np.floating))

    def test_auto_threshold_otsu_between_modes(self):
        """Тест что порог Оцу между модами."""
        rng = np.random.default_rng(42)
        low = rng.normal(1000, 100, size=700)
        high = rng.normal(5000, 100, size=300)
        data = np.concatenate([low, high])

        strategy = GatingStrategy()
        threshold = strategy._auto_threshold(data, method="otsu")

        # Порог должен быть между пиками
        assert 1500 < threshold < 4500

    def test_auto_threshold_percentile_method(self):
        """Тест перцентильного метода."""
        data = np.arange(100, dtype=float)

        strategy = GatingStrategy()
        threshold = strategy._auto_threshold(data, method="percentile")

        # По умолчанию ~95 перцентиль
        expected = np.percentile(data, 95)
        assert abs(threshold - expected) < 5

    def test_auto_threshold_invalid_method_raises(self):
        """Тест что невалидный метод вызывает ошибку."""
        strategy = GatingStrategy()

        with pytest.raises(ValueError):
            strategy._auto_threshold(np.array([1, 2, 3]), method="invalid_method")

    def test_auto_threshold_otsu_uniform_data_fallback(self):
        """Тест fallback к threshold_otsu для униформных данных без valleys."""
        # Униформные данные без чётких пиков/valleys
        data = np.linspace(0, 100, 1000)

        strategy = GatingStrategy()
        threshold = strategy._auto_threshold(data, method="otsu")

        # Должен вернуть валидный порог
        assert isinstance(threshold, (int, float, np.floating))
        assert 0 <= threshold <= 100

    def test_auto_threshold_otsu_fallback_to_threshold_otsu(self):
        """Тест fallback к threshold_otsu когда valleys пустой (монотонная гистограмма)."""
        # Данные со строго возрастающей гистограммой - каждый бин имеет больше значений
        # Это создает монотонную производную без локальных минимумов (valleys = [])
        data = []
        for i in range(256):
            data.extend([i * 100] * (i + 1))
        data = np.array(data, dtype=float)

        strategy = GatingStrategy()
        threshold = strategy._auto_threshold(data, method="otsu")

        # Должен вернуть валидный порог через threshold_otsu fallback
        assert isinstance(threshold, (int, float, np.floating))
        assert threshold > 0

    def test_auto_threshold_otsu_without_skimage(self):
        """Тест fallback когда skimage не установлен."""
        import src.data.gating as gating_module

        original_value = gating_module.HAS_SKIMAGE
        try:
            gating_module.HAS_SKIMAGE = False

            data = np.linspace(0, 100, 1000)
            strategy = GatingStrategy()
            threshold = strategy._auto_threshold(data, method="otsu")

            # Fallback к 70 перцентилю
            expected = np.percentile(data, 70)
            assert abs(threshold - expected) < 1
        finally:
            gating_module.HAS_SKIMAGE = original_value

    def test_module_handles_missing_skimage(self):
        """Тест что модуль корректно обрабатывает отсутствие skimage."""
        import src.data.gating as gating_module

        # Проверяем что HAS_SKIMAGE определен
        assert hasattr(gating_module, 'HAS_SKIMAGE')
        assert isinstance(gating_module.HAS_SKIMAGE, bool)

        # Если skimage установлен, HAS_SKIMAGE = True
        # Если нет - HAS_SKIMAGE = False
        # В любом случае модуль загрузился успешно
        try:
            from skimage.filters import threshold_otsu
            assert gating_module.HAS_SKIMAGE is True
        except ImportError:
            assert gating_module.HAS_SKIMAGE is False


# =============================================================================
# Тесты для GatingStrategy._density_gate
# =============================================================================

class TestGatingStrategyDensityGate:
    """Тесты для GatingStrategy._density_gate."""

    def test_density_gate_returns_boolean_mask(self):
        """Тест что возвращается boolean маска."""
        rng = np.random.default_rng(42)
        x = rng.normal(100, 20, size=1000)
        y = rng.normal(50, 10, size=1000)

        strategy = GatingStrategy()
        mask = strategy._density_gate(x, y)

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool

    def test_density_gate_fraction_parameter(self):
        """Тест параметра fraction."""
        rng = np.random.default_rng(42)
        x = rng.normal(100, 20, size=1000)
        y = rng.normal(50, 10, size=1000)

        strategy = GatingStrategy()

        mask_85 = strategy._density_gate(x, y, fraction=0.85)
        mask_95 = strategy._density_gate(x, y, fraction=0.95)

        # Больший fraction = больше событий (или равно)
        assert mask_95.sum() >= mask_85.sum()

    def test_density_gate_captures_dense_region(self):
        """Тест что захватывается плотный регион."""
        rng = np.random.default_rng(42)

        # Плотный кластер + разреженные точки
        x_dense = rng.normal(100, 5, size=900)
        y_dense = rng.normal(50, 5, size=900)
        x_sparse = rng.uniform(0, 200, size=100)
        y_sparse = rng.uniform(0, 100, size=100)

        x = np.concatenate([x_dense, x_sparse])
        y = np.concatenate([y_dense, y_sparse])

        strategy = GatingStrategy()
        mask = strategy._density_gate(x, y, fraction=0.85)

        # Плотный кластер должен быть в основном выбран
        dense_selected = mask[:900].sum()
        assert dense_selected > 900 * 0.7

    def test_density_gate_mask_length_matches_input(self):
        """Тест что длина маски соответствует входным данным."""
        rng = np.random.default_rng(42)
        x = rng.normal(100, 20, size=500)
        y = rng.normal(50, 10, size=500)

        strategy = GatingStrategy()
        mask = strategy._density_gate(x, y)

        assert len(mask) == 500


# =============================================================================
# Интеграционные тесты с разными сценариями данных
# =============================================================================

class TestGatingStrategyScenarios:
    """Интеграционные тесты с разными сценариями."""

    def test_inflamed_sample_high_macrophages(self, mock_fcs_data_inflamed):
        """Тест что воспаленный образец имеет больше макрофагов."""
        strategy = GatingStrategy()

        results = strategy.apply(mock_fcs_data_inflamed)

        macro_fraction = results.gates["macrophages"].fraction
        # Ожидаем повышенную фракцию (>5%)
        assert macro_fraction > 0.05

    def test_regenerating_sample_high_cd34(self, mock_fcs_data_regenerating):
        """Тест что регенерирующий образец имеет больше CD34+."""
        strategy = GatingStrategy()

        results = strategy.apply(mock_fcs_data_regenerating)

        cd34_fraction = results.gates["cd34_positive"].fraction
        # Ожидаем повышенную фракцию (>7%)
        assert cd34_fraction > 0.07

    def test_all_gate_fractions_between_0_and_1(self, mock_fcs_data_normal):
        """Тест что все фракции в диапазоне [0, 1]."""
        strategy = GatingStrategy()

        results = strategy.apply(mock_fcs_data_normal)

        for gate_name, gate_result in results.gates.items():
            assert 0 <= gate_result.fraction <= 1, \
                f"Gate {gate_name} has invalid fraction: {gate_result.fraction}"

    def test_all_masks_correct_length(self, mock_fcs_data_normal):
        """Тест что все маски имеют правильную длину."""
        strategy = GatingStrategy()
        n_events = len(mock_fcs_data_normal)

        results = strategy.apply(mock_fcs_data_normal)

        for gate_name, gate_result in results.gates.items():
            assert len(gate_result.mask) == n_events, \
                f"Gate {gate_name} mask has wrong length"


# =============================================================================
# Тесты для GatingStrategy.neutrophil_gate (расширенное гейтирование)
# =============================================================================

class TestNeutrophilGate:
    """Тесты для метода neutrophil_gate (CD66b+)."""

    def test_normal_data_approx_5_percent(self):
        """Тест что ~5% событий проходят нейтрофильный гейт."""
        rng = np.random.default_rng(60)
        # 95% — фоновый сигнал, 5% — высокий CD66b
        cd66b = np.concatenate([
            rng.exponential(4000, 950),
            rng.normal(120000, 25000, 50),
        ])
        strategy = GatingStrategy()
        mask = strategy.neutrophil_gate(cd66b)

        assert mask.shape == cd66b.shape
        assert mask.dtype == np.bool_
        fraction = mask.sum() / len(mask)
        assert 0.02 < fraction < 0.12

    def test_empty_array_shape_zero(self):
        """Тест что пустой массив → маска shape=(0,)."""
        strategy = GatingStrategy()
        mask = strategy.neutrophil_gate(np.array([]))
        assert mask.shape == (0,)
        assert mask.dtype == np.bool_

    def test_all_identical_all_false(self):
        """Тест что все одинаковые значения → все False."""
        strategy = GatingStrategy()
        cd66b = np.full(100, 5.0)
        mask = strategy.neutrophil_gate(cd66b)
        assert mask.sum() == 0

    def test_manual_threshold(self):
        """Тест с явно заданным threshold."""
        strategy = GatingStrategy()
        cd66b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mask = strategy.neutrophil_gate(cd66b, threshold=3.0)
        # > 3.0: только 4.0 и 5.0
        expected = np.array([False, False, False, True, True])
        np.testing.assert_array_equal(mask, expected)

    def test_single_element(self):
        """Тест с одним элементом."""
        strategy = GatingStrategy()
        cd66b = np.array([100.0])
        mask = strategy.neutrophil_gate(cd66b, percentile=95.0)
        assert mask.shape == (1,)
        assert mask.dtype == np.bool_

    def test_percentile_0_almost_all_true(self):
        """Тест что percentile=0 → почти все True (threshold = min)."""
        strategy = GatingStrategy()
        cd66b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mask = strategy.neutrophil_gate(cd66b, percentile=0.0)
        # threshold = min(1.0), mask = cd66b > 1.0 → [F,T,T,T,T]
        assert mask.sum() >= len(cd66b) - 1

    def test_percentile_100_all_false(self):
        """Тест что percentile=100 → все False."""
        strategy = GatingStrategy()
        cd66b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mask = strategy.neutrophil_gate(cd66b, percentile=100.0)
        assert mask.sum() == 0


# =============================================================================
# Тесты для GatingStrategy.endothelial_gate (расширенное гейтирование)
# =============================================================================

class TestEndothelialGate:
    """Тесты для метода endothelial_gate (CD31+)."""

    def test_normal_data_approx_3_percent(self):
        """Тест что ~3% событий проходят эндотелиальный гейт."""
        rng = np.random.default_rng(61)
        # 97% — фоновый сигнал, 3% — высокий CD31
        cd31 = np.concatenate([
            rng.exponential(3000, 970),
            rng.normal(100000, 20000, 30),
        ])
        strategy = GatingStrategy()
        mask = strategy.endothelial_gate(cd31)

        assert mask.shape == cd31.shape
        assert mask.dtype == np.bool_
        fraction = mask.sum() / len(mask)
        assert 0.01 < fraction < 0.10

    def test_empty_array_shape_zero(self):
        """Тест что пустой массив → маска shape=(0,)."""
        strategy = GatingStrategy()
        mask = strategy.endothelial_gate(np.array([]))
        assert mask.shape == (0,)
        assert mask.dtype == np.bool_

    def test_all_identical_all_false(self):
        """Тест что все одинаковые значения → все False."""
        strategy = GatingStrategy()
        cd31 = np.full(100, 5.0)
        mask = strategy.endothelial_gate(cd31)
        assert mask.sum() == 0

    def test_manual_threshold(self):
        """Тест с явно заданным threshold."""
        strategy = GatingStrategy()
        cd31 = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        mask = strategy.endothelial_gate(cd31, threshold=25.0)
        # > 25.0: 30, 40, 50
        expected = np.array([False, False, True, True, True])
        np.testing.assert_array_equal(mask, expected)

    def test_single_element(self):
        """Тест с одним элементом."""
        strategy = GatingStrategy()
        cd31 = np.array([50.0])
        mask = strategy.endothelial_gate(cd31, percentile=95.0)
        assert mask.shape == (1,)
        assert mask.dtype == np.bool_

    def test_percentile_0_almost_all_true(self):
        """Тест что percentile=0 → почти все True."""
        strategy = GatingStrategy()
        cd31 = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        mask = strategy.endothelial_gate(cd31, percentile=0.0)
        assert mask.sum() >= len(cd31) - 1

    def test_percentile_100_all_false(self):
        """Тест что percentile=100 → все False."""
        strategy = GatingStrategy()
        cd31 = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        mask = strategy.endothelial_gate(cd31, percentile=100.0)
        assert mask.sum() == 0


# =============================================================================
# Тесты для GatingStrategy.apply_extended (расширенное гейтирование)
# =============================================================================

class TestApplyExtended:
    """Тесты для метода apply_extended (9 каналов → 8 популяций)."""

    def test_ndarray_9_cols_returns_8_gates(self, mock_fcs_data_extended_normal):
        """Тест что ndarray с 9 столбцами → GatingResults с 8 гейтами."""
        strategy = GatingStrategy()
        data = mock_fcs_data_extended_normal.values
        results = strategy.apply_extended(data)
        assert isinstance(results, GatingResults)
        assert len(results.gates) == 8

    def test_dataframe_9_channels_returns_8_gates(self, mock_fcs_data_extended_normal):
        """Тест что DataFrame с 9 каналами → GatingResults с 8 гейтами."""
        strategy = GatingStrategy()
        results = strategy.apply_extended(mock_fcs_data_extended_normal)
        assert isinstance(results, GatingResults)
        assert len(results.gates) == 8

    def test_7_col_ndarray_raises_error(self):
        """Тест что ndarray с 7 столбцами → ошибка."""
        rng = np.random.default_rng(62)
        data = rng.uniform(0, 100000, (100, 7))
        strategy = GatingStrategy()
        with pytest.raises((IndexError, ValueError, KeyError)):
            strategy.apply_extended(data)

    def test_all_8_gate_keys_present(self, mock_fcs_data_extended_normal):
        """Тест что все 8 ключей гейтов присутствуют."""
        strategy = GatingStrategy()
        results = strategy.apply_extended(mock_fcs_data_extended_normal)
        expected_keys = {
            "non_debris", "singlets", "live_cells", "cd34_positive",
            "macrophages", "apoptotic", "neutrophils", "endothelial",
        }
        assert set(results.gates.keys()) == expected_keys

    def test_all_fractions_in_0_1(self, mock_fcs_data_extended_normal):
        """Тест что все фракции в диапазоне [0, 1]."""
        strategy = GatingStrategy()
        results = strategy.apply_extended(mock_fcs_data_extended_normal)
        for gate_name, gate in results.gates.items():
            assert 0 <= gate.fraction <= 1, \
                f"Gate {gate_name} fraction={gate.fraction} out of [0,1]"

    def test_neutrophils_parent_is_live_cells(self, mock_fcs_data_extended_normal):
        """Тест что neutrophils.parent == 'live_cells'."""
        strategy = GatingStrategy()
        results = strategy.apply_extended(mock_fcs_data_extended_normal)
        assert results.gates["neutrophils"].parent == "live_cells"

    def test_endothelial_parent_is_live_cells(self, mock_fcs_data_extended_normal):
        """Тест что endothelial.parent == 'live_cells'."""
        strategy = GatingStrategy()
        results = strategy.apply_extended(mock_fcs_data_extended_normal)
        assert results.gates["endothelial"].parent == "live_cells"

    def test_neutrophils_n_events_non_negative(self, mock_fcs_data_extended_normal):
        """Тест что neutrophils.n_events >= 0."""
        strategy = GatingStrategy()
        results = strategy.apply_extended(mock_fcs_data_extended_normal)
        assert results.gates["neutrophils"].n_events >= 0

    def test_endothelial_n_events_non_negative(self, mock_fcs_data_extended_normal):
        """Тест что endothelial.n_events >= 0."""
        strategy = GatingStrategy()
        results = strategy.apply_extended(mock_fcs_data_extended_normal)
        assert results.gates["endothelial"].n_events >= 0
