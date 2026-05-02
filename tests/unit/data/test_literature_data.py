"""Тесты для модуля literature_data.py.

Проверяет:
- Оцифрованные reference curves (Xue 2009, Flegg 2010)
- Маппинг переменных
- Phase breakpoints
- Метаданные
- Совместимость с ValidationRunner
"""

import numpy as np
import pytest

from src.data.dataset_loader import DatasetSource, TimeSeriesData
from src.data.literature_data import (
    ReferenceSource,
    get_flegg2010_metadata,
    get_flegg2010_reference,
    get_variable_mapping,
    get_xue2009_metadata,
    get_xue2009_phase_breakpoints,
    get_xue2009_reference,
)

# =============================================================================
# ReferenceSource enum
# =============================================================================


class TestReferenceSource:
    def test_enum_values(self):
        assert ReferenceSource.XUE_2009 == "Xue2009"
        assert ReferenceSource.FLEGG_2010 == "Flegg2010"


# =============================================================================
# Xue 2009 reference data
# =============================================================================


class TestXue2009Reference:
    @pytest.fixture
    def xue_data(self) -> TimeSeriesData:
        return get_xue2009_reference()

    def test_returns_time_series_data(self, xue_data: TimeSeriesData):
        assert isinstance(xue_data, TimeSeriesData)

    def test_time_points_range(self, xue_data: TimeSeriesData):
        """Время от 0 до 720 часов (30 дней)."""
        assert xue_data.time_points[0] == 0.0
        assert xue_data.time_points[-1] == 720.0
        assert len(xue_data.time_points) >= 6

    def test_time_points_monotonic(self, xue_data: TimeSeriesData):
        diffs = np.diff(xue_data.time_points)
        assert np.all(diffs > 0), "Time points must be strictly increasing"

    def test_contains_key_variables(self, xue_data: TimeSeriesData):
        """Должны быть основные переменные модели."""
        expected = {"M1", "M2", "F", "E", "C_PDGF", "C_VEGF", "rho_collagen", "O2"}
        assert expected.issubset(set(xue_data.values.keys()))

    def test_all_values_non_negative(self, xue_data: TimeSeriesData):
        for var, arr in xue_data.values.items():
            assert np.all(arr >= 0), f"Variable {var} has negative values"

    def test_values_match_time_points_length(self, xue_data: TimeSeriesData):
        n = len(xue_data.time_points)
        for var, arr in xue_data.values.items():
            assert len(arr) == n, f"Variable {var}: expected {n} points, got {len(arr)}"

    def test_units_present(self, xue_data: TimeSeriesData):
        for var in xue_data.values:
            assert var in xue_data.units, f"Missing unit for {var}"

    def test_metadata(self, xue_data: TimeSeriesData):
        assert xue_data.metadata is not None
        assert xue_data.metadata.source == DatasetSource.LOCAL
        assert (
            "xue" in xue_data.metadata.dataset_id.lower()
            or "xue" in xue_data.metadata.description.lower()
        )

    def test_macrophage_peak_in_inflammation(self, xue_data: TimeSeriesData):
        """Макрофаги должны иметь пик в фазе воспаления (24-168h)."""
        t = xue_data.time_points
        m_total = xue_data.values.get("M1", np.zeros_like(t)) + xue_data.values.get(
            "M2", np.zeros_like(t)
        )
        mask = (t >= 24) & (t <= 168)
        if np.any(mask):
            peak_in_inflammation = np.max(m_total[mask])
            initial = m_total[0]
            assert peak_in_inflammation > initial, "Macrophages should peak during inflammation"

    def test_collagen_increases_over_time(self, xue_data: TimeSeriesData):
        """Коллаген должен нарастать к концу."""
        collagen = xue_data.values.get("rho_collagen")
        if collagen is not None:
            assert collagen[-1] > collagen[0], "Collagen should increase over time"


# =============================================================================
# Flegg 2010 reference data
# =============================================================================


class TestFlegg2010Reference:
    @pytest.fixture
    def flegg_data(self) -> TimeSeriesData:
        return get_flegg2010_reference()

    def test_returns_time_series_data(self, flegg_data: TimeSeriesData):
        assert isinstance(flegg_data, TimeSeriesData)

    def test_time_points_range(self, flegg_data: TimeSeriesData):
        assert flegg_data.time_points[0] == 0.0
        assert flegg_data.time_points[-1] > 0.0

    def test_contains_wound_area(self, flegg_data: TimeSeriesData):
        """Должна быть хотя бы одна кривая wound area."""
        wound_keys = [k for k in flegg_data.values if "wound" in k.lower() or "area" in k.lower()]
        assert (
            len(wound_keys) > 0
        ), f"No wound area variables found in {list(flegg_data.values.keys())}"

    def test_wound_area_between_0_and_1(self, flegg_data: TimeSeriesData):
        """Wound area fraction должна быть в [0, 1]."""
        for key, arr in flegg_data.values.items():
            if "wound" in key.lower() or "area" in key.lower():
                assert np.all(arr >= 0) and np.all(arr <= 1.01), f"{key} should be fraction [0, 1]"

    def test_metadata(self, flegg_data: TimeSeriesData):
        assert flegg_data.metadata is not None
        assert flegg_data.metadata.source == DatasetSource.LOCAL


# =============================================================================
# Phase breakpoints
# =============================================================================


class TestPhaseBreakpoints:
    def test_returns_list(self):
        bps = get_xue2009_phase_breakpoints()
        assert isinstance(bps, list)
        assert len(bps) >= 2  # минимум 2 перехода фаз

    def test_breakpoint_structure(self):
        bps = get_xue2009_phase_breakpoints()
        for bp in bps:
            assert "time_hours" in bp
            assert "phase_before" in bp
            assert "phase_after" in bp
            assert isinstance(bp["time_hours"], (int, float))

    def test_breakpoints_ordered(self):
        bps = get_xue2009_phase_breakpoints()
        times = [bp["time_hours"] for bp in bps]
        assert times == sorted(times), "Breakpoints should be in chronological order"

    def test_inflammation_to_proliferation(self):
        """Переход inflammation→proliferation должен быть между 48-168h."""
        bps = get_xue2009_phase_breakpoints()
        for bp in bps:
            if (
                bp.get("phase_before") == "inflammation"
                and bp.get("phase_after") == "proliferation"
            ):
                t = float(bp["time_hours"])
                assert 48 <= t <= 168, f"inflammation→proliferation at {t}h is out of range"
                return
        pytest.fail("No inflammation→proliferation breakpoint found")


# =============================================================================
# Variable mapping
# =============================================================================


class TestVariableMapping:
    def test_returns_dict(self):
        mapping = get_variable_mapping()
        assert isinstance(mapping, dict)
        assert len(mapping) > 0

    def test_xue_variables_mapped(self):
        """Ключевые переменные Xue должны быть в маппинге."""
        mapping = get_variable_mapping()
        # Проверяем что есть маппинг для основных переменных Xue
        values = set(mapping.values())
        expected_targets = {"M_total", "F", "rho_collagen", "O2", "C_PDGF", "C_VEGF"}
        assert expected_targets.issubset(
            values
        ), f"Missing mappings for: {expected_targets - values}"


# =============================================================================
# Metadata
# =============================================================================


class TestMetadata:
    def test_xue2009_metadata(self):
        meta = get_xue2009_metadata()
        assert meta.dataset_id is not None
        assert meta.source == DatasetSource.LOCAL
        assert meta.species == "human"

    def test_flegg2010_metadata(self):
        meta = get_flegg2010_metadata()
        assert meta.dataset_id is not None
        assert meta.source == DatasetSource.LOCAL
