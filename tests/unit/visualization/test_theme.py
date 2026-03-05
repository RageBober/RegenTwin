"""Тесты для src/visualization/theme.py — консистентность констант."""

import re

import plotly.graph_objects as go

from src.visualization.theme import (
    AUXILIARY_VARS,
    CYTOKINE_COLORS,
    CYTOKINE_VARS,
    ECM_COLORS,
    ECM_VARS,
    PHASE_COLORS,
    PLOTLY_LAYOUT_DEFAULTS,
    POPULATION_COLORS,
    POPULATION_VARS,
    THERAPY_COLORS,
    VARIABLE_LABELS,
    apply_default_layout,
)

HEX_PATTERN = re.compile(r"^#[0-9a-fA-F]{6}$")


class TestPopulationColors:
    """Проверка цветов 8 клеточных популяций."""

    def test_all_8_populations_have_colors(self) -> None:
        expected = {"P", "Ne", "M1", "M2", "F", "Mf", "E", "S"}
        assert set(POPULATION_COLORS.keys()) == expected

    def test_colors_are_valid_hex(self) -> None:
        for name, color in POPULATION_COLORS.items():
            assert HEX_PATTERN.match(color), f"{name}: {color} не валидный hex"


class TestCytokineColors:
    """Проверка цветов 7 цитокинов."""

    def test_all_7_cytokines_have_colors(self) -> None:
        expected = {"C_TNF", "C_IL10", "C_PDGF", "C_VEGF", "C_TGFb", "C_MCP1", "C_IL8"}
        assert set(CYTOKINE_COLORS.keys()) == expected

    def test_colors_are_valid_hex(self) -> None:
        for name, color in CYTOKINE_COLORS.items():
            assert HEX_PATTERN.match(color), f"{name}: {color} не валидный hex"


class TestECMColors:
    """Проверка цветов 3 ECM компонентов."""

    def test_all_3_ecm_have_colors(self) -> None:
        expected = {"rho_collagen", "C_MMP", "rho_fibrin"}
        assert set(ECM_COLORS.keys()) == expected

    def test_colors_are_valid_hex(self) -> None:
        for name, color in ECM_COLORS.items():
            assert HEX_PATTERN.match(color), f"{name}: {color} не валидный hex"


class TestTherapyColors:
    """Проверка цветов 4 терапевтических сценариев."""

    def test_all_4_therapies_have_colors(self) -> None:
        expected = {"Control", "PRP", "PEMF", "PRP+PEMF"}
        assert set(THERAPY_COLORS.keys()) == expected


class TestPhaseColors:
    """Проверка цветов 4 фаз заживления."""

    def test_all_4_phases_have_colors(self) -> None:
        expected = {"hemostasis", "inflammation", "proliferation", "remodeling"}
        assert set(PHASE_COLORS.keys()) == expected


class TestVariableLabels:
    """Проверка подписей для всех 20 переменных."""

    def test_all_20_variables_have_labels(self) -> None:
        all_vars = POPULATION_VARS + CYTOKINE_VARS + ECM_VARS + AUXILIARY_VARS
        assert len(all_vars) == 20
        for var in all_vars:
            assert var in VARIABLE_LABELS, f"Нет подписи для {var}"

    def test_labels_are_nonempty_strings(self) -> None:
        for name, label in VARIABLE_LABELS.items():
            assert isinstance(label, str) and len(label) > 0, f"{name}: пустая подпись"


class TestVariableGroups:
    """Проверка корректности группировки переменных."""

    def test_population_vars_count(self) -> None:
        assert len(POPULATION_VARS) == 8

    def test_cytokine_vars_count(self) -> None:
        assert len(CYTOKINE_VARS) == 7

    def test_ecm_vars_count(self) -> None:
        assert len(ECM_VARS) == 3

    def test_auxiliary_vars_count(self) -> None:
        assert len(AUXILIARY_VARS) == 2

    def test_no_duplicates(self) -> None:
        all_vars = POPULATION_VARS + CYTOKINE_VARS + ECM_VARS + AUXILIARY_VARS
        assert len(all_vars) == len(set(all_vars))


class TestApplyDefaultLayout:
    """Проверка apply_default_layout."""

    def test_returns_figure(self) -> None:
        fig = go.Figure()
        result = apply_default_layout(fig)
        assert isinstance(result, go.Figure)

    def test_sets_height(self) -> None:
        fig = go.Figure()
        apply_default_layout(fig, height=600)
        assert fig.layout.height == 600

    def test_sets_template(self) -> None:
        fig = go.Figure()
        apply_default_layout(fig)
        assert fig.layout.template.layout.to_plotly_json() is not None

    def test_overrides_work(self) -> None:
        fig = go.Figure()
        apply_default_layout(fig, title="Test Title")
        assert fig.layout.title.text == "Test Title"
