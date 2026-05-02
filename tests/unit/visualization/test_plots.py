"""TDD тесты для src/visualization/plots.py — графики временных рядов."""

import plotly.graph_objects as go

from src.core.extended_sde import ExtendedSDETrajectory
from src.core.monte_carlo import MonteCarloResults
from src.core.wound_phases import PhaseIndicators
from src.visualization.plots import (
    plot_comparison,
    plot_cytokines,
    plot_ecm,
    plot_phases,
    plot_populations,
)


class TestPlotPopulations:
    """Тесты plot_populations — кривые роста 8 популяций."""

    def test_returns_figure(self, mock_extended_trajectory: ExtendedSDETrajectory) -> None:
        fig = plot_populations(mock_extended_trajectory)
        assert isinstance(fig, go.Figure)

    def test_default_8_traces(self, mock_extended_trajectory: ExtendedSDETrajectory) -> None:
        fig = plot_populations(mock_extended_trajectory)
        assert len(fig.data) == 8

    def test_subset_variables(self, mock_extended_trajectory: ExtendedSDETrajectory) -> None:
        fig = plot_populations(mock_extended_trajectory, variables=["P", "Ne", "F"])
        assert len(fig.data) == 3

    def test_single_variable(self, mock_extended_trajectory: ExtendedSDETrajectory) -> None:
        fig = plot_populations(mock_extended_trajectory, variables=["F"])
        assert len(fig.data) == 1

    def test_xaxis_has_time_label(self, mock_extended_trajectory: ExtendedSDETrajectory) -> None:
        fig = plot_populations(mock_extended_trajectory)
        assert "Время" in (fig.layout.xaxis.title.text or "")

    def test_yaxis_has_cells_label(self, mock_extended_trajectory: ExtendedSDETrajectory) -> None:
        fig = plot_populations(mock_extended_trajectory)
        assert "мкл" in (fig.layout.yaxis.title.text or "")

    def test_with_ci_adds_extra_traces(
        self,
        mock_extended_trajectory: ExtendedSDETrajectory,
        mock_mc_results: MonteCarloResults,
    ) -> None:
        fig = plot_populations(
            mock_extended_trajectory,
            show_ci=True,
            mc_results=mock_mc_results,
        )
        assert len(fig.data) > 8

    def test_without_mc_results_ci_ignored(
        self,
        mock_extended_trajectory: ExtendedSDETrajectory,
    ) -> None:
        fig = plot_populations(mock_extended_trajectory, show_ci=True)
        assert len(fig.data) == 8

    def test_custom_height(self, mock_extended_trajectory: ExtendedSDETrajectory) -> None:
        fig = plot_populations(mock_extended_trajectory, height=700)
        assert fig.layout.height == 700

    def test_traces_have_names(self, mock_extended_trajectory: ExtendedSDETrajectory) -> None:
        fig = plot_populations(mock_extended_trajectory)
        names = [t.name for t in fig.data]
        assert all(n is not None and len(n) > 0 for n in names)

    def test_hovertemplate_has_time_and_units(
        self,
        mock_extended_trajectory: ExtendedSDETrajectory,
    ) -> None:
        fig = plot_populations(mock_extended_trajectory, variables=["F"])
        hovertemplate = fig.data[0].hovertemplate or ""
        assert "Время" in hovertemplate
        assert "кл/мкл" in hovertemplate

    def test_ci_band_label_clarifies_variable(
        self,
        mock_extended_trajectory: ExtendedSDETrajectory,
        mock_mc_results: MonteCarloResults,
    ) -> None:
        fig = plot_populations(
            mock_extended_trajectory,
            show_ci=True,
            mc_results=mock_mc_results,
        )
        trace_names = [trace.name for trace in fig.data]
        assert "95% CI (N, Monte Carlo)" in trace_names


class TestPlotCytokines:
    """Тесты plot_cytokines — динамика 7 цитокинов."""

    def test_returns_figure_overlay(self, mock_extended_trajectory: ExtendedSDETrajectory) -> None:
        fig = plot_cytokines(mock_extended_trajectory)
        assert isinstance(fig, go.Figure)

    def test_overlay_7_traces(self, mock_extended_trajectory: ExtendedSDETrajectory) -> None:
        fig = plot_cytokines(mock_extended_trajectory, layout="overlay")
        assert len(fig.data) == 7

    def test_subplots_mode(self, mock_extended_trajectory: ExtendedSDETrajectory) -> None:
        fig = plot_cytokines(mock_extended_trajectory, layout="subplots")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 7

    def test_subset_variables(self, mock_extended_trajectory: ExtendedSDETrajectory) -> None:
        fig = plot_cytokines(mock_extended_trajectory, variables=["C_TNF", "C_IL10"])
        assert len(fig.data) == 2

    def test_overlay_yaxis_label(self, mock_extended_trajectory: ExtendedSDETrajectory) -> None:
        fig = plot_cytokines(mock_extended_trajectory, layout="overlay")
        yaxis_text = fig.layout.yaxis.title.text or ""
        assert "нг/мл" in yaxis_text or "Концентрация" in yaxis_text

    def test_subplots_has_title(self, mock_extended_trajectory: ExtendedSDETrajectory) -> None:
        fig = plot_cytokines(mock_extended_trajectory, layout="subplots")
        assert fig.layout.title.text == "Динамика цитокинов"

    def test_overlay_hovertemplate_has_time_and_units(
        self,
        mock_extended_trajectory: ExtendedSDETrajectory,
    ) -> None:
        fig = plot_cytokines(mock_extended_trajectory, variables=["C_TNF"], layout="overlay")
        hovertemplate = fig.data[0].hovertemplate or ""
        assert "Время" in hovertemplate
        assert "нг/мл" in hovertemplate


class TestPlotECM:
    """Тесты plot_ecm — динамика ECM (коллаген, MMP, фибрин)."""

    def test_returns_figure(self, mock_extended_trajectory: ExtendedSDETrajectory) -> None:
        fig = plot_ecm(mock_extended_trajectory)
        assert isinstance(fig, go.Figure)

    def test_three_traces(self, mock_extended_trajectory: ExtendedSDETrajectory) -> None:
        fig = plot_ecm(mock_extended_trajectory)
        assert len(fig.data) == 3

    def test_has_secondary_y(self, mock_extended_trajectory: ExtendedSDETrajectory) -> None:
        fig = plot_ecm(mock_extended_trajectory)
        # MMP на вторичной оси
        assert fig.layout.yaxis2 is not None

    def test_hovertemplates_include_units(
        self, mock_extended_trajectory: ExtendedSDETrajectory
    ) -> None:
        fig = plot_ecm(mock_extended_trajectory)
        assert "отн. ед." in (fig.data[0].hovertemplate or "")
        assert "нг/мл" in (fig.data[2].hovertemplate or "")


class TestPlotPhases:
    """Тесты plot_phases — цветовая полоса фаз заживления."""

    def test_returns_figure(self, mock_extended_trajectory: ExtendedSDETrajectory) -> None:
        fig = plot_phases(mock_extended_trajectory)
        assert isinstance(fig, go.Figure)

    def test_with_precomputed_phases(
        self,
        mock_extended_trajectory: ExtendedSDETrajectory,
        mock_phase_indicators: list[PhaseIndicators],
    ) -> None:
        fig = plot_phases(mock_extended_trajectory, phase_indicators=mock_phase_indicators)
        assert isinstance(fig, go.Figure)
        # Должны быть trace-ы для фаз + ключевых популяций
        assert len(fig.data) > 5

    def test_has_population_traces(
        self,
        mock_extended_trajectory: ExtendedSDETrajectory,
        mock_phase_indicators: list[PhaseIndicators],
    ) -> None:
        fig = plot_phases(mock_extended_trajectory, phase_indicators=mock_phase_indicators)
        trace_names = [t.name for t in fig.data if t.name]
        # Должны быть популяции Ne, M1, M2, F, E
        population_traces = [
            n
            for n in trace_names
            if any(p in n for p in ["Нейтрофилы", "M1", "M2", "Фибробласты", "Эндотелиальные"])
        ]
        assert len(population_traces) >= 3

    def test_late_transition_keeps_last_phase_segment(
        self,
        mock_extended_trajectory: ExtendedSDETrajectory,
        mock_phase_indicators_late_transition: list[PhaseIndicators],
    ) -> None:
        fig = plot_phases(
            mock_extended_trajectory,
            phase_indicators=mock_phase_indicators_late_transition,
        )
        phase_traces = [trace for trace in fig.data if getattr(trace, "fill", None) == "toself"]
        phase_names = [trace.name for trace in phase_traces]
        remodeling_trace = next(trace for trace in phase_traces if trace.name == "Remodeling")
        assert len(phase_traces) == 2
        assert "Proliferation" in phase_names
        assert "Remodeling" in phase_names
        assert max(remodeling_trace.x) > min(remodeling_trace.x)

    def test_population_hovertemplate_has_time_and_units(
        self,
        mock_extended_trajectory: ExtendedSDETrajectory,
        mock_phase_indicators: list[PhaseIndicators],
    ) -> None:
        fig = plot_phases(mock_extended_trajectory, phase_indicators=mock_phase_indicators)
        population_trace = next(trace for trace in fig.data if trace.name == "Фибробласты (F)")
        hovertemplate = population_trace.hovertemplate or ""
        assert "Время" in hovertemplate
        assert "кл/мкл" in hovertemplate


class TestPlotComparison:
    """Тесты plot_comparison — сравнение сценариев."""

    def test_returns_figure(self, mock_extended_trajectory: ExtendedSDETrajectory) -> None:
        results = {"Control": mock_extended_trajectory, "PRP": mock_extended_trajectory}
        fig = plot_comparison(results)
        assert isinstance(fig, go.Figure)

    def test_four_scenarios(self, mock_extended_trajectory: ExtendedSDETrajectory) -> None:
        results = {
            "Control": mock_extended_trajectory,
            "PRP": mock_extended_trajectory,
            "PEMF": mock_extended_trajectory,
            "PRP+PEMF": mock_extended_trajectory,
        }
        fig = plot_comparison(results)
        assert len(fig.data) == 4

    def test_custom_variable(self, mock_extended_trajectory: ExtendedSDETrajectory) -> None:
        results = {"Control": mock_extended_trajectory}
        fig = plot_comparison(results, variable="E")
        assert isinstance(fig, go.Figure)

    def test_show_all_populations(self, mock_extended_trajectory: ExtendedSDETrajectory) -> None:
        results = {"Control": mock_extended_trajectory, "PRP": mock_extended_trajectory}
        fig = plot_comparison(results, show_all_populations=True)
        assert isinstance(fig, go.Figure)
        # 8 популяций × 2 сценария = 16 traces
        assert len(fig.data) == 16

    def test_scenario_names_in_legend(
        self, mock_extended_trajectory: ExtendedSDETrajectory
    ) -> None:
        results = {"Control": mock_extended_trajectory, "PRP": mock_extended_trajectory}
        fig = plot_comparison(results)
        names = [t.name for t in fig.data]
        assert "Control" in names
        assert "PRP" in names

    def test_yaxis_title_includes_units(
        self, mock_extended_trajectory: ExtendedSDETrajectory
    ) -> None:
        results = {"Control": mock_extended_trajectory, "PRP": mock_extended_trajectory}
        fig = plot_comparison(results, variable="F")
        assert "кл/мкл" in (fig.layout.yaxis.title.text or "")

    def test_hovertemplate_has_time_and_units(
        self,
        mock_extended_trajectory: ExtendedSDETrajectory,
    ) -> None:
        results = {"Control": mock_extended_trajectory, "PRP": mock_extended_trajectory}
        fig = plot_comparison(results, variable="F")
        hovertemplate = fig.data[0].hovertemplate or ""
        assert "Время" in hovertemplate
        assert "кл/мкл" in hovertemplate
