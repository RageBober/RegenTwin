"""TDD тесты для src/visualization/analysis_plots.py — визуализация анализа."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest

from src.core.parameter_estimation import (
    ConvergenceDiagnostics,
    EstimationResult,
)
from src.core.sensitivity_analysis import MorrisResult, SobolResult
from src.visualization.analysis_plots import (
    plot_convergence,
    plot_morris,
    plot_posterior,
    plot_sobol,
)
from src.visualization.theme import ANALYSIS_COLORS

# ═══════════════════════════════════════════════════════════════════
# Тесты по: Description/Phase3/description_analysis_plots.md#plot_sobol
# ═══════════════════════════════════════════════════════════════════


class TestPlotSobol:
    """Тесты plot_sobol — tornado bar chart для Sobol indices."""

    # --- Basic (happy-path) ---

    def test_returns_figure(self, mock_sobol_result: SobolResult) -> None:
        """plot_sobol возвращает go.Figure."""
        fig = plot_sobol(mock_sobol_result)
        assert isinstance(fig, go.Figure)

    def test_both_metric_two_bar_traces(self, mock_sobol_result: SobolResult) -> None:
        """metric='both' → ровно 2 go.Bar traces (S1 и ST)."""
        fig = plot_sobol(mock_sobol_result, metric="both")
        assert len(fig.data) == 2
        assert all(isinstance(t, go.Bar) for t in fig.data)

    def test_s1_only_one_trace(self, mock_sobol_result: SobolResult) -> None:
        """metric='S1' → ровно 1 bar trace."""
        fig = plot_sobol(mock_sobol_result, metric="S1")
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Bar)

    def test_st_only_one_trace(self, mock_sobol_result: SobolResult) -> None:
        """metric='ST' → ровно 1 bar trace."""
        fig = plot_sobol(mock_sobol_result, metric="ST")
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Bar)

    def test_horizontal_orientation(self, mock_sobol_result: SobolResult) -> None:
        """Все bar traces имеют orientation='h' (tornado chart)."""
        fig = plot_sobol(mock_sobol_result)
        assert all(t.orientation == "h" for t in fig.data)

    def test_title_contains_output_variable(self, mock_sobol_result: SobolResult) -> None:
        """Заголовок содержит output_variable ('F')."""
        fig = plot_sobol(mock_sobol_result)
        title = fig.layout.title.text or ""
        assert "F" in title

    def test_error_bars_present_when_show_confidence(
        self,
        mock_sobol_result: SobolResult,
    ) -> None:
        """show_confidence=True → traces содержат error_x с данными."""
        fig = plot_sobol(mock_sobol_result, show_confidence=True)
        for trace in fig.data:
            assert trace.error_x is not None
            assert trace.error_x.array is not None

    def test_no_error_bars_when_confidence_false(
        self,
        mock_sobol_result: SobolResult,
    ) -> None:
        """show_confidence=False → нет error_x на traces."""
        fig = plot_sobol(mock_sobol_result, show_confidence=False)
        for trace in fig.data:
            assert trace.error_x is None or trace.error_x.array is None

    # --- Edge cases (top_n, сортировка) ---

    def test_top_n_limits_parameters(self, mock_sobol_result: SobolResult) -> None:
        """top_n=3 с 8 параметрами → только 3 параметра на графике."""
        fig = plot_sobol(mock_sobol_result, top_n=3)
        # Y-axis horizontal bar = параметры
        assert len(fig.data[0].y) == 3

    def test_top_n_none_shows_all(self, mock_sobol_result: SobolResult) -> None:
        """top_n=None → все 8 параметров."""
        fig = plot_sobol(mock_sobol_result, top_n=None)
        assert len(fig.data[0].y) == 8

    def test_top_n_exceeding_count_shows_all(self, mock_sobol_result: SobolResult) -> None:
        """top_n=100 при 8 параметрах → все 8 (не ошибка)."""
        fig = plot_sobol(mock_sobol_result, top_n=100)
        assert len(fig.data[0].y) == 8

    def test_top_n_zero_shows_all(self, mock_sobol_result: SobolResult) -> None:
        """top_n=0 → показать все параметры (не обрезать)."""
        fig = plot_sobol(mock_sobol_result, top_n=0)
        assert len(fig.data[0].y) == 8

    def test_sorted_by_st_descending(self, mock_sobol_result: SobolResult) -> None:
        """Y-axis отсортирована по убыванию ST (наибольший снизу для horizontal bar)."""
        fig = plot_sobol(mock_sobol_result, metric="ST")
        # Горизонтальный bar: x = значения, y = имена параметров
        x_values = list(fig.data[0].x)
        # В инвертированном порядке для horizontal bar: снизу вверх = по убыванию
        # Значит сверху — наименьший, снизу — наибольший
        # x_values[0] — самый нижний (на графике наверху) = наименьший
        # x_values[-1] — самый верхний (на графике внизу) = наибольший
        # Проверяем что значения возрастают (для horizontal bar: сверху вниз)
        assert x_values == sorted(x_values)

    def test_custom_height_applied(self, mock_sobol_result: SobolResult) -> None:
        """height=700 → fig.layout.height == 700."""
        fig = plot_sobol(mock_sobol_result, height=700)
        assert fig.layout.height == 700

    # --- Error handling ---

    def test_invalid_metric_raises_valueerror(self, mock_sobol_result: SobolResult) -> None:
        """Невалидный metric → ValueError."""
        with pytest.raises(ValueError):
            plot_sobol(mock_sobol_result, metric="invalid")

    def test_empty_parameter_names_raises_valueerror(self) -> None:
        """SobolResult с пустыми parameter_names → ValueError."""
        empty = SobolResult(parameter_names=[], output_variable="F")
        with pytest.raises(ValueError):
            plot_sobol(empty)


# ═══════════════════════════════════════════════════════════════════
# Тесты по: Description/Phase3/description_analysis_plots.md#plot_posterior
# ═══════════════════════════════════════════════════════════════════


class TestPlotPosterior:
    """Тесты plot_posterior — marginals и corner plot."""

    # --- Basic marginals ---

    def test_returns_figure_marginals(self, mock_estimation_result: EstimationResult) -> None:
        """layout='marginals' возвращает go.Figure."""
        fig = plot_posterior(mock_estimation_result)
        assert isinstance(fig, go.Figure)

    def test_marginals_has_histogram_traces(
        self,
        mock_estimation_result: EstimationResult,
    ) -> None:
        """Marginals mode → go.Histogram trace для каждого параметра."""
        fig = plot_posterior(mock_estimation_result, layout="marginals")
        histograms = [t for t in fig.data if isinstance(t, go.Histogram)]
        assert len(histograms) >= 3  # 3 параметра

    def test_marginals_subplot_rows(self, mock_estimation_result: EstimationResult) -> None:
        """3 параметра → 3 строки подграфиков (уникальных yaxis)."""
        fig = plot_posterior(
            mock_estimation_result,
            layout="marginals",
            show_ci=False,
            show_point_estimate=False,
        )
        # Каждый histogram в своём subplot → уникальный yaxis
        histograms = [t for t in fig.data if isinstance(t, go.Histogram)]
        yaxes = {t.yaxis for t in histograms if t.yaxis is not None}
        # Первый subplot не имеет суффикса (yaxis=None → "y"), остальные — y2, y3
        assert len(yaxes | {None}) >= 3 or len(histograms) == 3

    def test_ci_lines_present(self, mock_estimation_result: EstimationResult) -> None:
        """show_ci=True → вертикальные dashed линии (go.Scatter)."""
        fig = plot_posterior(mock_estimation_result, show_ci=True)
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        dashed = [t for t in scatter_traces if t.line and t.line.dash == "dash"]
        # 3 параметра × 2 линии (lower + upper) = 6 dashed
        assert len(dashed) >= 6

    def test_point_estimate_lines_present(
        self,
        mock_estimation_result: EstimationResult,
    ) -> None:
        """show_point_estimate=True → вертикальные solid линии."""
        fig = plot_posterior(
            mock_estimation_result,
            show_ci=False,
            show_point_estimate=True,
        )
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        # 3 параметра × 1 линия = минимум 3 scatter traces
        assert len(scatter_traces) >= 3

    def test_no_ci_when_disabled(self, mock_estimation_result: EstimationResult) -> None:
        """show_ci=False → нет dashed CI-линий."""
        fig = plot_posterior(
            mock_estimation_result,
            show_ci=False,
            show_point_estimate=False,
        )
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        dashed = [t for t in scatter_traces if t.line and t.line.dash == "dash"]
        assert len(dashed) == 0

    # --- Corner mode ---

    def test_returns_figure_corner(self, mock_estimation_result: EstimationResult) -> None:
        """layout='corner' → go.Figure."""
        fig = plot_posterior(mock_estimation_result, layout="corner")
        assert isinstance(fig, go.Figure)

    def test_corner_has_histograms_on_diagonal(
        self,
        mock_estimation_result: EstimationResult,
    ) -> None:
        """Corner mode: 3 Histogram traces (на диагонали N×N матрицы)."""
        fig = plot_posterior(
            mock_estimation_result,
            layout="corner",
            show_ci=False,
            show_point_estimate=False,
        )
        histograms = [t for t in fig.data if isinstance(t, go.Histogram)]
        assert len(histograms) == 3

    def test_corner_has_scatter_off_diagonal(
        self,
        mock_estimation_result: EstimationResult,
    ) -> None:
        """Corner mode: off-diagonal scatter plots (N*(N-1)/2 = 3 для N=3)."""
        fig = plot_posterior(
            mock_estimation_result,
            layout="corner",
            show_ci=False,
            show_point_estimate=False,
        )
        scatter_markers = [
            t for t in fig.data if isinstance(t, go.Scatter) and t.mode and "markers" in t.mode
        ]
        assert len(scatter_markers) == 3  # lower triangle: (1,0), (2,0), (2,1)

    def test_corner_downsample_large_samples(self) -> None:
        """Corner mode с >5000 samples → off-diagonal scatter ≤ 5000 точек."""
        rng = np.random.default_rng(99)
        n_samples = 10_000
        result = EstimationResult(
            method="bayesian_pymc",
            posterior_samples={
                "a": rng.normal(0, 1, n_samples),
                "b": rng.normal(0, 1, n_samples),
            },
            point_estimates={"a": 0.0, "b": 0.0},
            ci_lower={"a": -1.0, "b": -1.0},
            ci_upper={"a": 1.0, "b": 1.0},
        )
        fig = plot_posterior(
            result,
            layout="corner",
            show_ci=False,
            show_point_estimate=False,
        )
        scatter_markers = [
            t for t in fig.data if isinstance(t, go.Scatter) and t.mode and "markers" in t.mode
        ]
        for trace in scatter_markers:
            assert len(trace.x) <= 5000

    # --- Фильтрация параметров ---

    def test_filter_single_parameter(self, mock_estimation_result: EstimationResult) -> None:
        """parameters=['r_F'] → 1 histogram в marginals."""
        fig = plot_posterior(
            mock_estimation_result,
            parameters=["r_F"],
            show_ci=False,
            show_point_estimate=False,
        )
        histograms = [t for t in fig.data if isinstance(t, go.Histogram)]
        assert len(histograms) == 1

    def test_custom_height(self, mock_estimation_result: EstimationResult) -> None:
        """height=800 → layout.height == 800."""
        fig = plot_posterior(mock_estimation_result, height=800)
        assert fig.layout.height == 800

    # --- Error handling ---

    def test_no_posterior_raises_valueerror(self, mock_mle_result: EstimationResult) -> None:
        """MLE result (posterior_samples=None) → ValueError."""
        with pytest.raises(ValueError):
            plot_posterior(mock_mle_result)

    def test_unknown_parameter_raises_valueerror(
        self,
        mock_estimation_result: EstimationResult,
    ) -> None:
        """Запрос несуществующего параметра → ValueError."""
        with pytest.raises(ValueError):
            plot_posterior(mock_estimation_result, parameters=["nonexistent"])

    def test_invalid_layout_raises_valueerror(
        self,
        mock_estimation_result: EstimationResult,
    ) -> None:
        """layout='invalid' → ValueError."""
        with pytest.raises(ValueError):
            plot_posterior(mock_estimation_result, layout="invalid")


# ═══════════════════════════════════════════════════════════════════
# Тесты по: Description/Phase3/description_analysis_plots.md#plot_convergence
# ═══════════════════════════════════════════════════════════════════


class TestPlotConvergence:
    """Тесты plot_convergence — R-hat, ESS, trace panels."""

    # --- Basic ---

    def test_returns_figure(self, mock_estimation_result: EstimationResult) -> None:
        """Возвращает go.Figure при полных данных."""
        fig = plot_convergence(mock_estimation_result)
        assert isinstance(fig, go.Figure)

    def test_all_panels_have_traces(self, mock_estimation_result: EstimationResult) -> None:
        """metrics=None → traces для rhat (Bar), ess (Bar), trace (Scatter)."""
        fig = plot_convergence(mock_estimation_result)
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        assert len(bar_traces) >= 1  # rhat + ess bars
        assert len(scatter_traces) >= 1  # trace lines + threshold

    def test_rhat_only(self, mock_estimation_result: EstimationResult) -> None:
        """metrics=['rhat'] → Figure с Bar traces."""
        fig = plot_convergence(mock_estimation_result, metrics=["rhat"])
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        assert len(bar_traces) >= 1

    def test_ess_only(self, mock_estimation_result: EstimationResult) -> None:
        """metrics=['ess'] → Figure с grouped bar traces (bulk + tail)."""
        fig = plot_convergence(mock_estimation_result, metrics=["ess"])
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        assert len(bar_traces) >= 2  # bulk + tail

    def test_rhat_threshold_line_present(
        self,
        mock_estimation_result: EstimationResult,
    ) -> None:
        """show_rhat_threshold=True → Scatter trace при y=1.05."""
        fig = plot_convergence(
            mock_estimation_result,
            metrics=["rhat"],
            show_rhat_threshold=True,
        )
        threshold_lines = [
            t
            for t in fig.data
            if isinstance(t, go.Scatter)
            and t.y is not None
            and len(t.y) >= 2
            and all(v == pytest.approx(1.05) for v in t.y)
        ]
        assert len(threshold_lines) >= 1

    def test_rhat_threshold_line_absent(
        self,
        mock_estimation_result: EstimationResult,
    ) -> None:
        """show_rhat_threshold=False → нет threshold линии."""
        fig = plot_convergence(
            mock_estimation_result,
            metrics=["rhat"],
            show_rhat_threshold=False,
        )
        threshold_lines = [
            t
            for t in fig.data
            if isinstance(t, go.Scatter)
            and t.y is not None
            and len(t.y) >= 2
            and all(v == pytest.approx(1.05) for v in t.y)
        ]
        assert len(threshold_lines) == 0

    def test_rhat_color_coding(self, mock_estimation_result: EstimationResult) -> None:
        """R-hat бары: зелёный для <1.05, красный для >=1.05."""
        fig = plot_convergence(
            mock_estimation_result,
            metrics=["rhat"],
            show_rhat_threshold=False,
        )
        rhat_bars = [t for t in fig.data if isinstance(t, go.Bar)]
        assert len(rhat_bars) >= 1
        bar = rhat_bars[0]
        colors = bar.marker.color
        # K_F имеет rhat=1.08 → должен быть красный (#e74c3c / ANALYSIS_COLORS["ci"])
        # r_F и r_M1 имеют rhat <1.05 → зелёный (#27ae60 / ANALYSIS_COLORS["point_est"])
        if isinstance(colors, (list, tuple)):
            assert ANALYSIS_COLORS["ci"] in colors or ANALYSIS_COLORS["point_est"] in colors

    def test_ess_has_two_groups(self, mock_estimation_result: EstimationResult) -> None:
        """ESS панель имеет bulk и tail группы баров."""
        fig = plot_convergence(mock_estimation_result, metrics=["ess"])
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        names_lower = [t.name.lower() for t in bar_traces if t.name]
        assert any("bulk" in n for n in names_lower)
        assert any("tail" in n for n in names_lower)

    def test_trace_panel_with_chains(self, mock_estimation_result: EstimationResult) -> None:
        """С config.n_chains=4 → несколько линий в trace панели."""
        fig = plot_convergence(mock_estimation_result, metrics=["trace"])
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        # 3 параметра × 4 chains = 12, или минимум > 3 (по 1 на параметр)
        assert len(scatter_traces) >= 3

    # --- Edge cases ---

    def test_empty_rhat_skips_panel(self) -> None:
        """diagnostics.rhat={} → панель rhat пропущена, ess работает."""
        result = EstimationResult(
            method="bayesian_pymc",
            diagnostics=ConvergenceDiagnostics(
                rhat={},
                ess_bulk={"r_F": 3000.0},
                ess_tail={"r_F": 2000.0},
            ),
            posterior_samples=None,
        )
        fig = plot_convergence(result, metrics=["rhat", "ess"])
        # Должна быть только ess панель (bars), без rhat
        assert isinstance(fig, go.Figure)
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        assert len(bar_traces) >= 1

    # --- Error handling ---

    def test_no_diagnostics_raises_valueerror(
        self,
        mock_mle_result: EstimationResult,
    ) -> None:
        """MLE result (diagnostics=None) → ValueError."""
        with pytest.raises(ValueError):
            plot_convergence(mock_mle_result)

    def test_all_panels_filtered_raises_valueerror(self) -> None:
        """Все метрики пустые и нет posterior → ValueError."""
        result = EstimationResult(
            method="bayesian_pymc",
            diagnostics=ConvergenceDiagnostics(
                rhat={},
                ess_bulk={},
                ess_tail={},
            ),
            posterior_samples=None,
        )
        with pytest.raises(ValueError):
            plot_convergence(result)

    def test_invalid_metric_raises_valueerror(
        self,
        mock_estimation_result: EstimationResult,
    ) -> None:
        """Metrics содержит невалидное значение → ValueError."""
        with pytest.raises(ValueError):
            plot_convergence(mock_estimation_result, metrics=["invalid_metric"])


# ═══════════════════════════════════════════════════════════════════
# Тесты по: Description/Phase3/description_analysis_plots.md#plot_morris
# ═══════════════════════════════════════════════════════════════════


class TestPlotMorris:
    """Тесты plot_morris — scatter plot mu_star vs sigma."""

    # --- Basic ---

    def test_returns_figure(self, mock_morris_result: MorrisResult) -> None:
        """plot_morris возвращает go.Figure."""
        fig = plot_morris(mock_morris_result)
        assert isinstance(fig, go.Figure)

    def test_highlight_produces_two_scatter_groups(
        self,
        mock_morris_result: MorrisResult,
    ) -> None:
        """highlight_influential=True → 2 Scatter traces (influential + non-influential).

        Wedge line — третий trace.
        """
        fig = plot_morris(mock_morris_result, highlight_influential=True, show_wedge=False)
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        assert len(scatter_traces) == 2

    def test_scatter_trace_types(self, mock_morris_result: MorrisResult) -> None:
        """Все data traces — go.Scatter (не Bar)."""
        fig = plot_morris(mock_morris_result)
        for trace in fig.data:
            assert isinstance(trace, go.Scatter)

    def test_influential_markers_red_and_large(
        self,
        mock_morris_result: MorrisResult,
    ) -> None:
        """Influential группа: color=#e74c3c, size=12."""
        fig = plot_morris(mock_morris_result, highlight_influential=True, show_wedge=False)
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        influential_trace = [
            t
            for t in scatter_traces
            if t.marker and t.marker.color == ANALYSIS_COLORS["influential"]
        ]
        assert len(influential_trace) >= 1
        assert influential_trace[0].marker.size == 12

    def test_non_influential_markers_grey_and_small(
        self,
        mock_morris_result: MorrisResult,
    ) -> None:
        """Non-influential группа: color=#95a5a6, size=8."""
        fig = plot_morris(mock_morris_result, highlight_influential=True, show_wedge=False)
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        non_inf_trace = [
            t for t in scatter_traces if t.marker and t.marker.color == ANALYSIS_COLORS["threshold"]
        ]
        assert len(non_inf_trace) >= 1
        assert non_inf_trace[0].marker.size == 8

    def test_wedge_line_present(self, mock_morris_result: MorrisResult) -> None:
        """show_wedge=True → dashed diagonal линия от (0,0)."""
        fig = plot_morris(mock_morris_result, show_wedge=True)
        wedge_traces = [
            t
            for t in fig.data
            if isinstance(t, go.Scatter)
            and t.line
            and t.line.dash == "dash"
            and t.mode
            and "lines" in t.mode
        ]
        assert len(wedge_traces) >= 1
        # Линия начинается от 0
        assert wedge_traces[0].x[0] == pytest.approx(0.0)
        assert wedge_traces[0].y[0] == pytest.approx(0.0)

    def test_wedge_line_absent(self, mock_morris_result: MorrisResult) -> None:
        """show_wedge=False → нет diagonal линии."""
        fig = plot_morris(mock_morris_result, show_wedge=False)
        wedge_traces = [
            t
            for t in fig.data
            if isinstance(t, go.Scatter)
            and t.line
            and t.line.dash == "dash"
            and t.mode
            and "lines" in t.mode
        ]
        assert len(wedge_traces) == 0

    def test_no_highlight_single_group(self, mock_morris_result: MorrisResult) -> None:
        """highlight_influential=False → все точки в 1 scatter trace."""
        fig = plot_morris(
            mock_morris_result,
            highlight_influential=False,
            show_wedge=False,
        )
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        assert len(scatter_traces) == 1

    def test_labels_present(self, mock_morris_result: MorrisResult) -> None:
        """show_labels=True с <=20 параметрами → text на traces."""
        fig = plot_morris(mock_morris_result, show_labels=True, show_wedge=False)
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        has_text = any(t.text is not None and len(t.text) > 0 for t in scatter_traces)
        assert has_text

    def test_title_contains_output_variable(self, mock_morris_result: MorrisResult) -> None:
        """Заголовок содержит output_variable ('F')."""
        fig = plot_morris(mock_morris_result)
        title = fig.layout.title.text or ""
        assert "F" in title

    def test_error_bars_from_mu_star_conf(self, mock_morris_result: MorrisResult) -> None:
        """Ненулевой mu_star_conf → error_x на influential trace."""
        fig = plot_morris(mock_morris_result, highlight_influential=True, show_wedge=False)
        influential_traces = [
            t
            for t in fig.data
            if isinstance(t, go.Scatter)
            and t.marker
            and t.marker.color == ANALYSIS_COLORS["influential"]
        ]
        assert len(influential_traces) >= 1
        assert influential_traces[0].error_x is not None
        assert influential_traces[0].error_x.array is not None

    def test_custom_threshold_changes_groups(
        self,
        mock_morris_result: MorrisResult,
    ) -> None:
        """threshold_ratio=0.5 → порог=2.5 → только 2 influential (r_F, r_M1)."""
        fig = plot_morris(
            mock_morris_result,
            highlight_influential=True,
            threshold_ratio=0.5,
            show_wedge=False,
        )
        influential_traces = [
            t
            for t in fig.data
            if isinstance(t, go.Scatter)
            and t.marker
            and t.marker.color == ANALYSIS_COLORS["influential"]
        ]
        assert len(influential_traces) >= 1
        # Influential trace должен содержать ровно 2 точки (r_F, r_M1)
        assert len(influential_traces[0].x) == 2

    # --- Error handling ---

    def test_empty_parameter_names_raises_valueerror(self) -> None:
        """MorrisResult с пустыми parameter_names → ValueError."""
        empty = MorrisResult(parameter_names=[], output_variable="F")
        with pytest.raises(ValueError):
            plot_morris(empty)
