"""TDD тесты для src/visualization/spatial.py — пространственная визуализация."""

from pathlib import Path

import plotly.graph_objects as go
import pytest

from src.core.abm_model import ABMSnapshot, ABMTrajectory
from src.visualization.spatial import (
    animate_evolution,
    field_heatmap,
    heatmap_density,
    inflammation_map,
    scatter_agents,
)


class TestHeatmapDensity:
    """Тесты heatmap_density — 2D гистограмма агентов."""

    def test_returns_figure(self, mock_abm_snapshot: ABMSnapshot) -> None:
        fig = heatmap_density(mock_abm_snapshot)
        assert isinstance(fig, go.Figure)

    def test_has_heatmap_trace(self, mock_abm_snapshot: ABMSnapshot) -> None:
        fig = heatmap_density(mock_abm_snapshot)
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Heatmap)

    def test_filter_agent_types(self, mock_abm_snapshot: ABMSnapshot) -> None:
        fig = heatmap_density(mock_abm_snapshot, agent_types=["stem"])
        assert isinstance(fig, go.Figure)

    def test_custom_bin_size(self, mock_abm_snapshot: ABMSnapshot) -> None:
        fig = heatmap_density(mock_abm_snapshot, bin_size=20.0)
        assert isinstance(fig, go.Figure)

    def test_custom_height(self, mock_abm_snapshot: ABMSnapshot) -> None:
        fig = heatmap_density(mock_abm_snapshot, height=600)
        assert fig.layout.height == 600


class TestScatterAgents:
    """Тесты scatter_agents — scatter plot агентов."""

    def test_returns_figure(self, mock_abm_snapshot: ABMSnapshot) -> None:
        fig = scatter_agents(mock_abm_snapshot)
        assert isinstance(fig, go.Figure)

    def test_color_by_type_has_traces_per_type(self, mock_abm_snapshot: ABMSnapshot) -> None:
        fig = scatter_agents(mock_abm_snapshot, color_by="type")
        # У нас 3 типа в mock: stem, macro, fibro
        assert len(fig.data) >= 3

    def test_color_by_energy(self, mock_abm_snapshot: ABMSnapshot) -> None:
        fig = scatter_agents(mock_abm_snapshot, color_by="energy")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1  # Один scatter с colorscale

    def test_color_by_age(self, mock_abm_snapshot: ABMSnapshot) -> None:
        fig = scatter_agents(mock_abm_snapshot, color_by="age")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1


class TestInflammationMap:
    """Тесты inflammation_map — карта воспаления."""

    def test_returns_figure(self, mock_abm_snapshot: ABMSnapshot) -> None:
        fig = inflammation_map(mock_abm_snapshot)
        assert isinstance(fig, go.Figure)

    def test_has_heatmap_trace(self, mock_abm_snapshot: ABMSnapshot) -> None:
        fig = inflammation_map(mock_abm_snapshot)
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Heatmap)

    def test_colorscale_is_diverging(self, mock_abm_snapshot: ABMSnapshot) -> None:
        fig = inflammation_map(mock_abm_snapshot)
        hm = fig.data[0]
        assert hm.colorscale is not None


class TestFieldHeatmap:
    """Тесты field_heatmap — generic heatmap для полей."""

    def test_cytokine_field(self, mock_abm_snapshot: ABMSnapshot) -> None:
        fig = field_heatmap(mock_abm_snapshot, field="cytokine")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1

    def test_ecm_field(self, mock_abm_snapshot: ABMSnapshot) -> None:
        fig = field_heatmap(mock_abm_snapshot, field="ecm")
        assert isinstance(fig, go.Figure)

    def test_custom_colorscale(self, mock_abm_snapshot: ABMSnapshot) -> None:
        fig = field_heatmap(mock_abm_snapshot, field="cytokine", colorscale="Viridis")
        assert isinstance(fig, go.Figure)

    def test_invalid_field_raises(self, mock_abm_snapshot: ABMSnapshot) -> None:
        with pytest.raises(ValueError, match="Неизвестное поле"):
            field_heatmap(mock_abm_snapshot, field="invalid")


class TestAnimateEvolution:
    """Тесты animate_evolution — анимация ABM."""

    def test_returns_plotly_figure(self, mock_abm_trajectory: ABMTrajectory) -> None:
        fig = animate_evolution(mock_abm_trajectory)
        assert isinstance(fig, go.Figure)

    def test_has_frames(self, mock_abm_trajectory: ABMTrajectory) -> None:
        fig = animate_evolution(mock_abm_trajectory)
        assert len(fig.frames) == len(mock_abm_trajectory.snapshots)

    def test_has_play_button(self, mock_abm_trajectory: ABMTrajectory) -> None:
        fig = animate_evolution(mock_abm_trajectory)
        assert fig.layout.updatemenus is not None
        assert len(fig.layout.updatemenus) > 0

    def test_has_slider(self, mock_abm_trajectory: ABMTrajectory) -> None:
        fig = animate_evolution(mock_abm_trajectory)
        assert fig.layout.sliders is not None
        assert len(fig.layout.sliders) > 0

    def test_save_to_gif(self, mock_abm_trajectory: ABMTrajectory, tmp_path) -> None:
        output = tmp_path / "animation.gif"
        result = animate_evolution(mock_abm_trajectory, output_path=output, fps=2, dpi=50)
        assert isinstance(result, Path)
        assert result.exists()
        assert result.suffix == ".gif"
