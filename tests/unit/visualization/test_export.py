"""TDD тесты для src/visualization/export.py — экспорт результатов."""

from pathlib import Path

import plotly.graph_objects as go
import pytest

from src.core.extended_sde import ExtendedSDETrajectory, VARIABLE_NAMES
from src.visualization.export import ExportConfig, ReportExporter


class TestReportExporter:
    """Тесты базовых операций ReportExporter."""

    def test_create_default(self) -> None:
        exporter = ReportExporter()
        assert exporter.figure_count == 0
        assert exporter.trajectory_count == 0

    def test_create_with_config(self, tmp_path: Path) -> None:
        config = ExportConfig(output_dir=tmp_path)
        exporter = ReportExporter(config)
        assert exporter.figure_count == 0

    def test_add_figure(self) -> None:
        exporter = ReportExporter()
        fig = go.Figure(data=go.Scatter(x=[1, 2], y=[3, 4]))
        exporter.add_figure("test", fig)
        assert exporter.figure_count == 1

    def test_add_trajectory(self, mock_extended_trajectory: ExtendedSDETrajectory) -> None:
        exporter = ReportExporter()
        exporter.add_trajectory_data("run_1", mock_extended_trajectory)
        assert exporter.trajectory_count == 1

    def test_add_metadata(self) -> None:
        exporter = ReportExporter()
        exporter.add_metadata("therapy", "PRP")
        exporter.add_metadata("duration", "30 days")
        # Метаданные хранятся внутри
        assert exporter._metadata["therapy"] == "PRP"


class TestToPng:
    """Тесты to_png — экспорт фигур в PNG."""

    def test_creates_files(self, tmp_path: Path) -> None:
        config = ExportConfig(output_dir=tmp_path, width=400, height=300)
        exporter = ReportExporter(config)
        exporter.add_figure("test_fig", go.Figure(data=go.Scatter(x=[1], y=[2])))

        paths = exporter.to_png()
        assert len(paths) == 1
        assert paths[0].exists()
        assert paths[0].suffix == ".png"

    def test_returns_paths(self, tmp_path: Path) -> None:
        config = ExportConfig(output_dir=tmp_path, width=400, height=300)
        exporter = ReportExporter(config)
        exporter.add_figure("fig1", go.Figure())
        exporter.add_figure("fig2", go.Figure())

        paths = exporter.to_png()
        assert len(paths) == 2

    def test_custom_output_dir(self, tmp_path: Path) -> None:
        exporter = ReportExporter()
        exporter.add_figure("fig", go.Figure(data=go.Scatter(x=[1], y=[1])))

        custom_dir = tmp_path / "custom"
        paths = exporter.to_png(output_dir=custom_dir)
        assert paths[0].parent == custom_dir

    def test_empty_exporter_returns_empty(self, tmp_path: Path) -> None:
        config = ExportConfig(output_dir=tmp_path)
        exporter = ReportExporter(config)
        paths = exporter.to_png()
        assert paths == []


class TestToSvg:
    """Тесты to_svg — экспорт фигур в SVG."""

    def test_creates_svg(self, tmp_path: Path) -> None:
        config = ExportConfig(output_dir=tmp_path, width=400, height=300)
        exporter = ReportExporter(config)
        exporter.add_figure("test", go.Figure(data=go.Scatter(x=[1], y=[2])))

        paths = exporter.to_svg()
        assert len(paths) == 1
        assert paths[0].suffix == ".svg"
        assert paths[0].exists()


class TestToCsv:
    """Тесты to_csv — экспорт данных в CSV."""

    def test_creates_csv(
        self,
        tmp_path: Path,
        mock_extended_trajectory: ExtendedSDETrajectory,
    ) -> None:
        config = ExportConfig(output_dir=tmp_path)
        exporter = ReportExporter(config)
        exporter.add_trajectory_data("run_1", mock_extended_trajectory)

        paths = exporter.to_csv()
        assert len(paths) == 1
        assert paths[0].exists()
        assert paths[0].suffix == ".csv"

    def test_csv_has_correct_columns(
        self,
        tmp_path: Path,
        mock_extended_trajectory: ExtendedSDETrajectory,
    ) -> None:
        config = ExportConfig(output_dir=tmp_path)
        exporter = ReportExporter(config)
        exporter.add_trajectory_data("run_1", mock_extended_trajectory)

        paths = exporter.to_csv()
        content = paths[0].read_text(encoding="utf-8")
        header = content.split("\n")[0]
        columns = header.split(",")

        # time + 20 переменных = 21 колонка
        assert len(columns) == 21
        assert columns[0] == "time"
        for var in VARIABLE_NAMES:
            assert var in columns

    def test_csv_row_count(
        self,
        tmp_path: Path,
        mock_extended_trajectory: ExtendedSDETrajectory,
    ) -> None:
        config = ExportConfig(output_dir=tmp_path)
        exporter = ReportExporter(config)
        exporter.add_trajectory_data("run_1", mock_extended_trajectory)

        paths = exporter.to_csv()
        content = paths[0].read_text(encoding="utf-8")
        lines = [l for l in content.split("\n") if l.strip()]

        n_expected = len(mock_extended_trajectory.times) + 1  # header + data
        assert len(lines) == n_expected

    def test_empty_returns_empty(self, tmp_path: Path) -> None:
        config = ExportConfig(output_dir=tmp_path)
        exporter = ReportExporter(config)
        paths = exporter.to_csv()
        assert paths == []


class TestToPdf:
    """Тесты to_pdf — генерация PDF-отчёта."""

    def test_creates_pdf(self, tmp_path: Path) -> None:
        config = ExportConfig(output_dir=tmp_path)
        exporter = ReportExporter(config)
        exporter.add_metadata("test", "value")

        path = exporter.to_pdf()
        assert path.exists()
        assert path.suffix == ".pdf"

    def test_pdf_with_figures(self, tmp_path: Path) -> None:
        config = ExportConfig(output_dir=tmp_path, width=400, height=300)
        exporter = ReportExporter(config)
        exporter.add_figure("fig1", go.Figure(data=go.Scatter(x=[1], y=[2])))
        exporter.add_metadata("therapy", "PRP")

        path = exporter.to_pdf()
        assert path.exists()
        # PDF не пустой
        assert path.stat().st_size > 100

    def test_pdf_with_trajectory(
        self,
        tmp_path: Path,
        mock_extended_trajectory: ExtendedSDETrajectory,
    ) -> None:
        config = ExportConfig(output_dir=tmp_path, width=400, height=300)
        exporter = ReportExporter(config)
        exporter.add_trajectory_data("run_1", mock_extended_trajectory)

        path = exporter.to_pdf()
        assert path.exists()

    def test_custom_output_path(self, tmp_path: Path) -> None:
        exporter = ReportExporter()
        custom_path = tmp_path / "custom_report.pdf"
        path = exporter.to_pdf(output_path=custom_path)
        assert path == custom_path
        assert path.exists()
