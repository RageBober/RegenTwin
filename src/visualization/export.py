"""Экспорт визуализаций: PNG/SVG, CSV, PDF.

ReportExporter собирает Plotly-фигуры и данные траекторий,
экспортирует их в различные форматы.

Подробное описание: Description/Phase4/description_visualization.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from src.core.extended_sde import ExtendedSDETrajectory, VARIABLE_NAMES


@dataclass
class ExportConfig:
    """Конфигурация экспорта."""

    output_dir: Path = field(default_factory=lambda: Path("output"))
    image_format: str = "png"  # "png" | "svg" | "pdf"
    dpi: int = 150
    width: int = 1200
    height: int = 800
    csv_separator: str = ","


class ReportExporter:
    """Экспорт результатов симуляции в PNG, CSV, PDF.

    Usage:
        exporter = ReportExporter()
        exporter.add_figure("populations", fig_pop)
        exporter.add_trajectory_data("run_1", trajectory)
        paths = exporter.to_png()
        csv_paths = exporter.to_csv()
        pdf_path = exporter.to_pdf()
    """

    def __init__(self, config: ExportConfig | None = None) -> None:
        self._config = config or ExportConfig()
        self._figures: dict[str, go.Figure] = {}
        self._trajectories: dict[str, ExtendedSDETrajectory] = {}
        self._metadata: dict[str, str] = {}

    def add_figure(self, name: str, fig: go.Figure) -> None:
        """Зарегистрировать Plotly-фигуру для экспорта."""
        self._figures[name] = fig

    def add_trajectory_data(
        self,
        name: str,
        trajectory: ExtendedSDETrajectory,
    ) -> None:
        """Зарегистрировать данные траектории для CSV-экспорта."""
        self._trajectories[name] = trajectory

    def add_metadata(self, key: str, value: str) -> None:
        """Добавить метаданные для PDF-заголовка."""
        self._metadata[key] = value

    @property
    def figure_count(self) -> int:
        """Количество зарегистрированных фигур."""
        return len(self._figures)

    @property
    def trajectory_count(self) -> int:
        """Количество зарегистрированных траекторий."""
        return len(self._trajectories)

    def to_png(self, output_dir: Path | None = None) -> list[Path]:
        """Экспорт всех фигур в PNG.

        Requires: kaleido package.

        Returns:
            Список путей к сохранённым PNG файлам.
        """
        return self._export_images(output_dir, "png")

    def to_svg(self, output_dir: Path | None = None) -> list[Path]:
        """Экспорт всех фигур в SVG.

        Returns:
            Список путей к SVG файлам.
        """
        return self._export_images(output_dir, "svg")

    def _export_images(
        self,
        output_dir: Path | None,
        fmt: str,
    ) -> list[Path]:
        """Общий метод экспорта изображений."""
        out = Path(output_dir) if output_dir else self._config.output_dir
        out.mkdir(parents=True, exist_ok=True)

        paths: list[Path] = []
        for name, fig in self._figures.items():
            filepath = out / f"{name}.{fmt}"
            fig.write_image(
                str(filepath),
                width=self._config.width,
                height=self._config.height,
                scale=self._config.dpi / 72,
            )
            paths.append(filepath)

        return paths

    def to_csv(self, output_dir: Path | None = None) -> list[Path]:
        """Экспорт данных траекторий в CSV.

        Колонки: time, P, Ne, M1, M2, F, Mf, E, S,
                 C_TNF, C_IL10, ..., D, O2

        Returns:
            Список путей к CSV файлам.
        """
        out = Path(output_dir) if output_dir else self._config.output_dir
        out.mkdir(parents=True, exist_ok=True)

        paths: list[Path] = []
        for name, trajectory in self._trajectories.items():
            filepath = out / f"{name}.csv"
            self._write_trajectory_csv(filepath, trajectory)
            paths.append(filepath)

        return paths

    def _write_trajectory_csv(
        self,
        filepath: Path,
        trajectory: ExtendedSDETrajectory,
    ) -> None:
        """Запись одной траектории в CSV."""
        sep = self._config.csv_separator
        times = trajectory.times

        # Заголовок
        header = sep.join(["time"] + VARIABLE_NAMES)

        # Данные
        n_steps = len(times)
        rows: list[str] = [header]

        for i in range(n_steps):
            state = trajectory.states[i]
            values = [f"{times[i]:.4f}"]
            for var_name in VARIABLE_NAMES:
                val = getattr(state, var_name)
                values.append(f"{val:.6f}")
            rows.append(sep.join(values))

        filepath.write_text("\n".join(rows), encoding="utf-8")

    def to_pdf(self, output_path: Path | None = None) -> Path:
        """Генерация PDF-отчёта: титул → метаданные → графики → сводка.

        Requires: fpdf2 package.

        Returns:
            Путь к PDF файлу.
        """
        from fpdf import FPDF

        out_dir = self._config.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = Path(output_path) if output_path else out_dir / "report.pdf"

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        # Титульная страница
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 24)
        pdf.cell(0, 40, "RegenTwin Report", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.set_font("Helvetica", "", 12)
        pdf.cell(0, 10, "Tissue Regeneration Simulation Results", new_x="LMARGIN", new_y="NEXT", align="C")

        # Метаданные
        if self._metadata:
            pdf.ln(10)
            pdf.set_font("Helvetica", "B", 14)
            pdf.cell(0, 10, "Parameters", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 10)
            for key, value in self._metadata.items():
                pdf.cell(0, 7, f"{key}: {value}", new_x="LMARGIN", new_y="NEXT")

        # Графики (каждый на новой странице)
        if self._figures:
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                for name, fig in self._figures.items():
                    img_path = Path(tmpdir) / f"{name}.png"
                    fig.write_image(
                        str(img_path),
                        width=self._config.width,
                        height=self._config.height,
                        scale=2,
                    )

                    pdf.add_page()
                    pdf.set_font("Helvetica", "B", 14)
                    pdf.cell(0, 10, name.replace("_", " ").title(), new_x="LMARGIN", new_y="NEXT")
                    pdf.image(str(img_path), x=10, w=190)

        # Сводка траекторий
        if self._trajectories:
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 14)
            pdf.cell(0, 10, "Data Summary", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 9)

            for name, trajectory in self._trajectories.items():
                pdf.ln(5)
                pdf.set_font("Helvetica", "B", 11)
                pdf.cell(0, 8, f"Trajectory: {name}", new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", "", 9)

                stats = trajectory.get_statistics()
                pdf.cell(0, 6, f"Time points: {len(trajectory.times)}", new_x="LMARGIN", new_y="NEXT")
                pdf.cell(0, 6, f"Duration: {trajectory.times[-1]:.1f} h", new_x="LMARGIN", new_y="NEXT")

                # Таблица ключевых переменных
                for var in ["F", "M2", "rho_collagen", "C_TNF", "C_IL10"]:
                    if var in stats:
                        s = stats[var]
                        pdf.cell(
                            0, 5,
                            f"  {var}: final={s['final']:.2f}, "
                            f"mean={s['mean']:.2f}, max={s['max']:.2f}",
                            new_x="LMARGIN", new_y="NEXT",
                        )

        pdf.output(str(pdf_path))
        return pdf_path
