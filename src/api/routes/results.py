"""Endpoints для получения результатов и экспорта."""

from __future__ import annotations

import io
import uuid as _uuid_mod
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from src.api.models.schemas import (
    ExportFormat,
    ExportRequest,
    ResultsResponse,
    SimulationMode,
)
from src.api.services.simulation_service import SimulationService
from src.db.models import SimulationRecord
from src.db.session import get_db

router = APIRouter(prefix="/api/v1", tags=["results"])


def _validate_uuid(value: str) -> str:
    try:
        _uuid_mod.UUID(value)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail=f"Invalid ID format: {value}")
    return value


@router.get("/results/{simulation_id}", response_model=ResultsResponse)
async def get_results(
    simulation_id: str,
    db: Session = Depends(get_db),
) -> ResultsResponse:
    """Получить результаты завершённой симуляции."""
    _validate_uuid(simulation_id)
    record = db.get(SimulationRecord, simulation_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Simulation {simulation_id} not found")
    if record.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Simulation is {record.status}, not completed",
        )

    try:
        data = SimulationService.load_trajectory(simulation_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Result files not found")

    return ResultsResponse(
        simulation_id=simulation_id,
        mode=SimulationMode(record.mode),
        times=data["times"],
        variables=data["variables"],
    )


@router.post("/export/{simulation_id}")
async def export_results(
    simulation_id: str,
    request: ExportRequest,
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """Экспорт результатов симуляции в выбранном формате."""
    _validate_uuid(simulation_id)
    record = db.get(SimulationRecord, simulation_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Simulation {simulation_id} not found")
    if record.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Simulation is {record.status}, not completed",
        )

    try:
        data = SimulationService.load_trajectory(simulation_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Result files not found")

    # Построить траекторию из сохранённых данных для визуализации
    from src.core.extended_sde import ExtendedSDEState, ExtendedSDETrajectory, VARIABLE_NAMES
    from src.core.parameters import ParameterSet
    import numpy as np

    times = np.array(data["times"])
    states = []
    for i in range(len(times)):
        kwargs = {}
        for var_name in VARIABLE_NAMES:
            if var_name in data["variables"]:
                kwargs[var_name] = data["variables"][var_name][i]
            else:
                kwargs[var_name] = 0.0
        kwargs["t"] = data["times"][i]
        states.append(ExtendedSDEState(**kwargs))

    trajectory = ExtendedSDETrajectory(
        times=times,
        states=states,
        params=ParameterSet(),
    )

    # Экспорт через ReportExporter
    from src.visualization.export import ExportConfig, ReportExporter
    from src.visualization.plots import (
        plot_cytokines,
        plot_ecm,
        plot_phases,
        plot_populations,
    )

    with TemporaryDirectory() as tmpdir:
        config = ExportConfig(output_dir=Path(tmpdir), width=1000, height=600)
        exporter = ReportExporter(config)
        exporter.add_trajectory_data("simulation", trajectory)

        if request.format == ExportFormat.CSV:
            paths = exporter.to_csv()
            if not paths:
                raise HTTPException(status_code=500, detail="CSV generation failed")
            content = paths[0].read_text(encoding="utf-8")
            return StreamingResponse(
                io.BytesIO(content.encode("utf-8")),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={simulation_id}.csv"},
            )

        # Для графических форматов — добавить фигуры
        if request.include_populations:
            exporter.add_figure("populations", plot_populations(trajectory))
        if request.include_cytokines:
            exporter.add_figure("cytokines", plot_cytokines(trajectory))
        if request.include_ecm:
            exporter.add_figure("ecm", plot_ecm(trajectory))
        if request.include_phases:
            exporter.add_figure("phases", plot_phases(trajectory))

        if request.format == ExportFormat.PNG:
            paths = exporter.to_png()
            if not paths:
                raise HTTPException(status_code=500, detail="PNG generation failed")
            content_bytes = paths[0].read_bytes()
            return StreamingResponse(
                io.BytesIO(content_bytes),
                media_type="image/png",
                headers={"Content-Disposition": f"attachment; filename={simulation_id}.png"},
            )

        if request.format == ExportFormat.SVG:
            paths = exporter.to_svg()
            if not paths:
                raise HTTPException(status_code=500, detail="SVG generation failed")
            content_str = paths[0].read_text(encoding="utf-8")
            return StreamingResponse(
                io.BytesIO(content_str.encode("utf-8")),
                media_type="image/svg+xml",
                headers={"Content-Disposition": f"attachment; filename={simulation_id}.svg"},
            )

        if request.format == ExportFormat.PDF:
            pdf_path = exporter.to_pdf()
            content_bytes = pdf_path.read_bytes()
            return StreamingResponse(
                io.BytesIO(content_bytes),
                media_type="application/pdf",
                headers={"Content-Disposition": f"attachment; filename={simulation_id}.pdf"},
            )

        # ExportFormat enum уже валидирован Pydantic, этот код — fallback-защита
        raise HTTPException(status_code=400, detail=f"Unsupported format: {request.format}")
