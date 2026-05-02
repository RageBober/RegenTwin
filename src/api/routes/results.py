"""Endpoints for retrieving and exporting simulation results."""

from __future__ import annotations

import io
import uuid as uuid_mod
from pathlib import Path
from tempfile import TemporaryDirectory

_MODE_LABELS: dict[str, str] = {
    "extended": "SDE",
    "integrated": "SDE+ABM",
    "abm": "ABM",
    "mvp": "MVP",
}


def _export_filename(mode: str, is_extended_mc: bool, created_at: object, ext: str) -> str:
    """Build a human-readable export filename."""
    label = "Monte-Carlo" if is_extended_mc else _MODE_LABELS.get(mode, mode)
    ts = ""
    try:
        ts = "_" + created_at.strftime("%Y-%m-%d_%H-%M")  # type: ignore[union-attr]
    except Exception:
        pass
    return f"regentwin_{label}{ts}.{ext}"


from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from src.api.models.schemas import ExportFormat, ExportRequest, ResultsResponse, SimulationMode
from src.api.services.result_bundle import build_extended_trajectory, build_mc_mean_trajectory
from src.api.services.simulation_service import SimulationService
from src.db.models import SimulationRecord
from src.db.session import get_db

router = APIRouter(prefix="/api/v1", tags=["results"])


def _validate_uuid(value: str) -> str:
    try:
        uuid_mod.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid ID format: {value}") from exc
    return value


def _export_csv_content(data: dict) -> str:
    variables = data["variables"]
    headers = ["time", *variables.keys()]
    rows = [",".join(headers)]
    for idx, t in enumerate(data["times"]):
        values = [f"{float(t):.6f}"]
        for key in variables:
            values.append(f"{float(variables[key][idx]):.6f}")
        rows.append(",".join(values))
    return "\n".join(rows)


@router.get("/results/{simulation_id}", response_model=ResultsResponse)
async def get_results(
    simulation_id: str,
    db: Session = Depends(get_db),
) -> ResultsResponse:
    """Return results for a completed simulation."""
    _validate_uuid(simulation_id)
    record = db.get(SimulationRecord, simulation_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Simulation {simulation_id} not found")
    if record.status != "completed":
        raise HTTPException(status_code=400, detail=f"Simulation is {record.status}, not completed")

    try:
        data = SimulationService.load_trajectory(simulation_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Result files not found") from exc

    result_mode = data.get("mode", record.mode)
    return ResultsResponse(
        simulation_id=simulation_id,
        mode=SimulationMode(result_mode),
        times=data["times"],
        variables=data["variables"],
        metadata=data.get("metadata", {}),
    )


@router.post("/export/{simulation_id}")
async def export_results(
    simulation_id: str,
    request: ExportRequest,
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """Export a completed simulation result."""
    _validate_uuid(simulation_id)
    record = db.get(SimulationRecord, simulation_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Simulation {simulation_id} not found")
    if record.status != "completed":
        raise HTTPException(status_code=400, detail=f"Simulation is {record.status}, not completed")

    try:
        data = SimulationService.load_trajectory(simulation_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Result files not found") from exc

    result_mode = data.get("mode", record.mode)
    metadata = data.get("metadata", {})
    is_mc = metadata.get("n_trajectories", 1) > 1
    is_extended_mc = is_mc and metadata.get("extended_mc", False)

    if request.format == ExportFormat.CSV:
        content = _export_csv_content(data)
        fname = _export_filename(result_mode, is_extended_mc, record.created_at, "csv")
        return StreamingResponse(
            io.BytesIO(content.encode("utf-8")),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{fname}"'},
        )

    if is_mc and not is_extended_mc:
        raise HTTPException(
            status_code=400,
            detail=f"{request.format.value.upper()} export is not available for MVP Monte Carlo results (only CSV)",
        )

    if not is_mc and result_mode not in {
        SimulationMode.EXTENDED.value,
        SimulationMode.INTEGRATED.value,
    }:
        raise HTTPException(
            status_code=501,
            detail=f"{request.format.value.upper()} export is not available for mode {result_mode}",
        )

    trajectory = (
        build_mc_mean_trajectory(data)
        if is_extended_mc
        else build_extended_trajectory({**data, "mode": result_mode})
    )

    from src.visualization.export import ExportConfig, ReportExporter
    from src.visualization.plots import plot_cytokines, plot_ecm, plot_phases, plot_populations

    with TemporaryDirectory() as tmpdir:
        config = ExportConfig(output_dir=Path(tmpdir), width=1000, height=600)
        exporter = ReportExporter(config)
        exporter.add_trajectory_data("simulation", trajectory)

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
            fname = _export_filename(result_mode, is_extended_mc, record.created_at, "png")
            return StreamingResponse(
                io.BytesIO(content_bytes),
                media_type="image/png",
                headers={"Content-Disposition": f'attachment; filename="{fname}"'},
            )

        if request.format == ExportFormat.SVG:
            paths = exporter.to_svg()
            if not paths:
                raise HTTPException(status_code=500, detail="SVG generation failed")
            content_str = paths[0].read_text(encoding="utf-8")
            fname = _export_filename(result_mode, is_extended_mc, record.created_at, "svg")
            return StreamingResponse(
                io.BytesIO(content_str.encode("utf-8")),
                media_type="image/svg+xml",
                headers={"Content-Disposition": f'attachment; filename="{fname}"'},
            )

        if request.format == ExportFormat.PDF:
            pdf_path = exporter.to_pdf()
            content_bytes = pdf_path.read_bytes()
            fname = _export_filename(result_mode, is_extended_mc, record.created_at, "pdf")
            return StreamingResponse(
                io.BytesIO(content_bytes),
                media_type="application/pdf",
                headers={"Content-Disposition": f'attachment; filename="{fname}"'},
            )

    raise HTTPException(status_code=400, detail=f"Unsupported format: {request.format}")
