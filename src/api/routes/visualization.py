"""API endpoints для визуализации.

Возвращают Plotly JSON для потребления React/Plotly.js фронтенда.
Эндпоинты запускают симуляцию с переданными параметрами и возвращают
JSON-представление фигур.

Подробное описание: Description/Phase4/description_visualization.md
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.core.extended_sde import (
    VARIABLE_NAMES,
    ExtendedSDEModel,
    ExtendedSDEState,
    ExtendedSDETrajectory,
)
from src.core.parameters import ParameterSet
from src.core.sde_model import TherapyProtocol
from src.core.wound_phases import WoundPhaseDetector
from src.visualization.export import ExportConfig, ReportExporter
from src.visualization.plots import (
    plot_comparison,
    plot_cytokines,
    plot_ecm,
    plot_phases,
    plot_populations,
)
from src.visualization.spatial import (
    field_heatmap,
    heatmap_density,
    inflammation_map,
    scatter_agents,
)

router = APIRouter(prefix="/api/viz", tags=["visualization"])


# ── Pydantic models для запросов ────────────────────────────────────


class SimulationParams(BaseModel):
    """Параметры симуляции для визуализации."""

    # Начальные условия (ExtendedSDEState)
    P0: float = Field(default=500.0, ge=0, description="Начальные тромбоциты")
    Ne0: float = Field(default=200.0, ge=0, description="Начальные нейтрофилы")
    M1_0: float = Field(default=100.0, ge=0, description="Начальные M1")
    M2_0: float = Field(default=10.0, ge=0, description="Начальные M2")
    F0: float = Field(default=50.0, ge=0, description="Начальные фибробласты")
    Mf0: float = Field(default=0.0, ge=0, description="Начальные миофибробласты")
    E0: float = Field(default=20.0, ge=0, description="Начальные эндотелиальные")
    S0: float = Field(default=40.0, ge=0, description="Начальные стволовые")
    C_TNF0: float = Field(default=10.0, ge=0)
    C_IL10_0: float = Field(default=0.5, ge=0)
    D0: float = Field(default=5.0, ge=0, description="Начальный damage signal")
    O2_0: float = Field(default=80.0, ge=0, description="Начальный кислород")

    # Время
    t_max_hours: float = Field(default=720.0, gt=0, description="Время симуляции (часы)")
    dt: float = Field(default=0.1, gt=0, description="Шаг интегрирования (часы)")

    # Терапия
    prp_enabled: bool = False
    pemf_enabled: bool = False
    prp_intensity: float = Field(default=1.0, ge=0, le=2.0)
    pemf_frequency: float = Field(default=50.0, ge=1.0, le=100.0)
    pemf_intensity: float = Field(default=1.0, ge=0, le=2.0)

    # Seed
    random_seed: int | None = Field(default=42)


class PopulationsRequest(BaseModel):
    """Запрос для plot_populations."""

    simulation: SimulationParams = Field(default_factory=SimulationParams)
    variables: list[str] | None = Field(default=None, description="Подмножество популяций")
    height: int = Field(default=500, ge=200, le=1200)


class CytokinesRequest(BaseModel):
    """Запрос для plot_cytokines."""

    simulation: SimulationParams = Field(default_factory=SimulationParams)
    variables: list[str] | None = None
    layout: str = Field(default="overlay", pattern="^(overlay|subplots)$")
    height: int = Field(default=500, ge=200, le=1200)


class ECMRequest(BaseModel):
    """Запрос для plot_ecm."""

    simulation: SimulationParams = Field(default_factory=SimulationParams)
    height: int = Field(default=400, ge=200, le=1200)


class PhasesRequest(BaseModel):
    """Запрос для plot_phases."""

    simulation: SimulationParams = Field(default_factory=SimulationParams)
    height: int = Field(default=500, ge=200, le=1200)


class ComparisonRequest(BaseModel):
    """Запрос для plot_comparison."""

    simulation: SimulationParams = Field(default_factory=SimulationParams)
    variable: str = Field(default="F")
    show_all_populations: bool = False
    height: int = Field(default=500, ge=200, le=1200)


class ExportRequest(BaseModel):
    """Запрос для экспорта."""

    simulation: SimulationParams = Field(default_factory=SimulationParams)
    include_populations: bool = True
    include_cytokines: bool = True
    include_ecm: bool = True
    include_phases: bool = True


# ── Helper: запуск симуляции ────────────────────────────────────────


def _load_trajectory_from_cache(simulation_id: str) -> ExtendedSDETrajectory:
    """Загрузка траектории из сохранённого .npz файла."""
    from src.api.config import settings

    result_path = Path(settings.results_dir) / f"{simulation_id}.npz"
    if not result_path.exists():
        raise HTTPException(status_code=404, detail=f"Results not found for {simulation_id}")

    data = np.load(str(result_path))
    times = data["times"]
    states: list[ExtendedSDEState] = []
    for i in range(len(times)):
        kwargs: dict[str, float] = {}
        for var_name in VARIABLE_NAMES:
            kwargs[var_name] = float(data[var_name][i]) if var_name in data.files else 0.0
        kwargs["t"] = float(times[i])
        states.append(ExtendedSDEState(**kwargs))

    return ExtendedSDETrajectory(times=times, states=states, params=ParameterSet())


def _run_simulation(params: SimulationParams) -> ExtendedSDETrajectory:
    """Запуск ExtendedSDE с параметрами из запроса."""
    therapy = TherapyProtocol(
        prp_enabled=params.prp_enabled,
        prp_intensity=params.prp_intensity,
        pemf_enabled=params.pemf_enabled,
        pemf_frequency=params.pemf_frequency,
        pemf_intensity=params.pemf_intensity,
    )

    pset = ParameterSet(dt=params.dt, t_max=params.t_max_hours)

    model = ExtendedSDEModel(
        params=pset,
        therapy=therapy,
        rng_seed=params.random_seed,
    )

    initial_state = ExtendedSDEState(
        P=params.P0, Ne=params.Ne0, M1=params.M1_0, M2=params.M2_0,
        F=params.F0, Mf=params.Mf0, E=params.E0, S=params.S0,
        C_TNF=params.C_TNF0, C_IL10=params.C_IL10_0,
        D=params.D0, O2=params.O2_0,
    )

    return model.simulate(initial_state=initial_state)


def _run_comparison(params: SimulationParams) -> dict[str, ExtendedSDETrajectory]:
    """Запуск 4 сценариев для сравнения."""
    scenarios = {
        "Control": SimulationParams(**{**params.model_dump(), "prp_enabled": False, "pemf_enabled": False}),
        "PRP": SimulationParams(**{**params.model_dump(), "prp_enabled": True, "pemf_enabled": False}),
        "PEMF": SimulationParams(**{**params.model_dump(), "prp_enabled": False, "pemf_enabled": True}),
        "PRP+PEMF": SimulationParams(**{**params.model_dump(), "prp_enabled": True, "pemf_enabled": True}),
    }

    results = {}
    for name, scenario_params in scenarios.items():
        results[name] = _run_simulation(scenario_params)

    return results


# ── Endpoints: временные графики ────────────────────────────────────


@router.post("/populations")
async def viz_populations(request: PopulationsRequest) -> JSONResponse:
    """Кривые роста клеточных популяций → Plotly JSON."""
    trajectory = _run_simulation(request.simulation)
    fig = plot_populations(
        trajectory,
        variables=request.variables,
        height=request.height,
    )
    return JSONResponse(content=json.loads(fig.to_json()))


@router.post("/cytokines")
async def viz_cytokines(request: CytokinesRequest) -> JSONResponse:
    """Динамика цитокинов → Plotly JSON."""
    trajectory = _run_simulation(request.simulation)
    fig = plot_cytokines(
        trajectory,
        variables=request.variables,
        layout=request.layout,
        height=request.height,
    )
    return JSONResponse(content=json.loads(fig.to_json()))


@router.post("/ecm")
async def viz_ecm(request: ECMRequest) -> JSONResponse:
    """Динамика ECM → Plotly JSON."""
    trajectory = _run_simulation(request.simulation)
    fig = plot_ecm(trajectory, height=request.height)
    return JSONResponse(content=json.loads(fig.to_json()))


@router.post("/phases")
async def viz_phases(request: PhasesRequest) -> JSONResponse:
    """Фазы заживления → Plotly JSON."""
    trajectory = _run_simulation(request.simulation)
    fig = plot_phases(trajectory, height=request.height)
    return JSONResponse(content=json.loads(fig.to_json()))


@router.post("/comparison")
async def viz_comparison(request: ComparisonRequest) -> JSONResponse:
    """Сравнение 4 сценариев → Plotly JSON."""
    results = _run_comparison(request.simulation)
    fig = plot_comparison(
        results,
        variable=request.variable,
        show_all_populations=request.show_all_populations,
        height=request.height,
    )
    return JSONResponse(content=json.loads(fig.to_json()))


# ── Endpoints: графики из кэшированных результатов ─────────────────


@router.get("/from-result/{simulation_id}/populations")
async def viz_populations_cached(
    simulation_id: str,
    variables: str | None = None,
    height: int = 500,
) -> JSONResponse:
    """Кривые роста из сохранённых результатов (мгновенно, без пересчёта)."""
    trajectory = _load_trajectory_from_cache(simulation_id)
    var_list = variables.split(",") if variables else None
    fig = plot_populations(trajectory, variables=var_list, height=height)
    return JSONResponse(content=json.loads(fig.to_json()))


@router.get("/from-result/{simulation_id}/cytokines")
async def viz_cytokines_cached(
    simulation_id: str,
    layout: str = "overlay",
    height: int = 500,
) -> JSONResponse:
    """Динамика цитокинов из сохранённых результатов."""
    trajectory = _load_trajectory_from_cache(simulation_id)
    fig = plot_cytokines(trajectory, layout=layout, height=height)
    return JSONResponse(content=json.loads(fig.to_json()))


@router.get("/from-result/{simulation_id}/ecm")
async def viz_ecm_cached(simulation_id: str, height: int = 400) -> JSONResponse:
    """ECM из сохранённых результатов."""
    trajectory = _load_trajectory_from_cache(simulation_id)
    fig = plot_ecm(trajectory, height=height)
    return JSONResponse(content=json.loads(fig.to_json()))


@router.get("/from-result/{simulation_id}/phases")
async def viz_phases_cached(simulation_id: str, height: int = 500) -> JSONResponse:
    """Фазы заживления из сохранённых результатов."""
    trajectory = _load_trajectory_from_cache(simulation_id)
    fig = plot_phases(trajectory, height=height)
    return JSONResponse(content=json.loads(fig.to_json()))


# ── Endpoints: экспорт ──────────────────────────────────────────────


@router.post("/export/csv")
async def export_csv(request: ExportRequest) -> StreamingResponse:
    """Экспорт данных траектории в CSV."""
    trajectory = _run_simulation(request.simulation)

    with TemporaryDirectory() as tmpdir:
        config = ExportConfig(output_dir=Path(tmpdir))
        exporter = ReportExporter(config)
        exporter.add_trajectory_data("simulation", trajectory)
        paths = exporter.to_csv()

        if not paths:
            raise HTTPException(status_code=500, detail="CSV generation failed")

        content = paths[0].read_text(encoding="utf-8")

    return StreamingResponse(
        io.BytesIO(content.encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=simulation_data.csv"},
    )


@router.post("/export/png")
async def export_png(request: ExportRequest) -> StreamingResponse:
    """Экспорт графика популяций в PNG."""
    trajectory = _run_simulation(request.simulation)
    fig = plot_populations(trajectory)

    with TemporaryDirectory() as tmpdir:
        config = ExportConfig(output_dir=Path(tmpdir), width=1200, height=800)
        exporter = ReportExporter(config)
        exporter.add_figure("populations", fig)
        paths = exporter.to_png()

        if not paths:
            raise HTTPException(status_code=500, detail="PNG generation failed")

        content = paths[0].read_bytes()

    return StreamingResponse(
        io.BytesIO(content),
        media_type="image/png",
        headers={"Content-Disposition": "attachment; filename=populations.png"},
    )


@router.post("/export/pdf")
async def export_pdf(request: ExportRequest) -> StreamingResponse:
    """Экспорт полного PDF-отчёта."""
    trajectory = _run_simulation(request.simulation)

    with TemporaryDirectory() as tmpdir:
        config = ExportConfig(output_dir=Path(tmpdir), width=1000, height=600)
        exporter = ReportExporter(config)
        exporter.add_metadata("therapy_PRP", str(request.simulation.prp_enabled))
        exporter.add_metadata("therapy_PEMF", str(request.simulation.pemf_enabled))
        exporter.add_metadata("duration_hours", str(request.simulation.t_max_hours))

        if request.include_populations:
            exporter.add_figure("populations", plot_populations(trajectory))
        if request.include_cytokines:
            exporter.add_figure("cytokines", plot_cytokines(trajectory))
        if request.include_ecm:
            exporter.add_figure("ecm", plot_ecm(trajectory))
        if request.include_phases:
            exporter.add_figure("phases", plot_phases(trajectory))

        exporter.add_trajectory_data("simulation", trajectory)

        pdf_path = exporter.to_pdf()
        content = pdf_path.read_bytes()

    return StreamingResponse(
        io.BytesIO(content),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=regentwin_report.pdf"},
    )
