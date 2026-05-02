"""API endpoints for ABM spatial visualization."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.api.services.simulation_service import SimulationService
from src.core.abm_model import ABMConfig, ABMModel
from src.data.parameter_extraction import ModelParameters
from src.visualization.spatial import heatmap_density, inflammation_map, scatter_agents

router = APIRouter(prefix="/api/viz/spatial", tags=["spatial-visualization"])


class SpatialRequest(BaseModel):
    """Request for ABM spatial visualization."""

    simulation_id: str | None = None

    n_stem: int = Field(default=20, ge=0)
    n_macro: int = Field(default=30, ge=0)
    n_fibro: int = Field(default=15, ge=0)
    n_neutrophil: int = Field(default=40, ge=0)
    n_endothelial: int = Field(default=10, ge=0)

    t_max_hours: float = Field(default=48.0, gt=0, le=720)
    dt: float = Field(default=1.0, gt=0)

    timestep: int = Field(default=-1)
    bin_size: int = Field(default=10, ge=1, le=50)
    agent_types: list[str] | None = None
    color_by: str = Field(default="type", pattern="^(type|energy|age)$")
    height: int = Field(default=500, ge=200, le=1200)

    domain_size: float = Field(default=100.0, gt=0)
    random_seed: int | None = 42


def _run_abm(request: SpatialRequest):
    size = request.domain_size
    total_cells = (
        request.n_stem
        + request.n_macro
        + request.n_fibro
        + request.n_neutrophil
        + request.n_endothelial
    )
    stem_frac = request.n_stem / max(total_cells, 1)
    macro_frac = request.n_macro / max(total_cells, 1)

    config = ABMConfig(
        space_size=(size, size),
        initial_stem_cells=request.n_stem,
        initial_macrophages=request.n_macro,
        initial_fibroblasts=request.n_fibro,
        initial_neutrophils=request.n_neutrophil,
        initial_endothelial=request.n_endothelial,
        dt=request.dt,
        t_max=request.t_max_hours,
    )

    initial_params = ModelParameters(
        n0=float(total_cells),
        stem_cell_fraction=stem_frac,
        macrophage_fraction=macro_frac,
        apoptotic_fraction=0.05,
        c0=1.0,
        inflammation_level=0.5,
    )

    model = ABMModel(config, random_seed=request.random_seed)
    snapshot_interval = max(request.dt, request.t_max_hours / 10.0)
    return model.simulate(initial_params=initial_params, snapshot_interval=snapshot_interval)


def _load_snapshot(request: SpatialRequest):
    if request.simulation_id:
        return SimulationService.load_spatial_snapshot(
            request.simulation_id, timestep=request.timestep
        )

    trajectory = _run_abm(request)
    idx = request.timestep if request.timestep >= 0 else len(trajectory.snapshots) - 1
    idx = min(idx, len(trajectory.snapshots) - 1)
    return trajectory.snapshots[idx]


@router.post("/heatmap")
async def spatial_heatmap(request: SpatialRequest) -> JSONResponse:
    try:
        snapshot = _load_snapshot(request)
        fig = heatmap_density(
            snapshot,
            bin_size=request.bin_size,
            agent_types=request.agent_types,
            height=request.height,
        )
        return JSONResponse(content=fig.to_plotly_json())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/scatter")
async def spatial_scatter(request: SpatialRequest) -> JSONResponse:
    try:
        snapshot = _load_snapshot(request)
        fig = scatter_agents(snapshot, color_by=request.color_by, height=request.height)
        return JSONResponse(content=fig.to_plotly_json())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/inflammation")
async def spatial_inflammation(request: SpatialRequest) -> JSONResponse:
    try:
        snapshot = _load_snapshot(request)
        fig = inflammation_map(snapshot, height=request.height)
        return JSONResponse(content=fig.to_plotly_json())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
