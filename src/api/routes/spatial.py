"""API endpoints для пространственной визуализации ABM.

Запускают ABM симуляцию, берут snapshot/trajectory, и возвращают
Plotly JSON через функции из src.visualization.spatial.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.core.abm_model import ABMModel, ABMConfig
from src.data.parameter_extraction import ModelParameters
from src.visualization.spatial import (
    heatmap_density,
    inflammation_map,
    scatter_agents,
)

router = APIRouter(prefix="/api/viz/spatial", tags=["spatial-visualization"])


class SpatialRequest(BaseModel):
    """Запрос для spatial визуализации."""

    # ABM параметры
    n_stem: int = Field(default=20, ge=0, description="Стволовые клетки")
    n_macro: int = Field(default=30, ge=0, description="Макрофаги")
    n_fibro: int = Field(default=15, ge=0, description="Фибробласты")
    n_neutrophil: int = Field(default=40, ge=0, description="Нейтрофилы")
    n_endothelial: int = Field(default=10, ge=0, description="Эндотелиальные")

    # Время
    t_max_hours: float = Field(default=48.0, gt=0, le=720)
    dt: float = Field(default=1.0, gt=0)

    # Spatial options
    timestep: int = Field(default=-1, description="Индекс snapshot (-1 = последний)")
    bin_size: int = Field(default=10, ge=1, le=50)
    agent_types: list[str] | None = Field(default=None, description="Фильтр по типам")
    color_by: str = Field(default="type", pattern="^(type|energy|age)$")
    height: int = Field(default=500, ge=200, le=1200)

    # Domain
    domain_size: float = Field(default=100.0, gt=0, description="Размер домена (мкм)")

    random_seed: int | None = 42


def _run_abm(request: SpatialRequest):
    """Запуск ABM симуляции и возврат trajectory."""
    size = request.domain_size
    total_cells = (
        request.n_stem + request.n_macro + request.n_fibro
        + request.n_neutrophil + request.n_endothelial
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
    trajectory = model.simulate(
        initial_params=initial_params,
        snapshot_interval=snapshot_interval,
    )
    return trajectory


@router.post("/heatmap")
async def spatial_heatmap(request: SpatialRequest) -> JSONResponse:
    """2D density heatmap агентов → Plotly JSON."""
    try:
        trajectory = _run_abm(request)
        idx = request.timestep if request.timestep >= 0 else len(trajectory.snapshots) - 1
        idx = min(idx, len(trajectory.snapshots) - 1)
        snapshot = trajectory.snapshots[idx]

        fig = heatmap_density(
            snapshot,
            bin_size=request.bin_size,
            agent_types=request.agent_types,
            height=request.height,
        )
        return JSONResponse(content=json.loads(fig.to_json()))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scatter")
async def spatial_scatter(request: SpatialRequest) -> JSONResponse:
    """Scatter plot агентов → Plotly JSON."""
    try:
        trajectory = _run_abm(request)
        idx = request.timestep if request.timestep >= 0 else len(trajectory.snapshots) - 1
        idx = min(idx, len(trajectory.snapshots) - 1)
        snapshot = trajectory.snapshots[idx]

        fig = scatter_agents(
            snapshot,
            color_by=request.color_by,
            height=request.height,
        )
        return JSONResponse(content=json.loads(fig.to_json()))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/inflammation")
async def spatial_inflammation(request: SpatialRequest) -> JSONResponse:
    """Inflammation map → Plotly JSON."""
    try:
        trajectory = _run_abm(request)
        idx = request.timestep if request.timestep >= 0 else len(trajectory.snapshots) - 1
        idx = min(idx, len(trajectory.snapshots) - 1)
        snapshot = trajectory.snapshots[idx]

        fig = inflammation_map(snapshot, height=request.height)
        return JSONResponse(content=json.loads(fig.to_json()))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
