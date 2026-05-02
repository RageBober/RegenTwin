"""Visualization API endpoints."""

from __future__ import annotations

import asyncio
import io
from pathlib import Path
from tempfile import TemporaryDirectory

import plotly.graph_objects as go

HAS_PLOTLY = True

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.api.models.schemas import (
    ConvergenceVizRequest,
    MorrisVizRequest,
    PosteriorVizRequest,
    SobolVizRequest,
)
from src.api.services.result_bundle import build_extended_trajectory, build_mc_mean_trajectory
from src.api.services.simulation_service import SimulationService
from src.core.extended_sde import ExtendedSDEModel, ExtendedSDEState, ExtendedSDETrajectory
from src.core.parameter_estimation import (
    ConvergenceDiagnostics,
    EstimationConfig,
    EstimationResult,
)
from src.core.parameters import ParameterSet
from src.core.sde_model import TherapyProtocol
from src.core.sensitivity_analysis import MorrisResult, SobolResult
from src.db.models import AnalysisRecord
from src.db.session import get_db
from src.visualization.analysis_plots import (
    plot_convergence,
    plot_morris,
    plot_posterior,
    plot_sobol,
)
from src.visualization.export import ExportConfig, ReportExporter
from src.visualization.plots import (
    plot_comparison,
    plot_cytokines,
    plot_ecm,
    plot_phases,
    plot_populations,
)

router = APIRouter(prefix="/api/viz", tags=["visualization"])


class SimulationParams(BaseModel):
    """Simulation parameters for visualization-only requests."""

    P0: float = Field(default=500.0, ge=0)
    Ne0: float = Field(default=200.0, ge=0)
    M1_0: float = Field(default=100.0, ge=0)
    M2_0: float = Field(default=10.0, ge=0)
    F0: float = Field(default=50.0, ge=0)
    Mf0: float = Field(default=0.0, ge=0)
    E0: float = Field(default=20.0, ge=0)
    S0: float = Field(default=40.0, ge=0)
    C_TNF0: float = Field(default=10.0, ge=0)
    C_IL10_0: float = Field(default=0.5, ge=0)
    D0: float = Field(default=5.0, ge=0)
    O2_0: float = Field(default=80.0, ge=0)

    t_max_hours: float = Field(default=720.0, gt=0)
    dt: float = Field(default=0.1, gt=0)

    prp_enabled: bool = False
    pemf_enabled: bool = False
    prp_intensity: float = Field(default=1.0, ge=0, le=2.0)
    pemf_frequency: float = Field(default=50.0, ge=1.0, le=100.0)
    pemf_intensity: float = Field(default=1.0, ge=0, le=2.0)
    random_seed: int | None = Field(default=42)


class PopulationsRequest(BaseModel):
    simulation: SimulationParams = Field(default_factory=SimulationParams)
    variables: list[str] | None = None
    height: int = Field(default=500, ge=200, le=1200)


class CytokinesRequest(BaseModel):
    simulation: SimulationParams = Field(default_factory=SimulationParams)
    variables: list[str] | None = None
    layout: str = Field(default="overlay", pattern="^(overlay|subplots)$")
    height: int = Field(default=500, ge=200, le=1200)


class ECMRequest(BaseModel):
    simulation: SimulationParams = Field(default_factory=SimulationParams)
    height: int = Field(default=400, ge=200, le=1200)


class PhasesRequest(BaseModel):
    simulation: SimulationParams = Field(default_factory=SimulationParams)
    height: int = Field(default=500, ge=200, le=1200)


class ComparisonRequest(BaseModel):
    simulation: SimulationParams = Field(default_factory=SimulationParams)
    variable: str = Field(default="F")
    show_all_populations: bool = False
    height: int = Field(default=500, ge=200, le=1200)


class ExportRequest(BaseModel):
    simulation: SimulationParams = Field(default_factory=SimulationParams)
    include_populations: bool = True
    include_cytokines: bool = True
    include_ecm: bool = True
    include_phases: bool = True


def _load_result_from_cache(simulation_id: str) -> dict:
    try:
        return SimulationService.load_trajectory(simulation_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail=f"Results not found for {simulation_id}"
        ) from exc


def _load_extended_trajectory_from_cache(simulation_id: str) -> ExtendedSDETrajectory:
    result = _load_result_from_cache(simulation_id)
    try:
        return build_extended_trajectory(result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _plot_cached_lines(
    result: dict,
    series: list[str],
    *,
    height: int,
    title: str,
    yaxis_title: str,
) -> go.Figure:
    available = [name for name in series if name in result["variables"]]
    if not available:
        raise HTTPException(
            status_code=400, detail="Requested series are not available for this result"
        )

    fig = go.Figure()
    times = result["times"]
    for name in available:
        fig.add_trace(
            go.Scatter(
                x=times,
                y=result["variables"][name],
                mode="lines",
                name=name,
            )
        )

    fig.update_layout(
        height=height,
        title=title,
        margin=dict(l=50, r=20, t=50, b=50),
        legend=dict(orientation="h", y=-0.2),
    )
    fig.update_xaxes(title_text="Time (h)")
    fig.update_yaxes(title_text=yaxis_title)
    return fig


def _is_monte_carlo(result: dict) -> bool:
    """Return True if the result bundle contains Monte Carlo ensemble statistics."""
    return result.get("metadata", {}).get("n_trajectories", 1) > 1


def _is_extended_mc(result: dict) -> bool:
    """Return True if the result is an extended MC with all 20 variables."""
    return _is_monte_carlo(result) and result.get("metadata", {}).get("extended_mc", False)


def _fig_response(fig: go.Figure) -> JSONResponse:
    return JSONResponse(content=fig.to_plotly_json())


def _plot_mc_lines(
    result: dict,
    series: list[str],
    *,
    height: int,
    title: str,
    yaxis_title: str,
) -> go.Figure:
    """Plot Monte Carlo ensemble statistics with confidence bands."""
    variables = result["variables"]
    available = [name for name in series if name in variables]
    if not available:
        raise HTTPException(
            status_code=400, detail="Requested series not available in Monte Carlo results"
        )

    fig = go.Figure()
    times = result["times"]

    for name in available:
        if name.startswith("std_"):
            continue
        fig.add_trace(
            go.Scatter(
                x=times,
                y=variables[name],
                mode="lines",
                name=name,
                line=dict(width=2),
            )
        )

    # Добавляем CI-полосы для каждой серии mean_X, если есть q0.05_X и q0.95_X
    for name in available:
        if not name.startswith("mean_"):
            continue
        var_suffix = name[5:]  # "N", "C", "P", "C_TNF", etc.
        q05_key = f"q0.05_{var_suffix}"
        q95_key = f"q0.95_{var_suffix}"
        if q05_key in variables and q95_key in variables:
            upper = variables[q95_key]
            lower = variables[q05_key]
            fig.add_trace(
                go.Scatter(
                    x=list(times) + list(reversed(times)),
                    y=list(upper) + list(reversed(lower)),
                    fill="toself",
                    fillcolor="rgba(46,134,193,0.12)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name=f"90% CI ({var_suffix})",
                    showlegend=False,
                )
            )

    fig.update_layout(
        height=height,
        title=title,
        margin=dict(l=50, r=20, t=50, b=50),
        legend=dict(orientation="h", y=-0.2),
    )
    fig.update_xaxes(title_text="Time (h)")
    fig.update_yaxes(title_text=yaxis_title)
    return fig


def _run_simulation(params: SimulationParams) -> ExtendedSDETrajectory:
    therapy = TherapyProtocol(
        prp_enabled=params.prp_enabled,
        prp_intensity=params.prp_intensity,
        pemf_enabled=params.pemf_enabled,
        pemf_frequency=params.pemf_frequency,
        pemf_intensity=params.pemf_intensity,
    )

    pset = ParameterSet(dt=params.dt, t_max=params.t_max_hours)
    model = ExtendedSDEModel(params=pset, therapy=therapy, rng_seed=params.random_seed)
    initial_state = ExtendedSDEState(
        P=params.P0,
        Ne=params.Ne0,
        M1=params.M1_0,
        M2=params.M2_0,
        F=params.F0,
        Mf=params.Mf0,
        E=params.E0,
        S=params.S0,
        C_TNF=params.C_TNF0,
        C_IL10=params.C_IL10_0,
        D=params.D0,
        O2=params.O2_0,
    )
    return model.simulate(initial_state=initial_state)


def _run_comparison(params: SimulationParams) -> dict[str, ExtendedSDETrajectory]:
    scenarios = {
        "Control": SimulationParams(
            **{**params.model_dump(), "prp_enabled": False, "pemf_enabled": False}
        ),
        "PRP": SimulationParams(
            **{**params.model_dump(), "prp_enabled": True, "pemf_enabled": False}
        ),
        "PEMF": SimulationParams(
            **{**params.model_dump(), "prp_enabled": False, "pemf_enabled": True}
        ),
        "PRP+PEMF": SimulationParams(
            **{**params.model_dump(), "prp_enabled": True, "pemf_enabled": True}
        ),
    }
    return {name: _run_simulation(scenario) for name, scenario in scenarios.items()}


@router.post("/populations")
async def viz_populations(request: PopulationsRequest) -> JSONResponse:
    loop = asyncio.get_running_loop()
    trajectory = await loop.run_in_executor(None, _run_simulation, request.simulation)
    fig = plot_populations(trajectory, variables=request.variables, height=request.height)
    return _fig_response(fig)


@router.post("/cytokines")
async def viz_cytokines(request: CytokinesRequest) -> JSONResponse:
    loop = asyncio.get_running_loop()
    trajectory = await loop.run_in_executor(None, _run_simulation, request.simulation)
    fig = plot_cytokines(
        trajectory,
        variables=request.variables,
        layout=request.layout,
        height=request.height,
    )
    return _fig_response(fig)


@router.post("/ecm")
async def viz_ecm(request: ECMRequest) -> JSONResponse:
    loop = asyncio.get_running_loop()
    trajectory = await loop.run_in_executor(None, _run_simulation, request.simulation)
    fig = plot_ecm(trajectory, height=request.height)
    return _fig_response(fig)


@router.post("/phases")
async def viz_phases(request: PhasesRequest) -> JSONResponse:
    loop = asyncio.get_running_loop()
    trajectory = await loop.run_in_executor(None, _run_simulation, request.simulation)
    fig = plot_phases(trajectory, height=request.height)
    return _fig_response(fig)


@router.post("/comparison")
async def viz_comparison(request: ComparisonRequest) -> JSONResponse:
    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(None, _run_comparison, request.simulation)
    fig = plot_comparison(
        results,
        variable=request.variable,
        show_all_populations=request.show_all_populations,
        height=request.height,
    )
    return _fig_response(fig)


@router.get("/from-result/{simulation_id}/populations")
async def viz_populations_cached(
    simulation_id: str,
    variables: str | None = None,
    height: int = 500,
) -> JSONResponse:
    result = _load_result_from_cache(simulation_id)
    var_list = variables.split(",") if variables else None

    if _is_extended_mc(result):
        from src.visualization.theme import POPULATION_VARS

        pop_series = [f"mean_{v}" for v in POPULATION_VARS]
        fig = _plot_mc_lines(
            result,
            var_list or pop_series,
            height=height,
            title="Monte Carlo population dynamics (extended)",
            yaxis_title="Cell density",
        )
    elif _is_monte_carlo(result):
        fig = _plot_mc_lines(
            result,
            var_list or ["mean_N", "mean_C"],
            height=height,
            title="Monte Carlo population dynamics",
            yaxis_title="Cell density",
        )
    elif result["mode"] in {"extended", "integrated"}:
        trajectory = _load_extended_trajectory_from_cache(simulation_id)
        fig = plot_populations(trajectory, variables=var_list, height=height)
    elif result["mode"] == "mvp":
        fig = _plot_cached_lines(
            result,
            var_list or list(result["variables"].keys()),
            height=height,
            title="MVP population dynamics",
            yaxis_title="Cell density",
        )
    elif result["mode"] == "abm":
        fig = _plot_cached_lines(
            result,
            var_list or list(result["variables"].keys()),
            height=height,
            title="ABM population dynamics",
            yaxis_title="Agents",
        )
    else:
        raise HTTPException(
            status_code=400, detail=f"Unsupported cached result mode: {result['mode']}"
        )

    return _fig_response(fig)


@router.get("/from-result/{simulation_id}/cytokines")
async def viz_cytokines_cached(
    simulation_id: str,
    layout: str = "overlay",
    height: int = 500,
) -> JSONResponse:
    result = _load_result_from_cache(simulation_id)
    if _is_extended_mc(result):
        from src.visualization.theme import CYTOKINE_VARS

        cyt_series = [f"mean_{v}" for v in CYTOKINE_VARS]
        fig = _plot_mc_lines(
            result,
            cyt_series,
            height=height,
            title="Monte Carlo cytokine dynamics (extended)",
            yaxis_title="Concentration",
        )
    elif _is_monte_carlo(result):
        fig = _plot_mc_lines(
            result,
            ["mean_C"],
            height=height,
            title="Monte Carlo cytokine dynamics",
            yaxis_title="Concentration",
        )
    elif result["mode"] in {"extended", "integrated"}:
        trajectory = _load_extended_trajectory_from_cache(simulation_id)
        fig = plot_cytokines(trajectory, layout=layout, height=height)
    elif result["mode"] == "mvp":
        fig = _plot_cached_lines(
            result,
            ["C_TNF"],
            height=height,
            title="MVP cytokine dynamics",
            yaxis_title="Model value",
        )
    else:
        raise HTTPException(
            status_code=400, detail=f"Cytokine charts are not available for mode {result['mode']}"
        )

    return _fig_response(fig)


@router.get("/from-result/{simulation_id}/ecm")
async def viz_ecm_cached(simulation_id: str, height: int = 400) -> JSONResponse:
    result = _load_result_from_cache(simulation_id)
    if _is_extended_mc(result):
        mean_traj = build_mc_mean_trajectory(result)
        fig = plot_ecm(mean_traj, height=height)
    elif _is_monte_carlo(result):
        raise HTTPException(
            status_code=400,
            detail="ECM chart is not available for MVP Monte Carlo results",
        )
    else:
        trajectory = _load_extended_trajectory_from_cache(simulation_id)
        fig = plot_ecm(trajectory, height=height)
    return _fig_response(fig)


@router.get("/from-result/{simulation_id}/phases")
async def viz_phases_cached(simulation_id: str, height: int = 500) -> JSONResponse:
    result = _load_result_from_cache(simulation_id)
    if _is_extended_mc(result):
        mean_traj = build_mc_mean_trajectory(result)
        fig = plot_phases(mean_traj, height=height)
    elif _is_monte_carlo(result):
        raise HTTPException(
            status_code=400,
            detail="Phase chart is not available for MVP Monte Carlo results",
        )
    else:
        trajectory = _load_extended_trajectory_from_cache(simulation_id)
        fig = plot_phases(trajectory, height=height)
    return _fig_response(fig)


@router.get("/from-result/{simulation_id}/comparison")
async def viz_comparison_cached(
    simulation_id: str,
    variable: str = "F",
    show_all_populations: bool = False,
    height: int = 500,
) -> JSONResponse:
    try:
        params_dict = SimulationService.load_params(simulation_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    params = SimulationParams(**params_dict)
    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(None, _run_comparison, params)
    fig = plot_comparison(
        results,
        variable=variable,
        show_all_populations=show_all_populations,
        height=height,
    )
    return _fig_response(fig)


@router.post("/export/csv")
async def export_csv(request: ExportRequest) -> StreamingResponse:
    loop = asyncio.get_running_loop()
    trajectory = await loop.run_in_executor(None, _run_simulation, request.simulation)

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
    loop = asyncio.get_running_loop()
    trajectory = await loop.run_in_executor(None, _run_simulation, request.simulation)
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
    loop = asyncio.get_running_loop()
    trajectory = await loop.run_in_executor(None, _run_simulation, request.simulation)

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


# ── Analysis Visualization Helpers ──────────────────────────────


def _load_analysis_record(analysis_id: str, db: Session) -> AnalysisRecord:
    """Загрузить запись анализа из БД, проверить статус completed."""
    record = db.get(AnalysisRecord, analysis_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id} not found")
    if record.status != "completed":
        raise HTTPException(
            status_code=400, detail=f"Analysis not completed (status={record.status})"
        )
    if not record.result_json:
        raise HTTPException(status_code=400, detail="Analysis has no results")
    return record


def _dict_to_sobol_result(d: dict) -> SobolResult:
    """Реконструкция SobolResult из сериализованного dict."""
    n = len(d["parameters"])
    return SobolResult(
        parameter_names=d["parameters"],
        S1=np.array(d["S1"]),
        ST=np.array(d["ST"]),
        S1_conf=np.array(d["S1_conf"]) if d.get("S1_conf") else np.zeros(n),
        ST_conf=np.array(d["ST_conf"]) if d.get("ST_conf") else np.zeros(n),
        output_variable=d.get("output_variable", "F"),
        n_samples=d.get("n_samples", 0),
        n_model_runs=d.get("n_runs", 0),
    )


def _dict_to_morris_result(d: dict) -> MorrisResult:
    """Реконструкция MorrisResult из сериализованного dict."""
    n = len(d["parameters"])
    return MorrisResult(
        parameter_names=d["parameters"],
        mu_star=np.array(d["mu_star"]),
        sigma=np.array(d["sigma"]),
        mu_star_conf=np.array(d["mu_star_conf"]) if d.get("mu_star_conf") else np.zeros(n),
        mu=np.zeros(n),
        output_variable=d.get("output_variable", "F"),
        n_model_runs=d.get("n_runs", 0),
    )


def _dict_to_estimation_result(d: dict) -> EstimationResult:
    """Реконструкция EstimationResult из сериализованного dict."""
    posterior_samples = None
    if d.get("posterior_samples"):
        posterior_samples = {k: np.array(v) for k, v in d["posterior_samples"].items()}

    diagnostics = None
    if d.get("diagnostics"):
        diag = d["diagnostics"]
        diagnostics = ConvergenceDiagnostics(
            rhat=diag.get("rhat", {}),
            ess_bulk=diag.get("ess_bulk", {}),
            ess_tail=diag.get("ess_tail", {}),
            converged=diag.get("converged", False),
            warnings=diag.get("warnings", []),
        )

    config = None
    n_chains = d.get("n_chains", 1)
    if n_chains:
        config = EstimationConfig(n_chains=n_chains)

    return EstimationResult(
        method=d.get("method", ""),
        point_estimates=d.get("point_estimates", {}),
        ci_lower=d.get("ci_lower", {}),
        ci_upper=d.get("ci_upper", {}),
        posterior_samples=posterior_samples,
        diagnostics=diagnostics,
        config=config,
    )


# ── Analysis Visualization Endpoints ────────────────────────────


@router.post("/analysis/sobol")
async def viz_analysis_sobol(
    request: SobolVizRequest,
    db: Session = Depends(get_db),
) -> JSONResponse:
    """Sobol tornado bar chart из результатов анализа чувствительности."""
    record = _load_analysis_record(request.analysis_id, db)
    assert record.result_json is not None  # гарантировано _load_analysis_record
    data: dict = record.result_json
    if data.get("method") != "sobol":
        raise HTTPException(status_code=400, detail="Analysis is not a Sobol result")
    if data.get("error"):
        raise HTTPException(status_code=400, detail=data["error"])

    sobol = _dict_to_sobol_result(data)
    fig = plot_sobol(
        sobol,
        metric=request.metric,
        top_n=request.top_n,
        show_confidence=request.show_confidence,
        height=request.height,
    )
    return _fig_response(fig)


@router.post("/analysis/morris")
async def viz_analysis_morris(
    request: MorrisVizRequest,
    db: Session = Depends(get_db),
) -> JSONResponse:
    """Morris screening scatter plot (μ* vs σ)."""
    record = _load_analysis_record(request.analysis_id, db)
    assert record.result_json is not None
    data: dict = record.result_json
    if data.get("method") != "morris":
        raise HTTPException(status_code=400, detail="Analysis is not a Morris result")
    if data.get("error"):
        raise HTTPException(status_code=400, detail=data["error"])

    morris = _dict_to_morris_result(data)
    fig = plot_morris(
        morris,
        highlight_influential=request.highlight_influential,
        threshold_ratio=request.threshold_ratio,
        show_labels=request.show_labels,
        show_wedge=request.show_wedge,
        height=request.height,
    )
    return _fig_response(fig)


@router.post("/analysis/posterior")
async def viz_analysis_posterior(
    request: PosteriorVizRequest,
    db: Session = Depends(get_db),
) -> JSONResponse:
    """Posterior distributions (marginals / corner plot)."""
    record = _load_analysis_record(request.analysis_id, db)
    assert record.result_json is not None
    data: dict = record.result_json

    estimation = _dict_to_estimation_result(data)
    if estimation.posterior_samples is None:
        raise HTTPException(
            status_code=400,
            detail="No posterior samples available (MLE method does not produce them)",
        )

    fig = plot_posterior(
        estimation,
        parameters=request.parameters,
        layout=request.layout,
        show_ci=request.show_ci,
        show_point_estimate=request.show_point_estimate,
        n_bins=request.n_bins,
        height=request.height,
    )
    return _fig_response(fig)


@router.post("/analysis/convergence")
async def viz_analysis_convergence(
    request: ConvergenceVizRequest,
    db: Session = Depends(get_db),
) -> JSONResponse:
    """Convergence diagnostics (R-hat, ESS, trace plots)."""
    record = _load_analysis_record(request.analysis_id, db)
    assert record.result_json is not None
    data: dict = record.result_json

    estimation = _dict_to_estimation_result(data)
    if estimation.diagnostics is None:
        raise HTTPException(
            status_code=400,
            detail="No convergence diagnostics available",
        )

    fig = plot_convergence(
        estimation,
        metrics=request.metrics,
        show_rhat_threshold=request.show_rhat_threshold,
        height=request.height,
    )
    return _fig_response(fig)
