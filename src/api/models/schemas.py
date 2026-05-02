"""Pydantic v2 schemas for API requests and responses."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class SimulationMode(str, Enum):
    MVP = "mvp"
    EXTENDED = "extended"
    ABM = "abm"
    INTEGRATED = "integrated"


class SimulationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExportFormat(str, Enum):
    CSV = "csv"
    PNG = "png"
    SVG = "svg"
    PDF = "pdf"


class AnalysisType(str, Enum):
    SENSITIVITY = "sensitivity"
    ESTIMATION = "estimation"
    VALIDATION = "validation"


class SimulationRequest(BaseModel):
    """POST /api/v1/simulate."""

    mode: SimulationMode = SimulationMode.EXTENDED

    P0: float = Field(default=500.0, ge=0, le=1e7)
    Ne0: float = Field(default=200.0, ge=0, le=1e7)
    M1_0: float = Field(default=100.0, ge=0, le=1e7)
    M2_0: float = Field(default=10.0, ge=0, le=1e7)
    F0: float = Field(default=50.0, ge=0, le=1e7)
    Mf0: float = Field(default=0.0, ge=0, le=1e7)
    E0: float = Field(default=20.0, ge=0, le=1e7)
    S0: float = Field(default=40.0, ge=0, le=1e7)
    C_TNF0: float = Field(default=10.0, ge=0, le=1e5)
    C_IL10_0: float = Field(default=0.5, ge=0, le=1e5)
    C_PDGF0: float = Field(default=5.0, ge=0, le=1e5)
    C_VEGF0: float = Field(default=2.0, ge=0, le=1e5)
    C_TGFb0: float = Field(default=3.0, ge=0, le=1e5)
    C_MCP1_0: float = Field(default=5.0, ge=0, le=1e5)
    C_IL8_0: float = Field(default=8.0, ge=0, le=1e5)
    rho_collagen0: float = Field(default=0.1, ge=0, le=1e5)
    C_MMP0: float = Field(default=1.0, ge=0, le=1e5)
    rho_fibrin0: float = Field(default=5.0, ge=0, le=1e5)
    D0: float = Field(default=5.0, ge=0, le=100.0)
    O2_0: float = Field(default=80.0, ge=0, le=100.0)

    t_max_hours: float = Field(default=720.0, gt=0, le=8760)
    dt: float = Field(default=0.1, gt=0, le=100.0)

    prp_enabled: bool = False
    pemf_enabled: bool = False
    prp_intensity: float = Field(default=1.0, ge=0, le=2.0)
    pemf_frequency: float = Field(default=50.0, ge=1.0, le=100.0)
    pemf_intensity: float = Field(default=1.0, ge=0, le=2.0)

    random_seed: int | None = 42
    n_trajectories: int = Field(default=1, ge=1, le=1000)
    upload_id: str | None = None

    @model_validator(mode="after")
    def _validate_dt_vs_t_max(self) -> SimulationRequest:
        if self.dt >= self.t_max_hours:
            raise ValueError(f"dt ({self.dt}) must be less than t_max_hours ({self.t_max_hours})")
        return self


class SimulationResponse(BaseModel):
    simulation_id: str
    status: SimulationStatus
    created_at: datetime
    mode: SimulationMode


class SimulationStatusResponse(BaseModel):
    simulation_id: str
    status: SimulationStatus
    progress: float = 0.0
    message: str | None = None
    error_message: str | None = None
    created_at: datetime
    completed_at: datetime | None = None
    params_json: dict[str, Any] | None = None


class UploadResponse(BaseModel):
    upload_id: str
    filename: str
    status: str
    created_at: datetime
    metadata: dict[str, Any] | None = None


class ResultsResponse(BaseModel):
    simulation_id: str
    mode: SimulationMode
    times: list[float]
    variables: dict[str, list[float]]
    metadata: dict[str, Any] = {}


class ExportRequest(BaseModel):
    format: ExportFormat = ExportFormat.PDF
    include_populations: bool = True
    include_cytokines: bool = True
    include_ecm: bool = True
    include_phases: bool = True


class SensitivityRequest(BaseModel):
    simulation_params: SimulationRequest = Field(default_factory=SimulationRequest)
    parameters: list[str] = Field(
        default_factory=lambda: ["r_F", "K_F", "delta_F", "sigma_F"],
        min_length=1,
    )
    method: str = Field(default="sobol", pattern="^(sobol|morris)$")
    n_samples: int = Field(default=256, ge=64, le=4096)


class EstimationRequest(BaseModel):
    upload_id: str
    target_variable: str = "F"
    method: str = Field(default="mcmc", pattern="^(mcmc|optimization)$")
    n_samples: int = Field(default=1000, ge=100, le=50000)


class ValidationRequest(BaseModel):
    """POST /api/v1/analysis/validation."""

    dataset_id: str = Field(
        default="literature-xue2009",
        description="Dataset ID: literature-xue2009, literature-flegg2010, HPA-skin-baseline, GSE28914",
    )
    t_max: float = Field(default=720.0, ge=1.0, le=2000.0)
    dt: float = Field(default=0.1, ge=0.01, le=10.0)


class ValidationResponse(BaseModel):
    """Response for validation endpoint."""

    dataset_id: str
    overall_score: float
    elapsed_seconds: float
    initial_conditions: dict[str, float] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    validation: dict[str, Any] | None = None


class AnalysisResponse(BaseModel):
    analysis_id: str
    analysis_type: AnalysisType
    status: SimulationStatus
    created_at: datetime | None = None
    progress: float = 0.0
    result: dict[str, Any] | None = None


HealthStatus = Literal["ok", "degraded", "unhealthy", "skipped"]


class HealthCheckDetail(BaseModel):
    """Результат одной health-проверки (db / celery / redis / ...)."""

    status: HealthStatus
    latency_ms: float | None = None
    message: str | None = None


class HealthResponse(BaseModel):
    """Агрегированный ответ /api/v1/health."""

    status: Literal["ok", "degraded", "unhealthy"] = "ok"
    version: str
    uptime_seconds: float
    checks: dict[str, HealthCheckDetail] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    code: str | None = None


class ParameterBoundItem(BaseModel):
    """Границы одного параметра для анализа чувствительности."""

    name: str
    lower: float
    upper: float
    nominal: float
    group: str


class ParameterBoundsResponse(BaseModel):
    """GET /api/v1/parameters/bounds."""

    bounds: list[ParameterBoundItem]
    total: int


# ── Analysis Visualization Requests ─────────────────────────────


class SobolVizRequest(BaseModel):
    """POST /api/viz/analysis/sobol."""

    analysis_id: str
    metric: str = Field(default="both", pattern="^(S1|ST|both)$")
    top_n: int | None = Field(default=15, ge=0)
    show_confidence: bool = True
    height: int = Field(default=500, ge=200, le=1200)


class MorrisVizRequest(BaseModel):
    """POST /api/viz/analysis/morris."""

    analysis_id: str
    highlight_influential: bool = True
    threshold_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    show_labels: bool = True
    show_wedge: bool = True
    height: int = Field(default=500, ge=200, le=1200)


class PosteriorVizRequest(BaseModel):
    """POST /api/viz/analysis/posterior."""

    analysis_id: str
    parameters: list[str] | None = None
    layout: str = Field(default="marginals", pattern="^(marginals|corner)$")
    show_ci: bool = True
    show_point_estimate: bool = True
    n_bins: int = Field(default=40, ge=5, le=200)
    height: int = Field(default=600, ge=200, le=1200)


class ConvergenceVizRequest(BaseModel):
    """POST /api/viz/analysis/convergence."""

    analysis_id: str
    metrics: list[str] | None = None
    show_rhat_threshold: bool = True
    height: int = Field(default=500, ge=200, le=1200)
