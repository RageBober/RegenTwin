"""Pydantic v2 схемы для API запросов и ответов."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator


# ── Enums ─────────────────────────────────────────────────────────────


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


# ── Simulation ────────────────────────────────────────────────────────


class SimulationRequest(BaseModel):
    """POST /api/v1/simulate — запуск симуляции."""

    mode: SimulationMode = SimulationMode.EXTENDED

    # Начальные условия (те же поля что в visualization.SimulationParams)
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

    # Время
    t_max_hours: float = Field(default=720.0, gt=0, le=8760)
    dt: float = Field(default=0.1, gt=0, le=100.0)

    # Терапия
    prp_enabled: bool = False
    pemf_enabled: bool = False
    prp_intensity: float = Field(default=1.0, ge=0, le=2.0)
    pemf_frequency: float = Field(default=50.0, ge=1.0, le=100.0)
    pemf_intensity: float = Field(default=1.0, ge=0, le=2.0)

    random_seed: int | None = 42

    # Monte Carlo
    n_trajectories: int = Field(default=1, ge=1, le=1000)

    # Привязка к загруженным FCS-данным
    upload_id: str | None = None

    @model_validator(mode="after")
    def _validate_dt_vs_t_max(self) -> "SimulationRequest":
        if self.dt >= self.t_max_hours:
            raise ValueError(
                f"dt ({self.dt}) must be less than t_max_hours ({self.t_max_hours})"
            )
        return self


class SimulationResponse(BaseModel):
    """Ответ на POST /api/v1/simulate."""

    simulation_id: str
    status: SimulationStatus
    created_at: datetime
    mode: SimulationMode


class SimulationStatusResponse(BaseModel):
    """GET /api/v1/simulate/{id} — статус симуляции."""

    simulation_id: str
    status: SimulationStatus
    progress: float = 0.0
    message: str | None = None
    created_at: datetime
    completed_at: datetime | None = None
    params_json: dict | None = None


# ── Upload ────────────────────────────────────────────────────────────


class UploadResponse(BaseModel):
    """Ответ на POST/GET /api/v1/upload."""

    upload_id: str
    filename: str
    status: str
    created_at: datetime
    metadata: dict | None = None


# ── Results & Export ──────────────────────────────────────────────────


class ResultsResponse(BaseModel):
    """GET /api/v1/results/{id} — данные траектории."""

    simulation_id: str
    mode: SimulationMode
    times: list[float]
    variables: dict[str, list[float]]
    metadata: dict[str, str | float] = {}


class ExportRequest(BaseModel):
    """POST /api/v1/export/{id} — параметры экспорта."""

    format: ExportFormat = ExportFormat.PDF
    include_populations: bool = True
    include_cytokines: bool = True
    include_ecm: bool = True
    include_phases: bool = True


# ── Analysis ──────────────────────────────────────────────────────────


class SensitivityRequest(BaseModel):
    """POST /api/v1/analysis/sensitivity."""

    simulation_params: SimulationRequest = Field(default_factory=SimulationRequest)
    parameters: list[str] = Field(
        default_factory=lambda: ["r_F", "K_F", "delta_F", "sigma_F"],
        min_length=1,
    )
    method: str = Field(default="sobol", pattern="^(sobol|morris)$")
    n_samples: int = Field(default=256, ge=64, le=4096)


class EstimationRequest(BaseModel):
    """POST /api/v1/analysis/estimation."""

    upload_id: str
    target_variable: str = "F"
    method: str = Field(default="mcmc", pattern="^(mcmc|optimization)$")
    n_samples: int = Field(default=1000, ge=100, le=50000)


class AnalysisResponse(BaseModel):
    """Ответ на POST/GET /api/v1/analysis."""

    analysis_id: str
    analysis_type: AnalysisType
    status: SimulationStatus
    created_at: datetime
    progress: float = 0.0
    result: dict | None = None


# ── Health ────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    uptime_seconds: float


# ── Errors ────────────────────────────────────────────────────────────


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    code: str | None = None
