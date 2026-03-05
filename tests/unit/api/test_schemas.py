"""Тесты для Pydantic schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.models.schemas import (
    AnalysisResponse,
    AnalysisType,
    EstimationRequest,
    ExportFormat,
    ExportRequest,
    HealthResponse,
    ResultsResponse,
    SensitivityRequest,
    SimulationMode,
    SimulationRequest,
    SimulationResponse,
    SimulationStatus,
    SimulationStatusResponse,
    UploadResponse,
)


class TestSimulationRequest:
    def test_defaults(self) -> None:
        req = SimulationRequest()
        assert req.mode == SimulationMode.EXTENDED
        assert req.P0 == 500.0
        assert req.dt == 0.1
        assert req.t_max_hours == 720.0
        assert req.n_trajectories == 1
        assert req.upload_id is None

    def test_mvp_mode(self) -> None:
        req = SimulationRequest(mode="mvp")
        assert req.mode == SimulationMode.MVP

    def test_negative_initial_condition_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SimulationRequest(P0=-1.0)

    def test_zero_dt_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SimulationRequest(dt=0)

    def test_trajectories_bounds(self) -> None:
        with pytest.raises(ValidationError):
            SimulationRequest(n_trajectories=0)
        with pytest.raises(ValidationError):
            SimulationRequest(n_trajectories=1001)

    def test_therapy_fields(self) -> None:
        req = SimulationRequest(prp_enabled=True, prp_intensity=1.5)
        assert req.prp_enabled is True
        assert req.prp_intensity == 1.5

    def test_prp_intensity_bounds(self) -> None:
        with pytest.raises(ValidationError):
            SimulationRequest(prp_intensity=3.0)


class TestExportRequest:
    def test_defaults(self) -> None:
        req = ExportRequest()
        assert req.format == ExportFormat.PDF
        assert req.include_populations is True

    def test_csv_format(self) -> None:
        req = ExportRequest(format="csv")
        assert req.format == ExportFormat.CSV


class TestSensitivityRequest:
    def test_defaults(self) -> None:
        req = SensitivityRequest()
        assert req.method == "sobol"
        assert req.n_samples == 256
        assert len(req.parameters) > 0

    def test_invalid_method_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SensitivityRequest(method="invalid")

    def test_samples_bounds(self) -> None:
        with pytest.raises(ValidationError):
            SensitivityRequest(n_samples=10)


class TestEstimationRequest:
    def test_required_upload_id(self) -> None:
        with pytest.raises(ValidationError):
            EstimationRequest()

    def test_valid(self) -> None:
        req = EstimationRequest(upload_id="abc-123")
        assert req.target_variable == "F"
        assert req.method == "mcmc"


class TestResponseModels:
    def test_health_response(self) -> None:
        resp = HealthResponse(status="ok", version="0.1.0", uptime_seconds=1.5)
        assert resp.status == "ok"

    def test_simulation_status_enum(self) -> None:
        assert SimulationStatus.PENDING == "pending"
        assert SimulationStatus.COMPLETED == "completed"

    def test_analysis_type_enum(self) -> None:
        assert AnalysisType.SENSITIVITY == "sensitivity"
        assert AnalysisType.ESTIMATION == "estimation"
