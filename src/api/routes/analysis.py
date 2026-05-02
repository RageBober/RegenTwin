"""Endpoints for sensitivity analysis and parameter estimation."""

from __future__ import annotations

import uuid as uuid_mod

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.models.schemas import (
    AnalysisResponse,
    AnalysisType,
    EstimationRequest,
    SensitivityRequest,
    SimulationStatus,
    ValidationRequest,
    ValidationResponse,
)
from src.api.services.analysis_service import AnalysisService
from src.db.session import get_db

router = APIRouter(prefix="/api/v1", tags=["analysis"])


def _validate_uuid(value: str) -> str:
    try:
        uuid_mod.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid ID format: {value}") from exc
    return value


@router.post("/analysis/sensitivity", response_model=AnalysisResponse)
async def run_sensitivity(
    request: SensitivityRequest,
    db: Session = Depends(get_db),
) -> AnalysisResponse:
    """Run Sobol sensitivity analysis in the background."""
    service = AnalysisService(db)
    record = await service.run_sensitivity(request)
    return AnalysisResponse(
        analysis_id=record.id,
        analysis_type=AnalysisType.SENSITIVITY,
        status=SimulationStatus(record.status),
        created_at=record.created_at,
    )


@router.post("/analysis/estimation", response_model=AnalysisResponse)
async def run_estimation(
    request: EstimationRequest,
    db: Session = Depends(get_db),
) -> AnalysisResponse:
    """Run parameter estimation (MCMC / optimization) in the background."""
    service = AnalysisService(db)
    record = await service.run_estimation(request)
    return AnalysisResponse(
        analysis_id=record.id,
        analysis_type=AnalysisType.ESTIMATION,
        status=SimulationStatus(record.status),
        created_at=record.created_at,
    )


@router.get("/analysis/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis_status(
    analysis_id: str,
    db: Session = Depends(get_db),
) -> AnalysisResponse:
    """Return analysis status and results."""
    _validate_uuid(analysis_id)
    service = AnalysisService(db)
    record = service.get_analysis(analysis_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id} not found")
    return AnalysisResponse(
        analysis_id=record.id,
        analysis_type=AnalysisType(record.analysis_type),
        status=SimulationStatus(record.status),
        created_at=record.created_at,
        progress=record.progress or 0.0,
        result=record.result_json,
    )


@router.post("/analysis/{analysis_id}/cancel", response_model=AnalysisResponse)
async def cancel_analysis(
    analysis_id: str,
    db: Session = Depends(get_db),
) -> AnalysisResponse:
    """Cancel a running analysis."""
    _validate_uuid(analysis_id)
    service = AnalysisService(db)
    record = service.cancel_analysis(analysis_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id} not found")
    return AnalysisResponse(
        analysis_id=record.id,
        analysis_type=AnalysisType(record.analysis_type),
        status=SimulationStatus(record.status),
        created_at=record.created_at,
        progress=record.progress or 0.0,
        result=record.result_json,
    )


@router.post("/analysis/validation", response_model=ValidationResponse)
async def run_validation_endpoint(
    request: ValidationRequest,
) -> ValidationResponse:
    """Run model validation against reference datasets.

    Synchronous — typically completes in 2-5 seconds.
    """
    from src.analysis.validation_pipeline import PipelineConfig, ValidationPipeline

    config = PipelineConfig(
        t_max=request.t_max,
        dt=request.dt,
    )
    pipeline = ValidationPipeline(config=config)

    try:
        report = pipeline.run(request.dataset_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    result = ValidationResponse(
        dataset_id=report.dataset_id,
        overall_score=report.overall_score,
        elapsed_seconds=report.elapsed_seconds,
        initial_conditions=report.initial_conditions,
        errors=report.errors,
    )
    if report.validation_result is not None:
        result.validation = report.validation_result.get_summary()
    return result
