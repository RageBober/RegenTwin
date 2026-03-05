"""Endpoints для анализа чувствительности и параметрической идентификации."""

from __future__ import annotations

import uuid as _uuid_mod

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.models.schemas import (
    AnalysisResponse,
    AnalysisType,
    EstimationRequest,
    SensitivityRequest,
    SimulationStatus,
)
from src.api.services.analysis_service import AnalysisService
from src.db.session import get_db

router = APIRouter(prefix="/api/v1", tags=["analysis"])


def _validate_uuid(value: str) -> str:
    try:
        _uuid_mod.UUID(value)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail=f"Invalid ID format: {value}")
    return value


@router.post("/analysis/sensitivity", response_model=AnalysisResponse)
async def run_sensitivity(
    request: SensitivityRequest,
    db: Session = Depends(get_db),
) -> AnalysisResponse:
    """Запуск анализа чувствительности (Sobol/Morris)."""
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
    """Запуск параметрической идентификации (MCMC/optimization)."""
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
    """Статус и результаты анализа."""
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
