"""Health check endpoint."""

from __future__ import annotations

import time

from fastapi import APIRouter

from src.api.config import settings
from src.api.models.schemas import HealthResponse

router = APIRouter(tags=["health"])

_START_TIME = time.time()


@router.get("/api/v1/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        uptime_seconds=round(time.time() - _START_TIME, 2),
    )
