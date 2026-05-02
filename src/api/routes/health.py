"""Health check endpoints.

Три уровня:
- `/api/v1/health` — агрегированная информация (200, статус может быть degraded/unhealthy)
- `/api/v1/health/live` — liveness probe (200, если процесс жив)
- `/api/v1/health/ready` — readiness probe (503, если БД недоступна)
"""

from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.api.config import settings
from src.api.models.schemas import HealthCheckDetail, HealthResponse
from src.api.services.health_checks import (
    aggregate_status,
    check_celery,
    check_db,
    check_redis,
)

router = APIRouter(tags=["health"])

_START_TIME = time.time()


def _uptime() -> float:
    return round(time.time() - _START_TIME, 2)


async def _gather_checks() -> dict[str, HealthCheckDetail]:
    db_check, celery_check, redis_check = await asyncio.gather(
        check_db(), check_celery(), check_redis()
    )
    return {"db": db_check, "celery": celery_check, "redis": redis_check}


@router.get("/api/v1/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Агрегированный health-статус всех зависимостей."""
    checks = await _gather_checks()
    return HealthResponse(
        status=aggregate_status(checks),  # type: ignore[arg-type]
        version=settings.app_version,
        uptime_seconds=_uptime(),
        checks=checks,
    )


@router.get("/api/v1/health/live", response_model=HealthResponse)
async def health_live() -> HealthResponse:
    """Liveness probe: процесс жив. Без проверок зависимостей."""
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        uptime_seconds=_uptime(),
        checks={},
    )


@router.get("/api/v1/health/ready")
async def health_ready() -> JSONResponse:
    """Readiness probe: 200 если БД доступна, иначе 503."""
    checks = await _gather_checks()
    overall = aggregate_status(checks)
    db_ok = checks["db"].status == "ok"
    payload = HealthResponse(
        status=overall,  # type: ignore[arg-type]
        version=settings.app_version,
        uptime_seconds=_uptime(),
        checks=checks,
    )
    status_code = 200 if db_ok else 503
    return JSONResponse(status_code=status_code, content=payload.model_dump())
