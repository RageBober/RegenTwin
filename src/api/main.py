"""FastAPI application factory."""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from src.api.config import settings
from src.api.services.file_service import FileSizeExceededError
from src.db.session import create_tables
from src.utils.logging import setup_logging


def _reconcile_orphans() -> None:
    """Mark any DB records left in non-terminal states from a previous process as cancelled.

    The in-memory ``task_manager`` does not survive an API restart, so any
    ``running``/``pending``/``cancelling`` simulation or analysis row in the DB
    after startup refers to a thread that no longer exists.
    """
    from datetime import datetime

    from src.db.models import AnalysisRecord, SimulationRecord
    from src.db.session import SessionLocal

    db = SessionLocal()
    try:
        now = datetime.now(UTC)
        message = "Server restarted before completion"
        stale_states = ("running", "pending", "cancelling")

        sim_stale = (
            db.query(SimulationRecord).filter(SimulationRecord.status.in_(stale_states)).all()
        )
        for record in sim_stale:
            record.status = "cancelled"
            record.message = message
            record.completed_at = now

        analysis_stale = (
            db.query(AnalysisRecord).filter(AnalysisRecord.status.in_(stale_states)).all()
        )
        for record in analysis_stale:
            record.status = "cancelled"
            record.completed_at = now

        if sim_stale or analysis_stale:
            db.commit()
            logger.info(
                f"Reconciled {len(sim_stale)} orphan simulation(s) and "
                f"{len(analysis_stale)} orphan analysis record(s) on startup"
            )
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup: настроить логи, создать таблицы. Shutdown: cleanup running tasks."""
    setup_logging()
    create_tables()
    _reconcile_orphans()
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.results_dir).mkdir(parents=True, exist_ok=True)
    logger.info("RegenTwin API started")
    yield
    # Graceful shutdown: отменить все запущенные симуляции и анализы
    from src.api.services.analysis_service import analysis_task_manager
    from src.api.services.simulation_service import task_manager

    def _active_ids(mgr) -> list[str]:
        with mgr._lock:
            # Simulation manager keeps threads in ``_threads``; analysis manager
            # still uses the legacy ``_tasks`` mapping. Support both layouts.
            bucket = getattr(mgr, "_threads", None) or getattr(mgr, "_tasks", {})
            return list(bucket.keys())

    for mgr_name, mgr in [("simulation", task_manager), ("analysis", analysis_task_manager)]:
        for sid in _active_ids(mgr):
            mgr.cancel(sid)
            logger.info(f"Cancelled {mgr_name} task {sid} on shutdown")

    # Abort Celery tasks if enabled
    if settings.use_celery:
        from celery.contrib.abortable import AbortableAsyncResult

        from src.tasks.celery_app import celery_app

        for mgr_name, mgr in [("simulation", task_manager), ("analysis", analysis_task_manager)]:
            with mgr._lock:
                celery_ids = list(mgr._celery_task_ids.items())
            for sid, celery_id in celery_ids:
                AbortableAsyncResult(celery_id, app=celery_app).abort()
                logger.info(f"Aborted Celery {mgr_name} task {sid} on shutdown")

    logger.info("RegenTwin API shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="RegenTwin API",
        version=settings.app_version,
        description="Multiscale tissue regeneration modeling API",
        lifespan=lifespan,
    )

    # CORS — явный whitelist методов и заголовков
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "Accept"],
    )

    # Request logging
    @app.middleware("http")
    async def log_requests(request: Request, call_next):  # type: ignore[no-untyped-def]
        start = time.time()
        response = await call_next(request)
        duration = time.time() - start
        path = request.url.path
        message = f"{request.method} {path} -> {response.status_code} ({duration:.2f}s)"
        if path.startswith("/api/v1/health") or (
            request.method == "GET"
            and (path.startswith("/api/v1/simulate/") or path.startswith("/api/v1/analysis/"))
        ):
            logger.debug(message)
        else:
            logger.info(message)
        return response

    # ── Exception handlers ────────────────────────────────────────────

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"error": "validation_error", "detail": str(exc)},
        )

    @app.exception_handler(FileNotFoundError)
    async def not_found_handler(request: Request, exc: FileNotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={"error": "not_found", "detail": str(exc)},
        )

    @app.exception_handler(FileSizeExceededError)
    async def file_size_handler(request: Request, exc: FileSizeExceededError) -> JSONResponse:
        return JSONResponse(
            status_code=413,
            content={"error": "file_too_large", "detail": str(exc)},
        )

    @app.exception_handler(Exception)
    async def generic_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(f"Unhandled error on {request.method} {request.url.path}: {exc}")
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "detail": "An unexpected error occurred"},
        )

    # ── Роутеры ───────────────────────────────────────────────────────

    from src.api.routes.analysis import router as analysis_router
    from src.api.routes.health import router as health_router
    from src.api.routes.parameters import router as parameters_router
    from src.api.routes.results import router as results_router
    from src.api.routes.simulate import router as simulate_router
    from src.api.routes.spatial import router as spatial_router
    from src.api.routes.upload import router as upload_router
    from src.api.routes.visualization import HAS_PLOTLY
    from src.api.routes.visualization import router as viz_router

    app.include_router(health_router)
    app.include_router(upload_router)
    app.include_router(simulate_router)
    app.include_router(results_router)
    app.include_router(analysis_router)
    app.include_router(parameters_router)
    if HAS_PLOTLY:
        app.include_router(viz_router)
    else:
        logger.warning("Plotly not installed — visualization API disabled")
    app.include_router(spatial_router)

    from fastapi.responses import RedirectResponse

    @app.get("/", include_in_schema=False)
    async def root_redirect():
        return RedirectResponse(url="/docs")

    return app


app = create_app()
