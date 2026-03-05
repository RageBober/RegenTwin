"""FastAPI application factory."""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from src.api.config import settings
from src.api.services.file_service import FileSizeExceededError
from src.db.session import create_tables


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup: создать таблицы. Shutdown: cleanup running tasks."""
    create_tables()
    logger.info("RegenTwin API started")
    yield
    # Graceful shutdown: отменить все запущенные симуляции и анализы
    from src.api.services.simulation_service import task_manager
    from src.api.services.analysis_service import analysis_task_manager

    for mgr_name, mgr in [("simulation", task_manager), ("analysis", analysis_task_manager)]:
        with mgr._lock:
            active_ids = list(mgr._tasks.keys())
        for sid in active_ids:
            mgr.cancel(sid)
            logger.info(f"Cancelled {mgr_name} task {sid} on shutdown")

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
        logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({duration:.3f}s)")
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
    from src.api.routes.results import router as results_router
    from src.api.routes.simulate import router as simulate_router
    from src.api.routes.upload import router as upload_router
    from src.api.routes.spatial import router as spatial_router
    from src.api.routes.visualization import router as viz_router

    app.include_router(health_router)
    app.include_router(upload_router)
    app.include_router(simulate_router)
    app.include_router(results_router)
    app.include_router(analysis_router)
    app.include_router(viz_router)  # backward compat at /api/viz
    app.include_router(spatial_router)

    return app


app = create_app()
