"""Endpoints for starting and monitoring simulations."""

from __future__ import annotations

import asyncio
import shutil
import uuid as uuid_mod
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from src.api.models.schemas import (
    SimulationMode,
    SimulationRequest,
    SimulationResponse,
    SimulationStatus,
    SimulationStatusResponse,
)
from src.api.services.simulation_service import SimulationService, task_manager
from src.db.models import SimulationRecord
from src.db.session import SessionLocal, get_db

router = APIRouter(prefix="/api/v1", tags=["simulation"])

_WS_POLL_INTERVAL = 0.5


def _get_simulation_record(simulation_id: str) -> SimulationRecord | None:
    """Fetch a simulation record in an isolated DB session."""
    db = SessionLocal()
    try:
        return db.get(SimulationRecord, simulation_id)
    finally:
        db.close()


def _validate_uuid(value: str) -> str:
    try:
        uuid_mod.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid ID format: {value}") from exc
    return value


def _to_status_response(record: SimulationRecord) -> SimulationStatusResponse:
    return SimulationStatusResponse(
        simulation_id=record.id,
        status=SimulationStatus(record.status),
        progress=record.progress or 0.0,
        message=record.message,
        error_message=record.error_message,
        created_at=record.created_at,
        completed_at=record.completed_at,
        params_json=record.params_json,
    )


@router.delete("/simulations", status_code=200)
async def clear_simulations(db: Session = Depends(get_db)) -> dict:
    """Delete all simulation records and their result files."""
    from src.api.config import settings

    # Cancel any running tasks first (both thread-backed and Celery paths)
    for sid in task_manager.active_thread_ids():
        task_manager.cancel(sid)
    for sid in task_manager.active_celery_ids():
        task_manager.cancel_celery(sid)

    deleted = db.query(SimulationRecord).delete()
    db.commit()

    # Remove result files
    results_dir = Path(settings.results_dir)
    removed_files = 0
    if results_dir.exists():
        for f in results_dir.iterdir():
            try:
                if f.is_file():
                    f.unlink()
                    removed_files += 1
                elif f.is_dir():
                    shutil.rmtree(f)
                    removed_files += 1
            except OSError:
                pass

    return {"deleted": deleted, "files_removed": removed_files}


@router.get("/simulations", response_model=list[SimulationStatusResponse])
async def list_simulations(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
    status: SimulationStatus | None = None,
    db: Session = Depends(get_db),
) -> list[SimulationStatusResponse]:
    """List simulations with pagination and optional status filter."""
    query = db.query(SimulationRecord).order_by(SimulationRecord.created_at.desc())
    if status:
        query = query.filter(SimulationRecord.status == status.value)
    records = query.offset(skip).limit(limit).all()
    return [_to_status_response(record) for record in records]


@router.post("/simulate", response_model=SimulationResponse)
async def start_simulation(
    request: SimulationRequest,
    db: Session = Depends(get_db),
) -> SimulationResponse:
    """Start a new simulation in the background."""
    service = SimulationService(db)
    record = service.start_simulation(request)
    return SimulationResponse(
        simulation_id=record.id,
        status=SimulationStatus(record.status),
        created_at=record.created_at,
        mode=SimulationMode(record.mode),
    )


@router.get("/simulate/{simulation_id}", response_model=SimulationStatusResponse)
async def get_simulation_status(
    simulation_id: str,
    db: Session = Depends(get_db),
) -> SimulationStatusResponse:
    """Get the status of a simulation."""
    _validate_uuid(simulation_id)
    service = SimulationService(db)
    record = service.get_status(simulation_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Simulation {simulation_id} not found")
    return _to_status_response(record)


@router.post("/simulate/{simulation_id}/cancel")
async def cancel_simulation(simulation_id: str) -> dict[str, str]:
    """Request cooperative cancellation of a running simulation.

    Best-effort: если активной задачи в памяти нет (например после рестарта API),
    но в DB запись висит в running/pending/cancelling — пометить cancelled и
    вернуть 200, чтобы фронтенд вышел из running-state.
    """
    _validate_uuid(simulation_id)
    if task_manager.is_celery_task(simulation_id):
        if task_manager.cancel_celery(simulation_id):
            return {"status": "cancelling", "simulation_id": simulation_id}
    elif task_manager.cancel(simulation_id):
        return {"status": "cancelling", "simulation_id": simulation_id}

    # Задачи в памяти нет — возможно зомби-запись после рестарта.
    record = _get_simulation_record(simulation_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Simulation {simulation_id} not found")
    if SimulationService.mark_cancelled_in_db(simulation_id, "Cancelled (no active task)"):
        return {"status": "cancelled", "simulation_id": simulation_id}
    raise HTTPException(
        status_code=409,
        detail=f"Simulation already {record.status}",
    )


@router.websocket("/simulate/{simulation_id}/ws")
async def simulation_websocket(websocket: WebSocket, simulation_id: str) -> None:
    """Stream simulation progress and a terminal event over WebSocket."""
    try:
        uuid_mod.UUID(simulation_id)
    except (ValueError, AttributeError):
        await websocket.close(code=1008, reason="Invalid simulation ID format")
        return

    await websocket.accept()

    try:
        while True:
            # Celery-backend: poll progress from result backend
            if task_manager.is_celery_task(simulation_id):
                pct, msg = task_manager.get_celery_progress(simulation_id)
                await websocket.send_json(
                    {"event": "progress", "data": {"percent": round(pct, 1), "message": msg}}
                )
                celery_id = task_manager.get_celery_task_id(simulation_id)
                if celery_id:
                    from src.tasks.celery_app import celery_app

                    state = celery_app.AsyncResult(celery_id).state
                    if state == "SUCCESS":
                        await websocket.send_json(
                            {"event": "complete", "data": {"simulation_id": simulation_id}}
                        )
                        task_manager.cleanup(simulation_id)
                        break
                    if state in ("FAILURE", "REVOKED"):
                        await websocket.send_json(
                            {
                                "event": "failed" if state == "FAILURE" else "cancelled",
                                "data": {
                                    "simulation_id": simulation_id,
                                    "detail": msg or "Task ended",
                                },
                            }
                        )
                        task_manager.cleanup(simulation_id)
                        break
                await asyncio.sleep(_WS_POLL_INTERVAL)
                continue

            # Asyncio-backend: poll in-memory task_manager
            if task_manager.is_active(simulation_id):
                await websocket.send_json(
                    {
                        "event": "progress",
                        "data": {
                            "percent": round(task_manager.get_progress(simulation_id), 1),
                            "message": task_manager.get_message(simulation_id),
                        },
                    }
                )
                await asyncio.sleep(_WS_POLL_INTERVAL)
                continue

            record = _get_simulation_record(simulation_id)

            if record is None:
                await websocket.send_json(
                    {
                        "event": "not_found",
                        "data": {"detail": f"Simulation {simulation_id} not found"},
                    }
                )
                break

            await websocket.send_json(
                {
                    "event": "progress",
                    "data": {
                        "percent": round(record.progress or 0.0, 1),
                        "message": record.message,
                    },
                }
            )

            if record.status == "completed":
                await websocket.send_json(
                    {"event": "complete", "data": {"simulation_id": simulation_id}}
                )
            elif record.status == "cancelled":
                await websocket.send_json(
                    {"event": "cancelled", "data": {"simulation_id": simulation_id}}
                )
            else:
                await websocket.send_json(
                    {
                        "event": "failed",
                        "data": {
                            "simulation_id": simulation_id,
                            "detail": record.error_message or record.message or "Simulation failed",
                        },
                    }
                )
            break
    except WebSocketDisconnect:
        return
    except (ConnectionError, RuntimeError):
        try:
            await websocket.close()
        except Exception:
            pass
