"""Endpoints для запуска и управления симуляциями."""

from __future__ import annotations

import asyncio
import uuid as _uuid_mod

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
from src.db.session import get_db

router = APIRouter(prefix="/api/v1", tags=["simulation"])

_WS_POLL_INTERVAL = 0.5  # seconds
_WS_IDLE_TIMEOUT = 60.0  # seconds — закрыть WS если задача не найдена


def _validate_uuid(value: str) -> str:
    try:
        _uuid_mod.UUID(value)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail=f"Invalid ID format: {value}")
    return value


@router.get("/simulations", response_model=list[SimulationStatusResponse])
async def list_simulations(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
    status: SimulationStatus | None = None,
    db: Session = Depends(get_db),
) -> list[SimulationStatusResponse]:
    """Список всех симуляций с пагинацией и фильтрацией по статусу."""
    query = db.query(SimulationRecord).order_by(SimulationRecord.created_at.desc())
    if status:
        query = query.filter(SimulationRecord.status == status.value)
    records = query.offset(skip).limit(limit).all()
    return [
        SimulationStatusResponse(
            simulation_id=r.id,
            status=SimulationStatus(r.status),
            progress=r.progress or 0.0,
            message=r.message,
            created_at=r.created_at,
            completed_at=r.completed_at,
            params_json=r.params_json,
        )
        for r in records
    ]


@router.post("/simulate", response_model=SimulationResponse)
async def start_simulation(
    request: SimulationRequest,
    db: Session = Depends(get_db),
) -> SimulationResponse:
    """Запуск новой симуляции в фоновом режиме."""
    service = SimulationService(db)
    record = await service.start_simulation(request)
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
    """Статус симуляции."""
    _validate_uuid(simulation_id)
    service = SimulationService(db)
    record = service.get_status(simulation_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Simulation {simulation_id} not found")
    return SimulationStatusResponse(
        simulation_id=record.id,
        status=SimulationStatus(record.status),
        progress=record.progress or 0.0,
        message=record.message,
        created_at=record.created_at,
        completed_at=record.completed_at,
        params_json=record.params_json,
    )


@router.post("/simulate/{simulation_id}/cancel")
async def cancel_simulation(simulation_id: str) -> dict:
    """Отмена запущенной симуляции."""
    _validate_uuid(simulation_id)
    if task_manager.cancel(simulation_id):
        return {"status": "cancelled", "simulation_id": simulation_id}
    raise HTTPException(status_code=404, detail=f"Active simulation {simulation_id} not found")


@router.websocket("/simulate/{simulation_id}/ws")
async def simulation_websocket(websocket: WebSocket, simulation_id: str) -> None:
    """WebSocket для отслеживания прогресса симуляции в реальном времени."""
    # Валидация UUID до accept
    try:
        _uuid_mod.UUID(simulation_id)
    except (ValueError, AttributeError):
        await websocket.close(code=1008, reason="Invalid simulation ID format")
        return

    await websocket.accept()

    idle_elapsed = 0.0
    try:
        while True:
            progress = task_manager.get_progress(simulation_id)
            message = task_manager.get_message(simulation_id)
            is_active = task_manager.is_active(simulation_id)

            await websocket.send_json({
                "event": "progress",
                "data": {
                    "percent": round(progress, 1),
                    "message": message,
                },
            })

            if not is_active:
                # Задача завершена — отправить финальное событие
                await websocket.send_json({
                    "event": "complete" if progress >= 100 else "stopped",
                    "data": {"simulation_id": simulation_id},
                })
                break

            await asyncio.sleep(_WS_POLL_INTERVAL)

            # Защита от бесконечного polling несуществующей задачи
            if progress == 0.0 and not is_active:
                idle_elapsed += _WS_POLL_INTERVAL
                if idle_elapsed >= _WS_IDLE_TIMEOUT:
                    await websocket.send_json({
                        "event": "error",
                        "data": {"detail": "Simulation not found or not started"},
                    })
                    break
            else:
                idle_elapsed = 0.0
    except WebSocketDisconnect:
        pass
    except (ConnectionError, RuntimeError):
        # Клиент отключился или соединение сброшено
        try:
            await websocket.close()
        except Exception:
            pass
