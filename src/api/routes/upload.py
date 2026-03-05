"""Endpoints для загрузки файлов."""

from __future__ import annotations

import uuid as _uuid_mod

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from src.api.models.schemas import UploadResponse
from src.api.services.file_service import FileSizeExceededError, FileService
from src.db.session import get_db

router = APIRouter(prefix="/api/v1", tags=["upload"])


def _validate_uuid(value: str) -> str:
    """Проверить что строка является валидным UUID."""
    try:
        _uuid_mod.UUID(value)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail=f"Invalid ID format: {value}")
    return value


def _to_response(record) -> UploadResponse:  # type: ignore[no-untyped-def]
    return UploadResponse(
        upload_id=record.id,
        filename=record.filename,
        status=record.status,
        created_at=record.created_at,
        metadata=record.metadata_json,
    )


@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    description: str | None = None,
    db: Session = Depends(get_db),
) -> UploadResponse:
    """Загрузка .fcs файла (или изображения)."""
    service = FileService(db)
    try:
        record = service.save_upload(file, description)
    except FileSizeExceededError as exc:
        raise HTTPException(status_code=413, detail=str(exc))
    return _to_response(record)


@router.get("/upload/{upload_id}", response_model=UploadResponse)
async def get_upload_status(
    upload_id: str,
    db: Session = Depends(get_db),
) -> UploadResponse:
    """Статус загруженного файла."""
    _validate_uuid(upload_id)
    service = FileService(db)
    record = service.get_upload(upload_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Upload {upload_id} not found")
    return _to_response(record)
