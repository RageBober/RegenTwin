"""Сервис для обработки загруженных файлов."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import UploadFile
from loguru import logger
from sqlalchemy.orm import Session

from src.api.config import settings
from src.db.models import UploadRecord


class FileSizeExceededError(Exception):
    """Размер загружаемого файла превышает допустимый лимит."""


class FileService:
    """Сохранение загруженных файлов и парсинг FCS-метаданных."""

    def __init__(self, db: Session) -> None:
        self._db = db
        self._upload_dir = Path(settings.upload_dir)

    def save_upload(self, file: UploadFile, description: str | None = None) -> UploadRecord:
        """Сохранить файл на диск и создать запись в БД.

        Файл сначала записывается во временную директорию, и перемещается
        в целевую только после успешного commit в БД — для атомарности.
        """
        upload_id = str(uuid.uuid4())

        # Санитизация имени файла: uuid + оригинальный суффикс
        original_name = file.filename or "unknown"
        safe_suffix = Path(original_name).suffix.lower()
        safe_filename = f"{upload_id}{safe_suffix}"

        dest_dir = self._upload_dir / upload_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / safe_filename

        # Потоковая запись во временный файл с проверкой размера
        max_bytes = settings.max_upload_bytes
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / safe_filename
            total_written = 0

            with open(tmp_path, "wb") as f:
                while chunk := file.file.read(8192):
                    total_written += len(chunk)
                    if total_written > max_bytes:
                        raise FileSizeExceededError(
                            f"File exceeds maximum size of {max_bytes // (1024 * 1024)} MB"
                        )
                    f.write(chunk)

            # Создать запись в БД до перемещения файла
            record = UploadRecord(
                id=upload_id,
                filename=original_name,
                file_path=str(dest_path),
                description=description,
            )
            self._db.add(record)
            self._db.commit()

            # Только после успешного commit — перемещаем файл на место
            shutil.move(str(tmp_path), str(dest_path))

        logger.info(f"File saved: {original_name} -> {dest_path} ({total_written} bytes)")

        # Попытка парсинга FCS-метаданных
        self._try_parse_fcs(record)
        return record

    def _try_parse_fcs(self, record: UploadRecord) -> None:
        """Попытка парсинга FCS-файла для извлечения метаданных."""
        if not record.filename.lower().endswith(".fcs"):
            record.status = "ready"
            self._db.commit()
            return

        try:
            from src.data.fcs_parser import FCSLoader

            loader = FCSLoader()
            loader.load(record.file_path)
            meta = loader.get_metadata()

            record.metadata_json = {
                "n_events": meta.n_events,
                "n_channels": meta.n_channels,
                "channels": meta.channels,
                "cytometer": meta.cytometer,
                "fcs_version": meta.fcs_version,
            }
            record.status = "ready"
        except (OSError, ValueError, RuntimeError, KeyError, AttributeError) as exc:
            logger.warning(f"FCS parsing failed for {record.filename}: {exc}")
            record.status = "failed"
            record.metadata_json = {"error": str(exc)}

        self._db.commit()

    def get_upload(self, upload_id: str) -> UploadRecord | None:
        """Получить запись о загрузке."""
        return self._db.get(UploadRecord, upload_id)
