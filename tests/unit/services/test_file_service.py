"""Тесты для FileService."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

from fastapi import UploadFile
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api.services.file_service import FileService
from src.db.models import Base


def _make_session(tmp_path):  # type: ignore[no-untyped-def]
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _mock_settings(tmp_path):  # type: ignore[no-untyped-def]
    """Создать mock settings с upload_dir и max_upload_bytes."""
    class _FakeSettings:
        upload_dir = str(tmp_path)
        max_upload_bytes = 500 * 1024 * 1024  # 500 MB
    return _FakeSettings()


class TestFileServiceSave:
    def test_save_non_fcs_file(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        session = _make_session(tmp_path)

        upload = UploadFile(filename="data.csv", file=io.BytesIO(b"col1,col2\n1,2"))

        with patch("src.api.services.file_service.settings", _mock_settings(tmp_path)):
            service = FileService(session)
            record = service.save_upload(upload)

        assert record.filename == "data.csv"
        assert record.status == "ready"
        # Файл сохраняется с UUID-именем + суффиксом
        saved_dir = tmp_path / record.id
        assert saved_dir.exists()
        saved_files = list(saved_dir.iterdir())
        assert len(saved_files) == 1
        assert saved_files[0].suffix == ".csv"

    def test_save_fcs_file_with_parse_error(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        session = _make_session(tmp_path)

        upload = UploadFile(filename="sample.fcs", file=io.BytesIO(b"not a real fcs"))

        with patch("src.api.services.file_service.settings", _mock_settings(tmp_path)):
            service = FileService(session)
            record = service.save_upload(upload)

        assert record.filename == "sample.fcs"
        assert record.status == "failed"
        assert "error" in record.metadata_json


class TestFileServiceGet:
    def test_get_nonexistent(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        session = _make_session(tmp_path)
        service = FileService(session)
        assert service.get_upload("nonexistent") is None

    def test_get_existing(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        session = _make_session(tmp_path)
        upload = UploadFile(filename="test.csv", file=io.BytesIO(b"data"))

        with patch("src.api.services.file_service.settings", _mock_settings(tmp_path)):
            service = FileService(session)
            record = service.save_upload(upload)

        found = service.get_upload(record.id)
        assert found is not None
        assert found.filename == "test.csv"
