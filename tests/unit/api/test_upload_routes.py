"""Тесты для upload endpoints."""

from __future__ import annotations

import io
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.routes.upload import router
from src.db.models import Base
from src.db.session import get_db

_UUID_NONEXISTENT = "00000000-0000-0000-0000-000000000000"


def _setup():  # type: ignore[no-untyped-def]
    """Настройка тестового клиента с in-memory SQLite + StaticPool."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine)

    def override_get_db():  # type: ignore[no-untyped-def]
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_db] = override_get_db

    return TestClient(app), TestSession


class TestUploadEndpoint:
    @patch("src.api.services.file_service.FileService._try_parse_fcs")
    def test_upload_file(self, mock_parse, tmp_path) -> None:  # type: ignore[no-untyped-def]
        mock_parse.return_value = None
        client, _ = _setup()

        file_content = b"FCS mock data content"
        with patch("src.api.services.file_service.settings") as mock_settings:
            mock_settings.upload_dir = str(tmp_path)
            mock_settings.max_upload_bytes = 500 * 1024 * 1024
            resp = client.post(
                "/api/v1/upload",
                files={"file": ("test.fcs", io.BytesIO(file_content), "application/octet-stream")},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["filename"] == "test.fcs"
        assert "upload_id" in data

    def test_upload_missing_file(self) -> None:
        client, _ = _setup()
        resp = client.post("/api/v1/upload")
        assert resp.status_code == 422


class TestGetUploadStatus:
    def test_not_found(self) -> None:
        client, _ = _setup()
        resp = client.get(f"/api/v1/upload/{_UUID_NONEXISTENT}")
        assert resp.status_code == 404

    @patch("src.api.services.file_service.FileService._try_parse_fcs")
    def test_get_existing(self, mock_parse, tmp_path) -> None:  # type: ignore[no-untyped-def]
        mock_parse.return_value = None
        client, _ = _setup()

        with patch("src.api.services.file_service.settings") as mock_settings:
            mock_settings.upload_dir = str(tmp_path)
            mock_settings.max_upload_bytes = 500 * 1024 * 1024
            resp = client.post(
                "/api/v1/upload",
                files={"file": ("data.csv", io.BytesIO(b"a,b,c"), "text/csv")},
            )
        upload_id = resp.json()["upload_id"]

        resp = client.get(f"/api/v1/upload/{upload_id}")
        assert resp.status_code == 200
        assert resp.json()["upload_id"] == upload_id
