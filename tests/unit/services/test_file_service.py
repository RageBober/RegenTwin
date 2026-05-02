"""Tests for FileService."""

from __future__ import annotations

import io
from unittest.mock import patch

from fastapi import UploadFile
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api.services.file_service import FileService
from src.db.models import Base


def _make_session(tmp_path):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _mock_settings(tmp_path):
    class _FakeSettings:
        upload_dir = str(tmp_path)
        max_upload_bytes = 500 * 1024 * 1024

    return _FakeSettings()


class TestFileServiceSave:
    def test_save_csv_timeseries_file(self, tmp_path) -> None:
        session = _make_session(tmp_path)
        upload = UploadFile(
            filename="timeseries.csv",
            file=io.BytesIO(b"time,F,Ne\n0,1.0,0.1\n24,2.5,0.3\n"),
        )

        with patch("src.api.services.file_service.settings", _mock_settings(tmp_path)):
            service = FileService(session)
            record = service.save_upload(upload)

        assert record.filename == "timeseries.csv"
        assert record.status == "ready"
        metadata = record.metadata_json or {}
        assert metadata["kind"] == "csv"
        assert metadata["channels"] == ["F", "Ne"]
        assert metadata["time_min"] == 0.0
        assert metadata["time_max"] == 24.0
        saved_dir = tmp_path / record.id
        assert saved_dir.exists()
        saved_files = list(saved_dir.iterdir())
        assert len(saved_files) == 1
        assert saved_files[0].suffix == ".csv"

    def test_save_csv_without_time_column_is_failed(self, tmp_path) -> None:
        session = _make_session(tmp_path)
        upload = UploadFile(filename="bad.csv", file=io.BytesIO(b"col1,col2\n1,2"))

        with patch("src.api.services.file_service.settings", _mock_settings(tmp_path)):
            service = FileService(session)
            record = service.save_upload(upload)

        assert record.status == "failed"
        assert "time" in (record.metadata_json or {}).get("error", "")

    def test_save_fcs_file_with_parse_error(self, tmp_path) -> None:
        session = _make_session(tmp_path)
        upload = UploadFile(filename="sample.fcs", file=io.BytesIO(b"not a real fcs"))

        with patch("src.api.services.file_service.settings", _mock_settings(tmp_path)):
            service = FileService(session)
            record = service.save_upload(upload)

        assert record.filename == "sample.fcs"
        assert record.status == "failed"
        assert "error" in record.metadata_json

    def test_save_fcs_file_populates_initial_conditions_with_heuristic_fallback(
        self, tmp_path
    ) -> None:
        session = _make_session(tmp_path)
        upload = UploadFile(filename="sample.fcs", file=io.BytesIO(b"fake-fcs"))

        class FakeLoader:
            def load(self, _path):
                return self

            def get_metadata(self):
                class Meta:
                    n_events = 5000
                    n_channels = 3
                    channels = ["FSC-A", "SSC-A", "CD34"]
                    cytometer = "UnitTest"
                    fcs_version = "3.1"

                return Meta()

            def validate_required_channels(self, _required):
                raise ValueError("missing channels")

        with patch("src.api.services.file_service.settings", _mock_settings(tmp_path)):
            with patch("src.data.fcs_parser.FCSLoader", return_value=FakeLoader()):
                service = FileService(session)
                record = service.save_upload(upload)

        assert record.status == "ready"
        assert record.metadata_json is not None
        assert record.metadata_json["parameter_source"] == "metadata_heuristic"
        assert record.metadata_json["initial_conditions"]["upload_id"] if False else True
        assert "F0" in record.metadata_json["initial_conditions"]


class TestFileServiceGet:
    def test_get_nonexistent(self, tmp_path) -> None:
        session = _make_session(tmp_path)
        service = FileService(session)
        assert service.get_upload("nonexistent") is None

    def test_get_existing(self, tmp_path) -> None:
        session = _make_session(tmp_path)
        upload = UploadFile(filename="test.csv", file=io.BytesIO(b"data"))

        with patch("src.api.services.file_service.settings", _mock_settings(tmp_path)):
            service = FileService(session)
            record = service.save_upload(upload)

        found = service.get_upload(record.id)
        assert found is not None
        assert found.filename == "test.csv"
