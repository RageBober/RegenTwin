"""Тесты для SQLAlchemy ORM models."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.db.models import AnalysisRecord, Base, SimulationRecord, UploadRecord


def _make_session() -> Session:
    """In-memory SQLite session для тестов."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


class TestSimulationRecord:
    def test_create(self) -> None:
        session = _make_session()
        record = SimulationRecord(
            id="sim-1",
            mode="extended",
            params_json={"P0": 500.0},
        )
        session.add(record)
        session.commit()

        loaded = session.get(SimulationRecord, "sim-1")
        assert loaded is not None
        assert loaded.mode == "extended"
        assert loaded.status == "pending"
        assert loaded.progress == 0.0
        assert loaded.params_json == {"P0": 500.0}

    def test_update_status(self) -> None:
        session = _make_session()
        record = SimulationRecord(id="sim-2", mode="mvp", params_json={})
        session.add(record)
        session.commit()

        record.status = "completed"
        record.progress = 100.0
        session.commit()

        loaded = session.get(SimulationRecord, "sim-2")
        assert loaded.status == "completed"
        assert loaded.progress == 100.0

    def test_auto_created_at(self) -> None:
        session = _make_session()
        record = SimulationRecord(id="sim-3", mode="extended", params_json={})
        session.add(record)
        session.commit()
        assert record.created_at is not None


class TestUploadRecord:
    def test_create(self) -> None:
        session = _make_session()
        record = UploadRecord(
            id="upl-1",
            filename="test.fcs",
            file_path="/data/uploads/upl-1/test.fcs",
        )
        session.add(record)
        session.commit()

        loaded = session.get(UploadRecord, "upl-1")
        assert loaded.filename == "test.fcs"
        assert loaded.status == "processing"

    def test_metadata_json(self) -> None:
        session = _make_session()
        record = UploadRecord(
            id="upl-2",
            filename="data.fcs",
            file_path="/tmp/data.fcs",
            metadata_json={"n_events": 10000, "channels": ["FSC-A", "SSC-A"]},
        )
        session.add(record)
        session.commit()

        loaded = session.get(UploadRecord, "upl-2")
        assert loaded.metadata_json["n_events"] == 10000


class TestAnalysisRecord:
    def test_create(self) -> None:
        session = _make_session()
        record = AnalysisRecord(
            id="ana-1",
            analysis_type="sensitivity",
            params_json={"method": "sobol", "n_samples": 256},
        )
        session.add(record)
        session.commit()

        loaded = session.get(AnalysisRecord, "ana-1")
        assert loaded.analysis_type == "sensitivity"
        assert loaded.status == "pending"

    def test_store_result(self) -> None:
        session = _make_session()
        record = AnalysisRecord(
            id="ana-2",
            analysis_type="estimation",
            params_json={},
            result_json={"posterior_mean": {"r": 0.3, "K": 1e6}},
        )
        session.add(record)
        session.commit()

        loaded = session.get(AnalysisRecord, "ana-2")
        assert loaded.result_json["posterior_mean"]["r"] == 0.3
