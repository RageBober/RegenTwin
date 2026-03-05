"""Тесты для SimulationService и TaskManager."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.models.schemas import SimulationMode, SimulationRequest
from src.api.services.simulation_service import TaskManager
from src.db.models import Base, SimulationRecord


def _make_session():  # type: ignore[no-untyped-def]
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


class TestTaskManager:
    def test_register_and_progress(self) -> None:
        tm = TaskManager()
        mock_task = MagicMock()
        mock_task.done.return_value = False
        event = asyncio.Event()

        tm.register("sim-1", mock_task, event)
        assert tm.get_progress("sim-1") == 0.0

        tm.update_progress("sim-1", 50, 100, "Running...")
        assert tm.get_progress("sim-1") == 50.0
        assert tm.get_message("sim-1") == "Running..."

    def test_is_active(self) -> None:
        tm = TaskManager()
        mock_task = MagicMock()
        mock_task.done.return_value = False
        event = asyncio.Event()

        tm.register("sim-2", mock_task, event)
        assert tm.is_active("sim-2") is True

        mock_task.done.return_value = True
        assert tm.is_active("sim-2") is False

    def test_cancel(self) -> None:
        tm = TaskManager()
        mock_task = MagicMock()
        event = asyncio.Event()

        tm.register("sim-3", mock_task, event)
        assert tm.cancel("sim-3") is True
        assert event.is_set()
        mock_task.cancel.assert_called_once()

    def test_cancel_nonexistent(self) -> None:
        tm = TaskManager()
        assert tm.cancel("nonexistent") is False

    def test_cleanup(self) -> None:
        tm = TaskManager()
        mock_task = MagicMock()
        event = asyncio.Event()

        tm.register("sim-4", mock_task, event)
        tm.cleanup("sim-4")
        assert tm.get_progress("sim-4") == 0.0
        assert not tm.is_active("sim-4")

    def test_nonexistent_progress(self) -> None:
        tm = TaskManager()
        assert tm.get_progress("nonexistent") == 0.0
        assert tm.get_message("nonexistent") is None


class TestSimulationRequest:
    def test_request_defaults(self) -> None:
        req = SimulationRequest()
        assert req.mode == SimulationMode.EXTENDED
        assert req.n_trajectories == 1
        assert req.upload_id is None
