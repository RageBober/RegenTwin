"""Tests for Celery Monte Carlo fan-out (chord + group).

No Redis broker required — we assert task registration and dispatch-path
branching without actually enqueuing to a real broker.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.api.models.schemas import SimulationMode, SimulationRequest


@pytest.fixture
def db_session():
    """In-memory SQLite session for unit-tests."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from src.db.models import Base

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


class TestCeleryTaskRegistration:
    """Smoke tests: tasks are importable and registered under expected names."""

    def test_run_trajectory_task_registered(self):
        from src.tasks.simulation_tasks import run_trajectory_task

        assert run_trajectory_task.name == "regentwin.run_trajectory"  # type: ignore[attr-defined]

    def test_aggregate_monte_carlo_registered(self):
        from src.tasks.simulation_tasks import aggregate_monte_carlo

        assert aggregate_monte_carlo.name == "regentwin.aggregate_monte_carlo"  # type: ignore[attr-defined]

    def test_run_simulation_task_still_registered(self):
        """Prior monolith task must remain (used for single-trajectory path)."""
        from src.tasks.simulation_tasks import run_simulation_task

        assert run_simulation_task.name == "regentwin.run_simulation"  # type: ignore[attr-defined]


class TestCeleryFanoutDispatch:
    """Проверяем ветвление в start_simulation под use_celery=True."""

    def _make_service_and_request(self, n_trajectories: int, db_session):
        from src.api.services.simulation_service import SimulationService

        service = SimulationService(db_session)
        request = SimulationRequest(
            mode=SimulationMode.ABM,
            n_trajectories=n_trajectories,
            t_max_hours=6.0,
            dt=0.5,
            random_seed=123,
        )
        return service, request

    @patch("src.api.services.simulation_service.settings")
    def test_monte_carlo_dispatches_chord(self, mock_settings, db_session):
        """use_celery=True + n_trajectories>1 → chord() запускается."""
        mock_settings.use_celery = True

        service, request = self._make_service_and_request(4, db_session)

        with (
            patch("celery.chord") as mock_chord,
            patch("celery.group") as mock_group,
        ):
            mock_chord_result = MagicMock()
            mock_chord_result.id = "chord-task-id-xyz"
            mock_chord.return_value.return_value = mock_chord_result
            mock_group.return_value = MagicMock()

            record = service.start_simulation(request)

            assert mock_group.called, "group() должен быть вызван для fan-out"
            assert mock_chord.called, "chord() должен быть вызван"
            assert record.id is not None

    @patch("src.api.services.simulation_service.settings")
    def test_single_trajectory_uses_monolith_task(self, mock_settings, db_session):
        """n_trajectories=1 с use_celery=True — остаётся прежний monolith путь."""
        mock_settings.use_celery = True

        service, request = self._make_service_and_request(1, db_session)

        with (
            patch("src.tasks.simulation_tasks.run_simulation_task") as mock_task,
            patch("celery.chord") as mock_chord,
        ):
            mock_task.delay.return_value.id = "single-task-id"

            record = service.start_simulation(request)

            assert mock_task.delay.called
            assert not mock_chord.called
            assert record.id is not None
