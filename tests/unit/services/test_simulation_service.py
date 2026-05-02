"""Тесты для SimulationService, runtime hooks и TaskManager."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from src.api.models.schemas import SimulationMode, SimulationRequest
from src.api.services.simulation_service import SimulationService, TaskManager, task_manager


class TestTaskManager:
    def test_register_and_progress(self) -> None:
        tm = TaskManager()
        mock_task = MagicMock()
        mock_task.is_alive.return_value = True
        event = threading.Event()

        tm.register_thread("sim-1", mock_task, event)
        assert tm.get_progress("sim-1") == 0.0

        tm.update_progress("sim-1", 50, 100, "Running...")
        assert tm.get_progress("sim-1") == 50.0
        assert tm.get_message("sim-1") == "Running..."

    def test_is_active(self) -> None:
        tm = TaskManager()
        mock_task = MagicMock()
        mock_task.is_alive.return_value = True
        event = threading.Event()

        tm.register_thread("sim-2", mock_task, event)
        assert tm.is_active("sim-2") is True

        mock_task.is_alive.return_value = False
        assert tm.is_active("sim-2") is False

    def test_cancel(self) -> None:
        tm = TaskManager()
        mock_task = MagicMock()
        mock_task.is_alive.return_value = True
        event = threading.Event()

        tm.register_thread("sim-3", mock_task, event)
        assert tm.cancel("sim-3") is True
        assert event.is_set()
        assert tm.get_message("sim-3") == "Cancellation requested"

    def test_cancel_nonexistent(self) -> None:
        tm = TaskManager()
        assert tm.cancel("nonexistent") is False

    def test_cleanup(self) -> None:
        tm = TaskManager()
        mock_task = MagicMock()
        event = threading.Event()

        tm.register_thread("sim-4", mock_task, event)
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


class TestSimulationServiceRuntimeHooks:
    @staticmethod
    def _register_runtime_task(sim_id: str) -> None:
        mock_task = MagicMock()
        mock_task.is_alive.return_value = True
        task_manager.register_thread(sim_id, mock_task, threading.Event())

    @staticmethod
    def _cleanup_runtime_task(sim_id: str) -> None:
        task_manager.cleanup(sim_id)

    def test_run_mvp_sde_passes_progress_and_cancel_callbacks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.core import sde_model as sde_model_module

        service = SimulationService(MagicMock())
        sim_id = "mvp-hooks"
        self._register_runtime_task(sim_id)
        request = SimulationRequest(mode=SimulationMode.MVP, t_max_hours=4.0, dt=1.0)
        cancel_calls = {"count": 0}
        observed_progress: list[float] = []

        def cancel_callback() -> None:
            cancel_calls["count"] += 1

        def fake_update_progress(
            simulation_id: str, current: int, total: int, message: str | None = None
        ) -> None:
            if simulation_id == sim_id:
                observed_progress.append((current / total) * 100 if total else 0.0)

        def fake_simulate(self, initial_params, progress_callback=None, cancel_callback=None):  # type: ignore[no-untyped-def]
            assert progress_callback is not None
            assert cancel_callback is not None
            progress_callback(2, 4)
            cancel_callback()
            return MagicMock(name="mvp-trajectory")

        monkeypatch.setattr(sde_model_module.SDEModel, "simulate", fake_simulate)
        monkeypatch.setattr(task_manager, "update_progress", fake_update_progress)

        try:
            trajectory = service._run_mvp_sde(
                sim_id, request, cancel_callback, task_manager.update_progress
            )
        finally:
            self._cleanup_runtime_task(sim_id)

        assert trajectory is not None
        assert cancel_calls["count"] >= 3
        assert 50.0 in observed_progress
        assert observed_progress[-1] == 90.0

    def test_run_sde_passes_progress_and_cancel_callbacks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.core import extended_sde as extended_sde_module

        service = SimulationService(MagicMock())
        sim_id = "extended-hooks"
        self._register_runtime_task(sim_id)
        request = SimulationRequest(mode=SimulationMode.EXTENDED, t_max_hours=4.0, dt=1.0)
        cancel_calls = {"count": 0}
        observed_progress: list[float] = []

        def cancel_callback() -> None:
            cancel_calls["count"] += 1

        def fake_update_progress(
            simulation_id: str, current: int, total: int, message: str | None = None
        ) -> None:
            if simulation_id == sim_id:
                observed_progress.append((current / total) * 100 if total else 0.0)

        def fake_simulate(
            self, initial_state, t_span=None, progress_callback=None, cancel_callback=None
        ):  # type: ignore[no-untyped-def]
            assert progress_callback is not None
            assert cancel_callback is not None
            progress_callback(2, 4)
            cancel_callback()
            return MagicMock(name="extended-trajectory")

        monkeypatch.setattr(extended_sde_module.ExtendedSDEModel, "simulate", fake_simulate)
        monkeypatch.setattr(task_manager, "update_progress", fake_update_progress)

        try:
            trajectory = service._run_sde(
                sim_id, request, cancel_callback, task_manager.update_progress
            )
        finally:
            self._cleanup_runtime_task(sim_id)

        assert trajectory is not None
        assert cancel_calls["count"] >= 3
        assert 50.0 in observed_progress
        assert observed_progress[-1] == 90.0

    def test_run_abm_passes_progress_and_cancel_callbacks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.core import abm_model as abm_model_module

        service = SimulationService(MagicMock())
        sim_id = "abm-hooks"
        self._register_runtime_task(sim_id)
        request = SimulationRequest(mode=SimulationMode.ABM, t_max_hours=4.0, dt=1.0)
        cancel_calls = {"count": 0}
        observed_progress: list[float] = []

        def cancel_callback() -> None:
            cancel_calls["count"] += 1

        def fake_update_progress(
            simulation_id: str, current: int, total: int, message: str | None = None
        ) -> None:
            if simulation_id == sim_id:
                observed_progress.append((current / total) * 100 if total else 0.0)

        def fake_simulate_abm(
            initial_params,
            config=None,
            random_seed=None,
            snapshot_interval=24.0,
            progress_callback=None,
            cancel_callback=None,
        ):  # type: ignore[no-untyped-def]
            assert progress_callback is not None
            assert cancel_callback is not None
            progress_callback(2, 4)
            cancel_callback()
            return MagicMock(name="abm-trajectory")

        monkeypatch.setattr(abm_model_module, "simulate_abm", fake_simulate_abm)
        monkeypatch.setattr(task_manager, "update_progress", fake_update_progress)

        try:
            trajectory = service._run_abm(
                sim_id, request, cancel_callback, task_manager.update_progress
            )
        finally:
            self._cleanup_runtime_task(sim_id)

        assert trajectory is not None
        assert cancel_calls["count"] >= 3
        assert 50.0 in observed_progress
        assert observed_progress[-1] == 90.0
