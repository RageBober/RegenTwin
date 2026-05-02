"""Service for launching, tracking, and loading simulations."""

from __future__ import annotations

import os
import threading
import time
from datetime import UTC, datetime
from typing import Any

from loguru import logger
from sqlalchemy.orm import Session

from src.api.config import settings
from src.api.models.schemas import SimulationMode, SimulationRequest
from src.api.services.result_bundle import (
    build_abm_snapshot,
    build_extended_trajectory,
    load_result_bundle_for_simulation,
    result_path_for_record,
    save_result_bundle,
)
from src.db.models import SimulationRecord, UploadRecord


class SimulationCancelledError(RuntimeError):
    """Raised when a running simulation is cooperatively cancelled."""


class TaskManager:
    """In-memory registry of active simulation tasks."""

    def __init__(self) -> None:
        self._threads: dict[str, threading.Thread] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._celery_task_ids: dict[str, str] = {}
        self._progress: dict[str, float] = {}
        self._messages: dict[str, str] = {}
        self._lock = threading.Lock()

    def register_thread(
        self, sim_id: str, thread: threading.Thread, cancel_event: threading.Event
    ) -> None:
        """Register a background worker thread for a simulation."""
        with self._lock:
            self._threads[sim_id] = thread
            self._cancel_events[sim_id] = cancel_event
            self._progress[sim_id] = 0.0
            self._messages[sim_id] = "Queued"

    def cancel(self, sim_id: str) -> bool:
        with self._lock:
            event = self._cancel_events.get(sim_id)
            thread = self._threads.get(sim_id)
            if event is None or thread is None or not thread.is_alive():
                return False
            event.set()
            self._messages[sim_id] = "Cancellation requested"
            return True

    def get_progress(self, sim_id: str) -> float:
        with self._lock:
            return self._progress.get(sim_id, 0.0)

    def get_message(self, sim_id: str) -> str | None:
        with self._lock:
            return self._messages.get(sim_id)

    def update_progress(
        self, sim_id: str, current: int, total: int, message: str | None = None
    ) -> None:
        with self._lock:
            if sim_id not in self._threads and sim_id not in self._progress:
                return
            self._progress[sim_id] = (current / total) * 100 if total > 0 else 0.0
            if message:
                self._messages[sim_id] = message

    def set_state(self, sim_id: str, progress: float, message: str | None = None) -> None:
        with self._lock:
            if sim_id not in self._threads and sim_id not in self._progress:
                return
            self._progress[sim_id] = progress
            if message:
                self._messages[sim_id] = message

    def is_active(self, sim_id: str) -> bool:
        with self._lock:
            thread = self._threads.get(sim_id)
            if thread is not None and thread.is_alive():
                return True
            celery_id = self._celery_task_ids.get(sim_id)
        if celery_id is not None:
            from src.tasks.celery_app import celery_app

            state = celery_app.AsyncResult(celery_id).state
            return state in ("PENDING", "STARTED", "PROGRESS", "RETRY")
        return False

    def cleanup(self, sim_id: str) -> None:
        with self._lock:
            self._threads.pop(sim_id, None)
            self._cancel_events.pop(sim_id, None)
            self._celery_task_ids.pop(sim_id, None)
            self._progress.pop(sim_id, None)
            self._messages.pop(sim_id, None)

    def active_thread_ids(self) -> list[str]:
        with self._lock:
            return list(self._threads.keys())

    def active_celery_ids(self) -> list[str]:
        with self._lock:
            return list(self._celery_task_ids.keys())

    # ── Celery-specific methods ──────────────────────────────────────

    def register_celery(self, sim_id: str, celery_task_id: str) -> None:
        """Track a Celery task ID for progress polling."""
        with self._lock:
            self._celery_task_ids[sim_id] = celery_task_id
            self._progress[sim_id] = 0.0
            self._messages[sim_id] = "Queued (Celery)"

    def is_celery_task(self, sim_id: str) -> bool:
        with self._lock:
            return sim_id in self._celery_task_ids

    def get_celery_task_id(self, sim_id: str) -> str | None:
        with self._lock:
            return self._celery_task_ids.get(sim_id)

    def cancel_celery(self, sim_id: str) -> bool:
        """Abort a Celery task cooperatively via AbortableAsyncResult."""
        with self._lock:
            celery_id = self._celery_task_ids.get(sim_id)
        if celery_id is None:
            return False
        from celery.contrib.abortable import AbortableAsyncResult

        from src.tasks.celery_app import celery_app

        AbortableAsyncResult(celery_id, app=celery_app).abort()
        return True

    def get_celery_progress(self, sim_id: str) -> tuple[float, str | None]:
        """Poll Celery result backend for progress meta."""
        with self._lock:
            celery_id = self._celery_task_ids.get(sim_id)
        if celery_id is None:
            return 0.0, None
        from src.tasks.celery_app import celery_app

        result = celery_app.AsyncResult(celery_id)
        if result.state == "PROGRESS":
            meta = result.info or {}
            return meta.get("percent", 0.0), meta.get("message")
        if result.state == "SUCCESS":
            return 100.0, "Completed"
        if result.state == "FAILURE":
            return self._progress.get(sim_id, 0.0), str(result.info)
        return self._progress.get(sim_id, 0.0), self._messages.get(sim_id)


# Global singleton used by polling and websocket endpoints.
task_manager = TaskManager()


class SimulationService:
    """Business logic for simulation lifecycle and result loading."""

    def __init__(self, db: Session) -> None:
        self._db = db

    @staticmethod
    def _build_mvp_params(request: SimulationRequest):
        """Build basic model parameters from request initial conditions."""
        from src.data.parameter_extraction import ModelParameters

        total_cells = (
            request.P0 + request.Ne0 + request.M1_0 + request.M2_0 + request.F0 + request.S0
        )
        total_cytokines = request.C_TNF0 + request.C_IL10_0

        return ModelParameters(
            n0=total_cells,
            c0=total_cytokines,
            stem_cell_fraction=request.S0 / max(total_cells, 1.0),
            macrophage_fraction=(request.M1_0 + request.M2_0) / max(total_cells, 1.0),
            apoptotic_fraction=request.D0 / max(total_cells, 1.0),
            inflammation_level=min(request.C_TNF0 / max(total_cytokines, 0.01), 1.0),
        )

    def start_simulation(self, request: SimulationRequest) -> SimulationRecord:
        """Create a DB record and start the background simulation worker."""
        effective_request = self._resolve_request(request)
        record = SimulationRecord(
            mode=effective_request.mode.value,
            params_json=effective_request.model_dump(),
        )
        self._db.add(record)
        self._db.commit()
        self._db.refresh(record)

        if settings.use_celery:
            # Monte Carlo fan-out: N траекторий = N воркеров через chord(group, callback).
            # Для одиночных симуляций остаётся прежний monolith путь.
            if effective_request.n_trajectories > 1:
                # Derive per-trajectory seeds from request.random_seed so each worker
                # gets a deterministic, unique seed (reproducibility guarantee).
                import numpy as np
                from celery import chord, group

                from src.tasks.simulation_tasks import (
                    aggregate_monte_carlo,
                    run_trajectory_task,
                )

                base_seed = effective_request.random_seed
                if base_seed is not None:
                    rng = np.random.default_rng(base_seed)
                    seeds = [
                        int(rng.integers(0, 2**31)) for _ in range(effective_request.n_trajectories)
                    ]
                else:
                    seeds = [None] * effective_request.n_trajectories  # type: ignore[list-item]

                request_dump = effective_request.model_dump()
                header = group(
                    run_trajectory_task.s(record.id, i, seeds[i], request_dump)  # type: ignore[attr-defined]
                    for i in range(effective_request.n_trajectories)
                )
                callback = aggregate_monte_carlo.s(record.id, request_dump)  # type: ignore[attr-defined]
                chord_result = chord(header)(callback)
                task_manager.register_celery(record.id, chord_result.id)
            else:
                from src.tasks.simulation_tasks import run_simulation_task

                celery_result = run_simulation_task.delay(  # type: ignore[attr-defined]
                    record.id, effective_request.model_dump()
                )
                task_manager.register_celery(record.id, celery_result.id)
        else:
            cancel_event = threading.Event()
            thread = threading.Thread(
                target=self._run_in_background,
                args=(record.id, effective_request, cancel_event),
                name=f"sim-{record.id}",
                daemon=True,
            )
            task_manager.register_thread(record.id, thread, cancel_event)
            thread.start()
        return record

    def _resolve_request(self, request: SimulationRequest) -> SimulationRequest:
        """Apply server-side upload-derived initial conditions when present."""
        if request.upload_id is None:
            return request

        upload = self._db.get(UploadRecord, request.upload_id)
        if upload is None:
            raise ValueError(f"Upload {request.upload_id} not found")
        if upload.status != "ready":
            raise ValueError(f"Upload {request.upload_id} is {upload.status}, not ready")

        metadata = upload.metadata_json or {}
        initial_conditions = metadata.get("initial_conditions")
        if not isinstance(initial_conditions, dict) or not initial_conditions:
            raise ValueError(
                f"Upload {request.upload_id} does not contain derived initial conditions"
            )

        merged = request.model_dump()
        merged.update(initial_conditions)
        merged["upload_id"] = request.upload_id
        return SimulationRequest(**merged)

    def _run_in_background(
        self,
        sim_id: str,
        request: SimulationRequest,
        cancel_event: threading.Event,
    ) -> None:
        """Execute the simulation synchronously in a dedicated worker thread.

        Runs outside any asyncio event loop so it is not cancelled when the
        originating HTTP request completes. Timeout is enforced cooperatively
        via a watchdog ``threading.Timer`` that flips ``cancel_event``.
        """
        timed_out = threading.Event()
        watchdog: threading.Timer | None = None

        if settings.simulation_timeout and settings.simulation_timeout > 0:

            def _fire_timeout() -> None:
                timed_out.set()
                cancel_event.set()

            watchdog = threading.Timer(settings.simulation_timeout, _fire_timeout)
            watchdog.daemon = True
            watchdog.start()

        try:
            self._update_db_status(
                sim_id, "running", progress=5.0, message="Initializing simulation..."
            )
            task_manager.set_state(sim_id, 5.0, "Initializing simulation...")

            def cancel_callback() -> None:
                if cancel_event.is_set():
                    raise SimulationCancelledError("Simulation cancelled")

            result = self._execute_simulation(sim_id, request, cancel_callback)

            self._save_result(sim_id, request.mode.value, result)
            task_manager.set_state(sim_id, 100.0, "Simulation completed")
            self._update_db_status(
                sim_id,
                "completed",
                progress=100.0,
                message="Simulation completed",
            )
            logger.info(f"Simulation {sim_id} completed")
        except SimulationCancelledError:
            progress = task_manager.get_progress(sim_id)
            if timed_out.is_set():
                task_manager.set_state(sim_id, progress, "Simulation timed out")
                self._update_db_status(
                    sim_id,
                    "failed",
                    progress=progress,
                    error="Simulation timed out",
                    message="Simulation timed out",
                )
                logger.error(f"Simulation {sim_id} timed out after {settings.simulation_timeout}s")
            else:
                task_manager.set_state(sim_id, progress, "Simulation cancelled")
                self._update_db_status(
                    sim_id,
                    "cancelled",
                    progress=progress,
                    message="Simulation cancelled",
                )
                logger.info(f"Simulation {sim_id} cancelled (cooperative)")
        except Exception as exc:
            cancel_event.set()
            progress = task_manager.get_progress(sim_id)
            task_manager.set_state(sim_id, progress, str(exc))
            self._update_db_status(
                sim_id,
                "failed",
                progress=progress,
                error=str(exc),
                message=str(exc),
            )
            logger.exception(f"Simulation {sim_id} failed: {exc}")
        finally:
            if watchdog is not None:
                watchdog.cancel()
            task_manager.cleanup(sim_id)

    def _execute_simulation(
        self,
        sim_id: str,
        request: SimulationRequest,
        cancel_callback,
        progress_reporter=None,
    ):
        """Synchronously execute the requested simulation mode.

        Args:
            cancel_callback: ``() -> None`` that raises on cancellation.
            progress_reporter: ``(sim_id, current, total, message) -> None``.
                Falls back to ``task_manager.update_progress`` when *None*.
        """
        report = progress_reporter or (
            lambda sid, cur, tot, msg=None: task_manager.update_progress(sid, cur, tot, msg)
        )
        report(sim_id, 0, 100, "Initializing model...")

        if request.n_trajectories > 1:
            return self._run_monte_carlo(sim_id, request, cancel_callback, report)

        if request.mode == SimulationMode.MVP:
            return self._run_mvp_sde(sim_id, request, cancel_callback, report)
        if request.mode == SimulationMode.EXTENDED:
            return self._run_sde(sim_id, request, cancel_callback, report)
        if request.mode == SimulationMode.ABM:
            return self._run_abm(sim_id, request, cancel_callback, report)
        if request.mode == SimulationMode.INTEGRATED:
            return self._run_integrated(sim_id, request, cancel_callback, report)
        raise ValueError(f"Unsupported simulation mode: {request.mode}")

    def _run_mvp_sde(self, sim_id: str, request: SimulationRequest, cancel_callback, report):
        """Run the simplified MVP SDE model."""
        from src.core.sde_model import SDEConfig, SDEModel, TherapyProtocol

        therapy = TherapyProtocol(
            prp_enabled=request.prp_enabled,
            prp_intensity=request.prp_intensity,
            pemf_enabled=request.pemf_enabled,
            pemf_frequency=request.pemf_frequency,
            pemf_intensity=request.pemf_intensity,
        )

        config = SDEConfig(dt=request.dt, t_max=request.t_max_hours)
        model = SDEModel(config=config, therapy=therapy, random_seed=request.random_seed)
        initial_params = self._build_mvp_params(request)

        def on_progress(current_step: int, total_steps: int) -> None:
            overall = 10 + (current_step / max(total_steps, 1)) * 80
            report(
                sim_id,
                int(overall),
                100,
                f"Integrating MVP step {current_step}/{total_steps}...",
            )

        report(sim_id, 10, 100, "Running MVP SDE simulation...")
        cancel_callback()
        trajectory = model.simulate(
            initial_params=initial_params,
            progress_callback=on_progress,
            cancel_callback=cancel_callback,
        )
        cancel_callback()
        report(sim_id, 90, 100, "Saving results...")
        return trajectory

    def _run_sde(self, sim_id: str, request: SimulationRequest, cancel_callback, report):
        """Run the full extended SDE model."""
        from src.core.extended_sde import ExtendedSDEModel, ExtendedSDEState
        from src.core.parameters import ParameterSet
        from src.core.sde_model import TherapyProtocol

        therapy = TherapyProtocol(
            prp_enabled=request.prp_enabled,
            prp_intensity=request.prp_intensity,
            pemf_enabled=request.pemf_enabled,
            pemf_frequency=request.pemf_frequency,
            pemf_intensity=request.pemf_intensity,
        )

        params = ParameterSet()
        params.dt = request.dt
        params.t_max = request.t_max_hours

        model = ExtendedSDEModel(
            params=params,
            therapy=therapy,
            rng_seed=request.random_seed,
        )

        initial_state = ExtendedSDEState(
            P=request.P0,
            Ne=request.Ne0,
            M1=request.M1_0,
            M2=request.M2_0,
            F=request.F0,
            Mf=request.Mf0,
            E=request.E0,
            S=request.S0,
            C_TNF=request.C_TNF0,
            C_IL10=request.C_IL10_0,
            C_PDGF=request.C_PDGF0,
            C_VEGF=request.C_VEGF0,
            C_TGFb=request.C_TGFb0,
            C_MCP1=request.C_MCP1_0,
            C_IL8=request.C_IL8_0,
            rho_collagen=request.rho_collagen0,
            C_MMP=request.C_MMP0,
            rho_fibrin=request.rho_fibrin0,
            D=request.D0,
            O2=request.O2_0,
        )

        def on_progress(current_step: int, total_steps: int) -> None:
            overall = 10 + (current_step / max(total_steps, 1)) * 80
            report(
                sim_id,
                int(overall),
                100,
                f"Integrating SDE step {current_step}/{total_steps}...",
            )

        report(sim_id, 10, 100, "Running SDE simulation...")
        cancel_callback()
        trajectory = model.simulate(
            initial_state=initial_state,
            progress_callback=on_progress,
            cancel_callback=cancel_callback,
        )
        cancel_callback()
        report(sim_id, 90, 100, "Saving results...")
        return trajectory

    def _run_abm(self, sim_id: str, request: SimulationRequest, cancel_callback, report):
        """Run the ABM simulation."""
        from src.core.abm_model import ABMConfig, simulate_abm

        config = ABMConfig(t_max=request.t_max_hours)
        initial_params = self._build_mvp_params(request)

        def on_progress(current_step: int, total_steps: int) -> None:
            overall = 10 + (current_step / max(total_steps, 1)) * 80
            report(
                sim_id,
                int(overall),
                100,
                f"Running ABM step {current_step}/{total_steps}...",
            )

        report(sim_id, 10, 100, "Running ABM simulation...")
        cancel_callback()
        trajectory = simulate_abm(
            initial_params=initial_params,
            config=config,
            random_seed=request.random_seed,
            progress_callback=on_progress,
            cancel_callback=cancel_callback,
        )
        cancel_callback()
        report(sim_id, 90, 100, "Saving results...")
        return trajectory

    def _run_integrated(self, sim_id: str, request: SimulationRequest, cancel_callback, report):
        """Run the integrated SDE+ABM simulation."""
        from src.core.integration import IntegratedModel, create_default_integration_config
        from src.core.sde_model import TherapyProtocol

        therapy = TherapyProtocol(
            prp_enabled=request.prp_enabled,
            prp_intensity=request.prp_intensity,
            pemf_enabled=request.pemf_enabled,
            pemf_frequency=request.pemf_frequency,
            pemf_intensity=request.pemf_intensity,
        )

        t_max_days = request.t_max_hours / 24.0
        config = create_default_integration_config(t_max_days=t_max_days)

        model = IntegratedModel(
            config=config,
            therapy=therapy,
            random_seed=request.random_seed,
        )

        initial_params = self._build_mvp_params(request)

        report(sim_id, 10, 100, "Running integrated SDE+ABM simulation...")
        cancel_callback()
        trajectory = model.simulate(initial_params=initial_params)
        cancel_callback()
        report(sim_id, 90, 100, "Saving results...")
        return trajectory

    def _build_monte_carlo_configs(self, request: SimulationRequest):
        """Build (sde_cfg, abm_cfg, integ_cfg, ext_params, ext_state, model_type, therapy).

        Extracted so Celery fan-out workers can reuse the exact same assembly
        logic as the in-process Monte Carlo path.
        """
        from src.core.abm_model import ABMConfig
        from src.core.sde_model import SDEConfig, TherapyProtocol

        therapy = TherapyProtocol(
            prp_enabled=request.prp_enabled,
            prp_intensity=request.prp_intensity,
            pemf_enabled=request.pemf_enabled,
            pemf_frequency=request.pemf_frequency,
            pemf_intensity=request.pemf_intensity,
        )

        mode_to_model_type = {
            SimulationMode.MVP: "sde",
            SimulationMode.EXTENDED: "extended",
            SimulationMode.ABM: "abm",
            SimulationMode.INTEGRATED: "integrated",
        }
        model_type = mode_to_model_type.get(request.mode)
        if model_type is None:
            raise ValueError(f"Unsupported simulation mode: {request.mode}")

        sde_config = None
        abm_config = None
        integration_config = None
        extended_params = None
        extended_initial_state = None

        if model_type == "sde":
            sde_config = SDEConfig(dt=request.dt, t_max=request.t_max_hours)
        elif model_type == "extended":
            from src.core.extended_sde import ExtendedSDEState
            from src.core.parameters import ParameterSet

            extended_params = ParameterSet(dt=request.dt, t_max=request.t_max_hours)
            extended_initial_state = ExtendedSDEState(
                P=request.P0,
                Ne=request.Ne0,
                M1=request.M1_0,
                M2=request.M2_0,
                F=request.F0,
                Mf=request.Mf0,
                E=request.E0,
                S=request.S0,
                C_TNF=request.C_TNF0,
                C_IL10=request.C_IL10_0,
                C_PDGF=request.C_PDGF0,
                C_VEGF=request.C_VEGF0,
                C_TGFb=request.C_TGFb0,
                C_MCP1=request.C_MCP1_0,
                C_IL8=request.C_IL8_0,
                rho_collagen=request.rho_collagen0,
                C_MMP=request.C_MMP0,
                rho_fibrin=request.rho_fibrin0,
                D=request.D0,
                O2=request.O2_0,
            )
        elif model_type == "abm":
            abm_config = ABMConfig(t_max=request.t_max_hours)
        elif model_type == "integrated":
            from src.core.integration import create_default_integration_config

            integration_config = create_default_integration_config(
                t_max_days=request.t_max_hours / 24.0,
            )

        return (
            sde_config,
            abm_config,
            integration_config,
            extended_params,
            extended_initial_state,
            model_type,
            therapy,
        )

    def _run_monte_carlo(
        self,
        sim_id: str,
        request: SimulationRequest,
        cancel_callback,
        report,
    ):
        """Run Monte Carlo ensemble for any simulation mode (in-process)."""
        from src.core.monte_carlo import MonteCarloConfig, MonteCarloSimulator

        (
            sde_config,
            abm_config,
            integration_config,
            extended_params,
            extended_initial_state,
            model_type,
            therapy,
        ) = self._build_monte_carlo_configs(request)

        def on_progress(completed: int, total: int) -> None:
            overall = 10 + (completed / max(total, 1)) * 80
            report(sim_id, int(overall), 100, f"Monte Carlo trajectory {completed}/{total}...")

        # Троттлинг DB-коммитов для step-progress: гарантируем ≥0.5s между
        # апдейтами, чтобы 180 шагов × N траекторий не порождали 180×N отдельных
        # SessionLocal()+commit() — начало (step=0) и конец каждой траектории
        # (step=total-1) всегда форсируем, чтобы UI видел переходы.
        _last_commit_ts = [0.0]
        MIN_COMMIT_INTERVAL = 0.5

        def on_step_progress(
            trajectory_id: int,
            total_traj: int,
            step: int,
            total_steps: int,
        ) -> None:
            force = step == 0 or step >= total_steps - 1 or (trajectory_id == 0 and step <= 1)
            now = time.monotonic()
            if not force and (now - _last_commit_ts[0]) < MIN_COMMIT_INTERVAL:
                return
            _last_commit_ts[0] = now

            # Fractional progress within the current trajectory, blended into the
            # 10%–90% range so the UI never appears stuck between completions.
            traj_fraction = (trajectory_id + step / max(total_steps, 1)) / max(total_traj, 1)
            overall = 10 + traj_fraction * 80
            report(
                sim_id,
                int(overall),
                100,
                f"Monte Carlo trajectory {trajectory_id + 1}/{total_traj}"
                f" · step {step}/{total_steps}",
            )

        # Авто-параллелизация: по умолчанию используем ProcessPool с min(n_traj, cpu-1)
        # jobs для CPU-bound ABM/SDE; оставляем 1 ядро под UI/API-поток.
        cpu_available = os.cpu_count() or 2
        auto_n_jobs = min(request.n_trajectories, max(1, cpu_available - 1))
        use_mp = auto_n_jobs > 1

        mc_config = MonteCarloConfig(
            n_trajectories=request.n_trajectories,
            model_type=model_type,
            sde_config=sde_config,
            abm_config=abm_config,
            integration_config=integration_config,
            extended_params=extended_params,
            extended_initial_state=extended_initial_state,
            base_seed=request.random_seed,
            n_jobs=auto_n_jobs,
            use_multiprocessing=use_mp,
            progress_callback=on_progress,
            step_progress_callback=on_step_progress,
            cancel_callback=cancel_callback,
        )

        report(sim_id, 10, 100, f"Running Monte Carlo ({request.n_trajectories} trajectories)...")
        initial_params = self._build_mvp_params(request)
        simulator = MonteCarloSimulator(config=mc_config, therapy=therapy)
        results = simulator.run(initial_params)
        report(sim_id, 90, 100, "Saving results...")
        return results

    def _save_result(self, sim_id: str, mode: str, trajectory: Any) -> None:
        """Serialize a simulation result bundle to disk."""
        from src.db.session import SessionLocal

        db = SessionLocal()
        try:
            record = db.get(SimulationRecord, sim_id)
            if record is None:
                raise FileNotFoundError(f"Simulation {sim_id} not found")
            result_path = result_path_for_record(record)
        finally:
            db.close()

        save_result_bundle(result_path, mode, trajectory)
        self._update_result_path(sim_id, str(result_path))

    def _update_db_status(
        self,
        sim_id: str,
        status: str,
        progress: float | None = None,
        error: str | None = None,
        message: str | None = None,
    ) -> None:
        """Update the DB status from a fresh thread-safe session."""
        from src.db.session import SessionLocal

        db = SessionLocal()
        try:
            record = db.get(SimulationRecord, sim_id)
            if record:
                record.status = status
                if progress is not None:
                    record.progress = progress
                if message is not None:
                    record.message = message
                if error:
                    record.error_message = error
                    if message is None:
                        record.message = error
                if status in {"completed", "failed", "cancelled"}:
                    record.completed_at = datetime.now(UTC)
                db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def mark_cancelled_in_db(sim_id: str, message: str = "Cancelled") -> bool:
        """Best-effort: пометить запись cancelled в DB без активного thread.

        Используется когда `task_manager` не помнит задачу (например, после
        рестарта API или для уже завершённого процесса с залипшим статусом).
        Возвращает True если запись была обновлена, False если не найдена.
        """
        from src.db.session import SessionLocal

        db = SessionLocal()
        try:
            record = db.get(SimulationRecord, sim_id)
            if record is None:
                return False
            if record.status not in {"running", "pending", "cancelling"}:
                return False
            record.status = "cancelled"
            record.message = message
            record.completed_at = datetime.now(UTC)
            db.commit()
            return True
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def _update_result_path(self, sim_id: str, path: str) -> None:
        """Persist the result path in the DB."""
        from src.db.session import SessionLocal

        db = SessionLocal()
        try:
            record = db.get(SimulationRecord, sim_id)
            if record:
                record.result_path = path
                db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def get_status(self, sim_id: str) -> SimulationRecord | None:
        """Return the simulation status, overlaying live progress when active."""
        record = self._db.get(SimulationRecord, sim_id)
        if record and task_manager.is_celery_task(sim_id):
            pct, msg = task_manager.get_celery_progress(sim_id)
            record.progress = pct
            record.message = msg
        elif record and task_manager.is_active(sim_id):
            record.progress = task_manager.get_progress(sim_id)
            record.message = task_manager.get_message(sim_id)
        return record

    @staticmethod
    def load_params(sim_id: str) -> dict:
        """Load the original simulation params from the DB record."""
        from src.db.session import SessionLocal

        db = SessionLocal()
        try:
            record = db.get(SimulationRecord, sim_id)
            if record is None:
                raise FileNotFoundError(f"Simulation {sim_id} not found")
            return record.params_json
        finally:
            db.close()

    @staticmethod
    def load_trajectory(sim_id: str) -> dict[str, Any]:
        """Load the persisted result bundle for a simulation."""
        return load_result_bundle_for_simulation(sim_id)

    @staticmethod
    def load_extended_trajectory(sim_id: str):
        """Load a saved extended/integrated trajectory."""
        return build_extended_trajectory(load_result_bundle_for_simulation(sim_id))

    @staticmethod
    def load_spatial_snapshot(sim_id: str, timestep: int = -1):
        """Load a saved ABM snapshot for spatial visualization."""
        result = load_result_bundle_for_simulation(sim_id, include_snapshots=True)
        return build_abm_snapshot(result, timestep=timestep)
