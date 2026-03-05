"""Сервис для управления симуляциями."""

from __future__ import annotations

import asyncio
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.api.config import settings
from src.api.models.schemas import SimulationMode, SimulationRequest
from src.db.models import SimulationRecord


class TaskManager:
    """In-memory реестр запущенных задач с прогрессом."""

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task] = {}  # type: ignore[type-arg]
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._progress: dict[str, float] = {}
        self._messages: dict[str, str] = {}
        self._lock = threading.Lock()

    def register(
        self, sim_id: str, task: asyncio.Task, cancel_event: asyncio.Event  # type: ignore[type-arg]
    ) -> None:
        with self._lock:
            self._tasks[sim_id] = task
            self._cancel_events[sim_id] = cancel_event
            self._progress[sim_id] = 0.0

    def cancel(self, sim_id: str) -> bool:
        with self._lock:
            event = self._cancel_events.get(sim_id)
            task = self._tasks.get(sim_id)
        if event and task:
            event.set()
            task.cancel()
            return True
        return False

    def get_progress(self, sim_id: str) -> float:
        with self._lock:
            return self._progress.get(sim_id, 0.0)

    def get_message(self, sim_id: str) -> str | None:
        with self._lock:
            return self._messages.get(sim_id)

    def update_progress(self, sim_id: str, current: int, total: int, message: str | None = None) -> None:
        with self._lock:
            self._progress[sim_id] = (current / total) * 100 if total > 0 else 0.0
            if message:
                self._messages[sim_id] = message

    def is_active(self, sim_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(sim_id)
        return task is not None and not task.done()

    def cleanup(self, sim_id: str) -> None:
        with self._lock:
            self._tasks.pop(sim_id, None)
            self._cancel_events.pop(sim_id, None)
            self._progress.pop(sim_id, None)
            self._messages.pop(sim_id, None)


# Глобальный синглтон
task_manager = TaskManager()


class SimulationService:
    """Бизнес-логика симуляций: запуск, статус, результаты."""

    def __init__(self, db: Session) -> None:
        self._db = db

    @staticmethod
    def _build_mvp_params(request: SimulationRequest):  # type: ignore[no-untyped-def]
        """Построить ModelParameters из начальных условий запроса (для MVP/ABM)."""
        from src.data.parameter_extraction import ModelParameters

        total_cells = (
            request.P0 + request.Ne0 + request.M1_0
            + request.M2_0 + request.F0 + request.S0
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

    async def start_simulation(self, request: SimulationRequest) -> SimulationRecord:
        """Создать запись в БД и запустить фоновую задачу."""
        record = SimulationRecord(
            mode=request.mode.value,
            params_json=request.model_dump(),
        )
        self._db.add(record)
        self._db.commit()
        self._db.refresh(record)

        cancel_event = asyncio.Event()
        task = asyncio.create_task(
            self._run_in_background(record.id, request, cancel_event)
        )
        task_manager.register(record.id, task, cancel_event)

        return record

    async def _run_in_background(
        self,
        sim_id: str,
        request: SimulationRequest,
        cancel_event: asyncio.Event,
    ) -> None:
        """Фоновое выполнение симуляции в ThreadPool с таймаутом."""
        loop = asyncio.get_event_loop()
        try:
            # Обновить статус на running (с начальным прогрессом > 0 для polling)
            self._update_db_status(sim_id, "running", progress=5.0)

            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    self._execute_simulation,
                    sim_id,
                    request,
                ),
                timeout=settings.simulation_timeout,
            )

            # Сохранить результат
            self._save_result(sim_id, result)
            self._update_db_status(sim_id, "completed", progress=100.0)
            logger.info(f"Simulation {sim_id} completed")

        except asyncio.CancelledError:
            self._update_db_status(sim_id, "cancelled")
            logger.info(f"Simulation {sim_id} cancelled")
        except TimeoutError:
            self._update_db_status(sim_id, "failed", error="Simulation timed out")
            logger.error(f"Simulation {sim_id} timed out after {settings.simulation_timeout}s")
        except Exception as exc:
            self._update_db_status(sim_id, "failed", error=str(exc))
            logger.error(f"Simulation {sim_id} failed: {exc}")
        finally:
            task_manager.cleanup(sim_id)

    def _execute_simulation(self, sim_id: str, request: SimulationRequest):  # type: ignore[no-untyped-def]
        """Синхронное выполнение симуляции (запускается в потоке)."""
        task_manager.update_progress(sim_id, 0, 100, "Initializing model...")

        if request.mode == SimulationMode.MVP:
            return self._run_mvp_sde(sim_id, request)
        elif request.mode == SimulationMode.EXTENDED:
            return self._run_sde(sim_id, request)
        elif request.mode == SimulationMode.ABM:
            return self._run_abm(sim_id, request)
        elif request.mode == SimulationMode.INTEGRATED:
            return self._run_sde(sim_id, request)
        else:
            raise ValueError(f"Unsupported simulation mode: {request.mode}")

    def _run_mvp_sde(self, sim_id: str, request: SimulationRequest):  # type: ignore[no-untyped-def]
        """Запуск упрощённой 2-переменной MVP SDE модели (быстро)."""
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

        task_manager.update_progress(sim_id, 10, 100, "Running MVP SDE simulation...")
        trajectory = model.simulate(initial_params=initial_params)
        task_manager.update_progress(sim_id, 90, 100, "Saving results...")
        return trajectory

    def _run_sde(self, sim_id: str, request: SimulationRequest):  # type: ignore[no-untyped-def]
        """Запуск полной 20-переменной Extended SDE модели."""
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
            P=request.P0, Ne=request.Ne0, M1=request.M1_0, M2=request.M2_0,
            F=request.F0, Mf=request.Mf0, E=request.E0, S=request.S0,
            C_TNF=request.C_TNF0, C_IL10=request.C_IL10_0,
            D=request.D0, O2=request.O2_0,
        )

        task_manager.update_progress(sim_id, 10, 100, "Running SDE simulation...")

        def on_progress(current_step: int, total_steps: int) -> None:
            # Масштабируем прогресс модели (0-100%) в диапазон 10-90%
            model_pct = current_step / total_steps
            overall = 10 + model_pct * 80  # 10% → 90%
            task_manager.update_progress(
                sim_id,
                int(overall),
                100,
                f"Integrating SDE step {current_step}/{total_steps}...",
            )

        trajectory = model.simulate(
            initial_state=initial_state,
            progress_callback=on_progress,
        )
        task_manager.update_progress(sim_id, 90, 100, "Saving results...")
        return trajectory

    def _run_abm(self, sim_id: str, request: SimulationRequest):  # type: ignore[no-untyped-def]
        """Запуск ABM симуляции."""
        from src.core.abm_model import ABMConfig, simulate_abm

        task_manager.update_progress(sim_id, 5, 100, "Configuring ABM model...")

        config = ABMConfig(
            dt=request.dt,
            t_max=request.t_max_hours,
        )

        initial_params = self._build_mvp_params(request)

        task_manager.update_progress(sim_id, 10, 100, "Running ABM simulation...")
        trajectory = simulate_abm(
            initial_params=initial_params,
            config=config,
            random_seed=request.random_seed,
        )
        task_manager.update_progress(sim_id, 90, 100, "Saving results...")
        return trajectory

    def _save_result(self, sim_id: str, trajectory) -> None:  # type: ignore[no-untyped-def]
        """Сериализовать траекторию в .npz файл."""
        results_dir = Path(settings.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        result_path = results_dir / f"{sim_id}.npz"

        # Определяем тип траектории и извлекаем данные
        if hasattr(trajectory, "snapshots"):
            # ABMTrajectory
            times = trajectory.get_times()
            pop_dynamics = trajectory.get_population_dynamics()
            variables = {}
            for key, arr in pop_dynamics.items():
                variables[key] = arr
        elif hasattr(trajectory, "N_values"):
            # MVP SDETrajectory (2 переменных: N, C)
            times = trajectory.times
            variables = {
                "F": trajectory.N_values,   # Общая клеточная популяция → F для совместимости
                "C_TNF": trajectory.C_values,  # Общие цитокины → C_TNF для совместимости
            }
        else:
            # ExtendedSDETrajectory
            times = trajectory.times
            variables = {}
            from src.core.extended_sde import VARIABLE_NAMES
            for var_name in VARIABLE_NAMES:
                values = []
                for state in trajectory.states:
                    values.append(getattr(state, var_name, 0.0))
                variables[var_name] = np.array(values)

        np.savez(str(result_path), times=times, **variables)

        # Обновить путь в БД
        self._update_result_path(sim_id, str(result_path))

    def _update_db_status(
        self,
        sim_id: str,
        status: str,
        progress: float | None = None,
        error: str | None = None,
    ) -> None:
        """Обновить статус в БД (thread-safe через новый session)."""
        from src.db.session import SessionLocal

        db = SessionLocal()
        try:
            record = db.get(SimulationRecord, sim_id)
            if record:
                record.status = status
                if progress is not None:
                    record.progress = progress
                if error:
                    record.error_message = error
                if status in ("completed", "failed", "cancelled"):
                    record.completed_at = datetime.now(timezone.utc)
                db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def _update_result_path(self, sim_id: str, path: str) -> None:
        """Обновить путь к файлу результатов в БД."""
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
        """Получить статус симуляции."""
        record = self._db.get(SimulationRecord, sim_id)
        if record and task_manager.is_active(sim_id):
            record.progress = task_manager.get_progress(sim_id)
            record.message = task_manager.get_message(sim_id)
        return record

    @staticmethod
    def load_trajectory(sim_id: str) -> dict:
        """Загрузить траекторию из .npz файла."""
        result_path = Path(settings.results_dir) / f"{sim_id}.npz"
        if not result_path.exists():
            raise FileNotFoundError(f"Results not found for simulation {sim_id}")

        data = np.load(str(result_path))
        times = data["times"].tolist()
        variables = {}
        for key in data.files:
            if key != "times":
                variables[key] = data[key].tolist()
        return {"times": times, "variables": variables}
