"""Сервис для анализа чувствительности и параметрической идентификации."""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timezone

from loguru import logger
from sqlalchemy.orm import Session

from src.api.models.schemas import EstimationRequest, SensitivityRequest, SimulationMode
from src.db.models import AnalysisRecord


class AnalysisTaskManager:
    """In-memory реестр задач анализа."""

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task] = {}  # type: ignore[type-arg]
        self._progress: dict[str, float] = {}
        self._lock = threading.Lock()

    def register(self, analysis_id: str, task: asyncio.Task) -> None:  # type: ignore[type-arg]
        with self._lock:
            self._tasks[analysis_id] = task
            self._progress[analysis_id] = 0.0

    def cancel(self, analysis_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(analysis_id)
        if task and not task.done():
            task.cancel()
            return True
        return False

    def get_progress(self, analysis_id: str) -> float:
        with self._lock:
            return self._progress.get(analysis_id, 0.0)

    def update_progress(self, analysis_id: str, current: int, total: int) -> None:
        with self._lock:
            self._progress[analysis_id] = (current / total) * 100 if total > 0 else 0.0

    def cleanup(self, analysis_id: str) -> None:
        with self._lock:
            self._tasks.pop(analysis_id, None)
            self._progress.pop(analysis_id, None)


analysis_task_manager = AnalysisTaskManager()


class AnalysisService:
    """Бизнес-логика анализа: sensitivity и estimation."""

    def __init__(self, db: Session) -> None:
        self._db = db

    async def run_sensitivity(self, request: SensitivityRequest) -> AnalysisRecord:
        """Запуск анализа чувствительности в фоне."""
        record = AnalysisRecord(
            analysis_type="sensitivity",
            params_json=request.model_dump(),
        )
        self._db.add(record)
        self._db.commit()
        self._db.refresh(record)

        task = asyncio.create_task(
            self._sensitivity_background(record.id, request)
        )
        analysis_task_manager.register(record.id, task)

        return record

    async def _sensitivity_background(self, analysis_id: str, request: SensitivityRequest) -> None:
        """Фоновое выполнение анализа чувствительности."""
        loop = asyncio.get_event_loop()
        try:
            self._update_db_status(analysis_id, "running")

            result = await loop.run_in_executor(
                None,
                self._execute_sensitivity,
                analysis_id,
                request,
            )

            self._update_db_result(analysis_id, "completed", result)
            logger.info(f"Sensitivity analysis {analysis_id} completed")
        except Exception as exc:
            self._update_db_status(analysis_id, "failed", error=str(exc))
            logger.error(f"Sensitivity analysis {analysis_id} failed: {exc}")
        finally:
            analysis_task_manager.cleanup(analysis_id)

    def _execute_sensitivity(self, analysis_id: str, request: SensitivityRequest) -> dict:
        """Синхронное выполнение Sobol/Morris анализа."""
        import numpy as np

        from src.core.extended_sde import ExtendedSDEModel, ExtendedSDEState
        from src.core.parameters import ParameterSet
        from src.core.sde_model import TherapyProtocol

        params = request.simulation_params
        n_samples = request.n_samples
        param_names = request.parameters

        # Определить границы параметров
        bounds = self._get_parameter_bounds(param_names)

        try:
            from SALib.sample import saltelli as saltelli_sample
            from SALib.analyze import sobol as sobol_analyze

            problem = {
                "num_vars": len(param_names),
                "names": param_names,
                "bounds": bounds,
            }

            # Генерация сэмплов
            param_values = saltelli_sample.sample(problem, n_samples)
            n_runs = len(param_values)
            outputs = []

            for i, sample in enumerate(param_values):
                analysis_task_manager.update_progress(analysis_id, i, n_runs)

                # Создать ParameterSet с изменёнными параметрами
                ps = ParameterSet()
                for j, name in enumerate(param_names):
                    if hasattr(ps, name):
                        setattr(ps, name, sample[j])

                therapy = TherapyProtocol(
                    prp_enabled=params.prp_enabled,
                    pemf_enabled=params.pemf_enabled,
                )

                model = ExtendedSDEModel(params=ps, therapy=therapy, rng_seed=42)

                initial_state = ExtendedSDEState(
                    P=params.P0, Ne=params.Ne0, M1=params.M1_0, M2=params.M2_0,
                    F=params.F0, Mf=params.Mf0, E=params.E0, S=params.S0,
                    C_TNF=params.C_TNF0, C_IL10=params.C_IL10_0,
                    D=params.D0, O2=params.O2_0,
                )

                ps.dt = params.dt
                ps.t_max = params.t_max_hours

                try:
                    traj = model.simulate(initial_state=initial_state)
                    # Берём финальное значение фибробластов как целевой выход
                    final_F = float(traj.states[-1].F)
                    if not np.isfinite(final_F):
                        final_F = 0.0
                except Exception:
                    final_F = 0.0
                outputs.append(final_F)

            Y = np.array(outputs, dtype=np.float64)

            # Проверка: если все значения одинаковы, Sobol анализ невозможен
            if Y.std() == 0:
                return {
                    "method": request.method,
                    "parameters": param_names,
                    "S1": [0.0] * len(param_names),
                    "ST": [0.0] * len(param_names),
                    "S1_conf": [0.0] * len(param_names),
                    "ST_conf": [0.0] * len(param_names),
                    "n_samples": n_samples,
                    "n_runs": n_runs,
                    "warning": "All outputs identical — sensitivity indices undefined",
                }

            # Sobol анализ
            si = sobol_analyze.analyze(problem, Y)

            def _safe_list(arr):
                """Конвертировать numpy array в list, заменяя NaN на 0."""
                result = np.asarray(arr, dtype=np.float64)
                result = np.where(np.isfinite(result), result, 0.0)
                return result.tolist()

            return {
                "method": request.method,
                "parameters": param_names,
                "S1": _safe_list(si["S1"]),
                "ST": _safe_list(si["ST"]),
                "S1_conf": _safe_list(si["S1_conf"]),
                "ST_conf": _safe_list(si["ST_conf"]),
                "n_samples": n_samples,
                "n_runs": n_runs,
            }

        except ImportError:
            # SALib не установлен — вернуть заглушку
            logger.warning("SALib not installed, returning stub sensitivity result")
            return {
                "method": request.method,
                "parameters": param_names,
                "error": "SALib not installed",
                "n_samples": n_samples,
            }

    async def run_estimation(self, request: EstimationRequest) -> AnalysisRecord:
        """Запуск параметрической идентификации в фоне."""
        record = AnalysisRecord(
            analysis_type="estimation",
            params_json=request.model_dump(),
        )
        self._db.add(record)
        self._db.commit()
        self._db.refresh(record)

        task = asyncio.create_task(
            self._estimation_background(record.id, request)
        )
        analysis_task_manager.register(record.id, task)

        return record

    async def _estimation_background(self, analysis_id: str, request: EstimationRequest) -> None:
        """Фоновое выполнение параметрической идентификации."""
        loop = asyncio.get_event_loop()
        try:
            self._update_db_status(analysis_id, "running")

            result = await loop.run_in_executor(
                None,
                self._execute_estimation,
                analysis_id,
                request,
            )

            self._update_db_result(analysis_id, "completed", result)
            logger.info(f"Estimation {analysis_id} completed")
        except Exception as exc:
            self._update_db_status(analysis_id, "failed", error=str(exc))
            logger.error(f"Estimation {analysis_id} failed: {exc}")
        finally:
            analysis_task_manager.cleanup(analysis_id)

    def _execute_estimation(self, analysis_id: str, request: EstimationRequest) -> dict:
        """Синхронное выполнение MCMC/optimization."""
        # Загрузить FCS-данные из upload
        from src.db.models import UploadRecord
        from src.db.session import SessionLocal

        db = SessionLocal()
        try:
            upload = db.get(UploadRecord, request.upload_id)
            if upload is None:
                raise ValueError(f"Upload {request.upload_id} not found")
        finally:
            db.close()

        try:
            import emcee
            import numpy as np

            # Заглушка для MCMC — реальная реализация требует Phase 3
            logger.info(f"Running estimation with method={request.method}, n_samples={request.n_samples}")
            return {
                "method": request.method,
                "target_variable": request.target_variable,
                "upload_id": request.upload_id,
                "status": "stub",
                "message": "Full MCMC estimation requires Phase 3 (Analysis & Validation) completion",
                "n_samples": request.n_samples,
            }
        except ImportError:
            return {
                "method": request.method,
                "error": "emcee not installed",
                "n_samples": request.n_samples,
            }

    def get_analysis(self, analysis_id: str) -> AnalysisRecord | None:
        """Получить запись анализа."""
        record = self._db.get(AnalysisRecord, analysis_id)
        if record:
            progress = analysis_task_manager.get_progress(analysis_id)
            if progress > 0:
                record.progress = progress
        return record

    @staticmethod
    def _get_parameter_bounds(param_names: list[str]) -> list[list[float]]:
        """Получить границы параметров для sensitivity analysis."""
        # Типичные границы для параметров ParameterSet
        default_bounds: dict[str, list[float]] = {
            "r_F": [0.01, 0.1],
            "r_E": [0.005, 0.05],
            "r_S": [0.005, 0.03],
            "K_F": [1e5, 1e6],
            "K_E": [5e4, 5e5],
            "K_S": [5e3, 5e4],
            "delta_F": [0.001, 0.01],
            "delta_Ne": [0.01, 0.1],
            "delta_M": [0.005, 0.05],
            "k_switch": [0.005, 0.05],
            "sigma_F": [0.005, 0.05],
            "sigma_P": [0.01, 0.1],
            "sigma_TNF": [0.01, 0.1],
        }
        return [default_bounds.get(name, [0.1, 10.0]) for name in param_names]

    def _update_db_status(self, analysis_id: str, status: str, error: str | None = None) -> None:
        from src.db.session import SessionLocal

        db = SessionLocal()
        try:
            record = db.get(AnalysisRecord, analysis_id)
            if record:
                record.status = status
                if error:
                    record.result_json = {"error": error}
                if status in ("completed", "failed"):
                    record.completed_at = datetime.now(timezone.utc)
                db.commit()
        finally:
            db.close()

    def _update_db_result(self, analysis_id: str, status: str, result: dict) -> None:
        from src.db.session import SessionLocal

        db = SessionLocal()
        try:
            record = db.get(AnalysisRecord, analysis_id)
            if record:
                record.status = status
                record.progress = 100.0
                record.result_json = result
                record.completed_at = datetime.now(timezone.utc)
                db.commit()
        finally:
            db.close()
