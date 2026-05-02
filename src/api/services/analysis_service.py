"""Сервис для анализа чувствительности и параметрической идентификации."""

from __future__ import annotations

import asyncio
import threading
from datetime import UTC, datetime

from loguru import logger
from sqlalchemy.orm import Session

from src.api.models.schemas import EstimationRequest, SensitivityRequest
from src.db.models import AnalysisRecord


class AnalysisTaskManager:
    """In-memory реестр задач анализа."""

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task] = {}  # type: ignore[type-arg]
        self._celery_task_ids: dict[str, str] = {}
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
            self._celery_task_ids.pop(analysis_id, None)
            self._progress.pop(analysis_id, None)

    # ── Celery-specific methods ──────────────────────────────────────

    def register_celery(self, analysis_id: str, celery_task_id: str) -> None:
        with self._lock:
            self._celery_task_ids[analysis_id] = celery_task_id
            self._progress[analysis_id] = 0.0

    def is_celery_task(self, analysis_id: str) -> bool:
        with self._lock:
            return analysis_id in self._celery_task_ids

    def cancel_celery(self, analysis_id: str) -> bool:
        with self._lock:
            celery_id = self._celery_task_ids.get(analysis_id)
        if celery_id is None:
            return False
        from celery.contrib.abortable import AbortableAsyncResult

        from src.tasks.celery_app import celery_app

        AbortableAsyncResult(celery_id, app=celery_app).abort()
        return True


analysis_task_manager = AnalysisTaskManager()


class AnalysisService:
    """Бизнес-логика анализа: sensitivity и estimation."""

    def __init__(self, db: Session) -> None:
        self._db = db

    async def run_sensitivity(self, request: SensitivityRequest) -> AnalysisRecord:
        """Запуск анализа чувствительности в фоне."""
        from src.api.config import settings

        record = AnalysisRecord(
            analysis_type="sensitivity",
            params_json=request.model_dump(),
        )
        self._db.add(record)
        self._db.commit()
        self._db.refresh(record)

        if settings.use_celery:
            from src.tasks.simulation_tasks import run_sensitivity_task

            celery_result = run_sensitivity_task.delay(record.id, request.model_dump())  # type: ignore[attr-defined]
            analysis_task_manager.register_celery(record.id, celery_result.id)
        else:
            task = asyncio.create_task(self._sensitivity_background(record.id, request))
            analysis_task_manager.register(record.id, task)

        return record

    async def _sensitivity_background(self, analysis_id: str, request: SensitivityRequest) -> None:
        """Фоновое выполнение анализа чувствительности."""
        loop = asyncio.get_running_loop()
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
        except asyncio.CancelledError:
            self._update_db_status(analysis_id, "cancelled")
            logger.info(f"Sensitivity analysis {analysis_id} cancelled")
            raise
        except Exception as exc:
            self._update_db_status(analysis_id, "failed", error=str(exc))
            logger.error(f"Sensitivity analysis {analysis_id} failed: {exc}")
        finally:
            analysis_task_manager.cleanup(analysis_id)

    def _execute_sensitivity(self, analysis_id: str, request: SensitivityRequest) -> dict:
        """Синхронное выполнение Sobol/Morris анализа через SensitivityAnalyzer."""
        from src.core.extended_sde import ExtendedSDEModel
        from src.core.parameters import ParameterSet
        from src.core.sde_model import TherapyProtocol

        try:
            from src.core.sensitivity_analysis import (
                SensitivityAnalyzer,
                SensitivityConfig,
                SensitivityMethod,
            )
        except ImportError:
            logger.warning("SALib not installed, returning stub sensitivity result")
            return {
                "method": request.method,
                "parameters": request.parameters,
                "error": "SALib not installed",
                "n_samples": request.n_samples,
            }

        sim = request.simulation_params
        ps = ParameterSet()
        ps.dt = sim.dt
        ps.t_max = sim.t_max_hours

        # Bounds из единого источника
        bounds = ParameterSet.get_bounds(request.parameters)
        if not bounds:
            raise ValueError(
                f"None of the requested parameters {request.parameters} "
                "are valid sensitivity analysis parameters."
            )

        therapy = TherapyProtocol(
            prp_enabled=sim.prp_enabled,
            pemf_enabled=sim.pemf_enabled,
        )
        model = ExtendedSDEModel(
            params=ps,
            therapy=therapy,
            rng_seed=sim.random_seed or 42,
        )

        method = SensitivityMethod(request.method)
        config = SensitivityConfig(
            method=method,
            parameter_bounds=bounds,
            output_variables=["F"],
            t_span=(0.0, sim.t_max_hours),
            dt=sim.dt,
            rng_seed=sim.random_seed or 42,
        )
        analyzer = SensitivityAnalyzer(model=model, params=ps, config=config)

        def progress_cb(current: int, total: int) -> None:
            analysis_task_manager.update_progress(analysis_id, current, total)

        try:
            if method == SensitivityMethod.SOBOL:
                result = analyzer.run_sobol(
                    n_samples=request.n_samples,
                    progress_callback=progress_cb,
                )
                return {
                    "method": request.method,
                    "parameters": result.parameter_names,
                    "S1": result.S1.tolist(),
                    "ST": result.ST.tolist(),
                    "S1_conf": result.S1_conf.tolist(),
                    "ST_conf": result.ST_conf.tolist(),
                    "n_samples": result.n_samples,
                    "n_runs": result.n_model_runs,
                }
            else:  # MORRIS
                n_traj = max(10, request.n_samples // 32)
                result = analyzer.run_morris(
                    n_trajectories=n_traj,
                    progress_callback=progress_cb,
                )
                return {
                    "method": request.method,
                    "parameters": result.parameter_names,
                    "mu_star": result.mu_star.tolist(),
                    "sigma": result.sigma.tolist(),
                    "mu_star_conf": result.mu_star_conf.tolist(),
                    "n_samples": request.n_samples,
                    "n_runs": result.n_model_runs,
                }
        except ImportError:
            logger.warning("SALib not installed, returning stub sensitivity result")
            return {
                "method": request.method,
                "parameters": request.parameters,
                "error": "SALib not installed",
                "n_samples": request.n_samples,
            }

    async def run_estimation(self, request: EstimationRequest) -> AnalysisRecord:
        """Запуск параметрической идентификации в фоне."""
        from src.api.config import settings

        record = AnalysisRecord(
            analysis_type="estimation",
            params_json=request.model_dump(),
        )
        self._db.add(record)
        self._db.commit()
        self._db.refresh(record)

        if settings.use_celery:
            from src.tasks.simulation_tasks import run_estimation_task

            celery_result = run_estimation_task.delay(record.id, request.model_dump())  # type: ignore[attr-defined]
            analysis_task_manager.register_celery(record.id, celery_result.id)
        else:
            task = asyncio.create_task(self._estimation_background(record.id, request))
            analysis_task_manager.register(record.id, task)

        return record

    async def _estimation_background(self, analysis_id: str, request: EstimationRequest) -> None:
        """Фоновое выполнение параметрической идентификации."""
        loop = asyncio.get_running_loop()
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
        except asyncio.CancelledError:
            self._update_db_status(analysis_id, "cancelled")
            logger.info(f"Estimation {analysis_id} cancelled")
            raise
        except Exception as exc:
            self._update_db_status(analysis_id, "failed", error=str(exc))
            logger.error(f"Estimation {analysis_id} failed: {exc}")
        finally:
            analysis_task_manager.cleanup(analysis_id)

    def _execute_estimation(self, analysis_id: str, request: EstimationRequest) -> dict:
        """Синхронное выполнение MCMC/optimization через parameter_estimation ядро."""
        from src.core.parameter_estimation import EstimationConfig, estimate_parameters
        from src.db.models import UploadRecord
        from src.db.session import SessionLocal

        # 1. Загрузить upload record
        db = SessionLocal()
        try:
            upload = db.get(UploadRecord, request.upload_id)
            if upload is None:
                raise ValueError(f"Upload {request.upload_id} not found")
            file_path = upload.file_path
            upload_filename = upload.filename
        finally:
            db.close()

        # 2. Прочитать наблюдения (CSV с колонкой time; FCS — snapshot, не поддерживается)
        target = request.target_variable
        observed_data = _load_observed_timeseries(file_path, upload_filename, target)

        # 3. Маппинг API method → core method
        method_map = {"mcmc": "mcmc", "optimization": "mle"}
        core_method = method_map.get(request.method, "mcmc")

        # 4. Конфигурация
        config = EstimationConfig(
            observed_variables=[target],
            n_samples=request.n_samples,
            n_tune=min(request.n_samples // 2, 500),
        )

        # 5. Запуск
        logger.info(
            f"Estimation {analysis_id}: method={core_method}, "
            f"target={target}, n_samples={request.n_samples}"
        )
        result = estimate_parameters(
            observed_data=observed_data,
            method=core_method,
            estimated_param_names=None,
            config=config,
        )

        # 6. Сериализация EstimationResult → dict
        serialized: dict = {
            "method": result.method,
            "target_variable": target,
            "upload_id": request.upload_id,
            "point_estimates": result.point_estimates,
            "ci_lower": result.ci_lower,
            "ci_upper": result.ci_upper,
            "log_likelihood": result.log_likelihood,
            "aic": result.aic,
            "bic": result.bic,
            "n_observations": result.n_observations,
            "n_estimated_params": result.n_estimated_params,
            "elapsed_seconds": result.elapsed_seconds,
            "n_samples": request.n_samples,
            "diagnostics": {
                "converged": result.diagnostics.converged,
                "rhat": result.diagnostics.rhat,
                "ess_bulk": result.diagnostics.ess_bulk,
                "ess_tail": result.diagnostics.ess_tail,
                "warnings": result.diagnostics.warnings,
            }
            if result.diagnostics
            else None,
            "n_chains": result.config.n_chains if result.config else 1,
        }

        # Posterior samples для визуализации (plot_posterior / plot_convergence)
        if result.posterior_samples is not None:
            serialized["posterior_samples"] = {
                k: v.tolist() if hasattr(v, "tolist") else list(v)
                for k, v in result.posterior_samples.items()
            }

        return serialized

    def get_analysis(self, analysis_id: str) -> AnalysisRecord | None:
        """Получить запись анализа."""
        record = self._db.get(AnalysisRecord, analysis_id)
        if record:
            progress = analysis_task_manager.get_progress(analysis_id)
            if progress > 0:
                record.progress = progress
        return record

    def cancel_analysis(self, analysis_id: str) -> AnalysisRecord | None:
        """Cancel an active analysis and mark it as cancelled."""
        record = self._db.get(AnalysisRecord, analysis_id)
        if record is None:
            return None
        if record.status in {"completed", "failed", "cancelled"}:
            return record

        cancelled = analysis_task_manager.cancel(analysis_id)
        if not cancelled and analysis_task_manager.is_celery_task(analysis_id):
            cancelled = analysis_task_manager.cancel_celery(analysis_id)

        if cancelled:
            record.status = "cancelled"
            record.progress = record.progress or analysis_task_manager.get_progress(analysis_id)
            record.completed_at = datetime.now(UTC)
            self._db.commit()
            self._db.refresh(record)
            analysis_task_manager.cleanup(analysis_id)

        return record

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
                    record.completed_at = datetime.now(UTC)
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
                record.completed_at = datetime.now(UTC)
                db.commit()
        finally:
            db.close()


def _load_observed_timeseries(
    file_path: str,
    upload_filename: str,
    target: str,
) -> TimeSeriesData:
    """Load observed time-series from an uploaded CSV.

    Parameter estimation needs a time-series. FCS files contain a single
    cytometry snapshot and are rejected here with a clear message.
    """
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from src.data.dataset_loader import TimeSeriesData

    suffix = Path(upload_filename).suffix.lower()
    if suffix == ".fcs":
        raise ValueError(
            "Parameter estimation requires a time-series CSV "
            "(columns: time, <target>). FCS files contain a single snapshot "
            "and are only usable as initial conditions for simulation."
        )
    if suffix != ".csv":
        raise ValueError(
            f"Unsupported upload format '{suffix}'. Upload a CSV with a 'time' column."
        )

    df = pd.read_csv(file_path)
    if "time" not in df.columns:
        raise ValueError("Uploaded CSV must contain a 'time' column.")
    if target not in df.columns:
        raise ValueError(f"Target variable '{target}' not found in CSV columns: {list(df.columns)}")

    variable_columns = [c for c in df.columns if c != "time"]
    time_points = df["time"].to_numpy(dtype=np.float64)
    values = {col: df[col].to_numpy(dtype=np.float64) for col in variable_columns}
    units = {col: "cells/mm2" for col in variable_columns}

    return TimeSeriesData(time_points=time_points, values=values, units=units)
