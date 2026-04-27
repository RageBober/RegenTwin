"""Celery tasks for simulation and analysis execution."""

from __future__ import annotations

from celery.contrib.abortable import AbortableTask
from loguru import logger

from src.tasks.celery_app import celery_app


@celery_app.task(base=AbortableTask, bind=True, name="regentwin.run_simulation")
def run_simulation_task(self, sim_id: str, request_dict: dict) -> dict:
    """Execute a simulation inside a Celery worker."""
    from src.api.models.schemas import SimulationRequest
    from src.api.services.simulation_service import SimulationCancelledError, SimulationService
    from src.db.session import SessionLocal

    request = SimulationRequest(**request_dict)
    db = SessionLocal()
    service = SimulationService(db)
    try:
        service._update_db_status(sim_id, "running", progress=5.0, message="Initializing...")

        def cancel_callback() -> None:
            if self.is_aborted():
                raise SimulationCancelledError("Simulation cancelled via Celery")

        def progress_reporter(sid: str, current: int, total: int, message: str | None = None) -> None:
            pct = (current / total) * 100 if total > 0 else 0.0
            self.update_state(state="PROGRESS", meta={"percent": pct, "message": message or f"Step {current}/{total}"})

        result = service._execute_simulation(sim_id, request, cancel_callback, progress_reporter)

        service._save_result(sim_id, request.mode.value, result)
        service._update_db_status(sim_id, "completed", progress=100.0, message="Completed")
        logger.info(f"Celery simulation {sim_id} completed")
        return {"status": "completed", "sim_id": sim_id}
    except SimulationCancelledError:
        service._update_db_status(sim_id, "cancelled", message="Cancelled")
        logger.info(f"Celery simulation {sim_id} cancelled")
        return {"status": "cancelled", "sim_id": sim_id}
    except Exception as exc:
        service._update_db_status(sim_id, "failed", error=str(exc))
        logger.error(f"Celery simulation {sim_id} failed: {exc}")
        raise
    finally:
        db.close()


@celery_app.task(base=AbortableTask, bind=True, name="regentwin.run_sensitivity")
def run_sensitivity_task(self, analysis_id: str, request_dict: dict) -> dict:
    """Execute sensitivity analysis inside a Celery worker."""
    from src.api.models.schemas import SensitivityRequest
    from src.api.services.analysis_service import AnalysisService
    from src.db.session import SessionLocal

    request = SensitivityRequest(**request_dict)
    db = SessionLocal()
    service = AnalysisService(db)
    try:
        service._update_db_status(analysis_id, "running")

        result = service._execute_sensitivity(analysis_id, request)

        service._update_db_result(analysis_id, "completed", result)
        logger.info(f"Celery sensitivity {analysis_id} completed")
        return {"status": "completed", "analysis_id": analysis_id}
    except Exception as exc:
        service._update_db_status(analysis_id, "failed", error=str(exc))
        logger.error(f"Celery sensitivity {analysis_id} failed: {exc}")
        raise
    finally:
        db.close()


@celery_app.task(base=AbortableTask, bind=True, name="regentwin.run_estimation")
def run_estimation_task(self, analysis_id: str, request_dict: dict) -> dict:
    """Execute parameter estimation inside a Celery worker."""
    from src.api.models.schemas import EstimationRequest
    from src.api.services.analysis_service import AnalysisService
    from src.db.session import SessionLocal

    request = EstimationRequest(**request_dict)
    db = SessionLocal()
    service = AnalysisService(db)
    try:
        service._update_db_status(analysis_id, "running")

        result = service._execute_estimation(analysis_id, request)

        service._update_db_result(analysis_id, "completed", result)
        logger.info(f"Celery estimation {analysis_id} completed")
        return {"status": "completed", "analysis_id": analysis_id}
    except Exception as exc:
        service._update_db_status(analysis_id, "failed", error=str(exc))
        logger.error(f"Celery estimation {analysis_id} failed: {exc}")
        raise
    finally:
        db.close()
