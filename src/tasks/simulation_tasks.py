"""Celery tasks for simulation and analysis execution."""

from __future__ import annotations

from pathlib import Path

from celery.contrib.abortable import AbortableTask
from loguru import logger

from src.tasks.celery_app import celery_app


def _staging_dir(sim_id: str) -> Path:
    """Directory where Monte Carlo trajectory workers drop their per-trajectory bundles.

    Shared across workers via `settings.results_dir`. Aggregator reads from here
    and removes the directory once the final bundle is saved.
    """
    from src.api.config import settings

    path = Path(settings.results_dir) / ".mc_staging" / sim_id
    path.mkdir(parents=True, exist_ok=True)
    return path


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

        def progress_reporter(
            sid: str, current: int, total: int, message: str | None = None
        ) -> None:
            pct = (current / total) * 100 if total > 0 else 0.0
            self.update_state(
                state="PROGRESS",
                meta={"percent": pct, "message": message or f"Step {current}/{total}"},
            )

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


@celery_app.task(base=AbortableTask, bind=True, name="regentwin.run_trajectory")
def run_trajectory_task(
    self,
    sim_id: str,
    traj_id: int,
    seed: int | None,
    request_dict: dict,
) -> dict:
    """Run ONE Monte Carlo trajectory in this worker; drop result to staging dir.

    Returns a small JSON-serializable dict (no numpy arrays) pointing to the
    per-trajectory bundle written to ``.mc_staging/{sim_id}/traj_{traj_id}.npz``.
    The aggregator loads all bundles and merges into a single MonteCarloResults.
    """
    import numpy as np

    from src.api.models.schemas import SimulationRequest
    from src.api.services.simulation_service import SimulationCancelledError, SimulationService
    from src.core.monte_carlo import MonteCarloConfig, MonteCarloSimulator
    from src.db.session import SessionLocal

    request = SimulationRequest(**request_dict)

    def cancel_callback() -> None:
        if self.is_aborted():
            raise SimulationCancelledError(f"Trajectory {traj_id} cancelled via Celery")

    db = SessionLocal()
    try:
        service = SimulationService(db)
        sde_cfg, abm_cfg, integ_cfg, ext_params, ext_state, model_type, therapy = (
            service._build_monte_carlo_configs(request)  # type: ignore[attr-defined]
        )
        initial_params = service._build_mvp_params(request)  # type: ignore[attr-defined]
    finally:
        db.close()

    mc_config = MonteCarloConfig(
        n_trajectories=1,
        model_type=model_type,
        sde_config=sde_cfg,
        abm_config=abm_cfg,
        integration_config=integ_cfg,
        extended_params=ext_params,
        extended_initial_state=ext_state,
        base_seed=seed,
        n_jobs=1,
        use_multiprocessing=False,
        cancel_callback=cancel_callback,
    )

    simulator = MonteCarloSimulator(config=mc_config, therapy=therapy)
    simulator._seeds = [seed]
    result = simulator._run_single_trajectory(traj_id, initial_params, seed)

    bundle_path = _staging_dir(sim_id) / f"traj_{traj_id:05d}.npz"
    from typing import Any as _Any

    payload: dict[str, _Any] = {
        "trajectory_id": np.asarray(result.trajectory_id),
        "random_seed": np.asarray(-1 if result.random_seed is None else result.random_seed),
        "final_N": np.asarray(result.final_N),
        "final_C": np.asarray(result.final_C),
        "max_N": np.asarray(result.max_N),
        "growth_rate": np.asarray(result.growth_rate),
        "success": np.asarray(1 if result.success else 0),
        "computation_time": np.asarray(result.computation_time),
    }
    if result.sde_trajectory is not None:
        payload["sde_times"] = result.sde_trajectory.times
        payload["sde_N"] = result.sde_trajectory.N_values
        payload["sde_C"] = result.sde_trajectory.C_values
    if result.abm_trajectory is not None:
        payload["abm_times"] = np.asarray(result.abm_trajectory.get_times())
        dyn = result.abm_trajectory.get_population_dynamics()
        payload["abm_stem"] = dyn.get("stem", np.asarray([]))
        payload["abm_macro"] = dyn.get("macro", np.asarray([]))
        payload["abm_fibro"] = dyn.get("fibro", np.asarray([]))
    np.savez(bundle_path, **payload)  # type: ignore[arg-type]

    return {
        "traj_id": traj_id,
        "seed": seed,
        "success": result.success,
        "error": result.error_message,
        "bundle": str(bundle_path),
    }


@celery_app.task(name="regentwin.aggregate_monte_carlo")
def aggregate_monte_carlo(
    traj_dicts: list[dict],
    sim_id: str,
    request_dict: dict,
) -> dict:
    """Chord callback: load all per-trajectory bundles, aggregate, save final bundle."""
    import shutil

    import numpy as np

    from src.api.models.schemas import SimulationRequest
    from src.api.services.simulation_service import SimulationService
    from src.core.monte_carlo import MonteCarloConfig, MonteCarloSimulator, TrajectoryResult
    from src.core.sde_model import SDETrajectory
    from src.db.session import SessionLocal

    request = SimulationRequest(**request_dict)

    trajectories: list[TrajectoryResult] = []
    for entry in sorted(traj_dicts, key=lambda e: e["traj_id"]):
        bundle_path = Path(entry["bundle"])
        if not bundle_path.exists():
            trajectories.append(
                TrajectoryResult(
                    trajectory_id=int(entry["traj_id"]),
                    random_seed=entry.get("seed"),
                    success=False,
                    error_message=f"missing bundle: {bundle_path}",
                )
            )
            continue

        # numpy safe-load: binary bytecode is blocked by allow_pickle default
        data = np.load(bundle_path)
        tr = TrajectoryResult(
            trajectory_id=int(data["trajectory_id"]),
            random_seed=int(data["random_seed"]) if int(data["random_seed"]) >= 0 else None,
            final_N=float(data["final_N"]),
            final_C=float(data["final_C"]),
            max_N=float(data["max_N"]),
            growth_rate=float(data["growth_rate"]),
            success=bool(int(data["success"])),
            computation_time=float(data["computation_time"]),
            error_message=entry.get("error"),
        )
        if "sde_times" in data.files:
            tr.sde_trajectory = SDETrajectory(
                times=data["sde_times"],
                N_values=data["sde_N"],
                C_values=data["sde_C"],
            )
        trajectories.append(tr)

    stub_config = MonteCarloConfig(
        n_trajectories=len(trajectories),
        model_type=request.mode.value if hasattr(request.mode, "value") else str(request.mode),
    )
    simulator = MonteCarloSimulator(config=stub_config)
    mc_results = simulator._aggregate_trajectories(trajectories)

    db = SessionLocal()
    try:
        service = SimulationService(db)
        service._save_result(sim_id, "monte_carlo", mc_results)
        service._update_db_status(sim_id, "completed", progress=100.0, message="Completed")
    finally:
        db.close()

    staging = _staging_dir(sim_id)
    try:
        shutil.rmtree(staging, ignore_errors=True)
    except Exception:
        pass

    return {"status": "completed", "sim_id": sim_id, "n_trajectories": len(trajectories)}


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
