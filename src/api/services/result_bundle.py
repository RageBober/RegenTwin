"""Utilities for saving and loading simulation result bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from src.core.abm_model import ABMSnapshot
    from src.core.extended_sde import ExtendedSDETrajectory

from src.api.config import settings
from src.db.models import SimulationRecord
from src.db.session import SessionLocal

RESULT_BUNDLE_VERSION = 3


def result_path_for_record(record: SimulationRecord) -> Path:
    """Resolve the authoritative result path for a simulation record."""
    base = Path(settings.results_dir).resolve()
    if record.result_path:
        candidate = Path(record.result_path).resolve()
        if not str(candidate).startswith(str(base)):
            raise ValueError(f"Result path escapes results directory: {record.result_path}")
        return candidate
    return base / f"{record.id}.npz"


def save_result_bundle(result_path: Path, mode: str, trajectory: Any) -> None:
    """Persist a simulation result bundle to ``.npz``."""
    payload: dict[str, Any] = {}
    metadata: dict[str, Any] = {
        "version": RESULT_BUNDLE_VERSION,
        "mode": mode,
    }

    if hasattr(trajectory, "mean_N") and hasattr(trajectory, "quantiles_N"):
        # MonteCarloResults — ensemble statistics
        times = trajectory.times
        variables: dict[str, Any] = {
            "mean_N": np.asarray(trajectory.mean_N, dtype=np.float64),
            "std_N": np.asarray(trajectory.std_N, dtype=np.float64),
            "mean_C": np.asarray(trajectory.mean_C, dtype=np.float64),
            "std_C": np.asarray(trajectory.std_C, dtype=np.float64),
        }
        for q, vals in trajectory.quantiles_N.items():
            variables[f"q{q:.2f}_N"] = np.asarray(vals, dtype=np.float64)
        for q, vals in trajectory.quantiles_C.items():
            variables[f"q{q:.2f}_C"] = np.asarray(vals, dtype=np.float64)
        metadata["n_trajectories"] = trajectory.n_successful + trajectory.n_failed
        metadata["n_successful"] = trajectory.n_successful
        metadata["total_computation_time"] = trajectory.total_computation_time
        metadata["summary_statistics"] = trajectory.get_summary_statistics()
        metadata["supported_exports"] = ["csv"]

        # Extended MC: сохранить статистику для всех 20 переменных
        if hasattr(trajectory, "variable_means") and trajectory.variable_means:
            for var_name, arr in trajectory.variable_means.items():
                variables[f"mean_{var_name}"] = np.asarray(arr, dtype=np.float64)
            for var_name, arr in trajectory.variable_stds.items():
                variables[f"std_{var_name}"] = np.asarray(arr, dtype=np.float64)
            for var_name, q_dict in trajectory.variable_quantiles.items():
                for q, vals in q_dict.items():
                    variables[f"q{q:.2f}_{var_name}"] = np.asarray(vals, dtype=np.float64)
            metadata["extended_mc"] = True
            metadata["supported_exports"] = ["csv", "png", "svg", "pdf"]
    elif hasattr(trajectory, "snapshots"):
        times = trajectory.get_times()
        variables = trajectory.get_population_dynamics()
        metadata["snapshot_count"] = len(trajectory.snapshots)
        metadata["supported_exports"] = ["csv"]

        for idx, snapshot in enumerate(trajectory.snapshots):
            prefix = f"snapshot_{idx}"
            alive_agents = [agent for agent in snapshot.agents if agent.alive]
            payload[f"{prefix}_t"] = np.array([snapshot.t], dtype=np.float64)
            payload[f"{prefix}_x"] = np.asarray(
                [agent.x for agent in alive_agents], dtype=np.float64
            )
            payload[f"{prefix}_y"] = np.asarray(
                [agent.y for agent in alive_agents], dtype=np.float64
            )
            payload[f"{prefix}_type"] = np.asarray(
                [agent.agent_type for agent in alive_agents],
                dtype=np.str_,
            )
            payload[f"{prefix}_energy"] = np.asarray(
                [agent.energy for agent in alive_agents],
                dtype=np.float64,
            )
            payload[f"{prefix}_age"] = np.asarray(
                [agent.age for agent in alive_agents],
                dtype=np.float64,
            )
            payload[f"{prefix}_division_count"] = np.asarray(
                [agent.division_count for agent in alive_agents],
                dtype=np.int32,
            )
            payload[f"{prefix}_cytokine_field"] = np.asarray(
                snapshot.cytokine_field, dtype=np.float64
            )
            payload[f"{prefix}_ecm_field"] = np.asarray(snapshot.ecm_field, dtype=np.float64)
    elif hasattr(trajectory, "sde_trajectory"):
        # Integrated mode: extract the inner SDE trajectory
        inner_sde = trajectory.sde_trajectory
        if hasattr(inner_sde, "states"):
            from src.core.extended_sde import VARIABLE_NAMES as INT_VARS

            times = inner_sde.times
            variables = {
                name: np.asarray(
                    [getattr(state, name, 0.0) for state in inner_sde.states],
                    dtype=np.float64,
                )
                for name in INT_VARS
            }
        else:
            times = inner_sde.times
            variables = {
                "N": inner_sde.N_values,
                "C": inner_sde.C_values,
            }
        metadata["abm_snapshot_count"] = len(trajectory.abm_trajectory.snapshots)
        metadata["supported_exports"] = ["csv", "png", "svg", "pdf"]
    elif hasattr(trajectory, "N_values"):
        times = trajectory.times
        variables = {
            "N": trajectory.N_values,
            "C": trajectory.C_values,
        }
        metadata["supported_exports"] = ["csv"]
    else:
        from src.core.extended_sde import VARIABLE_NAMES

        times = trajectory.times
        variables = {
            name: np.asarray(
                [getattr(state, name, 0.0) for state in trajectory.states], dtype=np.float64
            )
            for name in VARIABLE_NAMES
        }
        metadata["supported_exports"] = ["csv", "png", "svg", "pdf"]

    payload["times"] = np.asarray(times, dtype=np.float64)
    metadata["series"] = list(variables.keys())
    for name, values in variables.items():
        payload[name] = np.asarray(values, dtype=np.float64)

    payload["__meta__"] = np.asarray(json.dumps(metadata))
    result_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(result_path), **payload)


def load_result_bundle_for_simulation(
    simulation_id: str,
    *,
    include_snapshots: bool = False,
) -> dict[str, Any]:
    """Load a saved result bundle using the DB record as source of truth."""
    db = SessionLocal()
    try:
        record = db.get(SimulationRecord, simulation_id)
        if record is None:
            raise FileNotFoundError(f"Simulation {simulation_id} not found")
        result_path = result_path_for_record(record)
        return load_result_bundle(
            result_path,
            default_mode=str(record.mode),
            include_snapshots=include_snapshots,
        )
    finally:
        db.close()


def load_result_bundle(
    result_path: Path,
    *,
    default_mode: str | None = None,
    include_snapshots: bool = False,
) -> dict[str, Any]:
    """Load a result bundle from disk."""
    if not result_path.exists():
        raise FileNotFoundError(f"Results not found: {result_path}")

    with np.load(str(result_path), allow_pickle=False) as data:
        if "__meta__" in data.files:
            raw_meta = data["__meta__"]
            meta_text = raw_meta.item() if raw_meta.shape == () else raw_meta.tolist()
            metadata = json.loads(str(meta_text))
        else:
            metadata = {
                "version": 1,
                "mode": default_mode or "extended",
                "series": [key for key in data.files if key != "times"],
            }

        series = metadata.get("series") or [
            key for key in data.files if key not in {"times", "__meta__"}
        ]
        result: dict[str, Any] = {
            "mode": metadata.get("mode", default_mode or "extended"),
            "times": data["times"].tolist(),
            "variables": {name: data[name].tolist() for name in series if name in data.files},
            "metadata": {
                key: value
                for key, value in metadata.items()
                if key not in {"version", "mode", "series"}
            },
        }

        if include_snapshots:
            snapshot_count = int(metadata.get("snapshot_count", 0))
            snapshots: list[dict[str, Any]] = []
            for idx in range(snapshot_count):
                prefix = f"snapshot_{idx}"
                division_key = f"{prefix}_division_count"
                division_count = (
                    data[division_key].tolist()
                    if division_key in data.files
                    else [0] * len(data[f"{prefix}_x"])
                )
                snapshots.append(
                    {
                        "t": float(data[f"{prefix}_t"][0]),
                        "x": data[f"{prefix}_x"].tolist(),
                        "y": data[f"{prefix}_y"].tolist(),
                        "type": data[f"{prefix}_type"].tolist(),
                        "energy": data[f"{prefix}_energy"].tolist(),
                        "age": data[f"{prefix}_age"].tolist(),
                        "division_count": division_count,
                        "cytokine_field": data[f"{prefix}_cytokine_field"].tolist(),
                        "ecm_field": data[f"{prefix}_ecm_field"].tolist(),
                    }
                )
            result["snapshots"] = snapshots

        return result


def build_extended_trajectory(result: dict[str, Any]) -> ExtendedSDETrajectory:
    """Reconstruct an ExtendedSDETrajectory from an extended result bundle."""
    from src.core.extended_sde import VARIABLE_NAMES, ExtendedSDEState, ExtendedSDETrajectory
    from src.core.parameters import ParameterSet

    if result["mode"] not in {"extended", "integrated"}:
        raise ValueError(f"Extended trajectory unavailable for mode {result['mode']}")

    times = np.asarray(result["times"], dtype=np.float64)
    states = []
    variables = result["variables"]
    missing = [name for name in VARIABLE_NAMES if name not in variables]
    if missing:
        raise ValueError(
            "Result was saved with a legacy variable format and cannot be displayed. "
            "Please run a new simulation."
        )
    for idx, t in enumerate(times):
        state_data = {name: float(variables[name][idx]) for name in VARIABLE_NAMES}
        state_data["t"] = float(t)
        states.append(ExtendedSDEState(**state_data))
    return ExtendedSDETrajectory(times=times, states=states, params=ParameterSet())


def build_mc_mean_trajectory(result: dict[str, Any]) -> ExtendedSDETrajectory:
    """Reconstruct an ExtendedSDETrajectory from MC mean values."""
    from src.core.extended_sde import VARIABLE_NAMES, ExtendedSDEState, ExtendedSDETrajectory
    from src.core.parameters import ParameterSet

    times = np.asarray(result["times"], dtype=np.float64)
    variables = result["variables"]
    states = []
    for idx in range(len(times)):
        state_data: dict[str, float] = {"t": float(times[idx])}
        for name in VARIABLE_NAMES:
            mean_key = f"mean_{name}"
            if mean_key in variables:
                state_data[name] = float(variables[mean_key][idx])
            else:
                state_data[name] = 0.0
        states.append(ExtendedSDEState(**state_data))
    return ExtendedSDETrajectory(times=times, states=states, params=ParameterSet())


def build_abm_snapshot(result: dict[str, Any], timestep: int = -1) -> ABMSnapshot:
    """Reconstruct an ABMSnapshot from a saved ABM bundle."""
    from src.core.abm_model import ABMSnapshot, AgentState

    if result["mode"] != "abm":
        raise ValueError(f"ABM snapshot unavailable for mode {result['mode']}")

    snapshots = result.get("snapshots") or []
    if not snapshots:
        raise FileNotFoundError("ABM snapshots were not saved for this simulation")

    if timestep < 0:
        idx = len(snapshots) - 1
    else:
        idx = min(timestep, len(snapshots) - 1)
    snap = snapshots[idx]
    division_counts = snap.get("division_count") or [0] * len(snap["x"])

    agents = [
        AgentState(
            agent_id=agent_idx,
            agent_type=str(agent_type),
            x=float(x),
            y=float(y),
            age=float(age),
            division_count=int(division_count),
            energy=float(energy),
            alive=True,
        )
        for agent_idx, (x, y, agent_type, energy, age, division_count) in enumerate(
            zip(
                snap["x"],
                snap["y"],
                snap["type"],
                snap["energy"],
                snap["age"],
                division_counts,
                strict=False,
            )
        )
    ]
    return ABMSnapshot(
        t=float(snap["t"]),
        agents=agents,
        cytokine_field=np.asarray(snap["cytokine_field"], dtype=np.float64),
        ecm_field=np.asarray(snap["ecm_field"], dtype=np.float64),
    )
