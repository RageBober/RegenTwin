"""Micro-benchmark for ABM+Monte Carlo performance.

Runs `MonteCarloSimulator.run()` in serial and parallel modes with a fixed
configuration, measures wall-clock time, and prints/saves the comparison.

Usage:
    python scripts/benchmark_abm_mc.py --n-trajectories 4 --t-max-hours 18 --repeats 3
    python scripts/benchmark_abm_mc.py --n-trajectories 4 --output output/bench.json

NOT part of CI — manual verification tool for the PRP/PEMF + MC speedup work.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.core.abm_model import ABMConfig
from src.core.monte_carlo import MonteCarloConfig, MonteCarloSimulator
from src.core.sde_model import TherapyProtocol
from src.data.parameter_extraction import ModelParameters


def _build_initial_params(n0: float = 10000.0) -> ModelParameters:
    """Neutral baseline parameters — no therapy, default demographics.

    Note: ABMModel maps n0 → agents via max(10, int(n0/100)), capped by
    config.max_agents. Default n0=10000 gives ~100 agents — достаточно для того
    чтобы полезная работа превышала process-spawn overhead.
    """
    return ModelParameters(
        n0=n0,
        stem_cell_fraction=0.05,
        macrophage_fraction=0.15,
        apoptotic_fraction=0.02,
        c0=5.0,
        inflammation_level=0.3,
    )


def _run_once(
    n_trajectories: int,
    t_max_hours: float,
    use_multiprocessing: bool,
    n_jobs: int,
    seed: int,
    therapy: TherapyProtocol | None,
    n0: float = 100000.0,
) -> float:
    """Return wall-clock seconds for one MC run."""
    abm_config = ABMConfig(t_max=t_max_hours)
    mc_config = MonteCarloConfig(
        n_trajectories=n_trajectories,
        model_type="abm",
        abm_config=abm_config,
        base_seed=seed,
        n_jobs=n_jobs,
        use_multiprocessing=use_multiprocessing,
    )
    simulator = MonteCarloSimulator(config=mc_config, therapy=therapy)
    initial_params = _build_initial_params(n0=n0)

    t0 = time.perf_counter()
    _ = simulator.run(initial_params)
    return time.perf_counter() - t0


def _measure(
    label: str,
    repeats: int,
    fn,
) -> dict[str, Any]:
    """Run fn() `repeats` times, return stats dict."""
    samples: list[float] = []
    for i in range(repeats):
        dt = fn()
        samples.append(dt)
        print(f"  [{label}] run {i + 1}/{repeats}: {dt:.2f}s")
    return {
        "label": label,
        "repeats": repeats,
        "mean_s": statistics.mean(samples),
        "min_s": min(samples),
        "max_s": max(samples),
        "stdev_s": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "samples_s": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trajectories", type=int, default=4)
    parser.add_argument("--t-max-hours", type=float, default=18.0)
    parser.add_argument(
        "--n0",
        type=float,
        default=100000.0,
        help="Начальная плотность клеток. n0/100 (≥10) → число ABM-агентов.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--with-therapy", action="store_true", help="Run with PRP+PEMF therapy enabled."
    )
    parser.add_argument(
        "--serial-only",
        action="store_true",
        help="Skip parallel run (useful for 'before' baseline).",
    )
    parser.add_argument("--parallel-only", action="store_true", help="Skip serial run.")
    parser.add_argument(
        "--output", type=str, default=None, help="Optional path to save JSON report."
    )
    args = parser.parse_args()

    cpu = os.cpu_count() or 2
    n_jobs = max(1, cpu - 1)

    therapy: TherapyProtocol | None = None
    if args.with_therapy:
        therapy = TherapyProtocol(
            prp_enabled=True,
            prp_intensity=1.0,
            pemf_enabled=True,
            pemf_frequency=15.0,
            pemf_intensity=1.0,
        )

    print(
        f"Config: n_trajectories={args.n_trajectories}, "
        f"t_max_hours={args.t_max_hours}, n0={args.n0} (~{max(10, int(args.n0 / 100))} agents), "
        f"repeats={args.repeats}, "
        f"cpu_count={cpu}, n_jobs={n_jobs}, "
        f"therapy={'ON' if args.with_therapy else 'OFF'}"
    )

    results: dict[str, Any] = {
        "config": {
            "n_trajectories": args.n_trajectories,
            "t_max_hours": args.t_max_hours,
            "repeats": args.repeats,
            "cpu_count": cpu,
            "n_jobs": n_jobs,
            "with_therapy": args.with_therapy,
        },
    }

    if not args.parallel_only:
        print("\n--- SERIAL (use_multiprocessing=False) ---")
        results["serial"] = _measure(
            "serial",
            args.repeats,
            lambda: _run_once(
                args.n_trajectories, args.t_max_hours, False, 1, args.seed, therapy, args.n0
            ),
        )

    if not args.serial_only:
        print(f"\n--- PARALLEL (use_multiprocessing=True, n_jobs={n_jobs}) ---")
        results["parallel"] = _measure(
            "parallel",
            args.repeats,
            lambda: _run_once(
                args.n_trajectories, args.t_max_hours, True, n_jobs, args.seed, therapy, args.n0
            ),
        )

    if "serial" in results and "parallel" in results:
        speedup = results["serial"]["mean_s"] / results["parallel"]["mean_s"]
        results["speedup_mean"] = speedup
        print(f"\n>>> Speedup (mean): {speedup:.2f}×")

    print("\nSummary:")
    print(
        json.dumps(
            {k: v for k, v in results.items() if k in {"config", "speedup_mean"}},
            indent=2,
        )
    )
    for k in ("serial", "parallel"):
        if k in results:
            r = results[k]
            print(
                f"  {k}: mean={r['mean_s']:.2f}s  min={r['min_s']:.2f}s  "
                f"max={r['max_s']:.2f}s  stdev={r['stdev_s']:.2f}s"
            )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nReport saved → {out_path}")


if __name__ == "__main__":
    main()
