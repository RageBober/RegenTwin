"""Baseline-профайл и numerical baseline для оптимизаций.

Запускается перед изменениями, чтобы зафиксировать:
1. Топ функций по cumtime/tottime (cProfile)
2. Финальные значения 20 SDE-переменных (для проверки эквивалентности)
3. Wall-clock на единый длинный прогон

Запуск:
    python scripts/perf_baseline.py --tag baseline
    python scripts/perf_baseline.py --tag phase1

Артефакты пишутся в output/profiling/<tag>_<scenario>.{txt,json}
"""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.abm_model import ABMConfig, ABMModel  # noqa: E402
from src.core.extended_sde import ExtendedSDEModel, ExtendedSDEState  # noqa: E402
from src.core.parameters import ParameterSet  # noqa: E402
from src.data.parameter_extraction import ModelParameters  # noqa: E402

OUT_DIR = PROJECT_ROOT / "output" / "profiling"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _initial_state() -> ExtendedSDEState:
    return ExtendedSDEState(
        P=200.0,
        Ne=100.0,
        M1=50.0,
        M2=10.0,
        F=80.0,
        Mf=5.0,
        E=20.0,
        S=10.0,
        C_TNF=5.0,
        C_IL10=1.0,
        C_PDGF=2.0,
        C_VEGF=2.0,
        C_TGFb=1.0,
        C_MCP1=3.0,
        C_IL8=4.0,
        rho_collagen=0.1,
        C_MMP=0.5,
        rho_fibrin=0.5,
        D=1.0,
        O2=40.0,
        t=0.0,
    )


def _model_params() -> ModelParameters:
    return ModelParameters(
        n0=10000.0,
        stem_cell_fraction=0.05,
        macrophage_fraction=0.03,
        apoptotic_fraction=0.02,
        c0=10.0,
        inflammation_level=0.5,
    )


def _run_sde(t_max: float, dt: float) -> tuple[float, dict]:
    p = ParameterSet()
    p.t_max = t_max
    p.dt = dt
    initial = _initial_state()
    model = ExtendedSDEModel(params=p, rng_seed=42)
    t0 = time.perf_counter()
    traj = model.simulate(initial)
    wall = time.perf_counter() - t0

    final = traj.states[-1]
    final_state = {
        "P": final.P,
        "Ne": final.Ne,
        "M1": final.M1,
        "M2": final.M2,
        "F": final.F,
        "Mf": final.Mf,
        "E": final.E,
        "S": final.S,
        "C_TNF": final.C_TNF,
        "C_IL10": final.C_IL10,
        "C_PDGF": final.C_PDGF,
        "C_VEGF": final.C_VEGF,
        "C_TGFb": final.C_TGFb,
        "C_MCP1": final.C_MCP1,
        "C_IL8": final.C_IL8,
        "rho_collagen": final.rho_collagen,
        "C_MMP": final.C_MMP,
        "rho_fibrin": final.rho_fibrin,
        "D": final.D,
        "O2": final.O2,
        "t": final.t,
    }
    return wall, final_state


def _run_abm(t_max: float, max_agents: int) -> tuple[float, dict]:
    cfg = ABMConfig()
    cfg.t_max = t_max
    cfg.max_agents = max_agents
    model = ABMModel(config=cfg, random_seed=42)
    t0 = time.perf_counter()
    traj = model.simulate(_model_params(), snapshot_interval=t_max)
    wall = time.perf_counter() - t0

    final = traj.snapshots[-1]
    counts = final.get_agent_count_by_type()
    final_summary = {
        "n_agents_alive": int(final.get_total_agents()),
        "agent_counts": {k: int(v) for k, v in counts.items()},
        "cytokine_sum": float(final.cytokine_field.sum()),
        "cytokine_max": float(final.cytokine_field.max()),
        "ecm_sum": float(final.ecm_field.sum()),
        "ecm_max": float(final.ecm_field.max()),
        "t": float(final.t),
    }
    return wall, final_summary


def _profile(target_fn, label: str, tag: str) -> None:
    profiler = cProfile.Profile()
    profiler.enable()
    wall, baseline = target_fn()
    profiler.disable()

    txt_path = OUT_DIR / f"{tag}_{label}.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write(f"# {label} | tag={tag}\n")
        f.write(f"wall_seconds: {wall:.4f}\n\n")
        stats = pstats.Stats(profiler, stream=f).sort_stats("cumulative")
        f.write("## Top 30 by cumulative time\n")
        stats.print_stats(30)
        f.write("\n## Top 30 by total time (self)\n")
        stats.sort_stats("tottime").print_stats(30)

    json_path = OUT_DIR / f"{tag}_{label}.json"
    json_path.write_text(
        json.dumps({"wall_seconds": wall, "baseline": baseline}, indent=2),
        encoding="utf-8",
    )

    print(f"  {label}: {wall:.3f}s  -> {txt_path.name}, {json_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="baseline", help="Tag for output files")
    parser.add_argument(
        "--scenario",
        choices=("sde", "abm", "all"),
        default="all",
        help="Which scenarios to run",
    )
    parser.add_argument("--sde-tmax", type=float, default=720.0)
    parser.add_argument("--sde-dt", type=float, default=0.01)
    parser.add_argument("--abm-tmax", type=float, default=72.0)
    parser.add_argument("--abm-max-agents", type=int, default=200)
    args = parser.parse_args()

    print(f"Tag: {args.tag}")
    print(f"Output dir: {OUT_DIR}")

    if args.scenario in ("sde", "all"):
        print(f"SDE: t_max={args.sde_tmax}h dt={args.sde_dt}h")
        _profile(
            lambda: _run_sde(args.sde_tmax, args.sde_dt),
            f"sde_{int(args.sde_tmax)}h",
            args.tag,
        )

    if args.scenario in ("abm", "all"):
        print(f"ABM: t_max={args.abm_tmax}h max_agents={args.abm_max_agents}")
        _profile(
            lambda: _run_abm(args.abm_tmax, args.abm_max_agents),
            f"abm_{int(args.abm_tmax)}h",
            args.tag,
        )


if __name__ == "__main__":
    main()
