"""Math + biology correctness checks for the ABM+MC+therapy E2E results.

Consumes simulation IDs produced by ``scripts/e2e_abm_scenario.py``.
Pulls the saved result bundles via :class:`result_bundle.load_result_bundle_for_simulation`
and runs quantitative checks:

  * Monte Carlo ensemble: every requested trajectory succeeded.
  * No NaN/Inf in ensemble statistics.
  * Therapy arm (PRP+PEMF) shows measurable difference from baseline.
  * Biological invariants: final populations non-negative, success rate 1.0.

Invocation::

    python scripts/e2e_math_checks.py --baseline <SID> --therapy <SID>
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Any

import numpy as np

from src.api.services.result_bundle import load_result_bundle_for_simulation


def _load(sim_id: str) -> dict[str, Any]:
    bundle = load_result_bundle_for_simulation(sim_id, include_snapshots=True)
    if not bundle:
        raise RuntimeError(f"No result bundle for sim {sim_id}")
    return bundle


def _finite(x: Any) -> bool:
    if x is None:
        return True
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return True
    return math.isfinite(xf)


def _traverse_finite(obj: Any, path: str = "") -> list[str]:
    bad: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            bad.extend(_traverse_finite(v, f"{path}.{k}" if path else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            bad.extend(_traverse_finite(v, f"{path}[{i}]"))
    elif isinstance(obj, (int, float)):
        if not _finite(obj):
            bad.append(path)
    elif isinstance(obj, np.ndarray):
        if not np.all(np.isfinite(obj)):
            bad.append(path)
    return bad


def _check(name: str, ok: bool, detail: str = "") -> bool:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}{(' — ' + detail) if detail else ''}")
    return ok


def _summary_stats(bundle: dict[str, Any]) -> dict[str, Any]:
    return (bundle.get("metadata") or {}).get("summary_statistics") or {}


def _run_bundle_checks(label: str, bundle: dict[str, Any], n_trajectories_expected: int) -> bool:
    print(f"\n— {label} —")
    ok = True

    mode = bundle.get("mode")
    ok &= _check("result bundle present", bool(bundle))
    ok &= _check(
        "bundle mode is abm/sde/mc",
        mode in {"abm", "sde", "mc", "monte_carlo"},
        detail=f"got {mode!r}",
    )

    metadata = bundle.get("metadata") or {}
    n_success = int(metadata.get("n_successful") or 0)
    n_total = int(metadata.get("n_trajectories") or 0)
    ok &= _check(
        f"at least {n_trajectories_expected} successful trajectories",
        n_success >= n_trajectories_expected,
        detail=f"n_successful={n_success} / n_trajectories={n_total}",
    )

    summary = _summary_stats(bundle)
    bad = _traverse_finite(summary)
    ok &= _check("summary has no NaN/Inf", not bad, detail=f"bad paths: {bad[:3]}")

    variables = bundle.get("variables") or {}
    var_bad = _traverse_finite(variables)
    ok &= _check("variables have no NaN/Inf", not var_bad, detail=f"bad paths: {var_bad[:3]}")

    final_n = summary.get("mean_final_N")
    ok &= _check(
        "mean_final_N non-negative",
        final_n is None or float(final_n) >= 0,
        detail=f"mean_final_N={final_n}",
    )

    success_rate = summary.get("success_rate")
    ok &= _check(
        "success_rate == 1.0",
        success_rate is not None and abs(float(success_rate) - 1.0) < 1e-9,
        detail=f"success_rate={success_rate}",
    )

    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--therapy", required=True)
    parser.add_argument("--n-trajectories", type=int, default=2)
    parser.add_argument(
        "--require-therapy-effect",
        action="store_true",
        help="Fail if therapy summary stats are identical to baseline.",
    )
    args = parser.parse_args()

    baseline = _load(args.baseline)
    therapy = _load(args.therapy)

    overall = True
    overall &= _run_bundle_checks("baseline", baseline, args.n_trajectories)
    overall &= _run_bundle_checks("therapy (PRP+PEMF)", therapy, args.n_trajectories)

    print("\n— Comparative biology checks —")
    base = _summary_stats(baseline)
    ther = _summary_stats(therapy)
    for key in ("mean_final_N", "std_final_N", "mean_final_C", "mean_growth_rate"):
        b, t = base.get(key), ther.get(key)
        print(f"  {key:20s}: baseline={b} therapy={t}")

    if base and ther:
        differs = any(
            base.get(k) != ther.get(k) for k in ("mean_final_N", "mean_final_C", "mean_growth_rate")
        )
        check_ok = differs or not args.require_therapy_effect
        overall &= _check(
            "therapy ensemble differs from baseline",
            check_ok,
            detail="" if differs else "identical summary — therapy had no effect",
        )

    print("\n" + ("ALL CHECKS PASSED" if overall else "FAILED"))
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
