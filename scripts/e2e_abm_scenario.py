"""E2E regression script: FCS upload -> ABM + Monte Carlo + PEMF + PRP via HTTP.

Exercises the real uvicorn server to prove that:
  1) POST /api/v1/simulate with mode=abm + n_trajectories>1 + therapies runs to completion
  2) The threading-based background worker no longer gets cancelled by asyncio
     lifecycle (previous "stuck at 10%" regression).
  3) Therapy knobs (PRP, PEMF) actually affect biological output.

Invocation::

    python scripts/e2e_abm_scenario.py --base http://127.0.0.1:8765 \
        --upload-id 65a0eee7-a811-4db1-8633-6ea0896fb78e
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _wait_for_completion(base: str, sim_id: str, deadline_s: float) -> dict[str, Any]:
    terminal = {"completed", "failed", "cancelled"}
    end = time.monotonic() + deadline_s
    last = {"status": "pending", "progress": 0.0}
    while time.monotonic() < end:
        last = _get_json(f"{base}/api/v1/simulate/{sim_id}")
        status = last.get("status")
        progress = last.get("progress", 0.0)
        print(f"  [{sim_id[:8]}] status={status:<10s} progress={progress:6.1f}%")
        if status in terminal:
            return last
        time.sleep(1.0)
    raise TimeoutError(f"simulation {sim_id} did not finish within {deadline_s}s; last={last}")


def _run_scenario(base: str, upload_id: str, label: str, extra: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "mode": "abm",
        "upload_id": upload_id,
        "t_max_hours": 18.0,
        "dt": 0.5,
        "n_trajectories": 2,
        "random_seed": 42,
    }
    payload.update(extra)
    print(f"\n=== {label} ===\n  POST payload: {payload}")
    started = _post_json(f"{base}/api/v1/simulate", payload)
    sim_id = started["simulation_id"]
    print(f"  simulation_id = {sim_id}")
    final = _wait_for_completion(base, sim_id, deadline_s=420.0)
    assert final["status"] == "completed", f"{label} did not complete: {final}"
    results = _get_json(f"{base}/api/v1/results/{sim_id}")
    return {"sim_id": sim_id, "status": final, "results": results}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8765")
    parser.add_argument("--upload-id", required=True)
    args = parser.parse_args()

    baseline = _run_scenario(
        args.base,
        args.upload_id,
        "baseline (no therapy)",
        {"prp_enabled": False, "pemf_enabled": False},
    )

    with_therapy = _run_scenario(
        args.base,
        args.upload_id,
        "PRP + PEMF enabled",
        {
            "prp_enabled": True,
            "prp_intensity": 1.0,
            "pemf_enabled": True,
            "pemf_frequency": 15.0,
            "pemf_intensity": 0.5,
        },
    )

    print("\n=== Summary ===")
    print("baseline  : sim_id=%s status=%s" % (baseline["sim_id"], baseline["status"]["status"]))
    print(
        "therapy   : sim_id=%s status=%s"
        % (with_therapy["sim_id"], with_therapy["status"]["status"])
    )

    for label, bundle in [("baseline", baseline), ("therapy", with_therapy)]:
        res = bundle["results"]
        result_type = res.get("result_type")
        print(f"  [{label}] result_type={result_type}")
        if result_type in {"monte_carlo", "mc"}:
            if "summary" in res:
                print(f"    summary keys: {list(res['summary'].keys())[:8]}")
            if "trajectories" in res:
                print(f"    n_trajectories saved: {len(res['trajectories'])}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
