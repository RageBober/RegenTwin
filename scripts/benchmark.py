"""Главный orchestrator бенчмарков RegenTwin.

Запускает pytest-benchmark, собирает CPU/RAM/OS info через psutil/cpuinfo,
пишет финальный отчёт в `output/benchmarks/<machine_id>_<timestamp>.json`.

Usage:
    uv run python scripts/benchmark.py --label "laptop-i5-1240p"
    uv run python scripts/benchmark.py --label "ryzen-9-7950x" --quick
    uv run python scripts/benchmark.py --label "any" --profile
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = PROJECT_ROOT / "output" / "benchmarks"


def _system_info() -> dict[str, Any]:
    """Собрать информацию о CPU/RAM/OS текущей машины."""
    try:
        import cpuinfo
        import psutil
    except ImportError as exc:
        raise SystemExit("Не установлены psutil/py-cpuinfo. Запустите `uv sync`.") from exc

    info = cpuinfo.get_cpu_info()
    vm = psutil.virtual_memory()
    return {
        "cpu_brand": info.get("brand_raw", "unknown"),
        "cpu_arch": info.get("arch", platform.machine()),
        "physical_cores": psutil.cpu_count(logical=False) or 0,
        "logical_cores": psutil.cpu_count(logical=True) or 0,
        "ram_gb": round(vm.total / (1024**3), 1),
        "os": f"{platform.system()}-{platform.release()}",
        "python": platform.python_version(),
    }


def _machine_id(system: dict[str, Any]) -> str:
    """Стабильный короткий идентификатор машины (для дедупликации отчётов)."""
    raw = f"{system['cpu_brand']}|{system['physical_cores']}|{system['ram_gb']}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _run_pytest_benchmark(quick: bool, raw_json: Path) -> int:
    """Запустить pytest-benchmark, сохранить результат в raw_json."""
    args = [
        sys.executable,
        "-m",
        "pytest",
        "tests/performance/test_benchmarks.py",
        "--benchmark-only",
        f"--benchmark-json={raw_json}",
        "-q",
        "--no-cov",
    ]
    if quick:
        args.extend(["-k", "small or sobol"])
    return subprocess.call(args, cwd=PROJECT_ROOT)


def _parse_pytest_benchmark(raw_json: Path) -> dict[str, Any]:
    """Извлечь группу-mean-stdev из raw pytest-benchmark JSON."""
    if not raw_json.exists():
        return {}
    data = json.loads(raw_json.read_text(encoding="utf-8"))
    groups: dict[str, Any] = {}
    for bench in data.get("benchmarks", []):
        group = bench.get("group") or bench["name"]
        stats = bench["stats"]
        groups[group] = {
            "name": bench["name"],
            "mean": round(stats["mean"], 6),
            "stddev": round(stats["stddev"], 6),
            "min": round(stats["min"], 6),
            "max": round(stats["max"], 6),
            "rounds": stats["rounds"],
        }
    return groups


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="Human-readable label, e.g. 'laptop-i5'")
    parser.add_argument(
        "--quick", action="store_true", help="Только быстрые бенчмарки (small, sobol)"
    )
    parser.add_argument("--out", type=Path, default=None, help="Override output JSON path")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Дополнительно прогнать py-spy/scalene через scripts/profile_hotspots.py",
    )
    args = parser.parse_args()

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S")
    raw_json = BENCH_DIR / f"_raw_{timestamp}.json"

    print(f"[1/3] Pytest-benchmark (quick={args.quick})")
    rc = _run_pytest_benchmark(args.quick, raw_json)
    if rc != 0:
        print(f"pytest-benchmark exited with code {rc}", file=sys.stderr)

    print("[2/3] System info collection")
    system = _system_info()
    machine_id = _machine_id(system)
    benchmarks = _parse_pytest_benchmark(raw_json)

    payload: dict[str, Any] = {
        "machine_id": machine_id,
        "label": args.label,
        "timestamp": dt.datetime.now(dt.UTC).isoformat(),
        "system": system,
        "benchmarks": benchmarks,
        "quick_mode": args.quick,
    }

    out_path = args.out or BENCH_DIR / f"{machine_id}_{timestamp}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[3/3] Wrote benchmark JSON: {out_path}")

    if args.profile:
        print("\nRunning hotspot profiles (py-spy + scalene)...")
        subprocess.call(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "profile_hotspots.py")],
            cwd=PROJECT_ROOT,
        )

    print(f"\nDone. machine_id={machine_id}, label={args.label}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
