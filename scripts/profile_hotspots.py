"""Профилирование горячих функций через py-spy и scalene.

Генерирует:
- output/benchmarks/profiles/<func>.svg (py-spy flamegraph для embed в MkDocs)
- output/benchmarks/profiles/<func>.speedscope.json (для интерактивного анализа)
- output/benchmarks/profiles/<func>.scalene.html (scalene line-level)

Usage:
    uv run python scripts/profile_hotspots.py
    uv run python scripts/profile_hotspots.py --only sde

WINDOWS:
    py-spy на Windows может потребовать запуска от admin (ReadProcessMemory).
    Если падает — используйте `--py-spy-nonblocking` (менее точный, но работает без admin).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGETS_DIR = PROJECT_ROOT / "scripts" / "_scalene_targets"
PROFILES_DIR = PROJECT_ROOT / "output" / "benchmarks" / "profiles"

TARGETS = {
    "sde": "sde_target.py",
    "abm": "abm_target.py",
    "mc": "mc_target.py",
    "sobol": "sobol_target.py",
}


def _run_py_spy(name: str, target: Path, nonblocking: bool) -> None:
    """Запустить py-spy record и сохранить flamegraph SVG + speedscope JSON."""
    if shutil.which("py-spy") is None:
        print("py-spy not found in PATH. Skipping flamegraph.", file=sys.stderr)
        return

    svg = PROFILES_DIR / f"{name}.svg"
    speedscope = PROFILES_DIR / f"{name}.speedscope.json"

    base_args: list[str] = ["py-spy", "record", "--rate", "100"]
    if nonblocking:
        base_args.append("--nonblocking")

    print(f"  py-spy → {svg.name}")
    subprocess.call(
        [*base_args, "--format", "flamegraph", "-o", str(svg), "--", sys.executable, str(target)],
        cwd=PROJECT_ROOT,
    )
    print(f"  py-spy → {speedscope.name}")
    subprocess.call(
        [
            *base_args,
            "--format",
            "speedscope",
            "-o",
            str(speedscope),
            "--",
            sys.executable,
            str(target),
        ],
        cwd=PROJECT_ROOT,
    )


def _run_scalene(name: str, target: Path) -> None:
    """Запустить scalene и сохранить HTML отчёт."""
    if shutil.which("scalene") is None:
        print("scalene not found in PATH. Skipping line-level profile.", file=sys.stderr)
        return

    html = PROFILES_DIR / f"{name}.scalene.html"
    print(f"  scalene → {html.name}")
    subprocess.call(
        [
            sys.executable,
            "-m",
            "scalene",
            "--html",
            "--outfile",
            str(html),
            "--no-browser",
            "--cpu-only",
            str(target),
        ],
        cwd=PROJECT_ROOT,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=list(TARGETS), help="Profile only this target")
    parser.add_argument(
        "--py-spy-nonblocking",
        action="store_true",
        help="Use non-blocking py-spy mode (no admin required on Windows)",
    )
    parser.add_argument(
        "--skip-scalene",
        action="store_true",
        help="Skip scalene profiling (faster, only flamegraph)",
    )
    args = parser.parse_args()

    PROFILES_DIR.mkdir(parents=True, exist_ok=True)

    targets_to_run = {args.only: TARGETS[args.only]} if args.only else TARGETS

    for name, target_file in targets_to_run.items():
        target_path = TARGETS_DIR / target_file
        if not target_path.exists():
            print(f"Missing target script: {target_path}", file=sys.stderr)
            continue

        print(f"\n[{name}] {target_file}")
        _run_py_spy(name, target_path, args.py_spy_nonblocking)
        if not args.skip_scalene:
            _run_scalene(name, target_path)

    print("\nDone. Profiles saved to:", PROFILES_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
