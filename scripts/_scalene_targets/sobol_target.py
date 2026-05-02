"""Profiler target: Sobol sensitivity analysis (Ishigami, N=128)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import saltelli


def main() -> None:
    problem = {
        "num_vars": 4,
        "names": ["x1", "x2", "x3", "x4"],
        "bounds": [[0.0, 1.0]] * 4,
    }
    samples = saltelli.sample(problem, N=128, calc_second_order=True)
    a, b = 7.0, 0.1
    outputs = (
        np.sin(samples[:, 0])
        + a * np.sin(samples[:, 1]) ** 2
        + b * samples[:, 2] ** 4 * np.sin(samples[:, 0])
    )
    sobol_analyze.analyze(problem, outputs, calc_second_order=True, print_to_console=False)


if __name__ == "__main__":
    main()
