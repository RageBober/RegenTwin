"""Profiler target: ABM simulate (500 max agents, 24h)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.abm_model import ABMConfig, ABMModel
from src.data.parameter_extraction import ModelParameters


def main() -> None:
    cfg = ABMConfig()
    cfg.t_max = 24.0
    cfg.max_agents = 500
    params = ModelParameters(
        n0=10000.0,
        stem_cell_fraction=0.05,
        macrophage_fraction=0.03,
        apoptotic_fraction=0.02,
        c0=10.0,
        inflammation_level=0.5,
    )
    model = ABMModel(config=cfg, random_seed=42)
    model.simulate(params, snapshot_interval=24.0)


if __name__ == "__main__":
    main()
