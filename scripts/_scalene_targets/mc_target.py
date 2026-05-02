"""Profiler target: Monte Carlo (4 траектории extended-SDE, последовательно)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.extended_sde import ExtendedSDEState
from src.core.monte_carlo import MonteCarloConfig, MonteCarloSimulator
from src.core.parameters import ParameterSet
from src.data.parameter_extraction import ModelParameters


def main() -> None:
    params = ParameterSet()
    params.t_max = 24.0
    initial = ExtendedSDEState(
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
    cfg = MonteCarloConfig(
        n_trajectories=4,
        model_type="extended",
        extended_params=params,
        extended_initial_state=initial,
        n_jobs=1,
        use_multiprocessing=False,
        base_seed=42,
    )
    sim = MonteCarloSimulator(config=cfg)
    sim.run(
        ModelParameters(
            n0=10000.0,
            stem_cell_fraction=0.05,
            macrophage_fraction=0.03,
            apoptotic_fraction=0.02,
            c0=10.0,
            inflammation_level=0.5,
        )
    )


if __name__ == "__main__":
    main()
