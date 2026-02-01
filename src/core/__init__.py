"""Математическое ядро RegenTwin.

Модули:
- sde_model: Стохастические дифференциальные уравнения (макроуровень)
- abm_model: Agent-Based модель (микроуровень)
- integration: Интеграция SDE + ABM
- monte_carlo: Monte Carlo симуляции

Подробное описание каждого модуля в Description/description_*.md
"""

# SDE Model
# ABM Model
from src.core.abm_model import (
    ABMConfig,
    ABMModel,
    ABMSnapshot,
    ABMTrajectory,
    Agent,
    AgentState,
    Fibroblast,
    Macrophage,
    StemCell,
    simulate_abm,
)

# Integration
from src.core.integration import (
    IntegratedModel,
    IntegratedState,
    IntegratedTrajectory,
    IntegrationConfig,
    create_default_integration_config,
    simulate_integrated,
)

# Monte Carlo
from src.core.monte_carlo import (
    MonteCarloConfig,
    MonteCarloResults,
    MonteCarloSimulator,
    TrajectoryResult,
    compare_therapies,
    run_monte_carlo,
    run_parameter_sweep,
)
from src.core.sde_model import (
    SDEConfig,
    SDEModel,
    SDEState,
    SDETrajectory,
    TherapyProtocol,
    simulate_sde,
)

__all__ = [
    # SDE
    "SDEConfig",
    "SDEModel",
    "SDEState",
    "SDETrajectory",
    "TherapyProtocol",
    "simulate_sde",
    # ABM
    "ABMConfig",
    "ABMModel",
    "ABMSnapshot",
    "ABMTrajectory",
    "Agent",
    "AgentState",
    "Fibroblast",
    "Macrophage",
    "StemCell",
    "simulate_abm",
    # Integration
    "IntegrationConfig",
    "IntegratedModel",
    "IntegratedState",
    "IntegratedTrajectory",
    "create_default_integration_config",
    "simulate_integrated",
    # Monte Carlo
    "MonteCarloConfig",
    "MonteCarloResults",
    "MonteCarloSimulator",
    "TrajectoryResult",
    "compare_therapies",
    "run_monte_carlo",
    "run_parameter_sweep",
]
