"""Математическое ядро RegenTwin.

Модули:
- sde_model: Стохастические дифференциальные уравнения (макроуровень)
- abm_model: Agent-Based модель (микроуровень)
- integration: Интеграция SDE + ABM
- monte_carlo: Monte Carlo симуляции
- numerical_utils: Утилиты для численной робастности
- parameters: Полный набор параметров модели (90+)
- extended_sde: Расширенная 20-переменная SDE система
- wound_phases: Детекция фаз заживления раны
- therapy_models: Механистические модели PRP и PEMF терапий
- sde_numerics: Продвинутые численные солверы (Milstein, IMEX, adaptive)
- robustness: Верификация робастности (conservation, convergence, SDE vs ABM)
- abm_spatial: Расширенные пространственные механики ABM (Phase 2.8)

Подробное описание каждого модуля в Description/Phase2/description_*.md
"""

# ABM Model
from src.core.abm_model import (
    ABMConfig,
    ABMModel,
    ABMSnapshot,
    ABMTrajectory,
    Agent,
    AgentState,
    EndothelialAgent,
    Fibroblast,
    KDTreeSpatialIndex,
    Macrophage,
    MyofibroblastAgent,
    NeutrophilAgent,
    StemCell,
    simulate_abm,
)

# Extended SDE
from src.core.extended_sde import (
    VARIABLE_NAMES,
    ExtendedSDEModel,
    ExtendedSDEState,
    ExtendedSDETrajectory,
    StateIndex,
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

# Numerical Utils
from src.core.numerical_utils import (
    DivergenceInfo,
    NumericalGuard,
    adaptive_timestep,
    clip_negative_concentrations,
    detect_divergence,
    handle_divergence,
)

# Parameters
from src.core.parameters import ParameterSet

# Robustness
from src.core.robustness import (
    ComparisonMetrics,
    ConservationChecker,
    ConservationReport,
    ConvergenceResult,
    ConvergenceVerifier,
    NaNHandler,
    PositivityEnforcer,
    SDEvsABMComparator,
    ViolationStats,
)

# SDE Model
from src.core.sde_model import (
    SDEConfig,
    SDEModel,
    SDEState,
    SDETrajectory,
    TherapyProtocol,
    simulate_sde,
)

# SDE Numerics
from src.core.sde_numerics import (
    FAST_INDICES,
    SLOW_INDICES,
    AdaptiveTimestepper,
    EulerMaruyamaSolver,
    IMEXSplitter,
    MilsteinSolver,
    SDESolver,
    SolverConfig,
    SolverType,
    StepResult,
    StochasticRungeKutta,
    create_solver,
)

# Therapy Models
from src.core.therapy_models import (
    PEMFConfig,
    PEMFEffects,
    PEMFModel,
    PRPConfig,
    PRPModel,
    PRPReleaseState,
    SynergyConfig,
    SynergyModel,
)

# ABM Extended (Phase 2.8)
from src.core.abm_spatial import (
    ChemotaxisEngine,
    ContactInhibitionEngine,
    EfferocytosisEngine,
    KDTreeNeighborSearch,
    MechanotransductionEngine,
    MultiCytokineField,
    PlateletAgent,
    SubcyclingManager,
)

# Wound Phases
from src.core.wound_phases import (
    PhaseIndicators,
    WoundPhase,
    WoundPhaseDetector,
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
    "EndothelialAgent",
    "Fibroblast",
    "KDTreeSpatialIndex",
    "Macrophage",
    "MyofibroblastAgent",
    "NeutrophilAgent",
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
    # Numerical Utils
    "DivergenceInfo",
    "NumericalGuard",
    "adaptive_timestep",
    "clip_negative_concentrations",
    "detect_divergence",
    "handle_divergence",
    # Parameters
    "ParameterSet",
    # Extended SDE
    "ExtendedSDEModel",
    "ExtendedSDEState",
    "ExtendedSDETrajectory",
    "StateIndex",
    "VARIABLE_NAMES",
    # Wound Phases
    "WoundPhase",
    "PhaseIndicators",
    "WoundPhaseDetector",
    # Therapy Models
    "PRPConfig",
    "PEMFConfig",
    "SynergyConfig",
    "PRPReleaseState",
    "PEMFEffects",
    "PRPModel",
    "PEMFModel",
    "SynergyModel",
    # SDE Numerics
    "SDESolver",
    "SolverType",
    "SolverConfig",
    "StepResult",
    "EulerMaruyamaSolver",
    "MilsteinSolver",
    "IMEXSplitter",
    "AdaptiveTimestepper",
    "StochasticRungeKutta",
    "FAST_INDICES",
    "SLOW_INDICES",
    "create_solver",
    # Robustness
    "ViolationStats",
    "ConservationReport",
    "ConvergenceResult",
    "ComparisonMetrics",
    "PositivityEnforcer",
    "NaNHandler",
    "ConservationChecker",
    "ConvergenceVerifier",
    "SDEvsABMComparator",
    # ABM Extended (Phase 2.8)
    "PlateletAgent",
    "ChemotaxisEngine",
    "ContactInhibitionEngine",
    "EfferocytosisEngine",
    "MechanotransductionEngine",
    "MultiCytokineField",
    "KDTreeNeighborSearch",
    "SubcyclingManager",
]
