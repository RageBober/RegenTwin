"""Визуализация результатов симуляции.

Содержит модули:
- theme — цветовая тема и layout defaults
- plots — графики динамики (рост, цитокины, популяции)
- spatial — пространственные визуализации (heatmap, анимации)
- export — экспорт в PNG, CSV, PDF

Подробное описание: Description/Phase4/description_visualization.md
"""

from src.visualization.analysis_plots import (
    plot_convergence,
    plot_morris,
    plot_posterior,
    plot_sobol,
)
from src.visualization.export import ExportConfig, ReportExporter
from src.visualization.plots import (
    plot_comparison,
    plot_cytokines,
    plot_ecm,
    plot_phases,
    plot_populations,
)
from src.visualization.spatial import (
    animate_evolution,
    field_heatmap,
    heatmap_density,
    inflammation_map,
    scatter_agents,
)
from src.visualization.theme import (
    ANALYSIS_COLORS,
    AUXILIARY_VARS,
    CYTOKINE_COLORS,
    CYTOKINE_VARS,
    ECM_COLORS,
    ECM_VARS,
    PHASE_COLORS,
    PLOTLY_LAYOUT_DEFAULTS,
    POPULATION_COLORS,
    POPULATION_VARS,
    THERAPY_COLORS,
    VARIABLE_LABELS,
    apply_default_layout,
)

__all__ = [
    # theme
    "ANALYSIS_COLORS",
    "POPULATION_COLORS",
    "CYTOKINE_COLORS",
    "ECM_COLORS",
    "THERAPY_COLORS",
    "PHASE_COLORS",
    "VARIABLE_LABELS",
    "POPULATION_VARS",
    "CYTOKINE_VARS",
    "ECM_VARS",
    "AUXILIARY_VARS",
    "PLOTLY_LAYOUT_DEFAULTS",
    "apply_default_layout",
    # analysis_plots
    "plot_sobol",
    "plot_posterior",
    "plot_convergence",
    "plot_morris",
    # plots
    "plot_populations",
    "plot_cytokines",
    "plot_ecm",
    "plot_phases",
    "plot_comparison",
    # spatial
    "heatmap_density",
    "scatter_agents",
    "inflammation_map",
    "field_heatmap",
    "animate_evolution",
    # export
    "ExportConfig",
    "ReportExporter",
]
