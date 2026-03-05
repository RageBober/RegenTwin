"""Визуализация результатов симуляции.

Содержит модули:
- theme — цветовая тема и layout defaults
- plots — графики динамики (рост, цитокины, популяции)
- spatial — пространственные визуализации (heatmap, анимации)
- export — экспорт в PNG, CSV, PDF

Подробное описание: Description/Phase4/description_visualization.md
"""

from src.visualization.theme import (
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
]
