"""Единая цветовая тема и стилевые константы для визуализации.

Централизованные цвета, подписи и layout defaults для всех графиков.
Все модули визуализации импортируют цвета отсюда.
"""

from __future__ import annotations

import plotly.graph_objects as go


# ── Цвета клеточных популяций (8) ──────────────────────────────────

POPULATION_COLORS: dict[str, str] = {
    "P": "#f39c12",      # Тромбоциты — оранжевый
    "Ne": "#e74c3c",     # Нейтрофилы — красный
    "M1": "#c0392b",     # M1 макрофаги — тёмно-красный
    "M2": "#27ae60",     # M2 макрофаги — зелёный
    "F": "#2e86c1",      # Фибробласты — синий
    "Mf": "#8e44ad",     # Миофибробласты — фиолетовый
    "E": "#1abc9c",      # Эндотелиальные — бирюзовый
    "S": "#d35400",      # Стволовые (CD34+) — тёмно-оранжевый
}

# ── Цвета цитокинов (7) ────────────────────────────────────────────

CYTOKINE_COLORS: dict[str, str] = {
    "C_TNF": "#e74c3c",   # TNF-α — красный
    "C_IL10": "#27ae60",  # IL-10 — зелёный
    "C_PDGF": "#2e86c1",  # PDGF — синий
    "C_VEGF": "#1abc9c",  # VEGF — бирюзовый
    "C_TGFb": "#8e44ad",  # TGF-β — фиолетовый
    "C_MCP1": "#f39c12",  # MCP-1 — оранжевый
    "C_IL8": "#d35400",   # IL-8 — тёмно-оранжевый
}

# ── Цвета ECM компонентов (3) ──────────────────────────────────────

ECM_COLORS: dict[str, str] = {
    "rho_collagen": "#2e86c1",  # Коллаген — синий
    "C_MMP": "#e74c3c",         # MMP — красный
    "rho_fibrin": "#f39c12",    # Фибрин — оранжевый
}

# ── Цвета терапевтических сценариев (4) ────────────────────────────

THERAPY_COLORS: dict[str, str] = {
    "Control": "#95a5a6",   # Без терапии — серый
    "PRP": "#e74c3c",       # PRP — красный
    "PEMF": "#2e86c1",      # PEMF — синий
    "PRP+PEMF": "#27ae60",  # Комбинированная — зелёный
}

# ── Цвета фаз заживления (4) ──────────────────────────────────────

PHASE_COLORS: dict[str, str] = {
    "hemostasis": "#e74c3c",      # Гемостаз — красный
    "inflammation": "#f39c12",    # Воспаление — оранжевый
    "proliferation": "#27ae60",   # Пролиферация — зелёный
    "remodeling": "#2e86c1",      # Ремоделирование — синий
}

# ── Человеко-читаемые подписи переменных ───────────────────────────

VARIABLE_LABELS: dict[str, str] = {
    # Клеточные популяции
    "P": "Тромбоциты (P)",
    "Ne": "Нейтрофилы (Ne)",
    "M1": "M1 макрофаги",
    "M2": "M2 макрофаги",
    "F": "Фибробласты (F)",
    "Mf": "Миофибробласты (Mf)",
    "E": "Эндотелиальные (E)",
    "S": "Стволовые (S, CD34+)",
    # Цитокины
    "C_TNF": "TNF-\u03b1",
    "C_IL10": "IL-10",
    "C_PDGF": "PDGF",
    "C_VEGF": "VEGF",
    "C_TGFb": "TGF-\u03b2",
    "C_MCP1": "MCP-1",
    "C_IL8": "IL-8",
    # ECM
    "rho_collagen": "Коллаген (\u03c1_c)",
    "C_MMP": "MMP",
    "rho_fibrin": "Фибрин (\u03c1_f)",
    # Вспомогательные
    "D": "Сигнал повреждения (D)",
    "O2": "Кислород (O\u2082)",
}

# ── Группы переменных ──────────────────────────────────────────────

POPULATION_VARS: list[str] = ["P", "Ne", "M1", "M2", "F", "Mf", "E", "S"]
CYTOKINE_VARS: list[str] = [
    "C_TNF", "C_IL10", "C_PDGF", "C_VEGF", "C_TGFb", "C_MCP1", "C_IL8",
]
ECM_VARS: list[str] = ["rho_collagen", "C_MMP", "rho_fibrin"]
AUXILIARY_VARS: list[str] = ["D", "O2"]

# ── Plotly layout defaults ─────────────────────────────────────────

PLOTLY_LAYOUT_DEFAULTS: dict = {
    "template": "plotly_white",
    "font": {"size": 13},
    "showlegend": True,
}


def apply_default_layout(
    fig: go.Figure,
    height: int = 500,
    **overrides: object,
) -> go.Figure:
    """Применить стандартный layout к Plotly-фигуре.

    Args:
        fig: Plotly Figure для обновления.
        height: Высота фигуры в пикселях.
        **overrides: Дополнительные параметры для fig.update_layout.

    Returns:
        Та же фигура с обновлённым layout.
    """
    layout_kwargs = {**PLOTLY_LAYOUT_DEFAULTS, "height": height, **overrides}
    fig.update_layout(**layout_kwargs)
    return fig
