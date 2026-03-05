"""Графики временных рядов для 20-переменной SDE модели.

Все функции возвращают plotly.graph_objects.Figure.
Нет зависимости от Streamlit — только Plotly.

Подробное описание: Description/Phase4/description_visualization.md
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.core.extended_sde import ExtendedSDETrajectory, VARIABLE_NAMES
from src.core.monte_carlo import MonteCarloResults
from src.core.wound_phases import PhaseIndicators, WoundPhase, WoundPhaseDetector
from src.visualization.theme import (
    CYTOKINE_COLORS,
    CYTOKINE_VARS,
    ECM_COLORS,
    ECM_VARS,
    PHASE_COLORS,
    POPULATION_COLORS,
    POPULATION_VARS,
    THERAPY_COLORS,
    VARIABLE_LABELS,
    apply_default_layout,
)


def plot_populations(
    trajectory: ExtendedSDETrajectory,
    variables: list[str] | None = None,
    show_ci: bool = False,
    mc_results: MonteCarloResults | None = None,
    height: int = 500,
) -> go.Figure:
    """Кривые роста клеточных популяций.

    Args:
        trajectory: Траектория 20-переменной SDE.
        variables: Подмножество из POPULATION_VARS. None = все 8.
        show_ci: Если True и mc_results задан, показать CI-полосы.
        mc_results: Результаты Monte Carlo для CI.
        height: Высота фигуры.

    Returns:
        Plotly Figure с одной линией на популяцию.
    """
    vars_to_plot = variables if variables is not None else POPULATION_VARS
    times = trajectory.times

    fig = go.Figure()

    for var in vars_to_plot:
        values = trajectory.get_variable(var)
        color = POPULATION_COLORS.get(var, "#333333")
        label = VARIABLE_LABELS.get(var, var)

        fig.add_trace(go.Scatter(
            x=times,
            y=values,
            mode="lines",
            name=label,
            line=dict(color=color, width=2),
        ))

    if show_ci and mc_results is not None:
        _add_ci_bands(fig, mc_results, times)

    fig.update_xaxes(title_text="Время (ч)")
    fig.update_yaxes(title_text="Клеток/мкл")

    return apply_default_layout(
        fig,
        height=height,
        title="Динамика клеточных популяций",
        legend=dict(orientation="h", y=-0.15),
    )


def _add_ci_bands(
    fig: go.Figure,
    mc_results: MonteCarloResults,
    times: np.ndarray,
) -> None:
    """Добавить CI-полосы из Monte Carlo на график."""
    ci_lower = mc_results.quantiles_N.get(0.05)
    ci_upper = mc_results.quantiles_N.get(0.95)

    if ci_lower is None or ci_upper is None:
        return

    mc_times = mc_results.times
    if len(mc_times) != len(ci_lower):
        return

    fig.add_trace(go.Scatter(
        x=np.concatenate([mc_times, mc_times[::-1]]),
        y=np.concatenate([ci_upper, ci_lower[::-1]]),
        fill="toself",
        fillcolor="rgba(46,134,193,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% CI (Monte Carlo)",
        showlegend=True,
    ))

    fig.add_trace(go.Scatter(
        x=mc_times,
        y=mc_results.mean_N,
        mode="lines",
        name="Среднее (MC)",
        line=dict(color="#2e86c1", width=2, dash="dash"),
    ))


def plot_cytokines(
    trajectory: ExtendedSDETrajectory,
    variables: list[str] | None = None,
    layout: str = "overlay",
    height: int = 500,
) -> go.Figure:
    """Динамика цитокинов.

    Args:
        trajectory: Траектория 20-переменной SDE.
        variables: Подмножество из CYTOKINE_VARS. None = все 7.
        layout: "overlay" — одни оси, "subplots" — по одному подграфику.
        height: Высота фигуры.

    Returns:
        Plotly Figure.
    """
    vars_to_plot = variables if variables is not None else CYTOKINE_VARS
    times = trajectory.times

    if layout == "subplots":
        n = len(vars_to_plot)
        fig = make_subplots(
            rows=n, cols=1,
            subplot_titles=[VARIABLE_LABELS.get(v, v) for v in vars_to_plot],
            vertical_spacing=0.05,
            shared_xaxes=True,
        )
        for i, var in enumerate(vars_to_plot, start=1):
            values = trajectory.get_variable(var)
            color = CYTOKINE_COLORS.get(var, "#333333")
            fig.add_trace(
                go.Scatter(
                    x=times, y=values,
                    mode="lines", name=VARIABLE_LABELS.get(var, var),
                    line=dict(color=color, width=2),
                    showlegend=False,
                ),
                row=i, col=1,
            )
            fig.update_yaxes(title_text="нг/мл", row=i, col=1)

        fig.update_xaxes(title_text="Время (ч)", row=n, col=1)
        return apply_default_layout(fig, height=max(height, n * 120))
    else:
        fig = go.Figure()
        for var in vars_to_plot:
            values = trajectory.get_variable(var)
            color = CYTOKINE_COLORS.get(var, "#333333")
            fig.add_trace(go.Scatter(
                x=times, y=values,
                mode="lines", name=VARIABLE_LABELS.get(var, var),
                line=dict(color=color, width=2),
            ))

        fig.update_xaxes(title_text="Время (ч)")
        fig.update_yaxes(title_text="Концентрация (нг/мл)")

        return apply_default_layout(
            fig,
            height=height,
            title="Динамика цитокинов",
            legend=dict(orientation="h", y=-0.15),
        )


def plot_ecm(
    trajectory: ExtendedSDETrajectory,
    height: int = 400,
) -> go.Figure:
    """Динамика ECM: коллаген, MMP, фибрин с dual axes.

    Коллаген и фибрин — левая ось (безразмерные, 0-1).
    MMP — правая ось (нг/мл).

    Returns:
        Plotly Figure с двумя осями Y.
    """
    times = trajectory.times

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Коллаген и фибрин — левая ось
    for var in ["rho_collagen", "rho_fibrin"]:
        values = trajectory.get_variable(var)
        color = ECM_COLORS[var]
        fig.add_trace(
            go.Scatter(
                x=times, y=values,
                mode="lines", name=VARIABLE_LABELS.get(var, var),
                line=dict(color=color, width=2),
            ),
            secondary_y=False,
        )

    # MMP — правая ось
    mmp_values = trajectory.get_variable("C_MMP")
    fig.add_trace(
        go.Scatter(
            x=times, y=mmp_values,
            mode="lines", name=VARIABLE_LABELS.get("C_MMP", "MMP"),
            line=dict(color=ECM_COLORS["C_MMP"], width=2, dash="dash"),
        ),
        secondary_y=True,
    )

    fig.update_xaxes(title_text="Время (ч)")
    fig.update_yaxes(title_text="Плотность (отн. ед.)", secondary_y=False)
    fig.update_yaxes(title_text="MMP (нг/мл)", secondary_y=True)

    return apply_default_layout(
        fig,
        height=height,
        title="Динамика внеклеточного матрикса (ECM)",
    )


def plot_phases(
    trajectory: ExtendedSDETrajectory,
    phase_indicators: list[PhaseIndicators] | None = None,
    height: int = 500,
) -> go.Figure:
    """Цветовая полоса фаз заживления + ключевые популяции.

    Верхняя часть — цветная полоса фаз (гемостаз → воспаление → пролиферация → ремоделирование).
    Нижняя часть — основные клеточные популяции.

    Args:
        trajectory: Траектория 20-переменной SDE.
        phase_indicators: Предвычисленные индикаторы. None → автодетекция.
        height: Высота фигуры.

    Returns:
        Plotly Figure.
    """
    times = trajectory.times

    if phase_indicators is None:
        detector = WoundPhaseDetector()
        phase_indicators = detector.detect_phase_trajectory(trajectory)

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.15, 0.85],
        vertical_spacing=0.05,
        shared_xaxes=True,
        subplot_titles=("Фазы заживления", "Клеточные популяции"),
    )

    # Цветовая полоса фаз
    _add_phase_bar(fig, times, phase_indicators, row=1)

    # Ключевые популяции
    key_vars = ["Ne", "M1", "M2", "F", "E"]
    for var in key_vars:
        values = trajectory.get_variable(var)
        color = POPULATION_COLORS.get(var, "#333333")
        fig.add_trace(
            go.Scatter(
                x=times, y=values,
                mode="lines", name=VARIABLE_LABELS.get(var, var),
                line=dict(color=color, width=2),
            ),
            row=2, col=1,
        )

    fig.update_xaxes(title_text="Время (ч)", row=2, col=1)
    fig.update_yaxes(title_text="Клеток/мкл", row=2, col=1)
    fig.update_yaxes(visible=False, row=1, col=1)

    return apply_default_layout(
        fig,
        height=height,
        title="Фазы заживления и клеточная динамика",
        legend=dict(orientation="h", y=-0.12),
    )


def _add_phase_bar(
    fig: go.Figure,
    times: np.ndarray,
    indicators: list[PhaseIndicators],
    row: int,
) -> None:
    """Добавить цветовую полосу фаз как горизонтальные прямоугольники."""
    if len(indicators) < 2 or len(times) < 2:
        return

    # Группируем последовательные одинаковые фазы
    current_phase = indicators[0].phase
    start_idx = 0

    for i in range(1, len(indicators)):
        if indicators[i].phase != current_phase or i == len(indicators) - 1:
            end_idx = i if indicators[i].phase != current_phase else i
            phase_name = current_phase.value
            color = PHASE_COLORS.get(phase_name, "#cccccc")

            t_start = times[start_idx] if start_idx < len(times) else times[-1]
            t_end = times[min(end_idx, len(times) - 1)]

            fig.add_trace(
                go.Scatter(
                    x=[t_start, t_end, t_end, t_start, t_start],
                    y=[0, 0, 1, 1, 0],
                    fill="toself",
                    fillcolor=color,
                    line=dict(width=0),
                    mode="lines",
                    name=phase_name.capitalize(),
                    showlegend=(start_idx == 0 or indicators[start_idx].phase != indicators[start_idx - 1].phase),
                    hoverinfo="text",
                    hovertext=f"{phase_name.capitalize()} ({t_start:.0f}-{t_end:.0f} ч)",
                ),
                row=row, col=1,
            )

            current_phase = indicators[i].phase
            start_idx = i


def plot_comparison(
    results: dict[str, ExtendedSDETrajectory],
    variable: str = "F",
    show_all_populations: bool = False,
    height: int = 500,
) -> go.Figure:
    """Сравнение терапевтических сценариев.

    Args:
        results: {"Control": traj1, "PRP": traj2, ...}
        variable: Какую переменную сравнивать (default: фибробласты).
        show_all_populations: Если True — все 8 популяций в subplots.
        height: Высота фигуры.

    Returns:
        Plotly Figure.
    """
    if show_all_populations:
        return _plot_comparison_all_populations(results, height)

    fig = go.Figure()

    for scenario_name, trajectory in results.items():
        values = trajectory.get_variable(variable)
        times = trajectory.times
        color = THERAPY_COLORS.get(scenario_name, "#333333")

        fig.add_trace(go.Scatter(
            x=times,
            y=values,
            mode="lines",
            name=scenario_name,
            line=dict(color=color, width=2),
        ))

    var_label = VARIABLE_LABELS.get(variable, variable)
    fig.update_xaxes(title_text="Время (ч)")
    fig.update_yaxes(title_text=var_label)

    return apply_default_layout(
        fig,
        height=height,
        title=f"Сравнение сценариев: {var_label}",
        legend=dict(orientation="h", y=-0.12),
    )


def _plot_comparison_all_populations(
    results: dict[str, ExtendedSDETrajectory],
    height: int,
) -> go.Figure:
    """Сравнение всех 8 популяций в subplots."""
    n = len(POPULATION_VARS)
    fig = make_subplots(
        rows=n, cols=1,
        subplot_titles=[VARIABLE_LABELS.get(v, v) for v in POPULATION_VARS],
        vertical_spacing=0.03,
        shared_xaxes=True,
    )

    for i, var in enumerate(POPULATION_VARS, start=1):
        for scenario_name, trajectory in results.items():
            values = trajectory.get_variable(var)
            times = trajectory.times
            color = THERAPY_COLORS.get(scenario_name, "#333333")

            fig.add_trace(
                go.Scatter(
                    x=times, y=values,
                    mode="lines", name=scenario_name,
                    line=dict(color=color, width=1.5),
                    showlegend=(i == 1),
                ),
                row=i, col=1,
            )

    fig.update_xaxes(title_text="Время (ч)", row=n, col=1)

    return apply_default_layout(
        fig,
        height=max(height, n * 100),
        title="Сравнение сценариев: все популяции",
    )
