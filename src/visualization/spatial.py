"""Пространственная визуализация для ABM модели.

Работает с ABMSnapshot и ABMTrajectory. Возвращает Plotly-фигуры.
Для анимации в файл используется matplotlib.animation.

Подробное описание: Description/Phase4/description_visualization.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from src.core.abm_model import ABMSnapshot, ABMTrajectory
from src.visualization.theme import (
    POPULATION_COLORS,
    VARIABLE_LABELS,
    apply_default_layout,
)

# Маппинг agent_type → визуальное имя
_AGENT_TYPE_LABELS: dict[str, str] = {
    "stem": "Стволовые (CD34+)",
    "macro": "Макрофаги",
    "fibro": "Фибробласты",
    "neutrophil": "Нейтрофилы",
    "endothelial": "Эндотелиальные",
    "myofibroblast": "Миофибробласты",
}

# Маппинг agent_type → цвет из POPULATION_COLORS
_AGENT_TYPE_COLORS: dict[str, str] = {
    "stem": POPULATION_COLORS["S"],
    "macro": POPULATION_COLORS["M1"],
    "fibro": POPULATION_COLORS["F"],
    "neutrophil": POPULATION_COLORS["Ne"],
    "endothelial": POPULATION_COLORS["E"],
    "myofibroblast": POPULATION_COLORS["Mf"],
}


def heatmap_density(
    snapshot: ABMSnapshot,
    bin_size: float = 10.0,
    agent_types: list[str] | None = None,
    height: int = 500,
) -> go.Figure:
    """2D density heatmap позиций агентов.

    Args:
        snapshot: Один ABM snapshot.
        bin_size: Размер ячейки в мкм.
        agent_types: Фильтр по типам. None = все.
        height: Высота фигуры.

    Returns:
        Plotly Figure с Heatmap trace.
    """
    agents = [a for a in snapshot.agents if a.alive]
    if agent_types:
        agents = [a for a in agents if a.agent_type in agent_types]

    if not agents:
        fig = go.Figure()
        fig.add_annotation(text="Нет агентов для отображения", showarrow=False)
        return apply_default_layout(fig, height=height)

    x_coords = np.array([a.x for a in agents])
    y_coords = np.array([a.y for a in agents])

    # Определяем границы сетки
    x_max = max(x_coords.max(), 100.0)
    y_max = max(y_coords.max(), 100.0)
    n_bins_x = max(1, int(np.ceil(x_max / bin_size)))
    n_bins_y = max(1, int(np.ceil(y_max / bin_size)))

    density, x_edges, y_edges = np.histogram2d(
        x_coords, y_coords,
        bins=[n_bins_x, n_bins_y],
        range=[[0, x_max], [0, y_max]],
    )

    fig = go.Figure(data=go.Heatmap(
        z=density.T,
        x=x_edges[:-1] + bin_size / 2,
        y=y_edges[:-1] + bin_size / 2,
        colorscale="YlOrRd",
        colorbar_title="Агентов",
        hovertemplate="X: %{x:.0f} мкм<br>Y: %{y:.0f} мкм<br>Количество: %{z}<extra></extra>",
    ))

    fig.update_xaxes(title_text="X (мкм)")
    fig.update_yaxes(title_text="Y (мкм)", scaleanchor="x", scaleratio=1)

    title_suffix = ""
    if agent_types:
        title_suffix = f" ({', '.join(agent_types)})"

    return apply_default_layout(
        fig,
        height=height,
        title=f"Плотность агентов (t={snapshot.t:.0f} ч){title_suffix}",
    )


def scatter_agents(
    snapshot: ABMSnapshot,
    color_by: str = "type",
    height: int = 500,
) -> go.Figure:
    """Scatter plot агентов, раскрашенных по типу/энергии/возрасту.

    Args:
        snapshot: ABM snapshot.
        color_by: "type", "energy", "age".
        height: Высота фигуры.

    Returns:
        Plotly Figure.
    """
    agents = [a for a in snapshot.agents if a.alive]

    if not agents:
        fig = go.Figure()
        fig.add_annotation(text="Нет агентов для отображения", showarrow=False)
        return apply_default_layout(fig, height=height)

    fig = go.Figure()

    if color_by == "type":
        # Группируем по типу
        types_seen: dict[str, tuple[list[float], list[float]]] = {}
        for a in agents:
            if a.agent_type not in types_seen:
                types_seen[a.agent_type] = ([], [])
            types_seen[a.agent_type][0].append(a.x)
            types_seen[a.agent_type][1].append(a.y)

        for atype, (xs, ys) in types_seen.items():
            color = _AGENT_TYPE_COLORS.get(atype, "#333333")
            label = _AGENT_TYPE_LABELS.get(atype, atype)
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode="markers",
                name=label,
                marker=dict(color=color, size=6, opacity=0.8),
            ))
    elif color_by == "energy":
        x = [a.x for a in agents]
        y = [a.y for a in agents]
        energy = [a.energy for a in agents]
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode="markers",
            marker=dict(
                color=energy,
                colorscale="Viridis",
                size=6,
                colorbar_title="Энергия",
                cmin=0, cmax=1,
            ),
            name="Энергия",
        ))
    elif color_by == "age":
        x = [a.x for a in agents]
        y = [a.y for a in agents]
        ages = [a.age for a in agents]
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode="markers",
            marker=dict(
                color=ages,
                colorscale="Plasma",
                size=6,
                colorbar_title="Возраст (ч)",
            ),
            name="Возраст",
        ))

    fig.update_xaxes(title_text="X (мкм)")
    fig.update_yaxes(title_text="Y (мкм)", scaleanchor="x", scaleratio=1)

    return apply_default_layout(
        fig,
        height=height,
        title=f"Агенты (t={snapshot.t:.0f} ч, N={len(agents)})",
    )


def inflammation_map(
    snapshot: ABMSnapshot,
    height: int = 500,
) -> go.Figure:
    """Карта воспаления: проксимированный TNF-α/IL-10 ratio.

    Использует cytokine_field как прокси для пространственного
    соотношения провоспалительных/антивоспалительных цитокинов.
    Высокие значения = воспаление, низкие = разрешение.

    Args:
        snapshot: ABM snapshot с cytokine_field.
        height: Высота фигуры.

    Returns:
        Plotly Heatmap с diverging colorscale.
    """
    field = snapshot.cytokine_field

    if field is None or field.size == 0:
        fig = go.Figure()
        fig.add_annotation(text="Нет данных цитокинового поля", showarrow=False)
        return apply_default_layout(fig, height=height)

    # Нормализуем поле для отображения уровня воспаления
    field_max = np.max(np.abs(field)) if np.max(np.abs(field)) > 0 else 1.0
    normalized = field / field_max

    fig = go.Figure(data=go.Heatmap(
        z=normalized,
        colorscale="RdYlGn_r",  # Красный = высокое воспаление
        zmin=-1, zmax=1,
        colorbar_title="Уровень<br>воспаления",
        hovertemplate="X: %{x}<br>Y: %{y}<br>Уровень: %{z:.2f}<extra></extra>",
    ))

    fig.update_xaxes(title_text="X")
    fig.update_yaxes(title_text="Y", scaleanchor="x", scaleratio=1)

    return apply_default_layout(
        fig,
        height=height,
        title=f"Карта воспаления (t={snapshot.t:.0f} ч)",
    )


def field_heatmap(
    snapshot: ABMSnapshot,
    field: str = "cytokine",
    colorscale: str | None = None,
    height: int = 400,
) -> go.Figure:
    """Heatmap для 2D-поля (цитокины или ECM).

    Args:
        snapshot: ABM snapshot.
        field: "cytokine" или "ecm".
        colorscale: Plotly colorscale. None = авто.
        height: Высота фигуры.

    Returns:
        Plotly Figure.
    """
    if field == "cytokine":
        data = snapshot.cytokine_field
        default_colorscale = "YlOrRd"
        title = f"Поле цитокинов (t={snapshot.t:.0f} ч)"
        colorbar_title = "нг/мл"
    elif field == "ecm":
        data = snapshot.ecm_field
        default_colorscale = "Greens"
        title = f"Внеклеточный матрикс (t={snapshot.t:.0f} ч)"
        colorbar_title = "отн. ед."
    else:
        raise ValueError(f"Неизвестное поле: {field}. Ожидается 'cytokine' или 'ecm'.")

    if data is None or data.size == 0:
        fig = go.Figure()
        fig.add_annotation(text=f"Нет данных поля '{field}'", showarrow=False)
        return apply_default_layout(fig, height=height)

    cs = colorscale if colorscale else default_colorscale

    fig = go.Figure(data=go.Heatmap(
        z=data,
        colorscale=cs,
        colorbar_title=colorbar_title,
    ))

    fig.update_xaxes(title_text="X")
    fig.update_yaxes(title_text="Y", scaleanchor="x", scaleratio=1)

    return apply_default_layout(fig, height=height, title=title)


def animate_evolution(
    trajectory: ABMTrajectory,
    output_path: Path | str | None = None,
    fps: int = 5,
    dpi: int = 100,
    show_fields: bool = True,
) -> go.Figure | Path:
    """Анимация эволюции ABM во времени.

    Если output_path=None, возвращает Plotly Figure с animation frames.
    Если output_path задан, записывает GIF через matplotlib.

    Args:
        trajectory: ABM траектория с несколькими snapshot.
        output_path: Путь для GIF/MP4. None = Plotly figure.
        fps: Кадров в секунду (для файла).
        dpi: Разрешение (для файла).
        show_fields: Наложить цитокиновое поле как фон.

    Returns:
        go.Figure (интерактивная) или Path (файл).
    """
    snapshots = trajectory.snapshots
    if not snapshots:
        fig = go.Figure()
        fig.add_annotation(text="Нет снимков для анимации", showarrow=False)
        return apply_default_layout(fig)

    if output_path is not None:
        return _save_animation_file(trajectory, Path(output_path), fps, dpi, show_fields)

    return _create_plotly_animation(trajectory, show_fields)


def _create_plotly_animation(
    trajectory: ABMTrajectory,
    show_fields: bool,
) -> go.Figure:
    """Plotly-анимация с кнопкой Play."""
    snapshots = trajectory.snapshots

    # Первый кадр
    first = snapshots[0]
    agents = [a for a in first.agents if a.alive]

    fig = go.Figure()

    # Данные первого кадра
    if show_fields and first.cytokine_field is not None and first.cytokine_field.size > 0:
        fig.add_trace(go.Heatmap(
            z=first.cytokine_field,
            colorscale="YlOrRd",
            opacity=0.3,
            showscale=False,
            name="Цитокины (фон)",
        ))

    # Агенты по типам
    types_data = _group_agents_by_type(agents)
    for atype, (xs, ys) in types_data.items():
        color = _AGENT_TYPE_COLORS.get(atype, "#333333")
        label = _AGENT_TYPE_LABELS.get(atype, atype)
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="markers",
            name=label,
            marker=dict(color=color, size=5, opacity=0.8),
        ))

    # Кадры анимации
    frames = []
    for snap in snapshots:
        frame_data = []

        if show_fields and snap.cytokine_field is not None and snap.cytokine_field.size > 0:
            frame_data.append(go.Heatmap(
                z=snap.cytokine_field,
                colorscale="YlOrRd",
                opacity=0.3,
                showscale=False,
            ))

        snap_agents = [a for a in snap.agents if a.alive]
        snap_types = _group_agents_by_type(snap_agents)
        for atype in types_data:
            xs, ys = snap_types.get(atype, ([], []))
            color = _AGENT_TYPE_COLORS.get(atype, "#333333")
            frame_data.append(go.Scatter(
                x=xs, y=ys,
                mode="markers",
                marker=dict(color=color, size=5, opacity=0.8),
            ))

        frames.append(go.Frame(
            data=frame_data,
            name=f"t={snap.t:.0f}",
        ))

    fig.frames = frames

    # Slider + кнопки
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=0,
            x=0.5,
            xanchor="center",
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, {"frame": {"duration": 200, "redraw": True}, "fromcurrent": True}]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]),
            ],
        )],
        sliders=[dict(
            active=0,
            steps=[
                dict(args=[[f.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                     label=f.name, method="animate")
                for f in frames
            ],
            x=0.1, len=0.8,
            currentvalue=dict(prefix="Время: ", visible=True),
        )],
    )

    fig.update_xaxes(title_text="X (мкм)")
    fig.update_yaxes(title_text="Y (мкм)", scaleanchor="x", scaleratio=1)

    return apply_default_layout(
        fig,
        height=600,
        title="Эволюция ABM модели",
    )


def _save_animation_file(
    trajectory: ABMTrajectory,
    output_path: Path,
    fps: int,
    dpi: int,
    show_fields: bool,
) -> Path:
    """Сохранение анимации в GIF через matplotlib."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    snapshots = trajectory.snapshots
    fig_mpl, ax = plt.subplots(figsize=(6, 6))

    def update(frame_idx: int) -> None:
        ax.clear()
        snap = snapshots[frame_idx]
        agents = [a for a in snap.agents if a.alive]

        if show_fields and snap.cytokine_field is not None and snap.cytokine_field.size > 0:
            ax.imshow(
                snap.cytokine_field.T, origin="lower",
                cmap="YlOrRd", alpha=0.3, aspect="equal",
                extent=[0, 100, 0, 100],
            )

        for atype, color_key in [("stem", "S"), ("macro", "M1"), ("fibro", "F"),
                                  ("neutrophil", "Ne"), ("endothelial", "E"), ("myofibroblast", "Mf")]:
            typed_agents = [a for a in agents if a.agent_type == atype]
            if typed_agents:
                xs = [a.x for a in typed_agents]
                ys = [a.y for a in typed_agents]
                color = POPULATION_COLORS.get(color_key, "#333333")
                ax.scatter(xs, ys, c=color, s=10, alpha=0.8,
                           label=_AGENT_TYPE_LABELS.get(atype, atype))

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel("X (мкм)")
        ax.set_ylabel("Y (мкм)")
        ax.set_title(f"ABM t={snap.t:.0f} ч (N={len(agents)})")
        ax.set_aspect("equal")
        if frame_idx == 0:
            ax.legend(fontsize=7, loc="upper right")

    anim = FuncAnimation(fig_mpl, update, frames=len(snapshots), interval=1000 // fps)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    anim.save(str(output_path), writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig_mpl)

    return output_path


def _group_agents_by_type(
    agents: list,
) -> dict[str, tuple[list[float], list[float]]]:
    """Группировка агентов по типу: {type: ([xs], [ys])}."""
    result: dict[str, tuple[list[float], list[float]]] = {}
    for a in agents:
        if a.agent_type not in result:
            result[a.agent_type] = ([], [])
        result[a.agent_type][0].append(a.x)
        result[a.agent_type][1].append(a.y)
    return result
