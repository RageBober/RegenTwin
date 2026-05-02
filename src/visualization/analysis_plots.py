"""Визуализация результатов анализа чувствительности и параметрической идентификации.

Четыре функции, каждая возвращает plotly.graph_objects.Figure:
- plot_sobol   — Tornado bar chart для S1/ST индексов Sobol
- plot_posterior — Маргинальные гистограммы / corner plots (Plotly-аналог ArviZ)
- plot_convergence — Сходимость MCMC/MC метрик по итерациям
- plot_morris  — Morris screening: μ* vs σ scatter plot

Подробное описание: Description/Phase3/description_analysis_plots.md
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.core.parameter_estimation import EstimationResult
from src.core.sensitivity_analysis import MorrisResult, SobolResult
from src.visualization.theme import (
    ANALYSIS_COLORS,
    VARIABLE_LABELS,
    apply_default_layout,
)

# ── Вспомогательные константы ─────────────────────────────────────

_CONVERGENCE_PANEL_TITLES: dict[str, str] = {
    "rhat": "R-hat по параметрам",
    "ess": "Effective Sample Size (ESS)",
    "trace": "Trace plots",
}


# ── plot_sobol ────────────────────────────────────────────────────


# Description: Description/Phase3/description_analysis_plots.md#plot_sobol
def plot_sobol(
    result: SobolResult,
    metric: str = "both",
    top_n: int | None = 15,
    show_confidence: bool = True,
    height: int = 500,
) -> go.Figure:
    """Tornado bar chart для Sobol чувствительности (S1 и/или ST).

    Горизонтальная столбчатая диаграмма, параметры отсортированы по убыванию
    выбранной метрики. Опционально с error bars (95% CI).

    Args:
        result: Результат Sobol анализа из SensitivityAnalyzer.run_sobol().
        metric: Какую метрику визуализировать:
            ``"S1"`` — только first-order,
            ``"ST"`` — только total-effect,
            ``"both"`` — S1 и ST рядом (grouped bar).
        top_n: Показать только top N параметров по ST. None — все.
        show_confidence: Показать error bars из S1_conf / ST_conf.
        height: Высота фигуры в пикселях.

    Returns:
        Plotly Figure с горизонтальными bar traces.

    Raises:
        ValueError: Если metric не в {``"S1"``, ``"ST"``, ``"both"``}.
        ValueError: Если result.parameter_names пуст.
    """
    if metric not in {"S1", "ST", "both"}:
        raise ValueError(f"metric должен быть 'S1', 'ST' или 'both', получено '{metric}'")
    if not result.parameter_names:
        raise ValueError("result.parameter_names не может быть пустым")

    # Сортировка по ST ascending: y[0] → ТОП в Plotly (наименьший сверху)
    sort_order = np.argsort(result.ST)
    n_params = len(result.parameter_names)

    # top_n: оставить только последние top_n (с наибольшим ST)
    if top_n is not None and 0 < top_n < n_params:
        sort_order = sort_order[-top_n:]

    names = [
        VARIABLE_LABELS.get(result.parameter_names[i], result.parameter_names[i])
        for i in sort_order
    ]
    s1_vals = result.S1[sort_order]
    st_vals = result.ST[sort_order]
    s1_conf = result.S1_conf[sort_order] if result.S1_conf is not None else None
    st_conf = result.ST_conf[sort_order] if result.ST_conf is not None else None

    fig = go.Figure()

    if metric in {"S1", "both"}:
        error_x = (
            {"type": "data", "array": s1_conf} if show_confidence and s1_conf is not None else None
        )
        fig.add_trace(
            go.Bar(
                name="S1",
                x=s1_vals,
                y=names,
                orientation="h",
                marker_color=ANALYSIS_COLORS["S1"],
                error_x=error_x,
            )
        )

    if metric in {"ST", "both"}:
        error_x = (
            {"type": "data", "array": st_conf} if show_confidence and st_conf is not None else None
        )
        fig.add_trace(
            go.Bar(
                name="ST",
                x=st_vals,
                y=names,
                orientation="h",
                marker_color=ANALYSIS_COLORS["ST"],
                error_x=error_x,
            )
        )

    title = f"Sobol Sensitivity — {result.output_variable}"
    return apply_default_layout(fig, height=height, title=title, barmode="group")


# ── plot_posterior ────────────────────────────────────────────────


# Description: Description/Phase3/description_analysis_plots.md#plot_posterior
def plot_posterior(
    result: EstimationResult,
    parameters: list[str] | None = None,
    layout: str = "marginals",
    show_ci: bool = True,
    show_point_estimate: bool = True,
    n_bins: int = 40,
    height: int = 600,
) -> go.Figure:
    """Маргинальные гистограммы / corner plot апостериорного распределения.

    ArviZ-style визуализация posterior samples из MCMC/Bayesian estimation,
    реализованная полностью в Plotly (без matplotlib/ArviZ).

    Args:
        result: Результат параметрической идентификации
            (``posterior_samples`` обязателен).
        parameters: Подмножество параметров для отображения.
            None — все из posterior_samples.
        layout: Режим визуализации:
            ``"marginals"`` — отдельные гистограммы в subplots (N rows × 1 col).
            ``"corner"`` — треугольная матрица (N × N): гистограммы на диагонали,
            2D scatter off-diagonal.
        show_ci: Показать вертикальные линии 95% CI (ci_lower, ci_upper).
        show_point_estimate: Показать вертикальную линию точечной оценки.
        n_bins: Число бинов гистограммы.
        height: Высота фигуры в пикселях.

    Returns:
        Plotly Figure.

    Raises:
        ValueError: Если ``result.posterior_samples is None`` (MLE метод).
        ValueError: Если запрошенный параметр отсутствует в posterior_samples.
    """
    if result.posterior_samples is None:
        raise ValueError(
            "MLE метод не предоставляет posterior samples. "
            "Используйте bayesian_pymc или mcmc_emcee."
        )
    if layout not in {"marginals", "corner"}:
        raise ValueError(f"layout должен быть 'marginals' или 'corner', получено '{layout}'")

    params = parameters if parameters is not None else list(result.posterior_samples.keys())
    for p in params:
        if p not in result.posterior_samples:
            available = list(result.posterior_samples.keys())
            raise ValueError(
                f"Параметр '{p}' отсутствует в posterior_samples. Доступные: {available}"
            )

    if layout == "marginals":
        n_rows = len(params)
        param_labels = [VARIABLE_LABELS.get(p, p) for p in params]
        fig = make_subplots(rows=n_rows, cols=1, subplot_titles=param_labels)

        for i, param in enumerate(params):
            row = i + 1
            samples = result.posterior_samples[param]
            fig.add_trace(
                go.Histogram(
                    x=samples,
                    nbinsx=n_bins,
                    marker_color=ANALYSIS_COLORS["posterior"],
                    name=param,
                    showlegend=False,
                ),
                row=row,
                col=1,
            )

            if show_ci and param in result.ci_lower and param in result.ci_upper:
                lo = result.ci_lower[param]
                hi = result.ci_upper[param]
                y_top = float(samples.max()) * 0.5
                for x_val in [lo, hi]:
                    fig.add_trace(
                        go.Scatter(
                            x=[x_val, x_val],
                            y=[0.0, y_top],
                            mode="lines",
                            line={"dash": "dash", "color": ANALYSIS_COLORS["ci"]},
                            showlegend=False,
                        ),
                        row=row,
                        col=1,
                    )

            if show_point_estimate and param in result.point_estimates:
                pe = result.point_estimates[param]
                y_top = float(samples.max()) * 0.5
                fig.add_trace(
                    go.Scatter(
                        x=[pe, pe],
                        y=[0.0, y_top],
                        mode="lines",
                        line={"dash": "solid", "color": ANALYSIS_COLORS["point_est"]},
                        showlegend=False,
                    ),
                    row=row,
                    col=1,
                )

        return apply_default_layout(
            fig, height=height, title="Posterior Distributions", showlegend=False
        )

    # layout == "corner"
    n_cols = len(params)
    fig = make_subplots(rows=n_cols, cols=n_cols)

    for i in range(n_cols):
        samples_i = result.posterior_samples[params[i]]
        # Diagonal histogram
        fig.add_trace(
            go.Histogram(
                x=samples_i,
                nbinsx=n_bins,
                marker_color=ANALYSIS_COLORS["posterior"],
                name=params[i],
                showlegend=False,
            ),
            row=i + 1,
            col=i + 1,
        )
        # Lower triangle scatter
        for j in range(i):
            samples_j = result.posterior_samples[params[j]]
            sx = samples_j.copy()
            sy = samples_i.copy()
            if len(sx) > 5000:
                rng = np.random.default_rng(0)
                idx = rng.choice(len(sx), size=5000, replace=False)
                sx = sx[idx]
                sy = sy[idx]
            fig.add_trace(
                go.Scatter(
                    x=sx,
                    y=sy,
                    mode="markers",
                    marker={"size": 2, "opacity": 0.3, "color": ANALYSIS_COLORS["posterior"]},
                    showlegend=False,
                ),
                row=i + 1,
                col=j + 1,
            )

    return apply_default_layout(fig, height=height, title="Corner Plot", showlegend=False)


# ── plot_convergence ─────────────────────────────────────────────


# Description: Description/Phase3/description_analysis_plots.md#plot_convergence
def plot_convergence(
    result: EstimationResult,
    metrics: list[str] | None = None,
    show_rhat_threshold: bool = True,
    height: int = 500,
) -> go.Figure:
    """Визуализация сходимости MCMC/MC по итерациям.

    Многопанельный график с метриками сходимости:
    - R-hat по параметрам (bar chart с threshold line при 1.05)
    - ESS bulk/tail по параметрам (grouped bar)
    - Trace plots: posterior samples vs iteration (если доступны)

    Args:
        result: Результат параметрической идентификации с diagnostics.
        metrics: Подмножество метрик для отображения:
            ``"rhat"`` — R-hat по параметрам,
            ``"ess"`` — ESS bulk и tail,
            ``"trace"`` — trace plots (samples vs iteration).
            None — все доступные.
        show_rhat_threshold: Показать горизонтальную линию R-hat = 1.05.
        height: Высота фигуры в пикселях.

    Returns:
        Plotly Figure с subplots.

    Raises:
        ValueError: Если ``result.diagnostics is None`` (MLE метод).
    """
    if result.diagnostics is None:
        raise ValueError("MLE метод не предоставляет диагностику сходимости.")

    _valid_metrics = {"rhat", "ess", "trace"}
    requested: list[str] = metrics if metrics is not None else ["rhat", "ess", "trace"]
    for m in requested:
        if m not in _valid_metrics:
            raise ValueError(f"Неизвестная метрика '{m}'. Допустимые: {sorted(_valid_metrics)}")

    diag = result.diagnostics
    active_panels: list[str] = []
    for m in requested:
        if m == "rhat" and diag.rhat:
            active_panels.append("rhat")
        elif m == "ess" and diag.ess_bulk:
            active_panels.append("ess")
        elif m == "trace" and result.posterior_samples:
            active_panels.append("trace")

    if not active_panels:
        raise ValueError("Нет доступных метрик для визуализации.")

    n_panels = len(active_panels)
    panel_titles = [_CONVERGENCE_PANEL_TITLES[p] for p in active_panels]
    fig = make_subplots(rows=n_panels, cols=1, subplot_titles=panel_titles)
    panel_row = {panel: i + 1 for i, panel in enumerate(active_panels)}

    # ── Панель rhat ──────────────────────────────────────────────
    if "rhat" in panel_row:
        row = panel_row["rhat"]
        param_names = list(diag.rhat.keys())
        rhat_vals = list(diag.rhat.values())
        colors = [
            ANALYSIS_COLORS["point_est"] if v < 1.05 else ANALYSIS_COLORS["ci"] for v in rhat_vals
        ]
        fig.add_trace(
            go.Bar(
                x=param_names,
                y=rhat_vals,
                marker_color=colors,
                name="R-hat",
                showlegend=False,
            ),
            row=row,
            col=1,
        )
        if show_rhat_threshold:
            fig.add_trace(
                go.Scatter(
                    x=[param_names[0], param_names[-1]],
                    y=[1.05, 1.05],
                    mode="lines",
                    line={"dash": "dash", "color": ANALYSIS_COLORS["threshold"]},
                    name="R-hat = 1.05",
                    showlegend=False,
                ),
                row=row,
                col=1,
            )

    # ── Панель ess ───────────────────────────────────────────────
    if "ess" in panel_row:
        row = panel_row["ess"]
        ess_param_names = list(diag.ess_bulk.keys())
        fig.add_trace(
            go.Bar(
                x=ess_param_names,
                y=list(diag.ess_bulk.values()),
                name="ESS bulk",
                marker_color=ANALYSIS_COLORS["convergence"],
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=ess_param_names,
                y=list(diag.ess_tail.values()),
                name="ESS tail",
                marker_color=ANALYSIS_COLORS["posterior"],
            ),
            row=row,
            col=1,
        )
        fig.update_layout(barmode="group")

    # ── Панель trace ─────────────────────────────────────────────
    if "trace" in panel_row:
        row = panel_row["trace"]
        posterior = result.posterior_samples  # не None: отфильтровано выше
        assert posterior is not None
        params_list = list(posterior.keys())
        n_chains = result.config.n_chains if result.config else 1
        for param in params_list:
            samples = posterior[param]
            n_per_chain = len(samples) // n_chains
            for chain_idx in range(n_chains):
                chain_samples = samples[chain_idx * n_per_chain : (chain_idx + 1) * n_per_chain]
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(chain_samples))),
                        y=chain_samples,
                        mode="lines",
                        opacity=0.7,
                        name=f"{param} chain {chain_idx + 1}",
                        showlegend=False,
                    ),
                    row=row,
                    col=1,
                )

    return apply_default_layout(fig, height=height * n_panels, title="Convergence Diagnostics")


# ── plot_morris ──────────────────────────────────────────────────


# Description: Description/Phase3/description_analysis_plots.md#plot_morris
def plot_morris(
    result: MorrisResult,
    highlight_influential: bool = True,
    threshold_ratio: float = 0.1,
    show_labels: bool = True,
    show_wedge: bool = True,
    height: int = 500,
) -> go.Figure:
    """Morris screening scatter plot: μ* vs σ.

    Scatter plot в пространстве (μ*, σ). Каждая точка — параметр.
    НЕ tornado chart: двумерное представление, где:

    - μ* (ось X) — общая важность параметра (|средний эффект|)
    - σ  (ось Y) — нелинейность / взаимодействия параметра

    Параметры выше диагонали σ = μ* имеют сильные нелинейности
    или взаимодействия; ниже — преимущественно линейные эффекты.

    Args:
        result: Результат Morris скрининга из
            SensitivityAnalyzer.run_morris().
        highlight_influential: Выделить цветом влиятельные параметры
            (по ``result.get_influential(threshold_ratio)``).
        threshold_ratio: Порог для get_influential (доля от max(μ*)).
        show_labels: Показать подписи параметров рядом с точками.
        show_wedge: Показать диагональную линию σ = μ*
            (граница линейность/нелинейность).
        height: Высота фигуры в пикселях.

    Returns:
        Plotly Figure со Scatter trace(s).

    Raises:
        ValueError: Если result.parameter_names пуст.
    """
    if not result.parameter_names:
        raise ValueError("result.parameter_names не может быть пустым")

    n_params = len(result.parameter_names)
    param_arr = np.array(result.parameter_names)

    if highlight_influential:
        influential_set = set(result.get_influential(threshold_ratio))
        inf_mask = np.array([n in influential_set for n in result.parameter_names])
        non_inf_mask = ~inf_mask
    else:
        inf_mask = np.zeros(n_params, dtype=bool)
        non_inf_mask = np.ones(n_params, dtype=bool)

    fig = go.Figure()

    # Non-influential (серые, size=8)
    if non_inf_mask.any():
        non_inf_names = param_arr[non_inf_mask]
        labels = [VARIABLE_LABELS.get(n, n) for n in non_inf_names]
        text = labels if show_labels and n_params <= 20 else None
        mode = "markers+text" if text else "markers"
        fig.add_trace(
            go.Scatter(
                x=result.mu_star[non_inf_mask],
                y=result.sigma[non_inf_mask],
                mode=mode,
                marker={"color": ANALYSIS_COLORS["threshold"], "size": 8},
                text=text,
                textposition="top center",
                name="Non-influential",
            )
        )

    # Influential (красные, size=12)
    if inf_mask.any():
        inf_names = param_arr[inf_mask]
        labels = [VARIABLE_LABELS.get(n, n) for n in inf_names]
        text = labels if show_labels else None
        mode = "markers+text" if text else "markers"
        has_conf = result.mu_star_conf is not None and np.any(result.mu_star_conf[inf_mask] > 0)
        error_x = {"type": "data", "array": result.mu_star_conf[inf_mask]} if has_conf else None
        fig.add_trace(
            go.Scatter(
                x=result.mu_star[inf_mask],
                y=result.sigma[inf_mask],
                mode=mode,
                marker={"color": ANALYSIS_COLORS["influential"], "size": 12},
                text=text,
                textposition="top center",
                name="Influential",
                error_x=error_x,
            )
        )

    # Wedge line σ = μ*
    if show_wedge:
        max_val = float(max(result.mu_star.max(), result.sigma.max())) * 1.1
        fig.add_trace(
            go.Scatter(
                x=[0.0, max_val],
                y=[0.0, max_val],
                mode="lines",
                line={"dash": "dash", "color": ANALYSIS_COLORS["threshold"]},
                name="σ = μ*",
                showlegend=False,
            )
        )

    title = f"Morris Screening — {result.output_variable}"
    return apply_default_layout(
        fig,
        height=height,
        title=title,
        xaxis_title="μ* (среднее |элементарных эффектов|)",
        yaxis_title="σ (СКО элементарных эффектов)",
    )
