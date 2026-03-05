"""RegenTwin — MVP интерфейс для симуляции регенерации тканей.

Streamlit-приложение для визуализации SDE/ABM моделей
с терапиями PRP и PEMF.
"""

import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

from src.core.sde_model import SDEConfig, SDEModel, TherapyProtocol
from src.core.abm_model import ABMConfig, ABMModel
from src.core.monte_carlo import MonteCarloConfig, MonteCarloSimulator
from src.core.therapy_models import PRPModel, PEMFModel, PRPConfig, PEMFConfig
from src.data.parameter_extraction import ModelParameters

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="RegenTwin — Tissue Regeneration Simulator",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a5276;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #566573;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 1.2rem;
        border-left: 4px solid #2e86c1;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🧬 RegenTwin</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Мультимасштабная симуляция регенерации тканей | SDE + Agent-Based Model'
    '</div>',
    unsafe_allow_html=True,
)


# ── Sidebar — Parameters ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Параметры симуляции")

    st.subheader("🔬 Начальные условия")
    n0 = st.number_input(
        "Начальная плотность клеток (клеток/мкл)",
        min_value=100.0, max_value=100000.0, value=5000.0, step=500.0,
    )
    c0 = st.number_input(
        "Начальная концентрация цитокинов (нг/мл)",
        min_value=0.1, max_value=100.0, value=5.0, step=1.0,
    )
    stem_frac = st.slider("Доля стволовых клеток (CD34+)", 0.01, 0.30, 0.05, 0.01)
    macro_frac = st.slider("Доля макрофагов", 0.05, 0.40, 0.15, 0.01)
    inflammation = st.slider("Уровень воспаления", 0.0, 1.0, 0.3, 0.05)

    st.subheader("⏱️ Время симуляции")
    t_max = st.slider("Длительность (дни)", 5, 90, 30, 5)
    dt = st.select_slider("Шаг интегрирования (дни)", [0.001, 0.005, 0.01, 0.05], value=0.01)

    st.subheader("💉 Терапия")
    prp_enabled = st.checkbox("PRP (Platelet-Rich Plasma)", value=False)
    pemf_enabled = st.checkbox("PEMF (Pulsed EM Field)", value=False)

    prp_start = prp_intensity = pemf_start = pemf_freq = pemf_intensity = None
    if prp_enabled:
        prp_start = st.slider("PRP — начало (день)", 0, t_max, 0)
        prp_intensity = st.slider("PRP — интенсивность", 0.1, 2.0, 1.0, 0.1)
    if pemf_enabled:
        pemf_start = st.slider("PEMF — начало (день)", 0, t_max, 0)
        pemf_freq = st.slider("PEMF — частота (Hz)", 10.0, 100.0, 50.0, 5.0)
        pemf_intensity = st.slider("PEMF — интенсивность", 0.1, 2.0, 1.0, 0.1)

    st.subheader("🔧 Параметры модели")
    growth_rate = st.slider("Скорость пролиферации r (1/день)", 0.05, 1.0, 0.3, 0.05)
    carrying_cap = st.number_input(
        "Carrying capacity K (клеток/см²)",
        min_value=1e4, max_value=1e8, value=1e6, step=1e5, format="%.0f",
    )
    sigma = st.slider("Стохастическая волатильность σ", 0.0, 0.3, 0.05, 0.01)


# ── Build configs ─────────────────────────────────────────────────────
def build_params() -> ModelParameters:
    return ModelParameters(
        n0=n0,
        stem_cell_fraction=stem_frac,
        macrophage_fraction=macro_frac,
        apoptotic_fraction=0.02,
        c0=c0,
        inflammation_level=inflammation,
    )


def build_sde_config() -> SDEConfig:
    return SDEConfig(
        r=growth_rate,
        K=carrying_cap,
        sigma_n=sigma,
        dt=dt,
        t_max=float(t_max),
    )


def build_therapy() -> TherapyProtocol:
    return TherapyProtocol(
        prp_enabled=prp_enabled,
        prp_start_time=float(prp_start) if prp_start is not None else 0.0,
        prp_intensity=prp_intensity if prp_intensity is not None else 1.0,
        pemf_enabled=pemf_enabled,
        pemf_start_time=float(pemf_start) if pemf_start is not None else 0.0,
        pemf_frequency=pemf_freq if pemf_freq is not None else 50.0,
        pemf_intensity=pemf_intensity if pemf_intensity is not None else 1.0,
    )


# ── Tabs ──────────────────────────────────────────────────────────────
tab_sde, tab_abm, tab_mc, tab_therapy, tab_about = st.tabs([
    "📈 SDE Симуляция",
    "🧫 ABM Модель",
    "🎲 Monte Carlo",
    "💊 Сравнение терапий",
    "📖 О проекте",
])

# ══════════════════════════════════════════════════════════════════════
# TAB 1: SDE Simulation
# ══════════════════════════════════════════════════════════════════════
with tab_sde:
    st.subheader("Стохастическое дифференциальное уравнение — динамика клеток")

    st.latex(r"dN_t = \left[rN_t\left(1 - \frac{N_t}{K}\right) + \alpha f(\text{PRP}) + \beta g(\text{PEMF}) - \delta N_t\right]dt + \sigma N_t\, dW_t")

    if st.button("▶ Запустить SDE", key="run_sde"):
        params = build_params()
        config = build_sde_config()
        therapy = build_therapy()

        with st.spinner("Симуляция SDE..."):
            model = SDEModel(config, therapy=therapy, random_seed=42)
            result = model.simulate(params)

        # Metrics row
        stats = result.get_statistics()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Финальная плотность N", f"{result.N_values[-1]:,.0f}")
        with col2:
            st.metric("Макс. плотность N", f"{result.N_values.max():,.0f}")
        with col3:
            st.metric("Финальные цитокины C", f"{result.C_values[-1]:.2f}")
        with col4:
            growth = (result.N_values[-1] - result.N_values[0]) / result.N_values[0] * 100
            st.metric("Рост клеток", f"{growth:+.1f}%")

        # Dual-axis plot
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Динамика клеточной плотности N(t)", "Динамика цитокинов C(t)"),
            vertical_spacing=0.12,
        )

        fig.add_trace(
            go.Scatter(
                x=result.times, y=result.N_values,
                mode="lines", name="N(t) — клетки",
                line=dict(color="#2e86c1", width=2),
                fill="tozeroy", fillcolor="rgba(46,134,193,0.1)",
            ),
            row=1, col=1,
        )
        # Carrying capacity line
        fig.add_hline(
            y=config.K, line_dash="dash", line_color="red",
            annotation_text=f"K = {config.K:.0e}",
            row=1, col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=result.times, y=result.C_values,
                mode="lines", name="C(t) — цитокины",
                line=dict(color="#e74c3c", width=2),
                fill="tozeroy", fillcolor="rgba(231,76,60,0.1)",
            ),
            row=2, col=1,
        )

        fig.update_layout(
            height=600, template="plotly_white",
            font=dict(size=13),
            showlegend=True,
        )
        fig.update_xaxes(title_text="Время (дни)", row=2, col=1)
        fig.update_yaxes(title_text="Клеток/мкл", row=1, col=1)
        fig.update_yaxes(title_text="нг/мл", row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

        # Phase portrait
        st.subheader("Фазовый портрет N-C")
        fig_phase = go.Figure()
        fig_phase.add_trace(go.Scatter(
            x=result.N_values, y=result.C_values,
            mode="lines", name="Траектория",
            line=dict(color="#8e44ad", width=1.5),
        ))
        fig_phase.add_trace(go.Scatter(
            x=[result.N_values[0]], y=[result.C_values[0]],
            mode="markers", name="Начало",
            marker=dict(size=12, color="green", symbol="star"),
        ))
        fig_phase.add_trace(go.Scatter(
            x=[result.N_values[-1]], y=[result.C_values[-1]],
            mode="markers", name="Конец",
            marker=dict(size=12, color="red", symbol="x"),
        ))
        fig_phase.update_layout(
            xaxis_title="N (клетки)", yaxis_title="C (цитокины)",
            height=400, template="plotly_white",
        )
        st.plotly_chart(fig_phase, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 2: ABM Simulation
# ══════════════════════════════════════════════════════════════════════
with tab_abm:
    st.subheader("Agent-Based модель клеточной динамики")
    st.markdown("""
    Микроскопическая модель дискретных агентов:
    **Стволовые клетки** (CD34+) · **Макрофаги** (CD14+/CD68+) · **Фибробласты** ·
    **Нейтрофилы** (CD66b+) · **Эндотелиальные** (CD31+) · **Миофибробласты** (α-SMA+)
    """)

    col_abm1, col_abm2 = st.columns(2)
    with col_abm1:
        abm_t_max = st.slider("Время ABM (часы)", 24, 720, 168, 24, key="abm_tmax")
        abm_dt = st.select_slider("Шаг ABM (часы)", [0.1, 0.5, 1.0, 2.0], value=1.0, key="abm_dt")
    with col_abm2:
        n_stem = st.number_input("Стволовые клетки", 5, 200, 50, 5)
        n_macro = st.number_input("Макрофаги", 5, 200, 30, 5)
        n_fibro = st.number_input("Фибробласты", 5, 200, 20, 5)

    if st.button("▶ Запустить ABM", key="run_abm"):
        abm_config = ABMConfig(
            dt=abm_dt,
            t_max=float(abm_t_max),
            initial_stem_cells=n_stem,
            initial_macrophages=n_macro,
            initial_fibroblasts=n_fibro,
        )
        params = build_params()

        # Snapshot every few hours for smooth visualization
        snap_interval = max(1.0, abm_t_max / 50)
        with st.spinner("ABM симуляция... (может занять время)"):
            abm_model = ABMModel(abm_config)
            abm_result = abm_model.simulate(params, snapshot_interval=snap_interval)

        snapshots = abm_result.snapshots

        if len(snapshots) > 1:
            # Extract time series of agent counts by type
            times_abm = [s.t for s in snapshots]
            counts_by_type: dict[str, list[int]] = {}

            for snap in snapshots:
                type_counts = snap.get_agent_count_by_type()
                for agent_type, count in type_counts.items():
                    if agent_type not in counts_by_type:
                        counts_by_type[agent_type] = []
                    counts_by_type[agent_type].append(count)

            # Pad shorter lists (if a type appears later)
            max_len = len(times_abm)
            for k in counts_by_type:
                while len(counts_by_type[k]) < max_len:
                    counts_by_type[k].insert(0, 0)

            # Totals
            total_agents = [sum(counts_by_type[k][i] for k in counts_by_type) for i in range(max_len)]

            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Всего агентов (начало)", total_agents[0])
            with col2:
                st.metric("Всего агентов (конец)", total_agents[-1])
            with col3:
                st.metric("Типов клеток", len(counts_by_type))

            # Stacked area chart
            fig_abm = go.Figure()
            colors = px.colors.qualitative.Set2
            for i, (agent_type, counts) in enumerate(counts_by_type.items()):
                fig_abm.add_trace(go.Scatter(
                    x=times_abm, y=counts,
                    mode="lines", name=agent_type,
                    stackgroup="agents",
                    line=dict(width=0.5, color=colors[i % len(colors)]),
                ))

            fig_abm.update_layout(
                title="Динамика клеточных популяций (ABM)",
                xaxis_title="Время (часы)",
                yaxis_title="Количество агентов",
                height=500, template="plotly_white",
                legend=dict(orientation="h", y=-0.15),
            )
            st.plotly_chart(fig_abm, use_container_width=True)

            # Spatial visualization for last snapshot
            last_snap = snapshots[-1]
            if last_snap.agents:
                st.subheader("Пространственное распределение агентов")
                x_coords = []
                y_coords = []
                agent_types = []
                type_names = {"stem": "Стволовые (CD34+)", "macro": "Макрофаги", "fibro": "Фибробласты",
                              "neutrophil": "Нейтрофилы", "endothelial": "Эндотелиальные", "myofibroblast": "Миофибробласты"}
                for agent in last_snap.agents:
                    if hasattr(agent, "x") and hasattr(agent, "y"):
                        x_coords.append(agent.x)
                        y_coords.append(agent.y)
                        agent_types.append(type_names.get(agent.agent_type, agent.agent_type))

                if x_coords:
                    fig_spatial = px.scatter(
                        x=x_coords, y=y_coords, color=agent_types,
                        labels={"x": "X (мкм)", "y": "Y (мкм)", "color": "Тип клетки"},
                        title=f"Пространственное распределение (t = {last_snap.t:.0f} ч)",
                        color_discrete_sequence=px.colors.qualitative.Set2,
                    )
                    fig_spatial.update_layout(
                        height=500, template="plotly_white",
                        xaxis=dict(scaleanchor="y", scaleratio=1),
                    )
                    st.plotly_chart(fig_spatial, use_container_width=True)

            # Heatmaps of cytokine and ECM fields
            st.subheader("Поля цитокинов и ECM")
            col_h1, col_h2 = st.columns(2)
            with col_h1:
                fig_cyto = go.Figure(data=go.Heatmap(
                    z=last_snap.cytokine_field,
                    colorscale="YlOrRd",
                    colorbar_title="нг/мл",
                ))
                fig_cyto.update_layout(
                    title=f"Поле цитокинов (t={last_snap.t:.0f}ч)",
                    height=400, template="plotly_white",
                    xaxis_title="X", yaxis_title="Y",
                )
                st.plotly_chart(fig_cyto, use_container_width=True)
            with col_h2:
                fig_ecm = go.Figure(data=go.Heatmap(
                    z=last_snap.ecm_field,
                    colorscale="Greens",
                    colorbar_title="отн. ед.",
                ))
                fig_ecm.update_layout(
                    title=f"Внеклеточный матрикс ECM (t={last_snap.t:.0f}ч)",
                    height=400, template="plotly_white",
                    xaxis_title="X", yaxis_title="Y",
                )
                st.plotly_chart(fig_ecm, use_container_width=True)
        else:
            st.warning("Слишком мало снимков для визуализации. Попробуйте увеличить время или уменьшить шаг.")

# ══════════════════════════════════════════════════════════════════════
# TAB 3: Monte Carlo
# ══════════════════════════════════════════════════════════════════════
with tab_mc:
    st.subheader("Monte Carlo анализ — ансамбль стохастических траекторий")

    col_mc1, col_mc2 = st.columns(2)
    with col_mc1:
        n_traj = st.slider("Количество траекторий", 10, 500, 50, 10)
    with col_mc2:
        confidence = st.slider("Уровень доверия (%)", 80, 99, 95, 1)

    if st.button("▶ Запустить Monte Carlo", key="run_mc"):
        sde_config = build_sde_config()
        mc_config = MonteCarloConfig(
            n_trajectories=n_traj,
            sde_config=sde_config,
        )
        params = build_params()

        progress = st.progress(0, "Запуск Monte Carlo...")
        sim = MonteCarloSimulator(mc_config)
        result = sim.run(params)
        progress.progress(100, "Готово!")

        # Summary stats
        summary = result.get_summary_statistics()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Средняя финальная N", f"{summary['mean_final_N']:,.0f}")
        with col2:
            st.metric("Std финальной N", f"{summary['std_final_N']:,.0f}")
        with col3:
            st.metric("Успешных траекторий", f"{summary['success_rate']:.0%}")
        with col4:
            st.metric("Средний рост", f"{summary.get('mean_growth_rate', 0):.2f}")

        # Confidence interval plot
        try:
            ci_lower, ci_upper = result.get_confidence_interval("N", confidence / 100)
            mean_n = result.mean_N

            fig_mc = go.Figure()

            # Individual trajectories (light)
            for i, traj in enumerate(result.trajectories[:20]):
                if traj.success:
                    fig_mc.add_trace(go.Scatter(
                        x=traj.times, y=traj.N_values,
                        mode="lines",
                        line=dict(color="rgba(46,134,193,0.1)", width=0.5),
                        showlegend=(i == 0),
                        name="Траектории",
                    ))

            # Confidence band
            times = result.trajectories[0].times if result.trajectories else np.array([])
            if len(times) > 0 and len(ci_lower) == len(times):
                fig_mc.add_trace(go.Scatter(
                    x=np.concatenate([times, times[::-1]]),
                    y=np.concatenate([ci_upper, ci_lower[::-1]]),
                    fill="toself",
                    fillcolor=f"rgba(231,76,60,0.15)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name=f"{confidence}% доверительный интервал",
                ))

            # Mean
            if len(times) > 0 and len(mean_n) == len(times):
                fig_mc.add_trace(go.Scatter(
                    x=times, y=mean_n,
                    mode="lines", name="Среднее",
                    line=dict(color="#e74c3c", width=3),
                ))

            fig_mc.update_layout(
                title=f"Monte Carlo: {n_traj} траекторий с {confidence}% CI",
                xaxis_title="Время (дни)",
                yaxis_title="Плотность клеток N",
                height=500, template="plotly_white",
            )
            st.plotly_chart(fig_mc, use_container_width=True)
        except Exception as e:
            st.error(f"Ошибка визуализации CI: {e}")

        # Distribution of final values
        try:
            final_dist = result.get_final_distribution("N")
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=final_dist, nbinsx=30,
                name="Финальная N",
                marker_color="#2e86c1",
                opacity=0.7,
            ))
            fig_hist.update_layout(
                title="Распределение финальной плотности клеток",
                xaxis_title="N (клеток/мкл)",
                yaxis_title="Частота",
                height=350, template="plotly_white",
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        except Exception as e:
            st.error(f"Ошибка гистограммы: {e}")


# ══════════════════════════════════════════════════════════════════════
# TAB 4: Therapy Comparison
# ══════════════════════════════════════════════════════════════════════
with tab_therapy:
    st.subheader("Сравнение терапевтических протоколов")
    st.markdown("""
    Сравнительный анализ четырёх сценариев:
    **Без терапии** · **PRP** · **PEMF** · **PRP + PEMF (комбинированная)**
    """)

    if st.button("▶ Сравнить терапии", key="run_therapy"):
        params = build_params()
        config = build_sde_config()

        therapies = {
            "Без терапии": TherapyProtocol(),
            "PRP": TherapyProtocol(
                prp_enabled=True, prp_start_time=0.0,
                prp_intensity=1.0,
            ),
            "PEMF": TherapyProtocol(
                pemf_enabled=True, pemf_start_time=0.0,
                pemf_frequency=50.0, pemf_intensity=1.0,
            ),
            "PRP + PEMF": TherapyProtocol(
                prp_enabled=True, prp_start_time=0.0,
                prp_intensity=1.0,
                pemf_enabled=True, pemf_start_time=0.0,
                pemf_frequency=50.0, pemf_intensity=1.0,
                synergy_factor=1.2,
            ),
        }

        results = {}
        colors_map = {
            "Без терапии": "#95a5a6",
            "PRP": "#e74c3c",
            "PEMF": "#2e86c1",
            "PRP + PEMF": "#27ae60",
        }

        with st.spinner("Запуск 4 симуляций..."):
            for name, therapy in therapies.items():
                model = SDEModel(config, therapy=therapy, random_seed=42)
                results[name] = model.simulate(params)

        # Comparison plot
        fig_comp = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Плотность клеток N(t)", "Цитокины C(t)"),
            vertical_spacing=0.12,
        )

        for name, res in results.items():
            fig_comp.add_trace(
                go.Scatter(
                    x=res.times, y=res.N_values,
                    mode="lines", name=name,
                    line=dict(color=colors_map[name], width=2),
                ),
                row=1, col=1,
            )
            fig_comp.add_trace(
                go.Scatter(
                    x=res.times, y=res.C_values,
                    mode="lines", name=name,
                    line=dict(color=colors_map[name], width=2, dash="dot"),
                    showlegend=False,
                ),
                row=2, col=1,
            )

        fig_comp.update_layout(
            height=600, template="plotly_white",
            legend=dict(orientation="h", y=-0.1),
        )
        fig_comp.update_xaxes(title_text="Время (дни)", row=2, col=1)
        fig_comp.update_yaxes(title_text="Клеток/мкл", row=1, col=1)
        fig_comp.update_yaxes(title_text="нг/мл", row=2, col=1)

        st.plotly_chart(fig_comp, use_container_width=True)

        # Comparison table
        st.subheader("Сравнительная таблица")
        import pandas as pd
        comp_data = []
        for name, res in results.items():
            growth = (res.N_values[-1] - res.N_values[0]) / res.N_values[0] * 100
            comp_data.append({
                "Протокол": name,
                "Финальная N": f"{res.N_values[-1]:,.0f}",
                "Макс N": f"{res.N_values.max():,.0f}",
                "Рост (%)": f"{growth:+.1f}%",
                "Финальная C": f"{res.C_values[-1]:.2f}",
                "Время до макс N (дни)": f"{res.times[np.argmax(res.N_values)]:.1f}",
            })
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

        # PRP/PEMF individual profiles
        st.subheader("Профили терапевтических эффектов")
        prp_model = PRPModel(PRPConfig())
        pemf_model = PEMFModel(PEMFConfig())

        t_therapy = np.linspace(0, float(t_max), 500)
        prp_effects = [prp_model.compute_release(t).theta_total for t in t_therapy]
        pemf_prolif = [pemf_model.compute_effects(t).proliferation for t in t_therapy]
        pemf_anti = [pemf_model.compute_effects(t).anti_inflammatory for t in t_therapy]

        fig_profiles = make_subplots(
            rows=1, cols=2,
            subplot_titles=("PRP: Высвобождение факторов роста", "PEMF: Эффекты"),
        )
        fig_profiles.add_trace(
            go.Scatter(x=t_therapy, y=prp_effects, name="PRP (суммарный)",
                       line=dict(color="#e74c3c", width=2)),
            row=1, col=1,
        )
        fig_profiles.add_trace(
            go.Scatter(x=t_therapy, y=pemf_prolif, name="PEMF (пролиферация)",
                       line=dict(color="#2e86c1", width=2)),
            row=1, col=2,
        )
        fig_profiles.add_trace(
            go.Scatter(x=t_therapy, y=pemf_anti, name="PEMF (антивоспаление)",
                       line=dict(color="#27ae60", width=2)),
            row=1, col=2,
        )
        fig_profiles.update_layout(height=350, template="plotly_white")
        fig_profiles.update_xaxes(title_text="Время (дни)")

        st.plotly_chart(fig_profiles, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 5: About
# ══════════════════════════════════════════════════════════════════════
with tab_about:
    st.subheader("О проекте RegenTwin")

    st.markdown("""
    **RegenTwin** — программный инструмент для мультимасштабной симуляции
    регенерации тканей с использованием данных flow cytometry.

    ### Математическая модель

    Система использует **двухуровневую** архитектуру:

    **Макроуровень — SDE (стохастические дифференциальные уравнения):**
    Уравнение Ланжевена для динамики клеточных популяций с логистическим ростом,
    терапевтическими факторами и стохастическим шумом.

    **Микроуровень — ABM (Agent-Based Model):**
    Дискретные агенты 6 типов с пространственным движением, хемотаксисом,
    делением, гибелью и межклеточными взаимодействиями.

    ### Типы агентов ABM
    | Тип | Маркер | Роль |
    |-----|--------|------|
    | Стволовые клетки | CD34+ | Дифференциация, самообновление |
    | Макрофаги | CD14+/CD68+ | Фагоцитоз, M1↔M2 поляризация |
    | Фибробласты | — | Продукция ECM, коллагена |
    | Нейтрофилы | CD66b+ | Первичный иммунный ответ |
    | Эндотелиальные | CD31+ | Ангиогенез |
    | Миофибробласты | α-SMA+ | Контракция раны |

    ### Терапевтические модели
    - **PRP (Platelet-Rich Plasma):** Многокомпонентная кинетика высвобождения
      PDGF, VEGF, TGF-β, EGF с различными временными константами
    - **PEMF (Pulsed Electromagnetic Field):** Частотно-зависимый эффект
      на пролиферацию, противовоспалительное действие и миграцию клеток
    - **Синергия PRP+PEMF:** Мультипликативный фактор совместного применения

    ### Технологический стек
    Python 3.11+ · NumPy/SciPy · Streamlit · Plotly · FastAPI

    ### Методы
    - Euler-Maruyama для SDE интегрирования
    - cKDTree для пространственного поиска соседей в ABM
    - Monte Carlo для статистического анализа
    - Hill-функции для хемотаксиса и рецепторной кинетики

    ---
    **Лицензия:** MIT · **Автор:** RegenTwin Team
    """)

    # Show project structure
    with st.expander("📁 Структура проекта"):
        st.code("""
regentwin/
├── src/
│   ├── core/              # 9 модулей, ~9000 строк
│   │   ├── sde_model.py       — SDE модель (Langevin)
│   │   ├── abm_model.py       — Agent-Based модель (6 типов)
│   │   ├── monte_carlo.py     — Monte Carlo симулятор
│   │   ├── therapy_models.py  — PRP / PEMF / Синергия
│   │   ├── extended_sde.py    — Расширенная многокомпонентная SDE
│   │   ├── wound_phases.py    — Детектор фаз заживления
│   │   ├── sde_numerics.py    — Euler-Maruyama, Milstein, Runge-Kutta
│   │   ├── integration.py     — SDE↔ABM связывание
│   │   ├── robustness.py      — Анализ робастности
│   │   └── numerical_utils.py — Математические утилиты
│   └── data/              # Парсинг flow cytometry
│       ├── fcs_parser.py      — FCS-файлы (FlowKit)
│       ├── gating.py          — Гейтинг популяций
│       └── parameter_extraction.py — Извлечение параметров
├── frontend/
│   └── app.py             — Streamlit интерфейс (этот файл)
├── tests/                 # Unit + Integration + Performance
└── Doks/                  # Математический фреймворк
        """, language="text")
