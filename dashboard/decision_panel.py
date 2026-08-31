import plotly.graph_objects as go
import streamlit as st
from typing import Any, Dict, List
from dashboard import theme


STRAT_COLORS = {
    "EXPLORE": "#00F0FF",
    "EXPLOIT": "#00FF9D",
    "PREDICT": "#FFB800",
    "BALANCED": "#A855F7",
    "SEQUENTIAL_SWEEP": "#6B7280",
}


def _fmt(v: Any, spec: str = ".2f") -> str:
    if v is None:
        return "N/A"
    try:
        return format(v, spec)
    except (ValueError, TypeError):
        return str(v)


def render_decision_panel(engine: Any) -> None:
    """Render current strategy status with interactive Polar Radar and Band Score charts."""
    snap = engine.get_snapshot()
    band_table = snap.get("band_scores_table", [])
    sel_bands = snap.get("selected_bands", [])
    curr_strat = snap.get("current_strategy", snap.get("selected_strategy", "BALANCED"))
    cur_step = snap.get("timestep", snap.get("current_step", 0))
    sim_time = snap.get("simulated_time_s", snap.get("simulation_time_s", 0.0))
    s_col = STRAT_COLORS.get(curr_strat, "#00F0FF")

    # 1. Header Banner
    st.markdown(
        f"""
        <div class='glass-card' style='padding:1rem 1.25rem; margin-bottom:1rem; border-left:4px solid {s_col} !important;'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <div>
                    <div style='font-size:1.3rem; font-weight:800; color:{theme.COLOR_TEXT}; font-family:"Outfit"; letter-spacing:0.04em;'>
                        COGNITIVE AI DECISION ENGINE
                    </div>
                    <div style='font-size:0.8rem; color:{theme.COLOR_TEXT_MUTED}; margin-top:0.1rem;'>
                        Reinforcement Learning Arbitrator & Multi-Stage Band Scoring
                    </div>
                </div>
                <div style='text-align:right;'>
                    <span style='font-size:0.75rem; font-weight:700; color:{s_col}; background:rgba(0,240,255,0.1); padding:0.3rem 0.75rem; border-radius:4px;'>
                        ACTIVE POLICY: {curr_strat}
                    </span>
                    <div style='font-family:monospace; font-size:0.8rem; color:#9CA3AF; margin-top:0.4rem;'>
                        Step {cur_step} &nbsp;·&nbsp; {sim_time:.2f}s Elapsed
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Main Visual Dashboard: Left (Meta-Strategy Q-Value Radar) | Right (Top Selected Band Scores)
    c1, c2 = st.columns([5, 5])

    # Left Column: Reinforcement Learning Meta-Arbitrator Polar Radar Chart
    with c1:
        st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#9CA3AF; margin-bottom:0.4rem; text-transform:uppercase;'>REINFORCEMENT LEARNING ARBITRATOR (Q-VALUES)</div>", unsafe_allow_html=True)
        q_vals = snap.get("meta_q_values")
        if q_vals is not None and len(q_vals) == 4:
            categories = ["EXPLORE", "EXPLOIT", "PREDICT", "BALANCED"]
            values = list(q_vals)
            # Close the polygon loop
            cat_closed = categories + [categories[0]]
            val_closed = values + [values[0]]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=val_closed,
                theta=cat_closed,
                fill="toself",
                fillcolor="rgba(0, 240, 255, 0.2)",
                line=dict(color="#00F0FF", width=2.5),
                marker=dict(size=8, color="#00FF9D"),
                name="Meta-Strategy Q-Values",
            ))

            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, max(max(values) * 1.2, 1.0)], tickfont=dict(color="#9CA3AF", size=9), gridcolor="rgba(0, 240, 255, 0.1)"),
                    angularaxis=dict(tickfont=dict(color="#F3F4F6", size=11, family="Outfit"), gridcolor="rgba(0, 240, 255, 0.1)"),
                    bgcolor="#07090E",
                ),
                paper_bgcolor="#07090E",
                plot_bgcolor="#07090E",
                showlegend=False,
                height=340,
                margin=dict(l=40, r=40, t=30, b=30),
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.info("Meta-Strategy Q-Values not available in this mode.")

    # Right Column: Top Selected Band Scores Horizontal Bar Chart
    with c2:
        st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#9CA3AF; margin-bottom:0.4rem; text-transform:uppercase;'>TOP 5 SELECTED BANDS (COMPOSITE AI SCORE)</div>", unsafe_allow_html=True)
        sel_rows_map = {r["Band"]: r for r in band_table if r["Band"] in sel_bands}
        if sel_bands:
            bands_plot = list(reversed(sel_bands))
            scores_plot = [_fmt(sel_rows_map.get(b, {}).get("Final Score", 0.0), ".3f") for b in bands_plot]
            try:
                num_scores = [float(s) if s != "N/A" else 0.0 for s in scores_plot]
            except ValueError:
                num_scores = [0.0] * len(bands_plot)

            bar_cols = ["#00FF9D" if i == len(bands_plot) - 1 else "#00F0FF" for i in range(len(bands_plot))]

            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                y=bands_plot,
                x=num_scores,
                orientation="h",
                marker=dict(color=bar_cols, line=dict(color="rgba(0,240,255,0.3)", width=1)),
                text=[f"Score: {s:.3f}" for s in num_scores],
                textposition="outside",
                hoverinfo="text",
                hovertext=[
                    f"Band: {b}<br>Final Score: {sel_rows_map.get(b,{}).get('Final Score','N/A')}<br>"
                    f"Activity: {sel_rows_map.get(b,{}).get('P(Active)','N/A')}<br>"
                    f"Temporal: {sel_rows_map.get(b,{}).get('Temporal Score','N/A')}"
                    for b in bands_plot
                ],
            ))

            fig_bar.update_layout(
                xaxis=dict(title="Composite Score", gridcolor="rgba(0, 240, 255, 0.1)", tickfont=dict(color="#9CA3AF")),
                yaxis=dict(title="Selected Band", tickfont=dict(color="#F3F4F6", size=12, family="Outfit")),
                paper_bgcolor="#07090E",
                plot_bgcolor="#07090E",
                height=340,
                margin=dict(l=50, r=60, t=30, b=30),
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No selected band scores available.")

    # 3. Interactive Full 50-Band Ranking Table
    st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)
    if snap.get("band_scores_available_for_all_bands", True) and band_table:
        with st.expander("INSPECT COMPLETE 50-BAND COGNITIVE PRIORITY RANKING TABLE", expanded=False):
            import pandas as pd
            st.dataframe(pd.DataFrame(band_table), use_container_width=True, height=280)
