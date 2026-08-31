"""Cognitive scheduler decision center, 50-band ranking table, belief map, and temporal predictions."""

from typing import Any, Dict, List
import plotly.graph_objects as go
import streamlit as st
from simulation.engine import SimulationEngine


def render_scheduler_view(engine: SimulationEngine) -> None:
    """Render the cognitive scheduler decision center, 50-band table, belief map, and temporal state."""
    snap = engine.get_snapshot()
    band_table = snap.get("band_scores_table", [])
    sel_bands = snap.get("selected_bands", [])
    curr_strat = snap.get("current_strategy", "BALANCED")

    # 1. Closed-Loop Process Trace
    st.markdown(
        """
        <div class='trace-container' style='padding:0.5rem; margin-top:0.4rem;'>
            <div class='trace-step trace-active'>1. OBSERVE (5 Channels)</div>
            <div class='trace-step trace-active'>2. UPDATE BELIEF (P(act))</div>
            <div class='trace-step trace-active'>3. ANALYZE TEMPORAL (PRI)</div>
            <div class='trace-step trace-active'>4. SCORE BANDS (50 Bands)</div>
            <div class='trace-step trace-active'>5. SELECT TOP-K (K=5)</div>
            <div class='trace-step trace-active'>6. SCAN & DETECT</div>
            <div class='trace-step trace-active'>7. REWARD & LEARN</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Cognitive Decision Center: Selected Bands & Detailed Breakdown
    st.markdown("<div class='channel-header' style='font-size:0.85rem; margin-top:0.6rem;'>COGNITIVE DECISION CENTER — SELECTED RECEIVER CHANNELS</div>", unsafe_allow_html=True)
    
    strat_colors = {
        "EXPLORE": "#00e5ff",
        "EXPLOIT": "#00c853",
        "PREDICT": "#ffab00",
        "BALANCED": "#a371f7",
        "SEQUENTIAL_SWEEP": "#8b949e",
    }
    s_col = strat_colors.get(curr_strat, "#00e5ff")

    top_cards_cols = st.columns(len(sel_bands) if sel_bands else 1)
    sel_rows_map = {r["Band"]: r for r in band_table if r["Band"] in sel_bands}

    for idx, b_name in enumerate(sel_bands):
        b_data = sel_rows_map.get(b_name, {})
        rank = b_data.get("Rank", idx + 1)
        p_act = b_data.get("P(Active)", 0.0)
        unc = b_data.get("Uncertainty", 1.0)
        t_score = b_data.get("Temporal Score", 0.0)
        f_score = b_data.get("Final Score", 0.0)
        reason = b_data.get("Reason", "Top-5 Rank Leader")

        with top_cards_cols[idx]:
            st.markdown(
                f"""
                <div class='decision-card' style='border-top:3px solid {s_col};'>
                    <div style='display:flex; justify-content:space-between;'>
                        <span style='font-weight:800; font-size:0.95rem; color:#e6edee;'>CH0{idx+1} → {b_name}</span>
                        <span style='font-size:0.7rem; color:#8b949e; font-family:monospace;'>Rank #{rank}</span>
                    </div>
                    <div style='font-size:0.68rem; color:#8b949e; margin-top:0.15rem;'>{b_data.get('Frequency Range', '')}</div>
                    <div style='display:grid; grid-template-columns:1fr 1fr; gap:0.2rem; font-size:0.68rem; font-family:monospace; margin-top:0.35rem; color:#8b949e; background-color:#0a0a0b; padding:0.35rem; border-radius:3px;'>
                        <div>P(Act): <strong style='color:#00c853;'>{p_act:.2f}</strong></div>
                        <div>Uncert: <strong style='color:#ffab00;'>{unc:.2f}</strong></div>
                        <div>Temp:   <strong style='color:#00e5ff;'>{t_score:.2f}</strong></div>
                        <div>Final:  <strong style='color:#e6edee;'>{f_score:.3f}</strong></div>
                    </div>
                    <div style='font-size:0.65rem; color:#c9d1d9; margin-top:0.3rem; font-style:italic;'>
                        {reason}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # 3. Interactive "Why was this band selected?" Inspector
    with st.expander("🔎 Detailed Cognitive Decision Inspector ('Why Was This Band Selected?')", expanded=False):
        insp_col1, insp_col2 = st.columns([3, 7])
        with insp_col1:
            inspect_band = st.selectbox(
                "SELECT BAND TO INSPECT",
                options=[r["Band"] for r in band_table],
                index=0,
                key="band_inspector_select",
            )
        target_row = next((r for r in band_table if r["Band"] == inspect_band), None)
        with insp_col2:
            if target_row:
                st.markdown(
                    f"""
                    <div class='callout-box' style='margin-top:0; border-left-color:{s_col};'>
                        <div style='font-weight:800; font-size:0.9rem; color:#00e5ff; margin-bottom:0.2rem;'>
                            INSPECTION REPORT: {inspect_band} ({target_row['Frequency Range']})
                        </div>
                        <div style='font-size:0.8rem; color:#c9d1d9; line-height:1.4;'>
                            • <strong>Current Status</strong>: <span style='color:#00c853; font-weight:700;'>{target_row['Selected']} (Rank #{target_row['Rank']})</span><br>
                            • <strong>Selection Rationale</strong>: {target_row['Reason']}<br>
                            • <strong>Active Meta-Strategy</strong>: <span style='color:{s_col}; font-weight:700;'>{curr_strat}</span><br>
                            • <strong>Score Breakdown</strong>: Activity Score = {target_row['Exploitation Score']:.3f} | Epistemic Uncertainty = {target_row['Exploration Score']:.3f} | Temporal Prediction = {target_row['Prediction Score']:.3f} → <strong>Final Score = {target_row['Final Score']:.3f}</strong><br>
                            • <strong>Last Intercepted</strong>: {target_row['Last Observed']} | <strong>Ground-Truth Leakage</strong>: ZERO (Decisions derived exclusively from observable history).
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # 4. Two Column Layout: Live 50-Band Priority Table & Bayesian Belief Map
    tbl_col, map_col = st.columns([6, 4])

    with tbl_col:
        st.markdown("<div class='channel-header' style='font-size:0.8rem;'>50-BAND LIVE COGNITIVE PRIORITY RANKING TABLE</div>", unsafe_allow_html=True)
        if band_table:
            st.dataframe(
                band_table,
                height=320,
                use_container_width=True,
                column_config={
                    "Rank": st.column_config.NumberColumn("Rank", width="small"),
                    "Band": st.column_config.TextColumn("Band", width="small"),
                    "Frequency Range": st.column_config.TextColumn("Frequency", width="medium"),
                    "P(Active)": st.column_config.NumberColumn("P(Act)", format="%.2f"),
                    "Uncertainty": st.column_config.NumberColumn("Unc", format="%.2f"),
                    "Temporal Score": st.column_config.NumberColumn("Temp", format="%.2f"),
                    "Final Score": st.column_config.NumberColumn("Final", format="%.3f"),
                    "Selected": st.column_config.TextColumn("Selected", width="small"),
                    "Reason": st.column_config.TextColumn("Reason", width="large"),
                },
            )

    with map_col:
        st.markdown("<div class='channel-header' style='font-size:0.8rem;'>REAL-TIME BAYESIAN BELIEF MAP (P(ACTIVE) PER BAND)</div>", unsafe_allow_html=True)
        
        # Build Real-Time Belief Bar Chart
        if band_table:
            bands_order = [f"F{i:02d}" for i in range(1, 51)]
            row_dict = {r["Band"]: r for r in band_table}
            p_acts = [row_dict.get(b, {}).get("P(Active)", 0.0) for b in bands_order]
            is_selected = [b in sel_bands for b in bands_order]
            bar_colors = ["#00c853" if sel else "#1f6feb" for sel in is_selected]

            fig_belief = go.Figure(go.Bar(
                x=bands_order,
                y=p_acts,
                marker=dict(color=bar_colors),
                hoverinfo="text",
                hovertext=[f"Band: {b}<br>P(Active): {p:.2f}<br>Status: {'SELECTED' if s else 'UNSELECTED'}" for b, p, s in zip(bands_order, p_acts, is_selected)],
            ))
            fig_belief.update_layout(
                height=320,
                margin=dict(l=30, r=10, t=10, b=40),
                paper_bgcolor="#0a0a0b",
                plot_bgcolor="#0a0a0b",
                xaxis=dict(gridcolor="#2d2d30", color="#8b949e", dtick=5),
                yaxis=dict(title="P(Active)", range=[0.0, 1.05], gridcolor="#2d2d30", color="#8b949e"),
            )
            st.plotly_chart(fig_belief, use_container_width=True)

    # 5. Temporal Prediction Intelligence Panel
    if hasattr(engine.scheduler, "temporal"):
        with st.expander("⏱ Temporal PRI Analysis & Recurrence Predictions", expanded=False):
            t_preds = engine.scheduler.temporal.get_state()
            active_preds = [p for p in t_preds if p.number_of_hits >= 2]
            if active_preds:
                t_table = [
                    {
                        "Band": p.band_id,
                        "Periodicity Score": f"{p.periodicity_score:.2f}",
                        "Estimated PRI (Timesteps)": f"{p.estimated_period:.1f}" if p.estimated_period else "Estimating...",
                        "Last Observed Hit": f"t={p.last_hit_timestep}" if p.last_hit_timestep is not None else "None",
                        "Predicted Next Active": f"t={p.predicted_next_active_time:.1f}" if p.predicted_next_active_time is not None else "N/A",
                        "Confidence": f"{p.prediction_confidence*100:.0f}%",
                        "Hits Recorded": p.number_of_hits,
                    }
                    for p in active_preds
                ]
                st.dataframe(t_table, use_container_width=True)
            else:
                st.info("Insufficient observation history for temporal PRI convergence (< 2 pulse hits on individual bands). Continue scanning to build PRI history.")
