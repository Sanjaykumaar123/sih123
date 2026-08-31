"""Cognitive Decision Engine View: strategy status and grounded "why this band" reasoning.

Duck-typed over PlaybackController (replay - only Final Score is real per selected band)
and the live SimulationEngine/OperationalEngine (real Activity/Uncertainty/Temporal per
all 50 bands, and real per-state Q-values). Whichever fields aren't real for the current
engine are rendered as N/A - never invented.
"""

from typing import Any, Dict, List
import streamlit as st

from dashboard.help import glossary_caption
from dashboard import theme

STRAT_COLORS = {
    "EXPLORE": "#00e5ff",
    "EXPLOIT": "#00c853",
    "PREDICT": "#ffab00",
    "BALANCED": "#a371f7",
    "SEQUENTIAL_SWEEP": "#8b949e",
}


def _fmt(v: Any, spec: str = ".2f") -> str:
    if v is None:
        return "N/A"
    try:
        return format(v, spec)
    except (ValueError, TypeError):
        return str(v)


def render_decision_panel(engine: Any) -> None:
    """Render current strategy status and per-selected-band decision reasoning."""
    snap = engine.get_snapshot()
    band_table = snap.get("band_scores_table", [])
    sel_bands = snap.get("selected_bands", [])
    curr_strat = snap.get("current_strategy", snap.get("selected_strategy", "BALANCED"))
    cur_step = snap.get("timestep", snap.get("current_step", 0))
    sim_time = snap.get("simulated_time_s", snap.get("simulation_time_s", 0.0))
    s_col = STRAT_COLORS.get(curr_strat, "#00e5ff")

    # 1. Strategy status header
    st.markdown(
        f"""
        <div style='display:flex; justify-content:space-between; align-items:center; margin-top:0.2rem;'>
            <span class='channel-header' style='font-size:0.85rem;'>WHY DID SMART SCAN SELECT THESE BANDS?</span> {theme.provenance_badge('REAL')}
            <span style='font-family:monospace; font-size:0.75rem; color:#8b949e;'>
                STEP: <strong style='color:#00e5ff;'>{cur_step}</strong> | TIME: <strong style='color:#e6edee;'>{sim_time:.2f}s</strong>
            </span>
        </div>
        <div style='margin-top:0.25rem;'>
            <span style='font-size:0.68rem; color:#8b949e;'>CURRENT COGNITIVE STRATEGY</span><br/>
            <span style='font-size:1.1rem; font-weight:800; color:{s_col};'>{curr_strat}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    glossary_caption(curr_strat.capitalize() if curr_strat.capitalize() in ("Explore", "Exploit", "Predict", "Balanced") else "Balanced")

    # 2. Per-band decision cards - only real values, N/A otherwise
    sel_rows_map = {r["Band"]: r for r in band_table if r["Band"] in sel_bands}
    if sel_bands:
        top_cards_cols = st.columns(len(sel_bands))
        for idx, b_name in enumerate(sel_bands):
            b_data = sel_rows_map.get(b_name, {})
            p_act = _fmt(b_data.get("P(Active)"))
            unc = _fmt(b_data.get("Uncertainty"))
            t_score = _fmt(b_data.get("Temporal Score"))
            f_score = _fmt(b_data.get("Final Score"), ".3f")
            reason = b_data.get("Reason", "N/A")

            with top_cards_cols[idx]:
                st.markdown(
                    f"""
                    <div class='decision-card' style='border-top:3px solid {s_col};'>
                        <div style='display:flex; justify-content:space-between;'>
                            <span style='font-weight:800; font-size:0.92rem; color:#e6edee;'>CH0{idx+1} → {b_name}</span>
                            <span style='font-size:0.68rem; color:#8b949e; font-family:monospace;'>#{b_data.get('Rank', idx+1)}</span>
                        </div>
                        <div style='font-size:0.65rem; color:#8b949e; margin-top:0.1rem;'>{b_data.get('Frequency Range', '')}</div>
                        <div style='display:grid; grid-template-columns:1fr 1fr; gap:0.15rem; font-size:0.65rem; font-family:monospace; margin-top:0.25rem; color:#8b949e; background-color:#0a0a0b; padding:0.25rem; border-radius:3px;'>
                            <div>Activity: <strong style='color:#00c853;'>{p_act}</strong></div>
                            <div>Uncertainty: <strong style='color:#ffab00;'>{unc}</strong></div>
                            <div>Temporal: <strong style='color:#00e5ff;'>{t_score}</strong></div>
                            <div>Final Score: <strong style='color:#e6edee;'>{f_score}</strong></div>
                        </div>
                        <div style='font-size:0.64rem; color:#c9d1d9; margin-top:0.2rem; font-style:italic;'>
                            {reason}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.info("No bands selected this step.")

    if not snap.get("band_scores_available_for_all_bands", True):
        st.caption(f"ℹ️ {snap.get('band_scores_note', '')}")

    # 3. Meta-strategy Q-values - real when the engine exposes them, honest note otherwise
    q_vals = snap.get("meta_q_values")
    st.markdown(
        f"<div style='margin-top:0.5rem;'>{theme.provenance_badge('REAL' if (q_vals is not None and len(q_vals) == 4) else 'NA')}</div>",
        unsafe_allow_html=True,
    )
    if q_vals is not None and len(q_vals) == 4:
        st.markdown(
            f"""
            <div style='background-color:#161618; border:1px solid #2d2d30; border-radius:4px; padding:0.45rem 0.8rem; font-family:monospace; font-size:0.75rem;'>
                Q(EXPLORE): <strong style='color:#00e5ff;'>{q_vals[0]:.2f}</strong> |
                Q(EXPLOIT): <strong style='color:#00c853;'>{q_vals[1]:.2f}</strong> |
                Q(PREDICT): <strong style='color:#ffab00;'>{q_vals[2]:.2f}</strong> |
                Q(BALANCED): <strong style='color:#a371f7;'>{q_vals[3]:.2f}</strong>
                <div style='font-size:0.65rem; color:#8b949e; margin-top:0.2rem; font-family:sans-serif;'>
                    Real Q-values for the current arbitrator state (meta-strategy level, not per-band).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style='background-color:#161618; border:1px solid #2d2d30; border-radius:4px; padding:0.45rem 0.8rem; font-family:monospace; font-size:0.72rem; color:#8b949e;'>
                Q-VALUES: N/A — {snap.get('q_value_note', 'Per-band Q-values are not exposed by the current 4-action meta-strategy Q-table.')}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 4. Full band ranking table, only when the engine actually has it for all bands
    if snap.get("band_scores_available_for_all_bands", True) and band_table:
        with st.expander("📊 INSPECT COMPLETE 50-BAND COGNITIVE PRIORITY RANKING TABLE", expanded=False):
            import pandas as pd
            st.dataframe(pd.DataFrame(band_table), use_container_width=True, height=280)
    elif band_table:
        with st.expander(f"📊 SELECTED-BAND SCORES ({len(band_table)} bands this step)", expanded=False):
            import pandas as pd
            st.dataframe(pd.DataFrame(band_table), use_container_width=True, height=180)
