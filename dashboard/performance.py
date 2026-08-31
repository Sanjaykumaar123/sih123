"""Live Performance and Cognitive Learning Monitor (ANALYTICS view).

Duck-typed over both runtimes: the live SimulationEngine (`.time_series` rows carry
"time_s"/"cumulative_reward"; `.decision_history` is a real list) and the replay
PlaybackController (which instead exposes `get_reward_timeseries()` /
`get_strategy_distribution()` built directly from the artifact - see
core/playback_controller.py). Both paths are real, computed data; nothing here is
invented.
"""

from typing import Any, Dict, List
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def _reward_series(engine: Any) -> List[Dict[str, Any]]:
    if hasattr(engine, "get_reward_timeseries"):
        return engine.get_reward_timeseries()
    return list(getattr(engine, "time_series", []))


def _strategy_counts(engine: Any) -> Dict[str, int]:
    if hasattr(engine, "get_strategy_distribution"):
        return engine.get_strategy_distribution()
    dec_hist = getattr(engine, "decision_history", [])
    counts: Dict[str, int] = {}
    for d in dec_hist:
        s = d.get("strategy")
        if s:
            counts[s] = counts.get(s, 0) + 1
    return counts


def render_performance_monitor(engine: Any) -> None:
    """Render live performance metrics, learning history, and strategy distributions."""
    snap = engine.get_snapshot()

    from dashboard import theme
    st.markdown(
        f"<div class='system-title'>COGNITIVE LEARNING & PERFORMANCE MONITOR</div> {theme.provenance_badge('REAL')}",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='system-subtitle'>REINFORCEMENT LEARNING REWARD, STRATEGY DISTRIBUTION, AND SENSING EFFICIENCY — THIS SESSION'S ACTIVE LIVE MISSION ONLY</div>", unsafe_allow_html=True)

    kpi_cols = st.columns(6)
    started = snap["total_scans"] > 0
    misses = snap["total_scans"] - snap["true_detections"] - snap["false_alarms"]
    ir = (snap["true_detections"] / snap["total_scans"] * 100.0) if started else 0.0
    if started:
        kpi_data = [
            ("True Detections", snap["true_detections"], "#00c853", "Confirmed Hits"),
            ("False Alarms", snap["false_alarms"], "#ffab00", f"{snap['pfa']*100:.2f}% Pfa"),
            ("Quiet Scans / Misses", max(0, misses), "#e6edee", "No Threshold Crossing"),
            ("Sensor Pd", f"{snap['sensor_pd']*100:.1f}%", "#e6edee", "On Tuned Channels"),
            ("Interception Rate", f"{ir:.2f}%", "#e6edee", "Channel Efficiency"),
            ("Cumulative Reward", f"{snap['cumulative_reward']:+.1f}", "#e6edee", f"Latest: {snap['latest_reward']:+.2f}"),
        ]
    else:
        kpi_data = [
            ("True Detections", "—", "#8b949e", "Mission not started"),
            ("False Alarms", "—", "#8b949e", "Mission not started"),
            ("Quiet Scans / Misses", "—", "#8b949e", "Mission not started"),
            ("Sensor Pd", "—", "#8b949e", "No scans yet"),
            ("Interception Rate", "—", "#8b949e", "No scans yet"),
            ("Cumulative Reward", "—", "#8b949e", "Mission not started"),
        ]
    for col, (lbl, val, color, sub) in zip(kpi_cols, kpi_data):
        with col:
            st.markdown(
                f"""<div class='metric-card'><div class='metric-lbl'>{lbl}</div>
                <div class='metric-val' style='color:{color};'>{val}</div>
                <div class='metric-imp imp-neutral'>{sub}</div></div>""",
                unsafe_allow_html=True,
            )

    c_r1, c_r2 = st.columns([6, 4])
    with c_r1:
        st.markdown("<div class='channel-header' style='font-size:0.8rem; margin-top:0.6rem;'>CUMULATIVE REWARD TIMELINE (LEARNING FEEDBACK)</div>", unsafe_allow_html=True)
        ts = _reward_series(engine)
        if ts:
            df_ts = pd.DataFrame(ts)
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatter(
                x=df_ts["time_s"], y=df_ts["cumulative_reward"], mode="lines",
                name="Cumulative Reward", line=dict(color="#00c853", width=2.5),
            ))
            fig_r.update_layout(
                height=260, margin=dict(l=30, r=10, t=10, b=30),
                paper_bgcolor="#0a0a0b", plot_bgcolor="#0a0a0b", font=dict(color="#c9d1d9"),
                xaxis=dict(title="Mission Time (s)", gridcolor="#00e5ff1a"),
                yaxis=dict(title="Reward", gridcolor="#00e5ff1a"),
            )
            st.plotly_chart(fig_r, use_container_width=True)
        else:
            st.info("No time-series reward data available yet.")

    with c_r2:
        st.markdown("<div class='channel-header' style='font-size:0.8rem; margin-top:0.6rem;'>META-STRATEGY DISTRIBUTION</div>", unsafe_allow_html=True)
        counts = _strategy_counts(engine)
        if counts:
            strat_series = pd.DataFrame({"Strategy": list(counts.keys()), "Decisions": list(counts.values())})
            fig_pie = px.pie(
                strat_series, names="Strategy", values="Decisions", hole=0.45,
                color_discrete_sequence=["#00e5ff", "#00c853", "#ffab00", "#a371f7"],
            )
            fig_pie.update_layout(
                height=260, margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="#0a0a0b", plot_bgcolor="#0a0a0b", font=dict(color="#c9d1d9"),
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No strategy distribution available yet.")
