"""Stage 10/11 & Step 9A: Professional Plotly figure builders for Cognitive RF Dashboard.

Pure presentation layer consuming verified operational evaluation results
(results/operational_evaluation_config_1.json) and live simulation runner outputs.
Zero fabricated data.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import plotly.graph_objects as go
import numpy as np

NUM_BANDS = 50
BAND_NAMES = [f"F{i:02d}" for i in range(1, NUM_BANDS + 1)]

# 0 = not scanned this step, 1 = scanned + miss, 2 = scanned + hit
_COLORSCALE = [
    (0.0, "#1e222b"), (0.34, "#1e222b"),
    (0.34, "#c0392b"), (0.67, "#c0392b"),
    (0.67, "#27ae60"), (1.0, "#27ae60"),
]


def get_band_freq_range(band_id: str, f_min: float = 500.0, f_max: float = 18000.0) -> tuple[float, float, float]:
    """Calculate min, max, and center MHz for a given band ID."""
    idx = int(band_id.replace("F", "")) - 1
    span = (f_max - f_min) / 50.0
    b_min = f_min + idx * span
    b_max = b_min + span
    b_center = (b_min + b_max) / 2.0
    return b_min, b_max, b_center


def spectrum_activity_map(
    time_series: List[Dict[str, Any]],
    current_t: int,
    window_steps: int = 100,
    strategy_view: str = "smart_scan",
) -> go.Figure:
    """Render interactive 2D Time-Frequency Activity Map (Bands F01-F50 vs Time)."""
    total_steps = len(time_series)
    start_t = max(0, current_t - window_steps + 1)
    end_t = min(total_steps, max(current_t + 1, start_t + window_steps))
    window_data = time_series[start_t:end_t]

    fig = go.Figure()

    # 1. Ground Truth Active Events
    gt_times, gt_bands, gt_hover = [], [], []
    for d in window_data:
        t_sec = d["simulated_time_s"]
        for b in d["env_active_bands"]:
            b_min, b_max, _ = get_band_freq_range(b)
            gt_times.append(t_sec)
            gt_bands.append(b)
            gt_hover.append(f"Band: {b} ({b_min:.0f}-{b_max:.0f} MHz)<br>Time: {t_sec:.2f}s (t={d['timestep']})<br>Status: Ground-Truth Active Pulse")

    if gt_times:
        fig.add_trace(go.Scatter(
            x=gt_times,
            y=gt_bands,
            mode="markers",
            name="Ground-Truth Active",
            marker=dict(symbol="square", size=9, color="rgba(110, 118, 129, 0.45)", line=dict(width=1, color="#2d2d30")),
            text=gt_hover,
            hoverinfo="text",
        ))

    # 2. Strategy Scans & Detections
    sel_key = "smart_scan_selected" if strategy_view == "smart_scan" else "open_loop_selected"
    true_key = "smart_scan_true_detections" if strategy_view == "smart_scan" else "open_loop_true_detections"
    fa_key = "smart_scan_false_alarms" if strategy_view == "smart_scan" else "open_loop_false_alarms"

    scan_times, scan_bands, scan_hover = [], [], []
    true_times, true_bands, true_hover = [], [], []
    fa_times, fa_bands, fa_hover = [], [], []

    for d in window_data:
        t_sec = d["simulated_time_s"]
        selected = d[sel_key]
        trues = set(d[true_key])
        fas = set(d[fa_key])

        for b in selected:
            b_min, b_max, _ = get_band_freq_range(b)
            if b in trues:
                true_times.append(t_sec)
                true_bands.append(b)
                true_hover.append(f"Band: {b} ({b_min:.0f}-{b_max:.0f} MHz)<br>Time: {t_sec:.2f}s (t={d['timestep']})<br><b>TRUE INTERCEPTION</b>")
            elif b in fas:
                fa_times.append(t_sec)
                fa_bands.append(b)
                fa_hover.append(f"Band: {b} ({b_min:.0f}-{b_max:.0f} MHz)<br>Time: {t_sec:.2f}s (t={d['timestep']})<br><b>FALSE ALARM</b> (Quiet Band)")
            else:
                scan_times.append(t_sec)
                scan_bands.append(b)
                scan_hover.append(f"Band: {b} ({b_min:.0f}-{b_max:.0f} MHz)<br>Time: {t_sec:.2f}s (t={d['timestep']})<br>Channel Monitoring (No Signal)")

    if scan_times:
        fig.add_trace(go.Scatter(
            x=scan_times,
            y=scan_bands,
            mode="markers",
            name="Scanned (Quiet)",
            marker=dict(symbol="circle-open", size=8, color="#00e5ff", line=dict(width=1.5)),
            text=scan_hover,
            hoverinfo="text",
        ))

    if fa_times:
        fig.add_trace(go.Scatter(
            x=fa_times,
            y=fa_bands,
            mode="markers",
            name="False Alarm",
            marker=dict(symbol="diamond", size=10, color="#ffab00", line=dict(width=1, color="#ffffff")),
            text=fa_hover,
            hoverinfo="text",
        ))

    if true_times:
        fig.add_trace(go.Scatter(
            x=true_times,
            y=true_bands,
            mode="markers",
            name="True Interception",
            marker=dict(symbol="star", size=12, color="#00c853", line=dict(width=1, color="#ffffff")),
            text=true_hover,
            hoverinfo="text",
        ))

    # 3. Current Selected Channel Highlight Marker
    curr_selected = window_data[current_t - start_t][sel_key] if (0 <= current_t - start_t < len(window_data)) else []
    curr_time_sec = current_t * 0.05
    curr_times = [curr_time_sec] * len(curr_selected)
    curr_hovers = [f"Band: {b}<br><b>CURRENTLY TUNED RECEIVER CHANNEL</b>" for b in curr_selected]

    if curr_times:
        fig.add_trace(go.Scatter(
            x=curr_times,
            y=curr_selected,
            mode="markers",
            name="Current Channels (K=5)",
            marker=dict(symbol="circle-dot", size=15, color="#00e5ff", line=dict(width=2, color="#ffffff")),
            text=curr_hovers,
            hoverinfo="text",
        ))

    # Current Timestep Vertical Cursor Line
    fig.add_vline(
        x=curr_time_sec,
        line_width=2,
        line_dash="dash",
        line_color="#d50000",
        annotation_text=f"t={curr_time_sec:.2f}s",
        annotation_position="top right",
    )

    fig.update_layout(
        title=dict(
            text=f"RF Spectrum Activity Map — {strategy_view.upper().replace('_', ' ')} (Window: {start_t*0.05:.1f}s - {end_t*0.05:.1f}s)",
            font=dict(color="#e6edee", size=15),
        ),
        xaxis=dict(
            title="Simulation Time (seconds)",
            range=[start_t * 0.05, max(end_t * 0.05, (start_t + 10) * 0.05)],
            gridcolor="#00e5ff1a",
            zerolinecolor="#2d2d30",
            tickfont=dict(color="#8b949e"),
        ),
        yaxis=dict(
            title="Frequency Band (F01=500MHz → F50=18000MHz)",
            categoryorder="array",
            categoryarray=list(reversed(BAND_NAMES)),
            gridcolor="#00e5ff1a",
            tickfont=dict(color="#8b949e", size=9),
            dtick=2,
        ),
        paper_bgcolor="#0a0a0b",
        plot_bgcolor="#161618",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#c9d1d9", size=11),
        ),
        height=520,
        margin=dict(l=55, r=25, t=50, b=45),
    )
    return fig


def cumulative_detections_chart(time_series: List[Dict[str, Any]], current_t: int) -> go.Figure:
    """Comparative cumulative true detections line chart over full mission."""
    times = [d["simulated_time_s"] for d in time_series]
    ol_cum = np.cumsum([len(d["open_loop_true_detections"]) for d in time_series])
    ss_cum = np.cumsum([len(d["smart_scan_true_detections"]) for d in time_series])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times,
        y=ol_cum,
        mode="lines",
        name="Conventional Open-Loop",
        line=dict(color="#8b949e", width=2, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=times,
        y=ss_cum,
        mode="lines",
        name="Intelligent Smart Scan",
        line=dict(color="#00c853", width=2.5),
    ))

    curr_time_sec = current_t * 0.05
    fig.add_vline(x=curr_time_sec, line_width=1.5, line_dash="dot", line_color="#d50000")

    fig.update_layout(
        title=dict(text="Cumulative True Interceptions (Full 30.0s Mission)", font=dict(color="#e6edee", size=13)),
        xaxis=dict(title="Time (s)", gridcolor="#00e5ff1a", tickfont=dict(color="#8b949e")),
        yaxis=dict(title="True Interceptions", gridcolor="#00e5ff1a", tickfont=dict(color="#8b949e")),
        paper_bgcolor="#0a0a0b",
        plot_bgcolor="#161618",
        legend=dict(font=dict(color="#c9d1d9", size=11), x=0.02, y=0.98),
        height=260,
        margin=dict(l=45, r=20, t=35, b=35),
    )
    return fig


def emitter_latency_bar_chart(emitter_records: List[Dict[str, Any]]) -> go.Figure:
    """Acquisition latency comparison per intercepted emitter."""
    intercepted = [r for r in emitter_records if r.get("first_intercept_time_s_ss") is not None or r.get("first_intercept_time_s_ol") is not None]
    ids = [r["emitter_id"] for r in intercepted]
    ol_lat = [(r["intercept_latency_s_ol"] if r["intercept_latency_s_ol"] is not None else 30.0) for r in intercepted]
    ss_lat = [(r["intercept_latency_s_ss"] if r["intercept_latency_s_ss"] is not None else 30.0) for r in intercepted]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Open Loop Latency (s)",
        x=ids,
        y=ol_lat,
        marker_color="#484f58",
    ))
    fig.add_trace(go.Bar(
        name="Smart Scan Latency (s)",
        x=ids,
        y=ss_lat,
        marker_color="#00c853",
    ))

    fig.update_layout(
        barmode="group",
        title=dict(text="First-Interception Acquisition Latency per Emitter (Lower is Better)", font=dict(color="#e6edee", size=13)),
        xaxis=dict(title="Emitter Class", tickfont=dict(color="#8b949e", size=10)),
        yaxis=dict(title="Latency from First Pulse (seconds)", gridcolor="#00e5ff1a", tickfont=dict(color="#8b949e")),
        paper_bgcolor="#0a0a0b",
        plot_bgcolor="#161618",
        legend=dict(font=dict(color="#c9d1d9", size=11), x=0.7, y=0.98),
        height=280,
        margin=dict(l=45, r=20, t=35, b=35),
    )
    return fig


# -----------------------------------------------------------------------------
# Legacy Stage 10 Compatibility Functions
# -----------------------------------------------------------------------------
def spectrum_waterfall(history: list, max_steps: int = 60) -> go.Figure:
    recent = history[-max_steps:] if history else []
    z = [[0] * len(recent) for _ in range(NUM_BANDS)]
    for col, record in enumerate(recent):
        for band_id, obs in record["observations"].items():
            row = int(band_id[1:]) - 1
            z[row][col] = 2 if obs.hit else 1

    fig = go.Figure(go.Heatmap(
        z=z, x=[r["t"] for r in recent], y=BAND_NAMES,
        colorscale=_COLORSCALE, zmin=0, zmax=2, showscale=False,
        hovertemplate="band=%{y}<br>t=%{x}<br>value=%{z}<extra></extra>",
    ))
    fig.update_layout(
        title="Spectrum Waterfall (green=hit, red=miss, dark=not scanned)",
        xaxis_title="Timestep", yaxis_title="Band",
        yaxis=dict(autorange="reversed"), height=650, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def belief_line_chart(belief_history: dict, band_ids: list) -> go.Figure:
    fig = go.Figure()
    for band_id in band_ids:
        points = belief_history.get(band_id, [])
        if not points:
            continue
        ts, probs = zip(*points)
        fig.add_trace(go.Scatter(x=list(ts), y=list(probs), mode="lines+markers", name=band_id))
    fig.update_layout(
        title="Bayesian Belief: P(active) over time",
        xaxis_title="Timestep", yaxis_title="P(active)", yaxis_range=[0, 1],
        height=350, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def q_value_bar_chart(q_values, selected_action: int) -> go.Figure:
    labels = ["EXPLORE", "EXPLOIT", "PREDICT", "BALANCED"]
    colors = ["#2980b9" if i != selected_action else "#f39c12" for i in range(4)]
    fig = go.Figure(go.Bar(x=labels, y=list(q_values), marker_color=colors))
    fig.update_layout(
        title="Q-Learning Arbitrator: Q-values for current state (selected in orange)",
        yaxis_title="Q-value", height=320, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def reward_history_chart(reward_history: list, window: int = 20) -> go.Figure:
    if not reward_history:
        return go.Figure()
    ts = [t for t, _ in reward_history]
    rewards = [r for _, r in reward_history]
    moving_avg = []
    for i in range(len(rewards)):
        lo = max(0, i - window + 1)
        moving_avg.append(sum(rewards[lo:i + 1]) / (i - lo + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=rewards, mode="lines", name="reward",
                              line=dict(color="#7f8c8d", width=1)))
    fig.add_trace(go.Scatter(x=ts, y=moving_avg, mode="lines", name=f"moving avg ({window})",
                              line=dict(color="#e74c3c", width=2)))
    fig.update_layout(title="Reward over time (this run)", xaxis_title="Timestep",
                       yaxis_title="Reward", height=300, margin=dict(l=40, r=20, t=40, b=40))
    return fig


def baseline_comparison_chart(aggregates: dict) -> go.Figure:
    schedulers = list(aggregates.keys())
    metrics = [("pd_mean", "Pd"), ("interception_rate_mean", "Interception Rate"),
               ("avg_reward_mean", "Avg Reward")]
    fig = go.Figure()
    for key, label in metrics:
        values = [aggregates[s].get(key, 0) for s in schedulers]
        values = [v if isinstance(v, (int, float)) else 0 for v in values]
        fig.add_trace(go.Bar(name=label, x=schedulers, y=values))
    fig.update_layout(barmode="group", title="Stage 8: Intelligent vs. Round Robin vs. Random-K",
                       height=380, margin=dict(l=40, r=20, t=40, b=40))
    return fig
