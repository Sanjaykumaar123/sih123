"""Stage 10/11 & Step 9A: Professional Plotly figure builders for Cognitive RF Dashboard.

Pure presentation layer consuming verified operational evaluation results
(results/operational_evaluation_config_1.json) and live simulation runner outputs.
Zero fabricated data.
"""

from __future__ import annotations
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots

NUM_BANDS = 50
BAND_NAMES = [f"F{i:02d}" for i in range(1, NUM_BANDS + 1)]


def get_band_freq_range(band_id: str, f_min: float = 500.0, f_max: float = 18000.0) -> tuple[float, float, float]:
    """Calculate min, max, and center MHz for a given band ID."""
    idx = int(band_id.replace("F", "")) - 1
    span = (f_max - f_min) / 50.0
    b_min = f_min + idx * span
    b_max = b_min + span
    b_center = (b_min + b_max) / 2.0
    return b_min, b_max, b_center


def spectrum_activity_map(time_series: List[Dict[str, Any]], current_t: int, strategy_view: str = "smart_scan", window_steps: int = 100) -> go.Figure:
    """Render full accumulated real mission history spectrum matrix from Step 0 to current_t."""
    total_steps = len(time_series)
    if not time_series:
        fig = go.Figure()
        fig.update_layout(title="No Spectrum Data Available", paper_bgcolor="#07090E", plot_bgcolor="#0F131D")
        return fig

    # Display full recorded mission history from Step 0 up to current_t
    end_t = min(total_steps, max(current_t + 1, 1))
    window_data = time_series[:end_t]

    x_steps = [d.get("timestep", d.get("step", idx)) for idx, d in enumerate(window_data)]
    sel_key = "smart_scan_selected" if strategy_view == "smart_scan" else "open_loop_selected"
    true_key = "smart_scan_true_detections" if strategy_view == "smart_scan" else "open_loop_true_detections"

    # Build 2D Matrix: 50 rows (Bands F01-F50) x N timesteps
    # Values: 0 = Dark (not scanned), 1 = Red (scanned miss), 2 = Green (scanned hit)
    z_matrix = []
    hover_matrix = []
    for b_name in BAND_NAMES:
        row = []
        h_row = []
        for d in window_data:
            selected = set(d.get(sel_key, d.get("selected_bands", [])))
            trues = set(d.get(true_key, d.get("hits", d.get("step_true_detections", []))))
            t_val = d.get("timestep", d.get("step", 0))

            if b_name in trues:
                row.append(2)
                h_row.append(f"<b>Band: {b_name}</b><br>Timestep: {t_val}<br>Status: <b style='color:#00FF9D;'>HIT (Pulse Intercepted)</b>")
            elif b_name in selected:
                row.append(1)
                h_row.append(f"<b>Band: {b_name}</b><br>Timestep: {t_val}<br>Status: <b style='color:#EF4444;'>MISS (Quiet Band)</b>")
            else:
                row.append(0)
                h_row.append(f"<b>Band: {b_name}</b><br>Timestep: {t_val}<br>Status: Not Scanned")
        z_matrix.append(row)
        hover_matrix.append(h_row)

    fig = go.Figure(data=go.Heatmap(
        x=x_steps,
        y=BAND_NAMES,
        z=z_matrix,
        text=hover_matrix,
        hoverinfo="text",
        zmin=0,
        zmax=2,
        colorscale=[
            [0.0, "#161922"],    # 0: Not Scanned (Dark)
            [0.5, "#EF4444"],    # 1: Scanned Miss (Red)
            [1.0, "#00FF9D"],    # 2: Scanned Hit (Green)
        ],
        showscale=False,
        xgap=1 if len(x_steps) <= 60 else 0,
        ygap=1,
    ))

    fig.update_layout(
        title=dict(
            text=f"REAL-TIME RF SPECTRUM WATERFALL (50 BANDS — STEPS 0 TO {x_steps[-1]})",
            font=dict(color="#F3F4F6", size=14, family="Outfit"),
        ),
        xaxis=dict(
            title="Mission Timestep",
            tickfont=dict(color="#9CA3AF", family="Inter"),
            title_font=dict(color="#9CA3AF", family="Inter"),
            dtick=10 if len(x_steps) <= 100 else 25,
            gridcolor="rgba(255,255,255,0.05)",
        ),
        yaxis=dict(
            title="Band (F01–F50)",
            categoryorder="array",
            categoryarray=list(reversed(BAND_NAMES)),
            tickfont=dict(color="#9CA3AF", size=9, family="Inter"),
            title_font=dict(color="#9CA3AF", family="Inter"),
            dtick=2,
            gridcolor="rgba(255,255,255,0.05)",
        ),
        paper_bgcolor="#07090E",
        plot_bgcolor="#07090E",
        height=520,
        margin=dict(l=55, r=25, t=45, b=45),
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
