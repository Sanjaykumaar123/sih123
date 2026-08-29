"""Stage 10: Plotly figure builders. Pure presentation -- every function
here takes already-computed data (from SimulationRunner / Stage 3-9
public outputs / saved results/ artifacts) and returns a go.Figure. No
scoring formula, belief model, or ML algorithm is computed here."""

from __future__ import annotations

import plotly.graph_objects as go

NUM_BANDS = 50
_BAND_NAMES = [f"F{i:02d}" for i in range(1, NUM_BANDS + 1)]

# 0 = not scanned this step, 1 = scanned + miss, 2 = scanned + hit
_COLORSCALE = [
    (0.0, "#1e222b"), (0.34, "#1e222b"),
    (0.34, "#c0392b"), (0.67, "#c0392b"),
    (0.67, "#27ae60"), (1.0, "#27ae60"),
]


def spectrum_waterfall(history: list, max_steps: int = 60) -> go.Figure:
    """Bands (F01-F50) vs. recent time: green=hit, red=miss, dark=not
    scanned this step. Ground truth is never used here -- only what the
    receiver actually returned for the bands actually selected."""
    recent = history[-max_steps:]
    z = [[0] * len(recent) for _ in range(NUM_BANDS)]
    for col, record in enumerate(recent):
        for band_id, obs in record["observations"].items():
            row = int(band_id[1:]) - 1
            z[row][col] = 2 if obs.hit else 1

    fig = go.Figure(go.Heatmap(
        z=z, x=[r["t"] for r in recent], y=_BAND_NAMES,
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
    """P(active) over time for the given bands (Stage 3 BeliefEngine
    output only)."""
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
    """Stage 6 Q-values for the current state, one bar per strategy."""
    labels = ["EXPLORE", "EXPLOIT", "PREDICT", "BALANCED"]
    colors = ["#2980b9" if i != selected_action else "#f39c12" for i in range(4)]
    fig = go.Figure(go.Bar(x=labels, y=list(q_values), marker_color=colors))
    fig.update_layout(
        title="Q-Learning Arbitrator: Q-values for current state (selected in orange)",
        yaxis_title="Q-value", height=320, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def reward_history_chart(reward_history: list, window: int = 20) -> go.Figure:
    """Actual per-step reward + its trailing moving average from this
    live run -- not a fabricated training curve."""
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
    """Stage 8's saved per-scheduler aggregates -- Pd / interception rate /
    avg reward, side by side. Losses are shown, not hidden."""
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
