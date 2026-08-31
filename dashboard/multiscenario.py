"""Multi-scenario aggregation, metric calculation, and Plotly visualization engine.

Computes exact aggregate statistics (mean, median, min, max, totals) across validated
TSRD operational evaluation artifacts. Zero fabricated numbers.
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import plotly.graph_objects as go

from dashboard.scenario_loader import ScenarioDescriptor


def calculate_aggregate_statistics(validated_scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Compute mathematical summary statistics across all validated scenarios."""
    n_scenarios = len(validated_scenarios)
    if n_scenarios == 0:
        return {"insufficient_data": True, "num_scenarios": 0}

    # Extract scenario-by-scenario arrays
    scenarios_list = sorted(validated_scenarios.keys(), key=lambda x: int(x.replace("config_", "")) if x.replace("config_", "").isdigit() else 999)
    
    ss_dets, ol_dets = [], []
    ss_emitters, ol_emitters = [], []
    ss_pd, ol_pd = [], []
    ss_cov, ol_cov = [], []
    ss_lat, ol_lat = [], []
    ss_pfa, ol_pfa = [], []

    for s_id in scenarios_list:
        data = validated_scenarios[s_id]
        m_sum = data.get("metrics_summary", {})
        ss_m = m_sum.get("smart_scan", {})
        ol_m = m_sum.get("baseline", {})

        ss_dets.append(ss_m.get("true_detections", 0))
        ol_dets.append(ol_m.get("true_detections", 0))

        ss_emitters.append(ss_m.get("unique_emitters_intercepted", 0))
        ol_emitters.append(ol_m.get("unique_emitters_intercepted", 0))

        ss_pd.append(ss_m.get("sensor_pd", 0.0))
        ol_pd.append(ol_m.get("sensor_pd", 0.0))

        ss_cov.append(ss_m.get("scenario_coverage", 0.0))
        ol_cov.append(ol_m.get("scenario_coverage", 0.0))

        # Latency in seconds (step * 0.05s)
        ss_lat_val = ss_m.get("avg_intercept_time", float("nan"))
        ol_lat_val = ol_m.get("avg_intercept_time", float("nan"))
        ss_lat.append(ss_lat_val * 0.05 if not math.isnan(ss_lat_val) else float("nan"))
        ol_lat.append(ol_lat_val * 0.05 if not math.isnan(ol_lat_val) else float("nan"))

        ss_pfa.append(ss_m.get("pfa", 0.0))
        ol_pfa.append(ol_m.get("pfa", 0.0))

    # Consistency counts
    det_adv_count = sum(1 for s, o in zip(ss_dets, ol_dets) if s > o)
    emit_strict_adv_count = sum(1 for s, o in zip(ss_emitters, ol_emitters) if s > o)
    emit_equal_or_better_count = sum(1 for s, o in zip(ss_emitters, ol_emitters) if s >= o)
    lat_adv_count = sum(1 for s, o in zip(ss_lat, ol_lat) if not math.isnan(s) and not math.isnan(o) and s < o)

    # Valid latencies for stats
    valid_ss_lat = [v for v in ss_lat if not math.isnan(v)]
    valid_ol_lat = [v for v in ol_lat if not math.isnan(v)]

    total_ss_dets = sum(ss_dets)
    total_ol_dets = sum(ol_dets)
    overall_det_imp = ((total_ss_dets - total_ol_dets) / total_ol_dets * 100.0) if total_ol_dets > 0 else 0.0

    return {
        "insufficient_data": False,
        "num_scenarios": n_scenarios,
        "scenarios_evaluated": scenarios_list,
        "total_horizon_s": n_scenarios * 30.0,
        "total_ss_true_detections": total_ss_dets,
        "total_ol_true_detections": total_ol_dets,
        "overall_detection_improvement_pct": overall_det_imp,
        "consistency": {
            "detection_advantage": (det_adv_count, n_scenarios),
            "emitter_strict_advantage": (emit_strict_adv_count, n_scenarios),
            "emitter_equal_or_better": (emit_equal_or_better_count, n_scenarios),
            "emitter_advantage": (emit_equal_or_better_count, n_scenarios),
            "latency_advantage": (lat_adv_count, n_scenarios),
        },
        "metrics": {
            "true_detections": {
                "smart_scan": {"mean": float(np.mean(ss_dets)), "median": float(np.median(ss_dets)), "min": int(np.min(ss_dets)), "max": int(np.max(ss_dets))},
                "open_loop": {"mean": float(np.mean(ol_dets)), "median": float(np.median(ol_dets)), "min": int(np.min(ol_dets)), "max": int(np.max(ol_dets))},
            },
            "unique_emitters": {
                "smart_scan": {"mean": float(np.mean(ss_emitters)), "median": float(np.median(ss_emitters)), "min": int(np.min(ss_emitters)), "max": int(np.max(ss_emitters))},
                "open_loop": {"mean": float(np.mean(ol_emitters)), "median": float(np.median(ol_emitters)), "min": int(np.min(ol_emitters)), "max": int(np.max(ol_emitters))},
            },
            "sensor_pd": {
                "smart_scan": {"mean": float(np.mean(ss_pd)), "median": float(np.median(ss_pd)), "min": float(np.min(ss_pd)), "max": float(np.max(ss_pd))},
                "open_loop": {"mean": float(np.mean(ol_pd)), "median": float(np.median(ol_pd)), "min": float(np.min(ol_pd)), "max": float(np.max(ol_pd))},
            },
            "scenario_coverage": {
                "smart_scan": {"mean": float(np.mean(ss_cov)), "median": float(np.median(ss_cov)), "min": float(np.min(ss_cov)), "max": float(np.max(ss_cov))},
                "open_loop": {"mean": float(np.mean(ol_cov)), "median": float(np.median(ol_cov)), "min": float(np.min(ol_cov)), "max": float(np.max(ol_cov))},
            },
            "avg_intercept_time_s": {
                "smart_scan": {"mean": float(np.mean(valid_ss_lat)) if valid_ss_lat else float("nan"), "median": float(np.median(valid_ss_lat)) if valid_ss_lat else float("nan"), "min": float(np.min(valid_ss_lat)) if valid_ss_lat else float("nan"), "max": float(np.max(valid_ss_lat)) if valid_ss_lat else float("nan")},
                "open_loop": {"mean": float(np.mean(valid_ol_lat)) if valid_ol_lat else float("nan"), "median": float(np.median(valid_ol_lat)) if valid_ol_lat else float("nan"), "min": float(np.min(valid_ol_lat)) if valid_ol_lat else float("nan"), "max": float(np.max(valid_ol_lat)) if valid_ol_lat else float("nan")},
            },
            "pfa": {
                "smart_scan": {"mean": float(np.mean(ss_pfa)), "median": float(np.median(ss_pfa)), "min": float(np.min(ss_pfa)), "max": float(np.max(ss_pfa))},
                "open_loop": {"mean": float(np.mean(ol_pfa)), "median": float(np.median(ol_pfa)), "min": float(np.min(ol_pfa)), "max": float(np.max(ol_pfa))},
            },
        },
    }


def build_multiscenario_comparison_table(validated_scenarios: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build structured rows for the multi-scenario comparison table."""
    scenarios_list = sorted(validated_scenarios.keys(), key=lambda x: int(x.replace("config_", "")) if x.replace("config_", "").isdigit() else 999)
    rows = []

    for s_id in scenarios_list:
        data = validated_scenarios[s_id]
        m_sum = data.get("metrics_summary", {})
        ss_m = m_sum.get("smart_scan", {})
        ol_m = m_sum.get("baseline", {})

        ss_lat_val = ss_m.get("avg_intercept_time", float("nan"))
        ol_lat_val = ol_m.get("avg_intercept_time", float("nan"))
        ss_lat_str = f"{ss_lat_val * 0.05:.2f}s" if not math.isnan(ss_lat_val) else "n/a"
        ol_lat_str = f"{ol_lat_val * 0.05:.2f}s" if not math.isnan(ol_lat_val) else "n/a"

        rows.append({
            "Scenario": data.get("scenario", f"{s_id}.h5"),
            "SS True Detections": ss_m.get("true_detections", 0),
            "OL True Detections": ol_m.get("true_detections", 0),
            "SS Emitters": ss_m.get("unique_emitters_intercepted", 0),
            "OL Emitters": ol_m.get("unique_emitters_intercepted", 0),
            "SS Sensor Pd": f"{ss_m.get('sensor_pd', 0.0) * 100:.1f}%",
            "OL Sensor Pd": f"{ol_m.get('sensor_pd', 0.0) * 100:.1f}%",
            "SS Coverage": f"{ss_m.get('scenario_coverage', 0.0) * 100:.2f}%",
            "OL Coverage": f"{ol_m.get('scenario_coverage', 0.0) * 100:.2f}%",
            "SS Avg Intercept": ss_lat_str,
            "OL Avg Intercept": ol_lat_str,
            "SS Pfa": f"{ss_m.get('pfa', 0.0) * 100:.2f}%",
            "OL Pfa": f"{ol_m.get('pfa', 0.0) * 100:.2f}%",
        })
    return rows


def build_validation_status_table(all_discovered: Dict[str, ScenarioDescriptor]) -> List[Dict[str, Any]]:
    """Build structured rows for the Scenario Validation Status table."""
    scenarios_list = sorted(all_discovered.keys(), key=lambda x: int(x.replace("config_", "")) if x.replace("config_", "").isdigit() else 999)
    rows = []

    for s_id in scenarios_list:
        desc = all_discovered[s_id]
        rows.append({
            "SCENARIO": desc.scenario_name,
            "DATA (H5)": "✓ Present" if desc.data_present else "— Missing",
            "EVALUATION": "✓ Complete" if desc.evaluation_present else "— Pending",
            "TESTED": "✓ Verified" if desc.tested else "—",
            "STATUS": desc.status,
        })
    return rows


# -----------------------------------------------------------------------------
# Multi-Scenario Plotly Visualizations
# -----------------------------------------------------------------------------
def plot_multiscenario_detections(validated_scenarios: Dict[str, Dict[str, Any]]) -> go.Figure:
    """Grouped bar chart comparing True Detections across all validated scenarios."""
    scenarios_list = sorted(validated_scenarios.keys(), key=lambda x: int(x.replace("config_", "")) if x.replace("config_", "").isdigit() else 999)
    names = [validated_scenarios[s].get("scenario", f"{s}.h5") for s in scenarios_list]
    
    ss_vals = [validated_scenarios[s]["metrics_summary"]["smart_scan"]["true_detections"] for s in scenarios_list]
    ol_vals = [validated_scenarios[s]["metrics_summary"]["baseline"]["true_detections"] for s in scenarios_list]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Conventional Open-Loop Sweep",
        x=names,
        y=ol_vals,
        marker_color="#484f58",
        text=ol_vals,
        textposition="auto",
    ))
    fig.add_trace(go.Bar(
        name="Cognitive Smart Scan",
        x=names,
        y=ss_vals,
        marker_color="#00c853",
        text=ss_vals,
        textposition="auto",
    ))

    fig.update_layout(
        barmode="group",
        title=dict(text="True Radar Interceptions per Scenario (Higher is Better)", font=dict(color="#e6edee", size=13)),
        xaxis=dict(title="TSRD Scenario", tickfont=dict(color="#8b949e", size=10)),
        yaxis=dict(title="True Detections (Count)", gridcolor="#00e5ff1a", tickfont=dict(color="#8b949e")),
        paper_bgcolor="#0a0a0b",
        plot_bgcolor="#161618",
        legend=dict(font=dict(color="#c9d1d9", size=11), x=0.02, y=0.98),
        height=320,
        margin=dict(l=45, r=20, t=40, b=40),
    )
    return fig


def plot_multiscenario_emitters(validated_scenarios: Dict[str, Dict[str, Any]]) -> go.Figure:
    """Grouped bar chart comparing Unique Emitters Intercepted across scenarios."""
    scenarios_list = sorted(validated_scenarios.keys(), key=lambda x: int(x.replace("config_", "")) if x.replace("config_", "").isdigit() else 999)
    names = [validated_scenarios[s].get("scenario", f"{s}.h5") for s in scenarios_list]
    
    ss_vals = [validated_scenarios[s]["metrics_summary"]["smart_scan"]["unique_emitters_intercepted"] for s in scenarios_list]
    ol_vals = [validated_scenarios[s]["metrics_summary"]["baseline"]["unique_emitters_intercepted"] for s in scenarios_list]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Conventional Open-Loop",
        x=names,
        y=ol_vals,
        marker_color="#484f58",
        text=ol_vals,
        textposition="auto",
    ))
    fig.add_trace(go.Bar(
        name="Cognitive Smart Scan",
        x=names,
        y=ss_vals,
        marker_color="#1f6feb",
        text=ss_vals,
        textposition="auto",
    ))

    fig.update_layout(
        barmode="group",
        title=dict(text="Unique Emitter Classes Intercepted per Scenario (Discovery Breadth)", font=dict(color="#e6edee", size=13)),
        xaxis=dict(title="TSRD Scenario", tickfont=dict(color="#8b949e", size=10)),
        yaxis=dict(title="Unique Emitter Classes", gridcolor="#00e5ff1a", tickfont=dict(color="#8b949e")),
        paper_bgcolor="#0a0a0b",
        plot_bgcolor="#161618",
        legend=dict(font=dict(color="#c9d1d9", size=11), x=0.02, y=0.98),
        height=320,
        margin=dict(l=45, r=20, t=40, b=40),
    )
    return fig


def plot_multiscenario_latency(validated_scenarios: Dict[str, Dict[str, Any]]) -> go.Figure:
    """Grouped bar chart comparing Average First-Intercept Time in seconds across scenarios."""
    scenarios_list = sorted(validated_scenarios.keys(), key=lambda x: int(x.replace("config_", "")) if x.replace("config_", "").isdigit() else 999)
    names = [validated_scenarios[s].get("scenario", f"{s}.h5") for s in scenarios_list]
    
    ss_vals = []
    ol_vals = []
    for s in scenarios_list:
        ss_t = validated_scenarios[s]["metrics_summary"]["smart_scan"].get("avg_intercept_time", float("nan"))
        ol_t = validated_scenarios[s]["metrics_summary"]["baseline"].get("avg_intercept_time", float("nan"))
        ss_vals.append(round(ss_t * 0.05, 2) if not math.isnan(ss_t) else 0.0)
        ol_vals.append(round(ol_t * 0.05, 2) if not math.isnan(ol_t) else 0.0)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Conventional Open-Loop",
        x=names,
        y=ol_vals,
        marker_color="#484f58",
        text=[f"{v:.2f}s" for v in ol_vals],
        textposition="auto",
    ))
    fig.add_trace(go.Bar(
        name="Cognitive Smart Scan",
        x=names,
        y=ss_vals,
        marker_color="#ffab00",
        text=[f"{v:.2f}s" for v in ss_vals],
        textposition="auto",
    ))

    fig.update_layout(
        barmode="group",
        title=dict(text="Average First-Intercept Time (Seconds — Lower is Faster)", font=dict(color="#e6edee", size=13)),
        xaxis=dict(title="TSRD Scenario", tickfont=dict(color="#8b949e", size=10)),
        yaxis=dict(title="Time to First Intercept (Seconds)", gridcolor="#00e5ff1a", tickfont=dict(color="#8b949e")),
        paper_bgcolor="#0a0a0b",
        plot_bgcolor="#161618",
        legend=dict(font=dict(color="#c9d1d9", size=11), x=0.02, y=0.98),
        height=320,
        margin=dict(l=45, r=20, t=40, b=40),
    )
    return fig
