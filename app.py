"""Stage 10: Streamlit dashboard for the Smart Scan Strategy prototype.

Launch: streamlit run app.py

This file is ONLY the visualization/integration layer. It runs the REAL
Stage 1-8 closed loop via dashboard/simulation_runner.py
(SimulationRunner, which itself only calls IntelligentSchedulerAdapter,
EvaluationMetrics, and env.notify_scan_results -- all Stage 6/7/8 code,
unmodified). No second scheduler, no new ML, no fabricated numbers.
"""

import json
import os
import pickle

import streamlit as st

from dashboard.simulation_runner import SimulationRunner
from dashboard import visualizations as viz

st.set_page_config(page_title="Smart Scan Strategy for EW", layout="wide")

STRATEGY_LABELS = ["EXPLORE", "EXPLOIT", "PREDICT", "BALANCED"]
LEVEL_LABELS = ["LOW", "MEDIUM", "HIGH"]


# ---------------------------------------------------------------- caching
@st.cache_data
def load_json_artifact(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_resource
def load_predictor(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def fmt(v):
    if isinstance(v, (int, float)):
        return f"{v:.3f}"
    return "n/a" if v == "insufficient_data" else str(v)


# ---------------------------------------------------------------- header
st.title("SMART SCAN STRATEGY FOR ELECTRONIC WARFARE")
st.caption("Adaptive ML-Based Spectrum Scanning under Limited Receiver Channels")

with st.expander("Judge demo script (talking points)"):
    st.markdown(
        "1. **Limited receiver** -- 50 bands, only 5 observed at a time.\n"
        "2. **Exploration** -- uncertain/stale bands are prioritized first.\n"
        "3. **Bayesian learning** -- every hit/miss updates P(active).\n"
        "4. **Temporal learning** -- repeated activity yields a period estimate.\n"
        "5. **Multiple strategies** -- explore/exploit/predict/balanced scores.\n"
        "6. **Reinforcement learning** -- Q-learning picks the trusted strategy.\n"
        "7. **Adaptive emitter** -- repeated detection triggers evasive behaviour.\n"
        "8. **Adaptation** -- scanning priorities shift as belief/temporal state changes.\n"
        "9. **Re-acquisition** -- the system searches again and finds it.\n"
        "10. **Predictive ML** -- Random Forest estimates interception time/rate.\n"
        "11. **Baseline comparison** -- Intelligent vs. Round Robin vs. Random-K, honestly."
    )

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Simulation Controls")
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=100000, value=42, step=1)
steps_per_action = st.sidebar.slider("Steps per action", 1, 100, 10)
auto_run = st.sidebar.checkbox("Auto-run (Start/Pause)", value=False)

if "runner" not in st.session_state:
    st.session_state.runner = SimulationRunner(seed=int(seed))

c1, c2, c3 = st.sidebar.columns(3)
if c1.button("Step"):
    st.session_state.runner.step()
if c2.button(f"Run {steps_per_action}"):
    st.session_state.runner.run(steps_per_action)
if c3.button("Reset"):
    st.session_state.runner.reset(int(seed))

runner = st.session_state.runner

# A fresh/just-reset runner has taken no steps yet (t == -1) -- Stage 5's
# BandScoringEngine has nothing to score until then. Take one real step so
# the dashboard has valid state to show immediately, rather than erroring.
if runner.t < 0:
    runner.step()

if auto_run:
    runner.step()
    st.rerun()

# ---------------------------------------------------------------- KPI cards
belief_state = runner.belief_state()
scores = runner.scores()
metrics = runner.metrics_summary()
last_record = runner.history[-1] if runner.history else None
current_strategy = last_record["strategy"].upper() if last_record else "-"

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total Bands", runner.config["num_bands"])
k2.metric("Receiver Channels (K)", runner.k)
k3.metric("Current Timestep", runner.t)
k4.metric("Current Strategy", current_strategy)
k5.metric("Detection Rate (Pd)", fmt(metrics["pd"]))
k6.metric("Avg Reward", fmt(metrics["avg_reward"]))

tabs = st.tabs([
    "Spectrum", "Band Priority", "Belief (Stage 3)", "Temporal (Stage 4)",
    "Q-Learning (Stage 6)", "Adaptive Evasion (Stage 7)", "Predictive ML (Stage 9)",
    "Baseline Comparison (Stage 8)", "Why This Band?", "Architecture", "Live Metrics",
])

# ---------------------------------------------------------------- Spectrum
with tabs[0]:
    if runner.history:
        st.plotly_chart(viz.spectrum_waterfall(runner.history), use_container_width=True)
    else:
        st.info("Step the simulation to populate the waterfall.")
    with st.expander("EVALUATION / DEBUG VIEW (ground truth -- not scheduler input)"):
        st.json(runner.last_ground_truth_debug or {})

# ---------------------------------------------------------------- Band Priority
with tabs[1]:
    st.write("Stage 5 strategy scores for every band (no new formula -- BandScoringEngine output).")
    selected_now = set(last_record["selected_bands"]) if last_record else set()
    rows = sorted(scores, key=lambda s: s.balanced_score, reverse=True)[:15]
    st.dataframe([
        {"Band": s.band_id, "Selected now": s.band_id in selected_now,
         "Exploration": round(s.exploration_score, 3), "Exploitation": round(s.exploitation_score, 3),
         "Prediction": round(s.prediction_score, 3), "Balanced": round(s.balanced_score, 3)}
        for s in rows
    ], use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- Belief
with tabs[2]:
    st.write("Stage 3: every observed hit or miss updates the probability that a band is active.")
    top_belief = sorted(belief_state, key=lambda b: b.activity_probability, reverse=True)[:15]
    st.dataframe([
        {"Band": b.band_id, "P(active)": round(b.activity_probability, 3),
         "Uncertainty": round(b.uncertainty, 4),
         "Staleness": b.staleness if b.staleness != float("inf") else "never observed",
         "Observations": b.hit_count + b.miss_count, "Hits": b.hit_count}
        for b in top_belief
    ], use_container_width=True, hide_index=True)
    chart_bands = last_record["selected_bands"] if last_record else []
    if chart_bands:
        st.plotly_chart(viz.belief_line_chart(runner.belief_history, chart_bands),
                         use_container_width=True)

# ---------------------------------------------------------------- Temporal
with tabs[3]:
    st.write("Stage 4: temporal pattern classification -- no fabricated predictions.")
    temporal_state = runner.temporal_state()
    periodic_first = sorted(temporal_state, key=lambda t: t.periodicity_score, reverse=True)[:15]
    st.dataframe([
        {"Band": t.band_id, "Behaviour": t.behaviour_type,
         "Estimated Period": round(t.estimated_period, 1) if t.estimated_period else "-",
         "Next Predicted Active": (round(t.predicted_next_active_time, 1)
                                    if t.predicted_next_active_time is not None else "-"),
         "Periodicity Score": round(t.periodicity_score, 3),
         "Confidence": round(t.prediction_confidence, 3)}
        for t in periodic_first
    ], use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- Q-Learning
with tabs[4]:
    state, q_values = runner.current_q_state_and_values()
    st.write("Stage 6 Q-learning arbitrator's CURRENT observable state:")
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Performance level", LEVEL_LABELS[state[0]])
    sc2.metric("Uncertainty level", LEVEL_LABELS[state[1]])
    sc3.metric("Detection level", LEVEL_LABELS[state[2]])
    selected_action = STRATEGY_LABELS.index(current_strategy) if current_strategy in STRATEGY_LABELS else 0
    st.plotly_chart(viz.q_value_bar_chart(q_values, selected_action), use_container_width=True)
    st.plotly_chart(viz.reward_history_chart(runner.reward_history), use_container_width=True)

# ---------------------------------------------------------------- Adaptive Evasion
with tabs[5]:
    e4 = runner.e4
    if e4 is None:
        st.info("No adaptive/evasive emitter configured.")
    else:
        if e4.is_evasive:
            st.warning("⚠ ADAPTIVE EVASION DETECTED -- emitter is currently in its evasive burst.")
        st.markdown(
            "Previous behaviour\n\n&darr;\n\nEmitter changes pattern\n\n&darr;\n\n"
            "Detection performance changes\n\n&darr;\n\nRL strategy adapts\n\n&darr;\n\n"
            "Scanning priorities change\n\n&darr;\n\nEmitter is re-acquired"
        )
        summary = runner.evasion_summary()
        ec1, ec2, ec3 = st.columns(3)
        ec1.metric("Evasion events (this run)", summary.get("evasion_events", 0))
        ec2.metric("Reacquired count", summary.get("reacquired_count", 0))
        ec3.metric("Avg reacquisition time", fmt(summary.get("reacquisition_time", "insufficient_data")))
        st.caption("These numbers are measured from THIS live run, via the existing Stage 7 "
                   "feedback interface -- no evasion event is artificially triggered.")

# ---------------------------------------------------------------- Predictive ML
with tabs[6]:
    stage9_results = load_json_artifact("results/stage9_results.json")
    predictor = load_predictor("results/stage9_predictor.pkl")
    if stage9_results is None or predictor is None:
        st.info("Run `python demo_stage9.py` once to generate results/stage9_results.json "
                "and results/stage9_predictor.pkl.")
    else:
        test_m = stage9_results["test_metrics"]
        st.write("Held-out TEST performance (from the saved Stage 9 experiment, not retrained here):")
        r2_time = test_m["intercept_time"]["random_forest"]["r2"]
        r2_rate = test_m["interception_rate"]["random_forest"]["r2"] if isinstance(test_m["interception_rate"], dict) else "n/a"
        pc1, pc2 = st.columns(2)
        pc1.metric("Intercept Time R2 (Random Forest)", fmt(r2_time))
        pc2.metric("Interception Rate R2 (Random Forest)", fmt(r2_rate))

        if last_record:
            st.write("Live prediction for the currently scanned bands:")
            live_rows = []
            for band_id, features in runner.last_features.items():
                pred = predictor.predict(features)
                live_rows.append({
                    "Band": band_id, "Predicted Intercept Time": pred["predicted_intercept_time"],
                    "Predicted Interception Rate": pred["predicted_interception_rate"],
                    "Quality": pred["prediction_quality"],
                })
            st.dataframe(live_rows, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- Baseline
with tabs[7]:
    stage8 = load_json_artifact("results/stage8_results.json")
    if stage8 is None:
        st.info("Run `python demo_stage8.py` once to generate results/stage8_results.json.")
    else:
        aggregates = stage8["aggregates"]
        st.plotly_chart(viz.baseline_comparison_chart(aggregates), use_container_width=True)
        st.dataframe([
            {"Scheduler": name, "Pd": fmt(agg.get("pd_mean")), "Interception Rate": fmt(agg.get("interception_rate_mean")),
             "Avg Reward": fmt(agg.get("avg_reward_mean")), "Redundant Scan Rate": fmt(agg.get("redundant_scan_rate_mean")),
             "Avg Intercept Time": fmt(agg.get("avg_intercept_time_mean")),
             "Reacquisition Time": fmt(agg.get("reacquisition_time_mean"))}
            for name, agg in aggregates.items()
        ], use_container_width=True, hide_index=True)
        st.caption("Losses are shown, not hidden -- see CLAUDE.md/Stage 8 report for the honest breakdown.")

# ---------------------------------------------------------------- Why This Band
with tabs[8]:
    if not last_record:
        st.info("Step the simulation first.")
    else:
        band_id = st.selectbox("Band", last_record["selected_bands"])
        b = next(x for x in belief_state if x.band_id == band_id)
        t = next(x for x in runner.temporal_state() if x.band_id == band_id)
        s = next(x for x in scores if x.band_id == band_id)
        st.subheader(f"WHY DID THE SYSTEM CHOOSE {band_id}?")
        st.markdown(f"""
- **Bayesian belief:** P(active) = `{b.activity_probability:.3f}`
- **Uncertainty:** `{b.uncertainty:.4f}`
- **Staleness:** `{b.staleness if b.staleness != float('inf') else 'never observed'}`
- **Temporal prediction:** behaviour=`{t.behaviour_type}`, confidence=`{t.prediction_confidence:.3f}`
- **Exploration score:** `{s.exploration_score:.3f}`
- **Exploitation score:** `{s.exploitation_score:.3f}`
- **Prediction score:** `{s.prediction_score:.3f}`
- **Selected strategy:** `{current_strategy}`
- **Final (selected-strategy) score:** `{getattr(s, current_strategy.lower() + '_score'):.3f}`
""")

# ---------------------------------------------------------------- Architecture
with tabs[9]:
    st.markdown("""
```
RF Environment (hidden ground truth)
        |
Limited Receiver (K of 50 bands)
        |
Detection Physics (SNR -> P_d, false alarms)
        |
Bayesian Belief (Stage 3)
        |
Temporal Prediction (Stage 4)
        |
Strategy Scoring: explore/exploit/predict/balanced (Stage 5)
        |
Q-Learning Arbitrator (Stage 6) --------> selects ONE strategy
        |
Band Selection (Top-K under that strategy)
        |
Receiver observes -> hit/miss -> reward
        |
   (feeds back into Belief + Temporal + Q-table + Adaptive Emitter)
        ^------------------------------------------------------------|
```
""")

# ---------------------------------------------------------------- Live Metrics
with tabs[10]:
    tracker = runner.metrics_tracker  # public attributes read directly; no
                                       # change made to Stage 8's EvaluationMetrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Scans", metrics["total_scans"])
    m2.metric("Hits", metrics["total_hits"])
    m3.metric("Misses", metrics["total_scans"] - metrics["total_hits"])
    m4.metric("False Alarms", tracker.false_detections)
    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Pd", fmt(metrics["pd"]))
    m6.metric("Interception Rate", fmt(metrics["interception_rate"]))
    m7.metric("Avg Reward", fmt(metrics["avg_reward"]))
    m8.metric("Redundant Scan Rate", fmt(metrics["redundant_scan_rate"]))
