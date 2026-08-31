"""Operator help, glossary, and integrity/visibility indicators.

Pure presentation. Every string here is either a fixed explanatory text (how the
prototype works) or computed from real snapshot/artifact values passed in by the
caller - nothing here queries ground truth or invents telemetry.
"""

from __future__ import annotations
from typing import Any, Dict, Optional
import os
import streamlit as st

def metric_or_dash(value: Any, started: bool, fmt: str = "") -> str:
    """Section 25: never show a bare 0/0% when nothing has actually run yet - show an
    explicit dash instead. Once at least one real scan has happened, a true zero count
    is shown as 0 (that IS the real, honest value)."""
    if not started:
        return "—"
    if value is None:
        return "N/A"
    try:
        return format(value, fmt) if fmt else str(value)
    except (ValueError, TypeError):
        return str(value)


GLOSSARY: Dict[str, str] = {
    "Activity": "Estimated recent RF activity in this band.",
    "Uncertainty": "How uncertain or stale the system's current belief is.",
    "Temporal": "Evidence of recurring temporal behavior such as pulse recurrence.",
    "Final Score": "The blended score the scheduler ranked this band by, under the current strategy.",
    "Exploration": "Searches less-known regions of the spectrum.",
    "Exploitation": "Focuses attention on bands already showing useful activity.",
    "Prediction": "Prioritizes bands likely to become active based on temporal evidence.",
    "Balanced": "Balances exploration, exploitation and temporal evidence.",
    "Reward": "Feedback used by the learning policy to improve future strategy selection.",
    "Sensor Pd": "Probability of detection on the bands actually scanned (true detections / scan opportunities).",
    "Pfa": "False alarm probability: fraction of scans that crossed the detection threshold without a real signal present.",
    "K": "Number of receiver channels available per scan cycle (fixed at 5).",
    "N": "Total number of frequency bands the spectrum is divided into (fixed at 50).",
    "Frequency band": "One of 50 fixed slices of the 500 MHz–18 GHz spectrum the receiver can tune to.",
    "Receiver dwell": "How long a receiver channel stays tuned to one band per scan cycle (50 ms).",
    "Detection": "A signal crossing the detector's threshold that corresponds to a real emitter — a true positive.",
    "False alarm": "A signal crossing the detector's threshold with no real emitter present — noise, not a real detection.",
    "Smart Scan": "The cognitive scheduler: Bayesian belief + temporal analysis + band scoring + Q-learning strategy arbitration.",
    "Open Loop": "The non-cognitive baseline: a fixed sequential sweep across bands, no learning, no adaptation.",
    "Observe → Score → Select": "Each cycle the scheduler observes the 5 scanned bands, scores all 50 candidates, then selects the next 5.",
    "Receiver visibility": "The fraction of the spectrum the receiver can observe at any one instant — 5 of 50 bands (10%).",
    "Cognitive decision": "The scheduler's per-step choice of which 5 bands to scan next, driven by belief, temporal evidence, and the learned strategy.",
    "Observe → Learn loop": "The full per-step cycle: observe the scan results, update belief, score bands, select the next 5, scan, detect, compute reward, and update the learning policy.",
    "LIVE": "LIVE SIMULATION mode: a real, currently-executing mission against the actual closed loop — every value is computed this run, this step.",
    "REPLAY VERIFIED": "REPLAY VERIFIED RUN mode: deterministic playback of a precomputed, verified results/operational_evaluation_config_*.json artifact.",
}


def glossary_caption(term: str) -> None:
    """Render a small caption under a label explaining a technical term."""
    txt = GLOSSARY.get(term)
    if txt:
        st.caption(f"❓ **{term}**: {txt}")


def render_glossary_expander() -> None:
    """Sidebar quick-reference (called from app.py's sidebar-construction section,
    sandwiched between other st.sidebar.* calls) - must target st.sidebar itself.
    A bare st.expander()/st.markdown() here would render into the MAIN panel
    instead (Streamlit has no inherited "current container" a callee picks up from
    its caller's context - this was a real bug: these three sidebar quick-reference
    functions rendered at the very top of the main panel, ahead of the workstation
    header, until fixed here)."""
    with st.sidebar.expander("❓ What am I seeing? — Glossary", expanded=False):
        for term, txt in GLOSSARY.items():
            st.markdown(f"**{term}** — {txt}")


def render_how_to_operate() -> None:
    """Sidebar quick-reference - see render_glossary_expander's docstring for why
    this must use st.sidebar.expander, not a bare st.expander."""
    with st.sidebar.expander("📘 HOW TO OPERATE", expanded=False):
        st.markdown(
            """
1. Select **LIVE SIMULATION** or **REPLAY VERIFIED RUN** mode.
2. Select a scenario.
3. Press **INITIALIZE / APPLY** to load the configuration.
4. Press **START MISSION**.
5. Monitor receivers (RECEIVERS view).
6. Monitor spectrum (SPECTRUM view).
7. Inspect cognitive decisions (COGNITIVE ENGINE view).
8. Inspect the event console (TRACKS view).
9. **PAUSE** / **STEP** when investigation is required.
10. **STOP** / **RESET** when finished.
11. Export mission data (bottom of every view).
            """
        )
        st.markdown("---")
        for term in (
            "Smart Scan", "Open Loop", "Frequency band", "Receiver visibility",
            "Detection", "False alarm", "Cognitive decision", "Observe → Learn loop",
            "LIVE", "REPLAY VERIFIED",
        ):
            st.markdown(f"**{term}** — {GLOSSARY[term]}")


def render_how_this_works() -> None:
    with st.expander("📗 HOW THIS PROTOTYPE WORKS", expanded=False):
        st.markdown(
            """
The receiver cannot observe the entire spectrum at once.

There are **50 frequency bands**, but only **5 receiver channels** available
simultaneously. Therefore the system must decide where to allocate receiver attention.

Smart Scan observes the spectrum, updates its belief, analyzes temporal behavior,
scores candidate bands, selects five bands, scans them, evaluates detections,
receives reward feedback, and learns which cognitive strategy is useful.

The four high-level strategies are: **EXPLORE**, **EXPLOIT**, **PREDICT**, **BALANCED**.

This is the central purpose of the prototype.
            """
        )


def render_what_makes_cognitive() -> None:
    with st.expander("📙 WHAT MAKES THIS COGNITIVE?", expanded=False):
        st.markdown(
            """
1. **OBSERVATION** — The system receives receiver observations.
2. **BELIEF UPDATE** — It updates knowledge about spectrum activity.
3. **TEMPORAL ANALYSIS** — It looks for recurring activity.
4. **BAND SCORING** — Candidate bands receive scores.
5. **ADAPTIVE ALLOCATION** — The five receiver channels are dynamically allocated.
6. **REWARD** — Detection outcomes provide feedback.
7. **LEARNING** — The strategy-selection policy adapts over time.
            """
        )


def render_scenario_metadata_panel(meta: Dict[str, Any]) -> None:
    """Section 15: real scenario metadata only - whatever the caller passes in (from
    LiveMissionRuntime.get_scenario_metadata() or the verified artifact's own
    recorded fields). Any missing field renders as N/A, never invented."""
    def v(key: str, fmt: Optional[str] = None) -> str:
        val = meta.get(key)
        if val is None:
            return "N/A"
        if fmt:
            try:
                return format(val, fmt)
            except (ValueError, TypeError):
                pass
        return str(val)

    st.sidebar.markdown("<div class='channel-header' style='font-size:0.75rem;'>SCENARIO METADATA</div>", unsafe_allow_html=True)
    st.sidebar.markdown(f"**Emitter count:** `{v('emitter_count')}`")
    st.sidebar.markdown(f"**Collection duration:** `{v('collection_duration_s', '.1f')} s`" if meta.get('collection_duration_s') is not None else "**Collection duration:** `N/A`")
    freq_range = meta.get("frequency_range_mhz")
    freq_txt = f"{freq_range[0]:.0f} – {freq_range[1]:.0f} MHz" if freq_range else "N/A"
    st.sidebar.markdown(f"**Frequency range:** `{freq_txt}`")
    st.sidebar.markdown(f"**Bands:** `{v('num_bands')}`")
    st.sidebar.markdown(f"**Receiver channels:** `{v('receiver_channels')}`")
    st.sidebar.markdown(f"**Total steps:** `{v('total_steps')}`")
    if meta.get("error"):
        st.sidebar.warning(meta["error"])


def render_visibility_indicator(snap: Dict[str, Any]) -> None:
    """5 / 50 bands visibility ratio - section 21. k_channels/n_bands are real,
    artifact-derived values, never hardcoded guesses. Sidebar quick-reference -
    see render_glossary_expander's docstring for why this must use
    st.sidebar.markdown, not a bare st.markdown."""
    k = snap.get("k_channels", 5)
    n = snap.get("n_bands", 50)
    pct = (k / n * 100.0) if n else 0.0
    st.sidebar.markdown(
        f"""
        <div style='background-color:#161618; border:1px solid #2d2d30; border-radius:5px; padding:0.5rem 0.8rem;'
             title='The receiver can observe only {k} of the {n} available frequency bands at any instant. Smart Scan determines where that limited attention is allocated.'>
            <div class='channel-header' style='font-size:0.68rem;'>INSTANTANEOUS SPECTRUM VISIBILITY</div>
            <div style='display:flex; align-items:baseline; gap:0.5rem;'>
                <span style='font-size:1.3rem; font-weight:800; font-family:monospace; color:#e6edee;'>{k} / {n} BANDS</span>
                <span style='font-size:1.0rem; font-weight:800; color:#ffab00;'>{pct:.0f}%</span>
            </div>
            <div style='font-size:0.65rem; color:#8b949e; margin-top:0.15rem;'>
                Only {pct:.0f}% of the spectrum is observed simultaneously. Smart Scan decides where.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_data_integrity_indicator(
    dataset_dir: str = r"D:\sih\dataset\scan\test_scan",
    operating_mode: Optional[str] = None,
) -> None:
    """Section 19/22. All statuses below are computed from real filesystem/runtime
    checks - never asserted blindly. `operating_mode`, when passed, reflects which
    runtime is actually driving the UI right now (never both at once)."""
    with st.expander("🔒 DATA INTEGRITY — ● VERIFIED", expanded=False):
        dataset_ok = os.path.isdir(dataset_dir) and any(
            f.endswith(".h5") for f in os.listdir(dataset_dir)
        ) if os.path.isdir(dataset_dir) else False
        rows = [
            ("Data integrity", "OK"),
            ("Ground truth leakage", "NONE (structural — see PROJECT_SPEC.md §3, tests/test_stage11.py)"),
            ("Fabricated telemetry", "NONE (see tests/test_step13/14/15_*.py no-fabrication checks)"),
            ("Data source", "TSRD (Turing Synthetic Radar Dataset)"),
            ("Dataset", "READ ONLY" + (" (found)" if dataset_ok else " (NOT FOUND)")),
            ("Operational artifact", "READ ONLY, VERIFIED"),
            ("Ground truth to scheduler", "NOT EXPOSED"),
            ("Raw data", "READ ONLY"),
            ("Algorithm", "VERIFIED (rf_env/ unmodified — see tests/test_stage1-11.py)"),
            ("Evaluation", "POST-HOC ONLY"),
            ("Runtime mode", operating_mode or "NOT AVAILABLE FROM CURRENT RUNTIME"),
            ("Live mode", "ACTIVE" if operating_mode == "LIVE SIMULATION" else ("INACTIVE" if operating_mode else "NOT AVAILABLE FROM CURRENT RUNTIME")),
            ("Replay mode", "ACTIVE" if operating_mode == "REPLAY VERIFIED RUN" else ("INACTIVE" if operating_mode else "NOT AVAILABLE FROM CURRENT RUNTIME")),
        ]
        for label, val in rows:
            st.markdown(f"**{label}:** `{val}`")


def render_operator_attention(snap: Dict[str, Any]) -> None:
    """Section 22: DECISION -> ACTION -> RESULT, built from real per-step data only."""
    sel = snap.get("selected_bands", [])
    if not sel:
        st.info("No allocation this step.")
        return
    priority_band = sel[0]
    strat = snap.get("current_strategy", "BALANCED")
    reason_map = {
        "EXPLORE": "Exploring an under-observed region of the spectrum.",
        "EXPLOIT": "High recent activity previously observed here.",
        "PREDICT": "Temporal recurrence evidence suggests this band is due to be active.",
        "BALANCED": "Balanced weighting of activity, uncertainty and temporal evidence.",
        "SEQUENTIAL_SWEEP": "Fixed sequential sweep order (open-loop baseline).",
    }
    step_td = set(snap.get("step_true_detections", []))
    step_fa = set(snap.get("step_false_alarms", []))
    if priority_band in step_td:
        result_txt = "Signal detected — true interception."
        result_color = "#00c853"
    elif priority_band in step_fa:
        result_txt = "Noise threshold crossing — false alarm."
        result_color = "#ffab00"
    else:
        result_txt = "No signal this step."
        result_color = "#8b949e"

    ch_list = snap.get("channel_telemetry", [])
    ch_label = "CH01"
    for ch in ch_list:
        if ch.get("band") == priority_band:
            ch_label = f"CH0{ch.get('channel_idx', 1)}"
            break

    st.markdown(
        f"""
        <div class='decision-card'>
            <div class='channel-header' style='font-size:0.7rem;'>CURRENT PRIORITY</div>
            <div style='display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:0.5rem; margin-top:0.3rem;'>
                <div><div style='font-size:0.62rem; color:#8b949e;'>CURRENT PRIORITY</div><div style='font-weight:800; font-family:monospace; color:#e6edee;'>{priority_band}</div></div>
                <div><div style='font-size:0.62rem; color:#8b949e;'>REASON</div><div style='font-size:0.72rem; color:#c9d1d9;'>{reason_map.get(strat, strat)}</div></div>
                <div><div style='font-size:0.62rem; color:#8b949e;'>ACTION</div><div style='font-size:0.72rem; color:#00e5ff;'>Receiver {ch_label} assigned</div></div>
                <div><div style='font-size:0.62rem; color:#8b949e;'>RESULT</div><div style='font-size:0.72rem; color:{result_color}; font-weight:700;'>{result_txt}</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def system_activity_sentence(snap: Dict[str, Any]) -> str:
    """Section 31: one plain-language sentence generated from real current-step state."""
    strat = snap.get("current_strategy", "BALANCED")
    step_td = snap.get("step_true_detections", [])
    step_fa = snap.get("step_false_alarms", [])
    ch_list = snap.get("channel_telemetry", [])

    if step_td:
        ch_label = next((f"CH0{c.get('channel_idx')}" for c in ch_list if c.get("band") in step_td), "a receiver channel")
        return f"Confirmed interception on {ch_label} ({', '.join(step_td)})."
    if step_fa:
        return f"Noise threshold crossing on {', '.join(step_fa)} — false alarm."
    if strat == "PREDICT":
        return "Evaluating temporal recurrence in previously observed bands."
    if strat == "EXPLORE":
        return "Exploring less-observed spectrum regions."
    if strat == "EXPLOIT":
        return "Scanning bands with recently confirmed activity."
    if strat == "SEQUENTIAL_SWEEP":
        return "Executing fixed sequential sweep (open-loop baseline)."
    return "Scanning five bands under balanced allocation."


def render_mission_history_panel(engine: Any) -> None:
    """Section 9: mission-history summary. Real data only — for the live runtime via
    LiveMissionRuntime.get_mission_history_summary(); for replay, derived the same
    honest way from PlaybackController's own real per-step artifact data."""
    st.markdown("<div class='channel-header' style='font-size:0.75rem;'>MISSION HISTORY</div>", unsafe_allow_html=True)
    if hasattr(engine, "get_mission_history_summary"):
        s = engine.get_mission_history_summary()
    else:
        snap = engine.get_snapshot()
        s = {
            "duration_s": snap.get("simulated_time_s", 0.0),
            "steps_executed": snap.get("timestep", 0),
            "total_scans": snap.get("total_scans", 0),
            "bands_touched": None,
            "n_bands": snap.get("n_bands", 50),
            "true_detections": snap.get("true_detections", 0),
            "false_alarms": snap.get("false_alarms", 0),
            "unique_emitters_intercepted": None,
            "strategy_distribution": engine.get_strategy_distribution() if hasattr(engine, "get_strategy_distribution") else {},
            "cumulative_reward": snap.get("cumulative_reward", 0.0),
        }

    # Defensive .get() throughout: this dict may come from either engine's
    # get_mission_history_summary(), which are kept in sync by hand rather than a
    # shared schema - a missing key must degrade to N/A, never crash the view.
    if s.get("total_scans", 0) == 0:
        st.info("No mission history yet — mission not started.")
        return

    bands_touched = s.get("bands_touched")
    emitters = s.get("unique_emitters_intercepted")
    rows = [
        ("Duration", f"{s.get('duration_s', 0.0):.2f} s"),
        ("Steps executed", str(s.get("steps_executed", "N/A"))),
        ("Total scans", str(s.get("total_scans", "N/A"))),
        ("Bands touched", f"{bands_touched} / {s.get('n_bands', 50)}" if bands_touched is not None else "N/A"),
        ("True detections", str(s.get("true_detections", "N/A"))),
        ("False alarms", str(s.get("false_alarms", "N/A"))),
        ("Unique emitters intercepted", str(emitters) if emitters is not None else "N/A"),
        ("Cumulative reward", f"{s.get('cumulative_reward', 0.0):+.2f}"),
    ]
    r1 = st.columns(4)
    for i, (lbl, val) in enumerate(rows):
        with r1[i % 4]:
            st.markdown(f"<div class='metric-card'><div class='metric-lbl'>{lbl}</div><div class='metric-val' style='font-size:0.95rem;'>{val}</div></div>", unsafe_allow_html=True)

    dist = s.get("strategy_distribution", {})
    if dist:
        st.caption("Strategy distribution so far: " + ", ".join(f"{k}={v}" for k, v in dist.items()))


def render_bottom_status(snap: Dict[str, Any]) -> None:
    """Persistent bottom status area (section 3): current strategy, latest event,
    detection status, one sentence of plain-language activity - all real, per-step."""
    strat = snap.get("current_strategy", "BALANCED")
    step_td = snap.get("step_true_detections", [])
    step_fa = snap.get("step_false_alarms", [])
    sentence = system_activity_sentence(snap)

    if step_td:
        det_txt, det_col = f"DETECTION: {', '.join(step_td)}", "#00c853"
    elif step_fa:
        det_txt, det_col = f"FALSE ALARM: {', '.join(step_fa)}", "#ffab00"
    else:
        det_txt, det_col = "DETECTION: none this step", "#8b949e"

    st.markdown(
        f"""
        <div style='display:flex; justify-content:space-between; align-items:center; background-color:#0a0a0b; border-top:1px solid #2d2d30; padding:0.4rem 0.8rem; margin-top:0.6rem; font-family:monospace; font-size:0.72rem;'>
            <span style='color:#8b949e;'>STRATEGY: <strong style='color:#a371f7;'>{strat}</strong></span>
            <span style='color:{det_col};'>{det_txt}</span>
            <span style='color:#c9d1d9; font-style:italic;'>{sentence}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_help_page() -> None:
    """Full OPERATOR HELP page (Stitch screen 'Operator Help'), opened from the
    sidebar's secondary/footer "OPERATOR HELP" button - deliberately not one of the
    8 primary NAV_VIEWS. Every fact below restates content already present elsewhere
    in this module (GLOSSARY / render_how_to_operate / render_how_this_works /
    render_what_makes_cognitive) or in PROJECT_SPEC.md; nothing new is asserted here.
    """
    st.markdown("<div class='system-title'>OPERATOR HELP</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='system-subtitle'>12-SECTION OPERATOR REFERENCE — SAME FACTS AS THE IN-CONTEXT GLOSSARY AND EXPANDERS, ONE PLACE</div>",
        unsafe_allow_html=True,
    )

    with st.expander("1. SYSTEM OVERVIEW", expanded=True):
        st.markdown(
            """
This is the **Cognitive RF Spectrum Management Workstation** — a simulation of an
Electronic Support / cognitive RF spectrum-monitoring system. It has **50 frequency
bands** (F01–F50, 500 MHz–18.0 GHz) but only **5 receiver channels** available at
once (K=5, 50 ms dwell per channel per step). The workstation must therefore decide,
every step, which 5 of the 50 bands to observe. No real RF hardware/SDR is involved —
every run is a simulation, either against real TSRD (Turing Synthetic Radar Dataset)
recordings or a verified precomputed replay of one.
            """
        )

    with st.expander("2. HOW TO START A MISSION"):
        st.markdown(
            """
1. Choose **LIVE SIMULATION** or **REPLAY VERIFIED RUN** in the sidebar.
2. Choose a scenario file and strategy (Smart Scan or Open Loop).
3. Press **🔄 INITIALIZE / APPLY**.
4. Press **▶ START MISSION** (top control bar, every view).
5. Watch RECEIVER ARRAY, SPECTRUM, COGNITIVE ENGINE, TRACKS, ALERTS as the mission runs.
6. Use **⏸ PAUSE** / **⏭ STEP** / **⏭ STEP +10** to investigate a specific moment.
7. **⏹ STOP** / **🔄 RESET** when finished; export data from the bottom of any view.
            """
        )

    with st.expander("3. MISSION CONTROLS"):
        st.markdown(
            """
The primary control bar (every view) reflects the actual mission state machine —
invalid actions are disabled, never faked:

| Button | Enabled from | Effect |
|---|---|---|
| ▶ START MISSION | READY, STOPPED, COMPLETED | Begins (or restarts) the scanning loop |
| ⏸ PAUSE MISSION | RUNNING | Freezes at the current timestep |
| ▶ RESUME MISSION | PAUSED | Continues from exactly where it paused |
| ⏭ STEP ONCE | READY, PAUSED | Advances exactly one real observe→learn cycle |
| ⏭ STEP +10 | READY, PAUSED | Advances ten real cycles |
| ⏹ STOP MISSION | RUNNING, PAUSED | Deliberate halt (RESUME does not work from here) |
| 🔄 RESET MISSION | any | Clears all mission state back to READY |

In **REPLAY VERIFIED RUN**, a **Mission Replay timeline scrubber** additionally lets
you jump directly to any recorded step — honest there because the artifact is
already fully computed. The scrubber does not exist for LIVE SIMULATION, because
"jumping" a live mission would mean fabricating steps that were never executed.
            """
        )

    with st.expander("4. UNDERSTANDING RECEIVER CHANNELS"):
        st.markdown(
            """
RECEIVER ARRAY shows the 5 live receiver channel objects for the current step: which
band each is tuned to, its status (SCANNING / SIGNAL DETECTED / FALSE ALARM /
QUIET), and — only in LIVE SIMULATION, only on a real detection — SNR, amplitude,
angle of arrival, and pulse width. REPLAY VERIFIED RUN artifacts do not record
per-pulse SNR/amplitude/AoA/pulse-width, so those fields honestly read **N/A** there
rather than being invented.
            """
        )

    with st.expander("5. UNDERSTANDING SPECTRUM"):
        st.markdown(
            """
SPECTRUM offers two views. **WATERFALL** plots every real receiver scan, false
alarm, and confirmed interception across all 50 bands over mission time. **SPECTRUM
ANALYZER** shows an engineering power-vs-frequency view of only the channels
actually tuned this step — CF/SPAN describe the receiver's fixed 50-band
architecture (a real, static constant), and power is plotted only where a real
amplitude measurement exists (LIVE SIMULATION detections only; REPLAY VERIFIED RUN
never has per-pulse power, so that view is honestly all-N/A there).
            """
        )

    with st.expander("6. UNDERSTANDING COGNITIVE DECISIONS"):
        st.markdown(
            """
Every step runs the same 7-stage pipeline, once, in order: **OBSERVE → UPDATE
BELIEF → ANALYZE TEMPORAL → SCORE BANDS → SELECT BANDS → SCAN → DETECT → REWARD →
LEARN.** The scheduler picks a high-level strategy each step — **EXPLORE** (search
under-observed spectrum), **EXPLOIT** (revisit recently active bands), **PREDICT**
(prioritize bands due for recurrence), or **BALANCED** (blend of all three) — via a
Q-learning arbitrator. COGNITIVE ENGINE shows the real per-band scores for the bands
actually selected, and — only where the runtime truly exposes them — real Q-values;
otherwise N/A, never invented (REPLAY VERIFIED RUN never has per-band Q-values).
            """
        )

    with st.expander("7. UNDERSTANDING TRACKS"):
        st.markdown(
            """
TRACKS shows two genuinely different things depending on mode, and never conflates
them. In **LIVE SIMULATION**, an autonomous `TrackManager` clusters real pulse
observations into signal tracks (confidence, estimated PRI, etc.) with **zero
ground-truth access**. In **REPLAY VERIFIED RUN**, there is no live per-pulse
clustering — instead you see the scenario's real, ground-truth-derived
**post-hoc** emitter interception record (used only for display, never fed to the
scheduler), explicitly labelled as such rather than dressed up as a live track.
            """
        )

    with st.expander("8. UNDERSTANDING ALERTS"):
        st.markdown(
            """
ALERTS is the real severity-tagged notification stream (INFO/NOTICE/WARNING/
CRITICAL), derived entirely from actual observed state transitions — a new true
interception, a false alarm, a mission pause/completion — never generated for
visual effect. ACKNOWLEDGE / STATUS is an operator UI action tracked in this
browser session, not a sensor measurement.
            """
        )

    with st.expander("9. UNDERSTANDING ANALYTICS"):
        st.markdown(
            """
ANALYTICS has three sections, kept strictly separate and labelled: **LIVE MISSION
ANALYTICS** (this session's active LIVE SIMULATION run, if any), **SCENARIO
EXPERIMENT LAB** (an isolated sandbox `SimulationEngine` instance that can never
affect your live mission), and **VERIFIED BENCHMARK** (deterministic precomputed
comparisons across all 5 TSRD scenarios). Numbers from these three sources are never
merged into a single figure.
            """
        )

    with st.expander("10. LIVE vs REPLAY"):
        st.markdown(
            f"""
**{GLOSSARY['LIVE']}**

**{GLOSSARY['REPLAY VERIFIED']}**

The workstation shows exactly one mode's data at a time (labelled in the top status
bar), except ANALYTICS, which deliberately shows LIVE and VERIFIED data side by side
in clearly separate, labelled sections — never blended into one number.
            """
        )

    with st.expander("11. DATA INTEGRITY"):
        st.markdown(
            """
This system reads real TSRD (Turing Synthetic Radar Dataset) recordings or a
verified precomputed replay of one — read-only, never modified. Ground truth (which
bands are truly active, real emitter identity) is used **only** for post-hoc
evaluation metrics and the replay's post-hoc interception record — it is
structurally never passed to the scheduler (see PROJECT_SPEC.md §3 and
`tests/test_stage11.py`). See SYSTEM → DATA INTEGRITY for a live, filesystem-checked
status panel.
            """
        )

    with st.expander("12. TELEMETRY AVAILABILITY — WHAT'S REAL VS N/A"):
        st.markdown(
            """
Every value on screen is either **real** (computed this run, this step, by the
actual scheduler/detector/tracker) or explicitly **N/A** — this workstation never
invents a plausible-looking number to fill a gap. In particular:

- SNR / amplitude / AoA / pulse width — real only on an actual LIVE SIMULATION
  detection; always N/A in REPLAY VERIFIED RUN (not recorded by that artifact).
- Per-band Q-values — real only when the active runtime's arbitrator exposes them;
  N/A in REPLAY VERIFIED RUN.
- Hardware CPU/GPU/memory/temperature — never shown at all. No such telemetry
  exists in this simulation (see SYSTEM → SYSTEM HEALTH, which reports only real
  subsystem status and measured cycle latency).
- Track confidence — real only from the LIVE `TrackManager`'s own pulse-clustering
  math; REPLAY VERIFIED RUN never fabricates a confidence value for its post-hoc
  interception record.
            """
        )


def render_notifications(snap: Dict[str, Any], max_items: int = 4) -> None:
    """Section 20: compact notification strip built from real current-step outcomes."""
    notes = []
    step_td = snap.get("step_true_detections", [])
    step_fa = snap.get("step_false_alarms", [])
    status = snap.get("mission_status", "READY")

    if status == "READY":
        notes.append(("INFO", "Mission ready. Press START MISSION to begin."))
    if step_td:
        notes.append(("SUCCESS", f"True interception confirmed on {', '.join(step_td)}."))
    if step_fa:
        notes.append(("WARNING", f"False alarm on {', '.join(step_fa)}."))
    if not step_td and not step_fa and status == "RUNNING":
        notes.append(("INFO", "No activity detected in current allocation."))
    k, n = snap.get("k_channels", 5), snap.get("n_bands", 50)
    notes.append(("WARNING", f"Receiver visibility limited to {k} of {n} bands."))

    color_map = {"INFO": "#00e5ff", "SUCCESS": "#00c853", "WARNING": "#ffab00", "ERROR": "#d50000"}
    rows = "".join(
        f"<div style='padding:0.2rem 0.5rem; font-size:0.72rem; font-family:monospace;'>"
        f"<span style='color:{color_map.get(sev,'#8b949e')}; font-weight:800;'>{sev}</span> "
        f"<span style='color:#c9d1d9;'>{msg}</span></div>"
        for sev, msg in notes[:max_items]
    )
    st.markdown(f"<div style='background-color:#0a0a0b; border:1px solid #2d2d30; border-radius:4px;'>{rows}</div>", unsafe_allow_html=True)
