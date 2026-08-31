"""COGNITIVE RF SPECTRUM MANAGEMENT WORKSTATION.

Production-grade operational Electronic Support / Cognitive RF Spectrum Monitoring
Workstation with two independent, clearly-labeled operating modes:

  LIVE SIMULATION    - core.live_mission.LiveMissionRuntime wraps the real, verified
                        closed loop (simulation.engine.SimulationEngine ->
                        rf_env.evaluation.IntelligentSchedulerAdapter ->
                        rf_env.receiver.Receiver -> rf_env.detection.DetectionModel).
                        Every value shown is computed live, this run, this step.

  REPLAY VERIFIED RUN - core.playback_controller.PlaybackController deterministically
                        replays a verified results/operational_evaluation_config_*.json
                        artifact.

The two are never mixed: each view shows exactly one mode's data at a time, labeled,
except ANALYTICS, which deliberately shows both side by side in clearly separate
sections. Neither mode fabricates telemetry - any field the active runtime doesn't
have real data for is shown as N/A rather than invented.

Launch: streamlit run app.py
"""

import json
import os
import time

import pandas as pd
import streamlit as st

from core.playback_controller import PlaybackController
from core.live_mission import LiveMissionRuntime, LiveMissionStatus
from core.state import EngineStatus
from data.scenario_loader import discover_scenarios, get_validated_scenarios
from dashboard import (
    live_operations,
    receiver_panel,
    decision_panel,
    spectrum,
    tracks,
    performance,
    event_console,
    system,
    help as ophelp,
    cognitive_pipeline,
    alerts,
    theme,
    jury_explainer,
)

DEFAULT_SCAN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "dataset", "scan", "test_scan"))
SCAN_DIR = r"D:\sih\dataset\scan\test_scan" if os.path.exists(r"D:\sih\dataset\scan\test_scan") else DEFAULT_SCAN_DIR
SCEN_OPTIONS = ["config_1.h5", "config_2.h5", "config_3.h5", "config_4.h5", "config_5.h5"]

# -----------------------------------------------------------------------------
# 1. Page Configuration & Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Cognitive RF Spectrum Management Workstation",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Stitch design-system CSS (colors/typography/spacing/radius/borders) - see
# dashboard/theme.py for the token source. Presentation only; nothing below reads or
# computes mission data.
st.markdown(theme.get_custom_css(), unsafe_allow_html=True)

t_render_start = time.perf_counter()


# -----------------------------------------------------------------------------
# 2. Session-state runtimes: one PlaybackController (cheap, eager) and at most one
#    LiveMissionRuntime (loads a real HDF5 scenario, so constructed lazily - only
#    once the operator actually engages LIVE SIMULATION mode).
# -----------------------------------------------------------------------------
if "playback_controller" not in st.session_state:
    st.session_state.playback_controller = PlaybackController(
        scenario_id="config_1.h5", speed=1.0, strategy_type="smart_scan",
    )
if "live_mission" not in st.session_state:
    st.session_state.live_mission = None
if "operating_mode" not in st.session_state:
    st.session_state.operating_mode = "LIVE SIMULATION"

controller: PlaybackController = st.session_state.playback_controller


def _init_live_mission(scenario_file: str, strategy_slug: str, k: int = 5, seed: int = 42) -> None:
    st.session_state.live_mission = LiveMissionRuntime(
        scenario_path=os.path.join(SCAN_DIR, scenario_file),
        strategy_type=strategy_slug, k_channels=k, n_bands=50, seed=seed,
    )


# -----------------------------------------------------------------------------
# 3. Scenario Discovery & Operator Sidebar
# -----------------------------------------------------------------------------
all_discovered = discover_scenarios()
validated_scenarios = get_validated_scenarios()

st.sidebar.markdown("<div class='system-title' style='font-size:0.95rem;'>OPERATOR CONSOLE</div>", unsafe_allow_html=True)
st.sidebar.markdown("<span class='tech-badge badge-success'>● SYSTEM HEALTHY</span>", unsafe_allow_html=True)

# Operator-oriented 8-item navigation (Stitch's left icon rail, approximated with a
# native Streamlit sidebar radio + icon glyphs - Streamlit has no icon-rail widget).
# HELP is deliberately NOT one of these 8 - it is a secondary/footer destination,
# rendered at the bottom of the sidebar (see "OPERATOR HELP" below).
NAV_VIEWS = ["SOLUTION EXPLAINER", "MISSION CONTROL", "SPECTRUM", "COGNITIVE ENGINE", "RECEIVER ARRAY", "TRACKS", "ALERTS", "ANALYTICS", "SYSTEM"]
if "show_help_page" not in st.session_state:
    st.session_state.show_help_page = False
# Programmatic navigation (e.g. Mission Control's "View all alerts ->" button):
# a widget-bound key (nav_view_radio) cannot be reassigned mid-script after the
# widget has already been instantiated this run - Streamlit raises
# StreamlitAPIException. The safe pattern is a plain, non-widget "pending nav"
# flag a button sets before calling st.rerun(); consumed here, BEFORE
# nav_view_radio is instantiated below, on the next run.
if "_pending_nav" in st.session_state:
    st.session_state["nav_view_radio"] = st.session_state.pop("_pending_nav")
nav_view = st.sidebar.radio(
    "NAVIGATION", options=NAV_VIEWS, index=0, key="nav_view_radio",
    format_func=lambda v: f"{theme.NAV_ICONS.get(v, '')}  {v}".strip(),
    on_change=lambda: st.session_state.update(show_help_page=False),
)
st.sidebar.markdown("---")

# --- Operating mode ---
st.sidebar.markdown("<div class='channel-header' style='font-size:0.75rem;'>OPERATING MODE</div>", unsafe_allow_html=True)
operating_mode = st.sidebar.radio(
    "OPERATING MODE", options=["LIVE SIMULATION", "REPLAY VERIFIED RUN"],
    index=0 if st.session_state.operating_mode == "LIVE SIMULATION" else 1,
    key="operating_mode_radio", label_visibility="collapsed",
    help="LIVE SIMULATION: a real, executing mission — new decisions generated this run. "
         "REPLAY VERIFIED RUN: deterministic playback of a verified precomputed artifact.",
)
st.session_state.operating_mode = operating_mode
mode_badge = "badge-live" if operating_mode == "LIVE SIMULATION" else "badge-primary"
st.sidebar.markdown(f"<span class='tech-badge {mode_badge}'>● {operating_mode} ACTIVE</span>", unsafe_allow_html=True)

if operating_mode == "LIVE SIMULATION" and st.session_state.live_mission is None:
    # Constructing LiveMissionRuntime loads a real TSRD HDF5 scenario file (measured
    # ~3-6 real seconds cold) - an explicit spinner here (rather than a silent
    # multi-second gap while only sidebar chrome has rendered) so a first-time
    # operator sees real progress feedback instead of what looks like a stalled or
    # broken page during that load.
    with st.spinner("INITIALIZING LIVE MISSION — loading TSRD scenario data..."):
        _init_live_mission("config_1.h5", "smart_scan")

st.sidebar.markdown("---")

# --- Scenario / strategy / speed ---
st.sidebar.markdown("<div class='channel-header' style='font-size:0.75rem;'>OPERATION CONFIGURATION</div>", unsafe_allow_html=True)
if operating_mode == "LIVE SIMULATION":
    lm = st.session_state.live_mission
    curr_scen = os.path.basename(lm.scenario_path) if lm else "config_1.h5"
    curr_strat = lm.strategy_type if lm else "smart_scan"
else:
    curr_scen = controller.scenario_name
    curr_strat = controller.strategy_type

sb_scen = st.sidebar.selectbox(
    "SCENARIO FILE", options=SCEN_OPTIONS,
    index=SCEN_OPTIONS.index(curr_scen) if curr_scen in SCEN_OPTIONS else 0,
    key="sb_scen_select",
)
sb_strat = st.sidebar.selectbox(
    "STRATEGY",
    options=["Smart Scan (Cognitive Q-Learning)", "Open Loop (Sequential Sweep)"],
    index=0 if curr_strat == "smart_scan" else 1,
    key="sb_strat_select",
)
strat_slug = "smart_scan" if "Smart" in sb_strat else "open_loop"

sb_speed = st.sidebar.select_slider(
    "SIMULATION SPEED", options=[0.5, 1.0, 2.0, 5.0, 10.0],
    value=float(st.session_state.live_mission.speed if (operating_mode == "LIVE SIMULATION" and st.session_state.live_mission) else controller.speed),
    format_func=lambda x: f"{x}x", key="sb_speed_slider",
)
if operating_mode == "LIVE SIMULATION" and st.session_state.live_mission is not None:
    st.session_state.live_mission.set_speed(sb_speed)
else:
    controller.set_speed(sb_speed)

# Section 11/12: changing scenario/strategy while a LIVE mission is actively RUNNING
# or PAUSED must not silently destroy it - require STOP -> RESET -> CHANGE first.
_live_mission_blocking = (
    operating_mode == "LIVE SIMULATION"
    and st.session_state.live_mission is not None
    and st.session_state.live_mission.mission_status in (LiveMissionStatus.RUNNING, LiveMissionStatus.PAUSED)
)
if _live_mission_blocking:
    st.sidebar.warning(
        f"Mission is {st.session_state.live_mission.mission_status} — "
        "STOP and RESET before changing scenario/strategy."
    )
# Step 17 section 12/15: REPLAY VERIFIED RUN is not blocked the same way (unlike a
# LIVE mission, replaying loses no real computed state - the artifact reloads
# instantly and deterministically), but the operator must still be told their
# current replay position is about to be discarded, not have it happen silently.
_replay_progress_at_risk = (
    operating_mode == "REPLAY VERIFIED RUN" and (controller.running or controller.current_step > 0)
)
if _replay_progress_at_risk:
    st.sidebar.info(
        f"Replay is at step {controller.current_step}/{controller.total_timesteps}. "
        "Applying a new scenario/strategy will reset this replay to step 0."
    )
if st.sidebar.button("🔄 INITIALIZE / APPLY", use_container_width=True, disabled=_live_mission_blocking):
    if operating_mode == "LIVE SIMULATION":
        with st.spinner(f"Loading {sb_scen}..."):
            _init_live_mission(sb_scen, strat_slug)
        st.sidebar.success(f"Live mission initialized: {sb_scen} ({strat_slug}).")
    else:
        controller.set_scenario(scenario_id=sb_scen, strategy_type=strat_slug)
        st.sidebar.success(f"Replay initialized: {sb_scen} ({strat_slug}).")
    st.rerun()

# --- Active engine for this rerun ---
engine = st.session_state.live_mission if operating_mode == "LIVE SIMULATION" else controller
snap0 = engine.get_snapshot()

st.sidebar.markdown("---")
st.sidebar.markdown("<div class='channel-header' style='font-size:0.75rem;'>SIMULATED RECEIVER ARRAY</div>", unsafe_allow_html=True)
st.sidebar.markdown(f"**Channels (K):** `{snap0.get('k_channels', 5)} Simultaneous`")
st.sidebar.markdown(f"**Frequency Range:** `500 MHz – 18.0 GHz`")
st.sidebar.markdown(f"**Bandwidth:** `17.5 GHz ({snap0.get('n_bands', 50)} Bands)`")
st.sidebar.markdown(f"**Dwell Time:** `50.0 ms per step`")

st.sidebar.markdown("---")
ophelp.render_visibility_indicator(snap0)
st.sidebar.markdown("---")

if operating_mode == "LIVE SIMULATION" and st.session_state.live_mission is not None:
    ophelp.render_scenario_metadata_panel(st.session_state.live_mission.get_scenario_metadata())
else:
    desc = all_discovered.get(sb_scen.replace(".h5", ""))
    ophelp.render_scenario_metadata_panel({
        "emitter_count": (desc.metrics_summary or {}).get("smart_scan", {}).get("unique_emitters_present") if desc and desc.metrics_summary else None,
        "collection_duration_s": desc.duration_s if desc else None,
        "frequency_range_mhz": (500.0, 18000.0) if desc else None,
        "num_bands": desc.num_bands if desc else None,
        "receiver_channels": desc.channels if desc else None,
        "total_steps": desc.num_steps if desc else None,
    })
st.sidebar.markdown("---")

st.sidebar.markdown("<div class='channel-header' style='font-size:0.75rem;'>SYSTEM TELEMETRY</div>", unsafe_allow_html=True)
st.sidebar.markdown(f"**Mission ID:** `{getattr(engine, 'mission_id', 'N/A')}`")
st.sidebar.markdown(f"**Data Source:** `TSRD ({'Live Execution' if operating_mode == 'LIVE SIMULATION' else 'Operational Replay'})`")
st.sidebar.markdown(f"**UI Latency:** `{getattr(controller, 'ui_latency_ms', 0.0):.1f} ms`")

# -----------------------------------------------------------------------------
# 3b. HELP - secondary/footer destination (deliberately not one of the 8 primary
# NAV_VIEWS). Quick-reference expanders stay inline here; the full 12-section
# operator guide (dashboard/help.py::render_help_page) opens in the main panel.
# -----------------------------------------------------------------------------
st.sidebar.markdown("---")
if st.sidebar.button("📖 OPERATOR HELP (Full Guide)", use_container_width=True, key="btn_open_help_page"):
    st.session_state.show_help_page = True
    st.rerun()
ophelp.render_how_to_operate()
ophelp.render_glossary_expander()

# -----------------------------------------------------------------------------
# 3c. HELP page short-circuit: sidebar (nav/mode/config) stays fully live and
# functional; only the main panel swaps to the full operator guide. Nothing below
# this block runs on a HELP-page render, including the LIVE/REPLAY auto-advance
# pacer at the very end of this file - the mission simply stops advancing while
# the operator is reading, exactly as PAUSE would, and resumes on the next click
# of a NAVIGATION item (see on_change= above) or the "← Back" button below.
# -----------------------------------------------------------------------------
if st.session_state.show_help_page:
    if st.button("← Back to workstation", key="btn_close_help_page"):
        st.session_state.show_help_page = False
        st.rerun()
    ophelp.render_help_page()
    st.stop()

# -----------------------------------------------------------------------------
# 5. Top status bar + primary control panel (every view) - ALWAYS the first thing
# rendered in the main panel, on every view, every mission state. Previously
# "SYSTEM READY" first-run guidance (below) and error banners rendered ahead of
# this, so the workstation's own identity/controls only appeared after a block of
# documentation-like content - fixed by moving this block first (Step 18).
# -----------------------------------------------------------------------------
live_operations.render_top_status_bar(engine)

# Mission Replay workspace: a timeline scrubber over the verified artifact. Only
# meaningful in REPLAY VERIFIED RUN - PlaybackController.step() can honestly jump to
# any recorded step (it is indexing an already-computed artifact); LiveMissionRuntime
# cannot ("jumping" a live mission would mean fabricating steps that were never
# actually executed, which is exactly what this project must never do). Uses
# PlaybackController only - no second replay engine.
if operating_mode == "REPLAY VERIFIED RUN":
    live_operations.render_replay_scrubber(controller)

# -----------------------------------------------------------------------------
# 4b. Error handling (section 20): surface a missing/corrupted scenario or artifact
# with an actionable message instead of silently rendering an all-N/A mission.
# -----------------------------------------------------------------------------
if operating_mode == "LIVE SIMULATION" and st.session_state.live_mission is not None:
    if getattr(st.session_state.live_mission.engine, "env", "missing") is None:
        st.error(
            f"⚠ SCENARIO ENVIRONMENT FAILED TO LOAD — `{sb_scen}` was not found or could "
            f"not be read from `{SCAN_DIR}`. Select a different scenario and press "
            "INITIALIZE / APPLY, or verify the dataset files are present."
        )
elif operating_mode == "REPLAY VERIFIED RUN" and not controller.time_series:
    load_err = getattr(controller, "artifact_load_error", None)
    reason = f" (reason: {load_err})" if load_err else " (file not found)"
    st.error(
        f"⚠ OPERATIONAL ARTIFACT UNAVAILABLE — no usable verified `operational_evaluation_"
        f"{sb_scen.replace('.h5','')}.json` in `results/`{reason}. Select a different "
        "scenario and press INITIALIZE / APPLY, or verify the artifact files are present."
    )

# -----------------------------------------------------------------------------
# 6. View Router
# -----------------------------------------------------------------------------
if nav_view == "SOLUTION EXPLAINER":
    jury_explainer.render_jury_explainer(engine)
elif nav_view == "MISSION CONTROL":
    # Mission Control redesign (Phase B): compact enterprise-console layout.
    # Every value below still comes from the exact same engine.get_snapshot()/
    # render_* calls as before - this block only changes WHERE/HOW they're laid
    # out, never what's computed. Large explanatory content ("HOW THIS PROTOTYPE
    # WORKS", the mission-history metrics recap that duplicated the KPI row, the
    # "why this band" decision-reasoning card, and the data-integrity checklist)
    # was moved out of this view - none of it was deleted: HOW THIS PROTOTYPE
    # WORKS/glossary content lives in the sidebar's "HOW TO OPERATE" expander and
    # in ophelp.render_help_page() section 1; band-selection reasoning has its
    # full, real version in the COGNITIVE ENGINE view's decision panel; data
    # integrity has its full, real version in the SYSTEM view.
    # Render Consolidated KPI Bar
    live_operations.render_kpi_bar(engine)
    snap = engine.get_snapshot()
    mc_current_step = snap.get("timestep", snap.get("current_step", 0))

    # Main operational workspace (redesign section E): Receiver Array (left) |
    # Live Spectrum (right, more visual weight) - existing render functions/
    # data only, no new computation. Receiver cards use compact=True here only
    # (Mission Control's reduced-hierarchy card); the standalone RECEIVER ARRAY
    # view still calls render_receiver_panel() directly with the unchanged default.
    st.markdown(theme.section_divider("MAIN OPERATIONAL WORKSPACE"), unsafe_allow_html=True)
    spectrum.render_live_spectrum_map(engine, show_ground_truth=False)

    st.markdown("<div class='channel-header' style='font-size:0.85rem; margin-top:1.2rem; margin-bottom:0.5rem; font-weight:700;'>RECEIVER HARDWARE ARRAY (K=5 Active Channels)</div>", unsafe_allow_html=True)
    live_operations.render_receiver_strip(engine, compact=True)

    # Secondary operational information (redesign section F/G/H): recent
    # decisions (left) | operator attention + alert summary (right). Concise,
    # curated views - not the full Cognitive Engine / Alerts pages.
    st.markdown(theme.section_divider("SECONDARY INFORMATION"), unsafe_allow_html=True)
    sec_c1, sec_c2 = st.columns([6, 4])
    with sec_c1:
        st.markdown(
            f"<div class='channel-header' style='font-size:0.75rem;'>RECENT DECISIONS &nbsp;"
            f"<span style='color:{theme.COLOR_TEXT_FAINT}; text-transform:none; font-weight:500;'>"
            f"(what the cognitive engine is doing right now)</span></div>",
            unsafe_allow_html=True,
        )
        if hasattr(engine, "get_decision_history"):
            dec_hist = engine.get_decision_history(window=6)
        else:
            dec_hist = list(getattr(engine, "decision_history", []))[:6]
        if dec_hist:
            def _dh(row, *keys, default="N/A"):
                for k in keys:
                    v = row.get(k)
                    if v is not None:
                        return v
                return default

            compact_rows = []
            for r in dec_hist:
                bands = _dh(r, "selected_bands", "Selected Bands", "Bands", default=[])
                bands_txt = ", ".join(bands) if isinstance(bands, list) else str(bands)
                reward_val = _dh(r, "step_reward", "Reward", "reward")
                compact_rows.append({
                    "Time (s)": _dh(r, "time_s", "Time (s)", "Time"),
                    "Strategy": _dh(r, "strategy", "Strategy"),
                    "Bands": bands_txt,
                    "Reward": f"{reward_val:+.2f}" if isinstance(reward_val, (int, float)) else reward_val,
                })
            st.dataframe(pd.DataFrame(compact_rows), use_container_width=True, height=210, hide_index=True)
        else:
            st.markdown(theme.empty_state("No decisions recorded yet — mission not started."), unsafe_allow_html=True)
    with sec_c2:
        alerts.render_attention_required(engine)
        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
        alerts.render_alert_summary_counts(engine)
        alerts.render_alerts_panel(engine, max_items=4)
        if st.button("View all alerts →", key="btn_mc_view_all_alerts", use_container_width=True):
            st.session_state["_pending_nav"] = "ALERTS"
            st.rerun()

    ophelp.render_bottom_status(snap)

elif nav_view == "SPECTRUM":
    st.markdown("---")
    st.markdown("<div class='system-title'>LIVE RF SPECTRUM</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='system-subtitle'>{engine.get_snapshot().get('n_bands', 50)} FREQUENCY BANDS • 500 MHz – 18 GHz • "
        f"{engine.get_snapshot().get('max_duration_s', 30.0):.0f} SECOND OPERATIONAL WINDOW • {operating_mode}</div>",
        unsafe_allow_html=True,
    )
    ts_override = None
    if operating_mode == "LIVE SIMULATION":
        st.markdown("<span class='tech-badge badge-live'>● LIVE TELEMETRY</span>", unsafe_allow_html=True)
        window_choice = st.radio(
            "SPECTRUM DATA WINDOW", options=["LIVE WINDOW", "MISSION HISTORY"], horizontal=True,
            key="spectrum_window_choice",
            help="LIVE WINDOW: most recent ~60 steps (3s). MISSION HISTORY: every real step "
                 "taken so far this mission, accumulated at the runtime layer.",
        )
        if window_choice == "MISSION HISTORY" and hasattr(engine, "get_mission_history_time_series"):
            ts_override = engine.get_mission_history_time_series()
            st.caption(f"Showing full accumulated mission history: {len(ts_override)} real steps recorded.")
        else:
            st.caption("Rolling window of the most recent steps.")
    else:
        st.markdown("<span class='tech-badge badge-primary'>● POST-HOC VERIFIED DATA</span>", unsafe_allow_html=True)

    spectrum_view_mode = st.radio(
        "SPECTRUM VIEW", options=["WATERFALL", "SPECTRUM ANALYZER"], horizontal=True,
        key="spectrum_view_mode",
        help="WATERFALL: time-frequency history across all 50 bands (unchanged). "
             "SPECTRUM ANALYZER: engineering power-vs-frequency view of the currently "
             "tuned channels only - never a fabricated trace across unobserved bands.",
    )
    if spectrum_view_mode == "WATERFALL":
        show_gt = st.checkbox("Overlay ground-truth activity (post-hoc validation only)", value=False, key="spectrum_show_gt")
        spectrum.render_live_spectrum_map(engine, show_ground_truth=show_gt, time_series_override=ts_override)
        st.caption("● RF ACTIVITY (ground truth, only when overlay enabled)   ◇ RECEIVER SCAN   ★ TRUE INTERCEPTION   ◆ FALSE ALARM   │ CURRENT TIME")
    else:
        spectrum.render_spectrum_analyzer(engine)
    st.markdown("---")
    spectrum.render_band_inspector(engine)
    ophelp.render_bottom_status(engine.get_snapshot())

elif nav_view == "COGNITIVE ENGINE":
    decision_panel.render_decision_panel(engine)

elif nav_view == "RECEIVER ARRAY":
    receiver_panel.render_receiver_panel(engine)

elif nav_view == "TRACKS":
    st.markdown("---")
    tracks.render_tracks_view(engine)
    st.markdown("---")
    event_console.render_event_console(engine)
    ophelp.render_bottom_status(engine.get_snapshot())

elif nav_view == "ALERTS":
    st.markdown("---")
    alerts.render_alerts_view(engine)
    ophelp.render_bottom_status(engine.get_snapshot())

elif nav_view == "ANALYTICS":
    lm = st.session_state.live_mission
    lm_active = lm is not None and lm.mission_status in (LiveMissionStatus.RUNNING, LiveMissionStatus.PAUSED)
    if lm is None or lm.get_snapshot()["total_scans"] == 0:
        st.info("No live mission data yet. Switch to LIVE SIMULATION mode, press START MISSION, and return here.")
    else:
        performance.render_performance_monitor(lm)

    st.markdown("---")
    st.markdown("<div style='font-size:1.1rem; font-weight:800; color:#00FF9D; font-family:\"Outfit\"; margin-bottom:0.5rem;'>SCENARIO EXPERIMENT LAB</div>", unsafe_allow_html=True)
    if "experiment_lab" not in st.session_state or st.session_state.experiment_lab is None:
        from simulation.engine import SimulationEngine as _SimEngine
        st.session_state.experiment_lab = _SimEngine(
            scenario_path=os.path.join(SCAN_DIR, "config_1.h5"), strategy_type="smart_scan", k_channels=5, seed=42,
        )
    system.render_scenario_lab(st.session_state.experiment_lab)

    st.markdown("---")
    st.markdown("<div class='system-title' style='font-size:1rem; color:#00e5ff;'>VERIFIED BENCHMARK (5-SCENARIO, DETERMINISTIC)</div>", unsafe_allow_html=True)
    system.render_benchmark_suite(validated_scenarios)
    ophelp.render_bottom_status(engine.get_snapshot())

elif nav_view == "SYSTEM":
    st.markdown("---")
    system.render_architecture_overview(engine, operating_mode=operating_mode)
    st.markdown("---")
    system.render_system_health(engine)
    system.render_health_matrix(engine, operating_mode=operating_mode)
    st.markdown("---")
    ophelp.render_data_integrity_indicator(operating_mode=operating_mode)
    ophelp.render_bottom_status(engine.get_snapshot())

    # Mission Reporting & Telemetry Export — Audit Trail (rendered ONLY on the last page: SYSTEM)
    st.markdown("---")
    st.markdown("<div class='channel-header' style='font-size:0.85rem;'>MISSION REPORTING & TELEMETRY EXPORT (AUDIT TRAIL)</div>", unsafe_allow_html=True)
    snap_exp = engine.get_snapshot()
    exp_c1, exp_c2, exp_c3 = st.columns(3)
    with exp_c1:
        st.download_button(
            "📥 EXPORT MISSION LOG (JSON)",
            data=json.dumps(engine.export_report_json(), indent=2, default=str),
            file_name=f"mission_report_{getattr(engine, 'mission_id', 'session')}_{snap_exp.get('scenario_name','scenario').replace('.h5','')}.json",
            mime="application/json", use_container_width=True, key="exp_m_report_btn",
        )
    with exp_c2:
        st.download_button(
            "📥 EXPORT EVENT TELEMETRY (CSV)",
            data=engine.export_events_csv(),
            file_name=f"rf_events_{getattr(engine, 'mission_id', 'session')}_{snap_exp.get('timestep', 0)}steps.csv",
            mime="text/csv", use_container_width=True, key="exp_m_events_btn",
        )
    with exp_c3:
        st.download_button(
            "📥 EXPORT TRACK / INTERCEPTION HISTORY (CSV)",
            data=engine.export_tracks_csv(),
            file_name=f"rf_tracks_{getattr(engine, 'mission_id', 'session')}_{snap_exp.get('timestep', 0)}steps.csv",
            mime="text/csv", use_container_width=True, key="exp_m_tracks_btn",
        )

    exp2_c1, exp2_c2, exp2_c3 = st.columns(3)
    with exp2_c1:
        if hasattr(engine, "get_decision_history"):
            dec_rows = engine.get_decision_history(window=600)
        else:
            dec_rows = list(getattr(engine, "decision_history", []))
        dec_csv = pd.DataFrame(dec_rows).to_csv(index=False) if dec_rows else "No decision data recorded."
        st.download_button(
            "📥 EXPORT DECISION TRACE (CSV)", data=dec_csv,
            file_name=f"decision_trace_{getattr(engine, 'mission_id', 'session')}.csv",
            mime="text/csv", use_container_width=True, key="exp_m_decisions_btn",
        )
    with exp2_c2:
        band_counts = dict(getattr(engine, "band_scan_counts", {}))
        receiver_csv = pd.DataFrame(list(band_counts.items()), columns=["Band", "Scan_Count"]).to_csv(index=False) if band_counts else "No receiver utilization data recorded."
        st.download_button(
            "📥 EXPORT RECEIVER UTILIZATION (CSV)", data=receiver_csv,
            file_name=f"receiver_utilization_{getattr(engine, 'mission_id', 'session')}.csv",
            mime="text/csv", use_container_width=True, key="exp_m_receiver_btn",
        )
    with exp2_c3:
        if hasattr(engine, "get_mission_history_summary"):
            summary = engine.get_mission_history_summary()
        else:
            summary = {"note": "Mission summary is only computed for the LIVE runtime; use EXPORT MISSION LOG for REPLAY."}
        st.download_button(
            "📥 EXPORT MISSION SUMMARY (JSON)", data=json.dumps(summary, indent=2, default=str),
            file_name=f"mission_summary_{getattr(engine, 'mission_id', 'session')}.json",
            mime="application/json", use_container_width=True, key="exp_m_summary_btn",
        )

# Record UI render latency (attributed to the replay controller for backward-compat
# with the sidebar telemetry line; harmless if the live runtime is currently active)
t_render_end = time.perf_counter()
controller.ui_latency_ms = (t_render_end - t_render_start) * 1000.0

# -----------------------------------------------------------------------------
# 8. Real-time execution loop pacer — thread-free, rerun-driven (section 19).
#    Each rerun advances at most ONE real timestep; the full mission is never
#    executed inline on a single rerun (section 17).
# -----------------------------------------------------------------------------
if operating_mode == "LIVE SIMULATION":
    lm = st.session_state.live_mission
    if lm is not None and lm.mission_status == LiveMissionStatus.RUNNING:
        time.sleep(0.02)
        lm.advance_time_tick()
        st.rerun()
else:
    if controller.running:
        if controller.current_step < controller.total_timesteps - 1:
            spd = max(0.5, float(controller.speed))
            delay_s = max(0.015, 0.05 / spd)
            time.sleep(delay_s)
            controller.step(1)
            st.rerun()
        else:
            controller.mission_completed = True
            controller.running = False
            st.rerun()
