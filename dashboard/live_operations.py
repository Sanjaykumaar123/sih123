"""Live Operations View: Top system status bar, operational mission controls, live KPI bar, and receiver hardware objects."""

from typing import Any, Dict, List, Optional
import time
import streamlit as st
from core.state import EngineStatus
from dashboard import alerts as _alerts
from dashboard import theme


def render_live_operations(engine: Any) -> None:
    """Render top system status bar, mission control panel, receiver panel, and live KPI bar.
    Composite of render_top_status_bar + render_receiver_strip + render_kpi_bar - kept as
    one call for backward compatibility with existing call sites/tests."""
    render_top_status_bar(engine)
    render_receiver_strip(engine)
    render_kpi_bar(engine)


def render_top_status_bar(engine: Any) -> None:
    """Top system status bar + primary mission control panel (sections 3-5)."""
    snap = engine.get_snapshot()

    # 1. Top System Status Bar. Note: core.state.EngineStatus spells the terminal
    # state "COMPLETE" while core.live_mission.LiveMissionStatus (per Step 14 section
    # 3) spells it "COMPLETED" - both are treated as the same logical state below so
    # neither runtime's completion silently fails to disable START/STEP.
    status_str = snap.get("mission_status", snap.get("status", EngineStatus.READY))
    COMPLETE_STATES = (EngineStatus.COMPLETE, "COMPLETED")
    # Canonical STATUS vocabulary (dashboard/theme.py) - reconciled 1:1 with the
    # exact states this status_str can actually take, so this header never
    # disagrees with the STATUS dict used elsewhere.
    status_map = {
        EngineStatus.IDLE: (theme.COLOR_PRIMARY, "IDLE"),
        EngineStatus.READY: (theme.COLOR_PRIMARY, "READY"),
        EngineStatus.RUNNING: (theme.COLOR_NOMINAL, "RUNNING"),
        EngineStatus.PAUSED: (theme.COLOR_CAUTION, "PAUSED"),
        EngineStatus.STOPPED: (theme.COLOR_TEXT_MUTED, "STOPPED"),
        EngineStatus.COMPLETE: (theme.COLOR_COGNITIVE, "MISSION COMPLETE"),
        "COMPLETED": (theme.COLOR_COGNITIVE, "MISSION COMPLETE"),
        EngineStatus.ERROR: (theme.COLOR_CRITICAL, "ERROR"),
    }
    st_color, st_label = status_map.get(status_str, (theme.COLOR_TEXT_MUTED, status_str))

    max_dur = snap.get("max_duration_s", 30.0)
    max_steps = snap.get("max_steps", snap.get("total_timesteps", 600))
    current_step = snap.get("timestep", snap.get("current_step", 0))
    operating_mode = snap.get("operating_mode", "REPLAY VERIFIED RUN")

    # Active alerts (Step 17 section 2): the same real, severity-filtered actionable
    # stream render_attention_required already uses (WARNING/CRITICAL only) - counted
    # here, not recomputed differently, so the cockpit badge and the OPERATOR
    # ATTENTION panel can never disagree.
    _alert_rows = engine.get_alerts(limit=50) if hasattr(engine, "get_alerts") else _alerts._replay_alerts(engine)
    active_alert_count = sum(1 for a in _alert_rows if a.get("severity") in _alerts.ACTIONABLE_SEVERITIES)
    alert_color = theme.COLOR_CRITICAL if active_alert_count > 0 else theme.COLOR_NOMINAL

    # Global Shell header (section 6): application identity + mission ID + mode +
    # system health + alerts only - detailed telemetry (time/step/progress/
    # strategy/visibility/latency, previously crammed into a stat-tile row here)
    # now lives in Mission Control's KPI cards instead, per "Do NOT squeeze all
    # telemetry into a single line. Detailed telemetry belongs in KPI/stat
    # cards." A soft lift (SHADOW_SM), not a glow - "very restrained shadows".
    st.markdown(
        f"""
        <div style='background-color:{theme.COLOR_PANEL}; border:{theme.BORDER}; border-radius:{theme.RADIUS_CARD}; padding:0.65rem 1.0rem; margin-bottom:{theme.SPACE[4]}; box-shadow:{theme.SHADOW_SM};'>
            <div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;'>
                <div>
                    <span style='font-size:0.95rem; font-weight:700; color:{theme.COLOR_TEXT}; letter-spacing:0.02em;'>RF MISSION CONTROL</span>
                    <span style='font-family:{theme.FONT_MONO}; font-size:0.66rem; color:{theme.COLOR_TEXT_MUTED}; margin-left:0.7rem;'>MISSION: {getattr(engine, 'mission_id', 'N/A')}</span>
                </div>
                <div>
                    <span class='tech-badge' style='background-color:{theme.COLOR_NOMINAL}18; color:{theme.COLOR_NOMINAL}; border:1px solid {theme.COLOR_NOMINAL}55;'>
                        ● SYSTEM HEALTHY
                    </span>
                    <span class='tech-badge' style='background-color:{st_color}18; color:{st_color}; border:1px solid {st_color}55;'>
                        ● {st_label}
                    </span>
                    <span class='tech-badge {"badge-live" if operating_mode == "LIVE SIMULATION" else "badge-primary"}'>{operating_mode}</span>
                    <span class='tech-badge' style='background-color:{alert_color}18; color:{alert_color}; border:1px solid {alert_color}55;'
                          title='Actionable (WARNING/CRITICAL) alerts only - see the ALERTS view for the full stream.'>
                        ⚠ ALERTS: {active_alert_count}
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    is_complete = status_str in COMPLETE_STATES or current_step >= max_steps - 1

    # Mission Completion Banner
    if is_complete:
        st.success(f"🏁 **MISSION COMPLETE — {max_dur:.2f} SECOND OPERATIONAL RUN COMPLETE.** (Press RESET to re-initialize)")

    # 2. Mission Operational Controls, regrouped by hierarchy (Mission Control
    # redesign): [ START ](primary) [ STEP ][ STEP+10 ][ PAUSE ][ RESUME ][ RESET ]
    # (secondary) [ STOP ](destructive) Speed. Every button below keeps its exact
    # original key/condition/on-click call - only the column order (purely visual
    # grouping) changed; the enable/disable state machine is byte-for-byte unchanged.
    ctl_cols = st.columns([1.3, 1.1, 1.2, 1.15, 1.3, 1.2, 1.2, 2.0])
    is_running = status_str == EngineStatus.RUNNING
    is_paused = status_str == EngineStatus.PAUSED

    with ctl_cols[0]:
        # Per the state table: enabled at READY/STOPPED/COMPLETED (COMPLETED restarts
        # via an implicit reset - see LiveMissionRuntime.start_mission), disabled only
        # while the mission is actually RUNNING or PAUSED (use RESUME there instead).
        # Stitch: primary action gets a solid-cyan fill when it's actually the
        # available next action (styled via the .st-key-btn_ops_start CSS rule's
        # :not(:disabled) selector in theme.py - Streamlit's own type="primary"
        # was tried first but its stock red (#FF4B4B) internal stylesheet won the
        # cascade over a plain color override, so this button intentionally stays
        # the default kind and gets its cyan treatment purely from our own key-
        # scoped CSS instead).
        if st.button("▶ START", use_container_width=True, disabled=is_running or is_paused, key="btn_ops_start",
                      help="Begins the live RF scanning loop."):
            engine.start()
            st.rerun()
    with ctl_cols[1]:
        # STEP is only meaningful from READY/PAUSED - disabled while RUNNING so manual
        # single-step and auto-advance are never mixed (section 7 / 19).
        if st.button("⏭ STEP", use_container_width=True, disabled=is_running or is_complete, key="btn_ops_step1",
                      help="Advance exactly one simulation timestep (one full observe-decide-scan-learn cycle)."):
            engine.step(num_steps=1)
            st.rerun()
    with ctl_cols[2]:
        if st.button("⏭ STEP +10", use_container_width=True, disabled=is_running or is_complete, key="btn_ops_step10",
                      help="Advance ten simulation timesteps for rapid inspection."):
            engine.step(num_steps=10)
            st.rerun()
    with ctl_cols[3]:
        if st.button("⏸ PAUSE", use_container_width=True, disabled=(not is_running), key="btn_ops_pause",
                      help="Freeze the mission at the current timestep. Nothing more executes until RESUME."):
            engine.pause()
            st.rerun()
    with ctl_cols[4]:
        if st.button("▶ RESUME", use_container_width=True, disabled=(not is_paused), key="btn_ops_resume",
                      help="Continue the mission from exactly where it was paused."):
            engine.resume()
            st.rerun()
    with ctl_cols[5]:
        if st.button("🔄 RESET", use_container_width=True, key="btn_ops_reset",
                      help="Return the live mission to READY and clear all current mission state (timestep, telemetry, events, detections, learning progress)."):
            engine.reset()
            st.rerun()
    with ctl_cols[6]:
        if st.button("⏹ STOP", use_container_width=True, disabled=(status_str in (EngineStatus.STOPPED, EngineStatus.READY, EngineStatus.IDLE)), key="btn_ops_stop",
                      help="Deliberately halt the mission. Unlike PAUSE, RESUME does not work from here — press START MISSION to continue, or RESET for a clean run."):
            engine.stop()
            st.rerun()
    with ctl_cols[7]:
        spd = st.select_slider(
            "SPEED",
            options=[0.5, 1.0, 2.0, 5.0, 10.0],
            value=float(snap.get("speed_multiplier", snap.get("speed", 1.0))),
            format_func=lambda x: f"{x}x",
            key="ops_speed_slider",
        )
        if hasattr(engine, "clock"):
            engine.clock.set_speed(spd)
        if hasattr(engine, "set_speed"):
            engine.set_speed(spd)


def render_receiver_strip(engine: Any, compact: bool = False) -> None:
    """5 receiver channel cards (section 8). `compact` is passed straight through
    to receiver_panel.render_receiver_panel - see that function's docstring for
    what changes (Mission Control redesign uses compact=True; the standalone
    RECEIVER ARRAY view keeps calling render_receiver_panel directly with the
    default, unchanged)."""
    from dashboard.receiver_panel import render_receiver_panel
    render_receiver_panel(engine, compact=compact)


def render_kpi_bar(engine: Any) -> None:
    """Live KPI cards (section 13), Mission Control redesign: a primary row of 6
    prominent theme.kpi_card()s (Mission Time / Progress / Scans / Detections /
    Active Tracks / Cumulative Reward - the "dynamic runtime info stays visually
    prominent" design principle) plus a visually quieter secondary row (Cycles /
    False Alarms / Emitters / Current Reward) using the existing, smaller
    .metric-card styling. All 9 original metrics are still shown - none removed,
    only regrouped by operator priority. Section 25 still holds: before any scan
    has actually happened, cards show '—' / 'Mission not started' rather than a
    misleading bare 0 - every value is read from the same snapshot fields as
    before, nothing here is computed or invented differently."""
    snap = engine.get_snapshot()
    current_step = snap.get("timestep", snap.get("current_step", 0))
    max_steps = snap.get("max_steps", snap.get("total_timesteps", 600))
    sim_time_s = snap.get("simulated_time_s", snap.get("simulation_time_s", 0.0))
    total_scans = snap.get("total_scans", 0)
    started = total_scans > 0
    # Same formula as render_top_status_bar's PROGRESS stat tile - one real,
    # already-computed value, just also surfaced here as a KPI card.
    progress_pct = (current_step / max(1, max_steps - 1)) * 100.0

    def dash(v: Any, fmt: str = "") -> str:
        if not started:
            return "—"
        return format(v, fmt) if fmt else str(v)

    cum_val = snap.get("cumulative_reward", 0.0)
    reward_dir = "up" if (started and cum_val > 0) else ("down" if (started and cum_val < 0) else "neutral")

    # KPI visual hierarchy (Mission Control redesign section 7): DETECTIONS /
    # ACTIVE TRACKS / PROGRESS are the highest-priority "what is happening"
    # signals and get the louder 'hero' card treatment; MISSION TIME / SCANS /
    # CUMULATIVE REWARD stay at the standard size. Same 6 real fields/values as
    # before - only which ones visually dominate changed.
    primary = [
        ("MISSION TIME", f"{sim_time_s:.2f}s", f"{current_step} steps", "neutral", "◷", theme.COLOR_PRIMARY, "standard"),
        ("PROGRESS", f"{progress_pct:.0f}%", f"{current_step} / {max_steps}", "neutral", "▤", theme.COLOR_PRIMARY, "hero"),
        ("SCANS", dash(total_scans), (f"K={snap.get('k_channels', 5)} dwells" if started else "Mission not started"), "neutral", "◉", theme.COLOR_PRIMARY, "standard"),
        ("DETECTIONS", dash(snap.get("true_detections", 0)), ("Confirmed hits" if started else "Mission not started"), ("up" if started else "neutral"), "✓", theme.COLOR_NOMINAL, "hero"),
        ("ACTIVE TRACKS", dash(snap.get("active_tracks_count", 0)), (f"{snap.get('total_tracks_count', 0)} formed" if started else "Mission not started"), "neutral", "◈", theme.COLOR_COGNITIVE, "hero"),
        ("CUMULATIVE REWARD", (f"{cum_val:+.1f}" if started else "—"), ("Total score" if started else "Mission not started"), reward_dir, "Σ", theme.COLOR_PRIMARY, "standard"),
    ]
    p_cols = st.columns(6)
    for col, (lbl, val, delta, ddir, icon, icon_color, size) in zip(p_cols, primary):
        with col:
            st.markdown(theme.kpi_card(lbl, val, delta, ddir, icon, icon_color, size=size), unsafe_allow_html=True)

    st.markdown("<div class='channel-header' style='font-size:0.68rem; margin-top:0.6rem;'>MISSION DETAILS</div>", unsafe_allow_html=True)

    cycles_val = snap.get("health", {}).get("total_cycles_executed", current_step)
    pfa_val = snap.get("pfa", 0.0)
    emitters_val = snap.get("unique_emitters_count")
    rew_val = snap.get("latest_reward", 0.0)
    secondary = [
        ("Cycles", dash(cycles_val), "Cognitive steps"),
        ("False Alarms", dash(snap.get("false_alarms", 0)), (f"{pfa_val*100:.2f}% Pfa" if started else "—")),
        ("Emitters", (dash(emitters_val) if emitters_val is not None else "N/A"), ("Acquired" if started else "Mission not started")),
        ("Current Reward", (f"{rew_val:+.2f}" if started else "—"), "Step reward"),
    ]
    s_cols = st.columns(4)
    for col, (lbl, val, sub) in zip(s_cols, secondary):
        with col:
            st.markdown(
                f"""
                <div class='metric-card metric-card-quiet'>
                    <div class='metric-lbl'>{lbl}</div>
                    <div class='metric-val' style='font-size:0.98rem;'>{val}</div>
                    <div class='metric-imp imp-neutral'>{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_replay_scrubber(controller: Any) -> None:
    """Mission Replay workspace: a real timeline scrubber over the verified
    operational-evaluation artifact PlaybackController already loaded (no second
    replay engine). Jumping the scrubber is honest here because PlaybackController is
    indexing an already-computed, verified time series - it does not execute anything
    new. The equivalent capability does NOT exist (and must not be added) for the
    live runtime, which can only ever advance by actually running its closed loop."""
    total = max(1, controller.total_timesteps - 1)
    snap = controller.get_snapshot()
    st.markdown(
        "<div class='channel-header' style='font-size:0.75rem; margin-top:0.2rem;'>MISSION REPLAY TIMELINE</div>",
        unsafe_allow_html=True,
    )
    rc1, rc2 = st.columns([8, 2])
    with rc1:
        new_step = st.slider(
            "TIMELINE", min_value=0, max_value=total,
            value=min(int(controller.current_step), total),
            key="replay_timeline_scrubber", label_visibility="collapsed",
            disabled=(not controller.time_series),
        )
    with rc2:
        st.markdown(
            f"<div style='font-family:monospace; font-size:0.78rem; text-align:right; padding-top:0.3rem;'>"
            f"t={snap.get('simulated_time_s', 0.0):.2f}s&nbsp;&nbsp;STEP {snap.get('timestep', 0)}/{total}</div>",
            unsafe_allow_html=True,
        )
    if controller.time_series and new_step != controller.current_step:
        # Scrubbing implies an explicit pause - never fight the auto-advance loop.
        controller.running = False
        controller.paused = True
        controller.mission_completed = new_step >= total
        controller.current_step = new_step
        controller.last_update_time = time.perf_counter()
        st.rerun()
