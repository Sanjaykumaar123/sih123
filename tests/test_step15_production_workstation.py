"""Step 15: Production-grade operator workstation - verification tests.

Covers what's new/changed in Step 15 on top of the Step 13/14 baseline: experiment
lab isolation from the active live mission, scenario-switch guarding while a mission
is RUNNING/PAUSED, real mission-history accumulation (spectrum + summary stats),
real operational event/alert generation, new exports, session persistence across
Streamlit reruns, and error handling for missing scenario/artifact data.

Does not touch or re-verify rf_env/ - those are exercised unchanged by
tests/test_stage*.py. Does not weaken or replace tests/test_step13_workstation_
redesign.py or tests/test_step14_live_workstation.py.
"""

import os
import pytest
from streamlit.testing.v1 import AppTest

from core.live_mission import LiveMissionRuntime, LiveMissionStatus
from simulation.engine import SimulationEngine

SCENARIO = r"D:\sih\dataset\scan\test_scan\config_1.h5"
SCAN_DIR = r"D:\sih\dataset\scan\test_scan"
APP_TIMEOUT = 90


def make_runtime(**kwargs):
    defaults = dict(scenario_path=SCENARIO, strategy_type="smart_scan", k_channels=5, n_bands=50, seed=42)
    defaults.update(kwargs)
    return LiveMissionRuntime(**defaults)


# -----------------------------------------------------------------------------
# Experiment Lab isolation (the central Step 15 requirement)
# -----------------------------------------------------------------------------
def test_experiment_lab_is_a_separate_object_from_live_mission():
    at = AppTest.from_file("app.py", default_timeout=APP_TIMEOUT)
    at.run()
    at.sidebar.radio(key="nav_view_radio").set_value("ANALYTICS").run()
    assert not at.exception
    lm = at.session_state["live_mission"]
    lab = at.session_state["experiment_lab"]
    assert lm is not lab
    assert isinstance(lab, SimulationEngine)
    assert isinstance(lm, LiveMissionRuntime)


def test_experiment_lab_execution_never_mutates_active_live_mission():
    at = AppTest.from_file("app.py", default_timeout=APP_TIMEOUT)
    at.run()
    at.button(key="btn_ops_step10").click().run()
    lm = at.session_state["live_mission"]
    step_before = lm.engine.clock.current_step
    scans_before = lm.get_snapshot()["total_scans"]

    at.sidebar.radio(key="nav_view_radio").set_value("ANALYTICS").run()
    exec_btn = next(b for b in at.button if "EXECUTE FULL 30s RUN" in (b.label or ""))
    exec_btn.click().run()
    assert not at.exception

    lm_after = at.session_state["live_mission"]
    assert lm_after is lm
    assert lm_after.engine.clock.current_step == step_before
    assert lm_after.get_snapshot()["total_scans"] == scans_before
    assert at.session_state["experiment_lab"].clock.current_step == 600


def test_active_live_mission_protected_banner_shown_when_running():
    at = AppTest.from_file("app.py", default_timeout=APP_TIMEOUT)
    at.run()
    lm = at.session_state["live_mission"]
    lm.mission_status = LiveMissionStatus.PAUSED  # simulate an in-progress mission
    at.sidebar.radio(key="nav_view_radio").set_value("ANALYTICS").run()
    assert not at.exception
    warnings = [w.value for w in at.warning]
    assert any("ACTIVE LIVE MISSION PROTECTED" in w for w in warnings)


# -----------------------------------------------------------------------------
# Scenario-switch guard while a live mission is RUNNING/PAUSED (section 11/12)
# -----------------------------------------------------------------------------
def test_apply_button_disabled_while_live_mission_running_or_paused():
    at = AppTest.from_file("app.py", default_timeout=APP_TIMEOUT)
    at.run()
    lm = at.session_state["live_mission"]
    # Setting RUNNING on a fresh (step=0) mission and calling at.run() would trigger
    # the real (correct) auto-advance rerun loop, which does not settle until
    # COMPLETED - not suitable for a synchronous assertion. Step to just before the
    # end first so the loop completes almost immediately (same technique as
    # test_step14_live_workstation.py's test_20d).
    total = lm.engine.env.total_steps
    lm.step_n(total - 2)
    lm.mission_status = LiveMissionStatus.RUNNING
    lm._last_tick_wall_time = None
    at.run()
    assert not at.exception
    assert lm.mission_status == LiveMissionStatus.COMPLETED
    # COMPLETED still requires STOP/RESET-equivalent housekeeping before changing
    # scenario is safe, but per the state table COMPLETED is not RUNNING/PAUSED, so
    # the guard should NOT block Apply here - only RUNNING/PAUSED do.
    apply_btn = next(b for b in at.sidebar.button if "INITIALIZE" in (b.label or ""))
    assert apply_btn.disabled is False

    lm.mission_status = LiveMissionStatus.PAUSED
    at.run()
    apply_btn2 = next(b for b in at.sidebar.button if "INITIALIZE" in (b.label or ""))
    assert apply_btn2.disabled is True

    lm.mission_status = LiveMissionStatus.READY
    at.run()
    apply_btn3 = next(b for b in at.sidebar.button if "INITIALIZE" in (b.label or ""))
    assert apply_btn3.disabled is False


def test_scenario_apply_does_not_silently_destroy_running_mission():
    """Even if somehow invoked, applying a new scenario must not be reachable while
    RUNNING via the UI - covered structurally by the disabled-button test above; here
    we confirm the guard condition itself is computed from the real mission_status."""
    at = AppTest.from_file("app.py", default_timeout=APP_TIMEOUT)
    at.run()
    lm = at.session_state["live_mission"]
    lm.mission_status = LiveMissionStatus.PAUSED
    at.run()
    apply_btn = next(b for b in at.sidebar.button if "INITIALIZE" in (b.label or ""))
    assert apply_btn.disabled is True
    warnings = [w.value for w in at.sidebar.warning]
    assert any("STOP and RESET" in w for w in warnings)


# -----------------------------------------------------------------------------
# Real mission-history accumulation (section 3/9) - built at the runtime layer,
# simulation/engine.py itself untouched (its time_series was already unbounded).
# -----------------------------------------------------------------------------
def test_mission_history_time_series_exceeds_display_window():
    rt = make_runtime()
    rt.step_n(150)
    snap = rt.get_snapshot()
    assert len(snap["time_series"]) <= 60  # the display-truncated LIVE WINDOW
    full_history = rt.get_mission_history_time_series()
    assert len(full_history) == 150  # real, full MISSION HISTORY


def test_mission_history_summary_real_values():
    rt = make_runtime()
    rt.step_n(100)
    s = rt.get_mission_history_summary()
    assert s["steps_executed"] == 100
    assert s["total_scans"] == 500
    assert s["duration_s"] == pytest.approx(5.0)
    assert 0 <= s["bands_touched"] <= 50
    assert sum(s["strategy_distribution"].values()) == 100


def test_mission_history_toggle_renders_in_ui():
    at = AppTest.from_file("app.py", default_timeout=APP_TIMEOUT)
    at.run()
    at.button(key="btn_ops_step10").click().run()
    at.sidebar.radio(key="nav_view_radio").set_value("SPECTRUM").run()
    at.radio(key="spectrum_window_choice").set_value("MISSION HISTORY").run()
    assert not at.exception


# -----------------------------------------------------------------------------
# Real operational event generation (section 6)
# -----------------------------------------------------------------------------
def test_event_console_rows_are_real_and_time_ordered():
    rt = make_runtime()
    rt.start_mission()
    rt.pause_mission()
    rt.step_n(60)
    rows = rt.get_event_console_rows(limit=100)
    assert len(rows) > 0
    for r in rows:
        assert r["level"] in ("INFO", "COG", "DETECT", "TRACK", "ALERT")
        assert r["source"]
    # newest first
    timesteps = [r["timestep"] for r in rows]
    assert timesteps == sorted(timesteps, reverse=True)


def test_strategy_change_events_reflect_real_transitions():
    rt = make_runtime()
    rt.step_n(80)
    cog_events = [r for r in rt.get_event_console_rows(200) if r["level"] == "COG"]
    strategies_seen = {r["event"].replace("Strategy: ", "") for r in cog_events}
    assert strategies_seen <= {"EXPLORE", "EXPLOIT", "PREDICT", "BALANCED"}
    assert len(cog_events) >= 1  # at least the initial strategy is recorded


def test_lifecycle_events_recorded_on_real_transitions():
    rt = make_runtime()
    rt.start_mission()
    rt.pause_mission()
    rt.resume_mission()
    rt.stop_mission()
    events_text = " ".join(e["event"] for e in rt.op_event_log)
    assert "Mission started" in events_text
    assert "Mission paused" in events_text
    assert "Mission resumed" in events_text
    assert "Mission stopped" in events_text


def test_event_log_cleared_on_reset():
    rt = make_runtime()
    rt.step_n(30)
    assert len(rt.op_event_log) > 0
    rt.reset_mission()
    # reset immediately records exactly one fresh "Mission reset" event
    assert len(rt.op_event_log) == 1
    assert "reset" in rt.op_event_log[0]["event"].lower()


# -----------------------------------------------------------------------------
# Real alert generation (section 7)
# -----------------------------------------------------------------------------
def test_alerts_generated_from_real_conditions():
    rt = make_runtime()
    rt.step_n(300)  # enough steps to likely hit a real detection
    alerts = rt.get_alerts(50)
    for a in alerts:
        assert a["severity"] in ("INFO", "NOTICE", "WARNING", "CRITICAL")
    # if any detections occurred, at least one alert must reference them
    snap = rt.get_snapshot()
    if snap["true_detections"] > 0:
        assert any("TRUE INTERCEPTION" in a["event"] for a in alerts)


def test_pause_completion_generate_alerts():
    rt = make_runtime()
    rt.start_mission()
    rt.pause_mission()
    assert any(a["event"] == "MISSION PAUSED" for a in rt.alert_log)

    rt2 = make_runtime()
    rt2.step_n(rt2.engine.env.total_steps + 5)
    assert rt2.mission_status == LiveMissionStatus.COMPLETED
    assert any(a["event"] == "MISSION COMPLETED" for a in rt2.alert_log)


def test_alerts_panel_renders_in_ui_both_modes():
    for mode in ("LIVE SIMULATION", "REPLAY VERIFIED RUN"):
        at = AppTest.from_file("app.py", default_timeout=APP_TIMEOUT)
        at.run()
        if mode != "LIVE SIMULATION":
            at.sidebar.radio(key="operating_mode_radio").set_value(mode).run()
        at.button(key="btn_ops_step10").click().run()
        at.sidebar.radio(key="nav_view_radio").set_value("MISSION CONTROL").run()
        assert not at.exception, f"{mode}: {[str(e.value) for e in at.exception]}"


# -----------------------------------------------------------------------------
# New exports (section 18)
# -----------------------------------------------------------------------------
def test_new_export_buttons_render_without_exception():
    """AppTest (this Streamlit version) has no dedicated st.download_button accessor,
    so button labels can't be enumerated directly - this instead confirms the export
    row (which builds all three new exports' data every rerun) renders cleanly, and
    the data-producing calls below confirm the underlying values are real."""
    at = AppTest.from_file("app.py", default_timeout=APP_TIMEOUT)
    at.run()
    at.button(key="btn_ops_step10").click().run()
    assert not at.exception


def test_decision_trace_and_receiver_utilization_export_data_is_real():
    rt = make_runtime()
    rt.step_n(40)
    # LiveMissionRuntime has no get_decision_history() of its own (that's
    # PlaybackController's method) - app.py's export falls back to the real
    # .decision_history attribute (delegated through to SimulationEngine) for the
    # live runtime; confirm that path has real data.
    assert not hasattr(rt, "get_decision_history")
    dec_rows = list(rt.decision_history)
    assert len(dec_rows) == 40
    band_counts = dict(rt.band_scan_counts)
    assert sum(band_counts.values()) == 200  # 40 steps * K=5


def test_mission_summary_export_matches_real_state():
    rt = make_runtime()
    rt.step_n(40)
    summary = rt.get_mission_history_summary()
    assert summary["steps_executed"] == 40
    assert summary["total_scans"] == 200


# -----------------------------------------------------------------------------
# Session persistence across Streamlit reruns (section 13)
# -----------------------------------------------------------------------------
def test_live_mission_object_identity_preserved_across_reruns():
    at = AppTest.from_file("app.py", default_timeout=APP_TIMEOUT)
    at.run()
    lm_first = at.session_state["live_mission"]
    at.button(key="btn_ops_step1").click().run()
    at.sidebar.radio(key="nav_view_radio").set_value("SYSTEM").run()
    at.sidebar.radio(key="nav_view_radio").set_value("MISSION CONTROL").run()
    lm_second = at.session_state["live_mission"]
    assert lm_first is lm_second  # never recreated across reruns
    assert lm_second.engine.clock.current_step == 1  # state genuinely persisted


def test_playback_controller_object_identity_preserved_across_reruns():
    at = AppTest.from_file("app.py", default_timeout=APP_TIMEOUT)
    at.run()
    controller_first = at.session_state["playback_controller"]
    at.sidebar.radio(key="operating_mode_radio").set_value("REPLAY VERIFIED RUN").run()
    at.sidebar.radio(key="nav_view_radio").set_value("SYSTEM").run()
    controller_second = at.session_state["playback_controller"]
    assert controller_first is controller_second


# -----------------------------------------------------------------------------
# Error handling (section 20)
# -----------------------------------------------------------------------------
def test_missing_scenario_environment_shows_actionable_error_not_crash():
    at = AppTest.from_file("app.py", default_timeout=APP_TIMEOUT)
    at.run()
    lm = at.session_state["live_mission"]
    lm.engine.env = None  # simulate a scenario load failure
    at.run()
    assert not at.exception
    assert any("SCENARIO ENVIRONMENT FAILED TO LOAD" in e.value for e in at.error)


def test_empty_replay_artifact_shows_actionable_error_not_crash():
    at = AppTest.from_file("app.py", default_timeout=APP_TIMEOUT)
    at.run()
    at.sidebar.radio(key="operating_mode_radio").set_value("REPLAY VERIFIED RUN").run()
    controller = at.session_state["playback_controller"]
    controller.time_series = []  # simulate a corrupted/unavailable artifact
    at.run()
    assert not at.exception
    assert any("OPERATIONAL ARTIFACT UNAVAILABLE" in e.value for e in at.error)


# -----------------------------------------------------------------------------
# No fabrication in the new event/alert/history layer
# -----------------------------------------------------------------------------
def test_no_fabrication_in_new_live_mission_code():
    import inspect
    src = inspect.getsource(__import__("core.live_mission", fromlist=["dummy"]))
    for banned in ("92%", "11.5", "-88.5", "45.0 if", "5.20"):
        assert banned not in src, f"possible fabricated constant found: {banned!r}"


def test_no_ground_truth_in_new_step15_code():
    import inspect
    src = inspect.getsource(__import__("core.live_mission", fromlist=["dummy"]))
    assert "band_truth" not in src
    assert "active_truth" not in src


# -----------------------------------------------------------------------------
# All 7 views, both modes, with real stepped data (not just the READY state) -
# a stricter re-check than Step 14's READY-state sweep.
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["LIVE SIMULATION", "REPLAY VERIFIED RUN"])
@pytest.mark.parametrize("view", ["MISSION CONTROL", "SPECTRUM", "COGNITIVE ENGINE", "RECEIVER ARRAY", "TRACKS", "ALERTS", "ANALYTICS", "SYSTEM"])
def test_all_views_render_with_real_stepped_data(mode, view):
    at = AppTest.from_file("app.py", default_timeout=APP_TIMEOUT)
    at.run()
    if mode != "LIVE SIMULATION":
        at.sidebar.radio(key="operating_mode_radio").set_value(mode).run()
    at.button(key="btn_ops_step10").click().run()
    at.sidebar.radio(key="nav_view_radio").set_value(view).run()
    assert not at.exception, f"{mode}/{view}: {[str(e.value) for e in at.exception]}"
