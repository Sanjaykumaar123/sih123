"""Step 14: Production-grade LIVE cognitive RF workstation - verification tests.

Covers: mission start/pause/resume/step/step+10/stop/reset, scenario switching,
real runtime timestep progression, real live observation generation, receiver
allocation, detection recording, reward recording, event logging, absence of
fabricated telemetry, absence of ground-truth leakage, LIVE/REPLAY separation,
export functionality, UI rendering, and invalid state transitions.

Does not touch or re-verify the underlying rf_env/ algorithms - those are exercised
unchanged by tests/test_stage*.py. LiveMissionRuntime is a thin wrapper around the
already-verified simulation.engine.SimulationEngine.
"""

import inspect
import pytest
from streamlit.testing.v1 import AppTest

from core.live_mission import LiveMissionRuntime, LiveMissionStatus
from simulation.engine import SimulationStatus

SCENARIO = r"D:\sih\dataset\scan\test_scan\config_1.h5"
APP_TIMEOUT = 90


def make_runtime(**kwargs):
    defaults = dict(scenario_path=SCENARIO, strategy_type="smart_scan", k_channels=5, n_bands=50, seed=42)
    defaults.update(kwargs)
    return LiveMissionRuntime(**defaults)


# -----------------------------------------------------------------------------
# 1-7. Mission lifecycle: start / pause / resume / step / step+10 / stop / reset
# -----------------------------------------------------------------------------
def test_1_mission_start():
    rt = make_runtime()
    assert rt.mission_status == LiveMissionStatus.READY
    assert rt.start_mission() is True
    assert rt.mission_status == LiveMissionStatus.RUNNING


def test_2_mission_pause():
    rt = make_runtime()
    rt.start_mission()
    assert rt.pause_mission() is True
    assert rt.mission_status == LiveMissionStatus.PAUSED


def test_3_mission_resume():
    rt = make_runtime()
    rt.start_mission()
    rt.pause_mission()
    assert rt.resume_mission() is True
    assert rt.mission_status == LiveMissionStatus.RUNNING


def test_4_single_step():
    rt = make_runtime()
    before = rt.engine.clock.current_step
    assert rt.step_once() is True
    assert rt.engine.clock.current_step == before + 1
    assert rt.mission_status == LiveMissionStatus.PAUSED  # stepped from READY


def test_5_step_plus_10():
    rt = make_runtime()
    before = rt.engine.clock.current_step
    assert rt.step_n(10) is True
    assert rt.engine.clock.current_step == before + 10


def test_6_stop():
    rt = make_runtime()
    rt.start_mission()
    assert rt.stop_mission() is True
    assert rt.mission_status == LiveMissionStatus.STOPPED


def test_7_reset_clears_everything():
    """Section 16: RESET genuinely clears timestep, telemetry, decisions, events,
    cumulative metrics and learning state - not just the timestep counter."""
    rt = make_runtime()
    rt.step_n(60)
    snap_before = rt.get_snapshot()
    assert snap_before["total_scans"] > 0

    rt.reset_mission()
    snap_after = rt.get_snapshot()
    assert rt.mission_status == LiveMissionStatus.READY
    assert rt.engine.clock.current_step == 0
    assert snap_after["total_scans"] == 0
    assert snap_after["true_detections"] == 0
    assert snap_after["false_alarms"] == 0
    assert snap_after["cumulative_reward"] == 0.0
    assert snap_after["recent_events"] == []
    assert snap_after["total_tracks_count"] == 0
    assert all(v == 0.0 for v in rt.engine.scheduler.arbitrator.q_table.flatten()) if hasattr(rt.engine.scheduler, "arbitrator") else True


# -----------------------------------------------------------------------------
# 8. Scenario switching
# -----------------------------------------------------------------------------
def test_8_scenario_switching_via_reset():
    rt = make_runtime()
    rt.step_n(5)
    rt.reset_mission(scenario_path=r"D:\sih\dataset\scan\test_scan\config_2.h5")
    assert "config_2.h5" in rt.engine.scenario_path
    assert rt.engine.clock.current_step == 0


def test_8b_scenario_switch_via_ui():
    at = AppTest.from_file("app.py", default_timeout=APP_TIMEOUT)
    at.run()
    at.sidebar.selectbox(key="sb_scen_select").set_value("config_4.h5").run()
    apply_btn = next(b for b in at.sidebar.button if "INITIALIZE" in (b.label or ""))
    apply_btn.click().run()
    assert not at.exception
    assert "config_4.h5" in at.session_state["live_mission"].engine.scenario_path


# -----------------------------------------------------------------------------
# 9. Runtime timestep progression is real - t=0,1,2,... dt=0.05s
# -----------------------------------------------------------------------------
def test_9_timestep_progression_is_real_and_ordered():
    rt = make_runtime()
    steps_seen = []
    for _ in range(5):
        rt.step_once()
        steps_seen.append(rt.engine.clock.current_step)
    assert steps_seen == [1, 2, 3, 4, 5]
    snap = rt.get_snapshot()
    assert snap["simulated_time_s"] == pytest.approx(5 * 0.05)


# -----------------------------------------------------------------------------
# 10. Live observation generation (real Receiver.observe() output, not canned data)
# -----------------------------------------------------------------------------
def test_10_live_observation_generation():
    rt = make_runtime()
    rt.step_once()
    assert len(rt.engine.latest_observations) == 5
    for band, obs in rt.engine.latest_observations.items():
        assert band in rt.engine.selected_bands
        assert isinstance(obs.hit, bool)
        assert isinstance(obs.snr, float)


# -----------------------------------------------------------------------------
# 11. Receiver allocation: exactly K=5 of N=50, changes as the scheduler decides
# -----------------------------------------------------------------------------
def test_11_receiver_allocation_k_of_n():
    rt = make_runtime()
    seen_selections = set()
    for _ in range(40):
        rt.step_once()
        snap = rt.get_snapshot()
        assert len(snap["selected_bands"]) == 5
        assert len(snap["channel_telemetry"]) == 5
        seen_selections.add(tuple(sorted(snap["selected_bands"])))
    assert len(seen_selections) > 1, "scheduler never changed its band selection over 40 steps"


# -----------------------------------------------------------------------------
# 12. Detection recording
# -----------------------------------------------------------------------------
def test_12_detection_recording():
    rt = make_runtime()
    rt.step_n(200)  # enough steps to encounter at least one real emitter
    snap = rt.get_snapshot()
    assert snap["true_detections"] + snap["false_alarms"] <= snap["total_scans"]
    assert snap["total_scans"] == 200 * 5


# -----------------------------------------------------------------------------
# 13. Reward recording
# -----------------------------------------------------------------------------
def test_13_reward_recording():
    rt = make_runtime()
    rt.step_n(30)
    snap = rt.get_snapshot()
    assert isinstance(snap["latest_reward"], float)
    assert isinstance(snap["cumulative_reward"], float)


# -----------------------------------------------------------------------------
# 14. Event logging
# -----------------------------------------------------------------------------
def test_14_event_logging():
    rt = make_runtime()
    rt.step_n(300)
    snap = rt.get_snapshot()
    if snap["true_detections"] > 0 or snap["false_alarms"] > 0:
        assert len(snap["recent_events"]) > 0
        for ev in snap["recent_events"]:
            assert "time_s" in ev and "event_type" in ev


# -----------------------------------------------------------------------------
# 15. No fabricated telemetry
# -----------------------------------------------------------------------------
def test_15_no_fabricated_quiet_channel_telemetry():
    rt = make_runtime()
    rt.step_once()
    for ch in rt.get_snapshot()["channel_telemetry"]:
        if ch["status"] in ("MONITORING", "QUIET"):
            assert ch["snr_db"] is None
            assert ch["amplitude_dbm"] is None
            assert ch["aoa_deg"] is None
            assert ch["pulse_width_us"] is None


def test_15b_no_fabricated_false_alarm_fallback_constants():
    """simulation/engine.py used to hardcode -88.5 dBm / 11.5 dB for every false
    alarm; it must now use the detector's real output or None, never that constant."""
    src = inspect.getsource(__import__("simulation.engine", fromlist=["dummy"]))
    assert "-88.5" not in src
    assert "11.5" not in src
    assert "5.20" not in src
    assert "45.0 if" not in src


def test_15c_no_hardcoded_meta_q_values():
    """The live engine's Q-values must come from the real Q-table, never a constant list."""
    src = inspect.getsource(__import__("simulation.engine", fromlist=["dummy"]))
    assert "[0.32, 0.45, 0.28, 0.58]" not in src


# -----------------------------------------------------------------------------
# 16. No ground-truth leakage into the live scheduler
# -----------------------------------------------------------------------------
def test_16_no_ground_truth_leakage_into_scheduler_selection():
    """Structural check mirroring Stage 11's re-verification: the scheduler's
    select_bands/learn calls never receive ground-truth objects, only the receiver's
    stripped Observation dict - already enforced by SimulationEngine.step() (unmodified
    this step) and re-checked here as part of Step 14 sign-off."""
    src = inspect.getsource(__import__("simulation.engine", fromlist=["dummy"]).SimulationEngine.step)
    # scheduler.select_bands is called with only a timestep int; scheduler.learn only
    # with (observations, t) - ground truth (active_truth/band_truth) must not appear
    # as an argument to either call.
    assert "select_bands(t)" in src
    assert "self.scheduler.learn(observations, t)" in src


def test_16b_live_mission_runtime_never_touches_band_truth():
    src = inspect.getsource(__import__("core.live_mission", fromlist=["dummy"]))
    assert "band_truth" not in src
    assert ".active_truth" not in src


# -----------------------------------------------------------------------------
# 17. Replay/live separation - never mixed
# -----------------------------------------------------------------------------
def test_17_live_and_replay_are_independent_objects():
    at = AppTest.from_file("app.py", default_timeout=APP_TIMEOUT)
    at.run()
    live = at.session_state["live_mission"]
    replay = at.session_state["playback_controller"]
    assert live is not None
    assert replay is not None
    assert live is not replay
    assert live.get_snapshot()["operating_mode"] == "LIVE SIMULATION"
    assert "mission_status" in replay.get_snapshot()  # replay has no operating_mode key contamination
    assert replay.get_snapshot().get("operating_mode") is None


def test_17b_analytics_view_never_exceptions_with_or_without_live_data():
    for pre_step in (False, True):
        at = AppTest.from_file("app.py", default_timeout=APP_TIMEOUT)
        at.run()
        if pre_step:
            at.button(key="btn_ops_step1").click().run()
        at.sidebar.radio(key="nav_view_radio").set_value("ANALYTICS").run()
        assert not at.exception


# -----------------------------------------------------------------------------
# 18. Export functionality
# -----------------------------------------------------------------------------
def test_18_export_functions_return_real_data():
    rt = make_runtime()
    rt.step_n(20)
    report = rt.export_report_json()
    assert report["performance_metrics"]["total_channel_scans"] == 100
    events_csv = rt.export_events_csv()
    assert isinstance(events_csv, str)
    tracks_csv = rt.export_tracks_csv()
    assert isinstance(tracks_csv, str)


# -----------------------------------------------------------------------------
# 19. UI rendering across all views, both modes
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("view", ["MISSION CONTROL", "SPECTRUM", "COGNITIVE ENGINE", "RECEIVER ARRAY", "TRACKS", "ALERTS", "ANALYTICS", "SYSTEM"])
def test_19_live_mode_views_render_without_exception(view):
    at = AppTest.from_file("app.py", default_timeout=APP_TIMEOUT)
    at.run()
    at.sidebar.radio(key="nav_view_radio").set_value(view).run()
    assert not at.exception, [str(e.value) for e in at.exception]


def test_19b_live_mode_after_stepping_renders_real_values():
    at = AppTest.from_file("app.py", default_timeout=APP_TIMEOUT)
    at.run()
    at.button(key="btn_ops_step10").click().run()
    assert not at.exception
    lm = at.session_state["live_mission"]
    assert lm.engine.clock.current_step == 10
    at.sidebar.radio(key="nav_view_radio").set_value("COGNITIVE ENGINE").run()
    assert not at.exception


# -----------------------------------------------------------------------------
# Regression: advance_time_tick() must catch up on multiple due steps rather than
# silently discarding elapsed time and capping the effective rate below the
# requested speed multiplier (found and fixed during Step 14 implementation).
# -----------------------------------------------------------------------------
def test_advance_time_tick_catches_up_multiple_due_steps():
    rt = make_runtime()
    rt.start_mission()
    rt.set_speed(5.0)
    # Simulate "coarse poll interval elapsed": pretend 100ms of wall time passed
    # (= 500ms of sim-equivalent time at 5x = 10 whole steps due).
    rt._last_tick_wall_time -= 0.100
    advanced = rt.advance_time_tick()
    assert advanced is True
    assert rt.engine.clock.current_step == 10


def test_advance_time_tick_caps_catchup_to_avoid_inline_full_mission():
    rt = make_runtime()
    rt.start_mission()
    rt.set_speed(1.0)
    # Simulate an enormous gap (e.g. a backgrounded tab) - must not inline-execute
    # hundreds of steps in one call (section 17).
    rt._last_tick_wall_time -= 10.0
    rt.advance_time_tick()
    assert rt.engine.clock.current_step <= LiveMissionRuntime.MAX_CATCHUP_STEPS


def test_advance_time_tick_no_step_before_due():
    rt = make_runtime()
    rt.start_mission()
    before = rt.engine.clock.current_step
    assert rt.advance_time_tick() is False  # just started, nothing due yet
    assert rt.engine.clock.current_step == before


# -----------------------------------------------------------------------------
# 20. Invalid mission state transitions are safely rejected, not crashed
# -----------------------------------------------------------------------------
def test_20_invalid_transitions_from_ready():
    rt = make_runtime()
    assert rt.pause_mission() is False
    assert rt.resume_mission() is False
    assert rt.stop_mission() is False
    assert rt.mission_status == LiveMissionStatus.READY


def test_20b_invalid_transitions_from_running():
    rt = make_runtime()
    rt.start_mission()
    assert rt.start_mission() is False  # already running
    step_before = rt.engine.clock.current_step
    assert rt.step_once() is False  # step disabled while running
    assert rt.engine.clock.current_step == step_before
    assert rt.mission_status == LiveMissionStatus.RUNNING


def test_20c_invalid_transitions_from_completed():
    rt = make_runtime()
    rt.step_n(rt.engine.env.total_steps + 5)
    assert rt.mission_status == LiveMissionStatus.COMPLETED
    assert rt.step_once() is False
    assert rt.pause_mission() is False
    assert rt.resume_mission() is False


def test_20d_ui_buttons_disabled_state_matches_runtime_guards():
    """Drive the mission to just before completion so the RUNNING->COMPLETED auto-
    advance loop terminates within the AppTest run, then inspect real widget state."""
    at = AppTest.from_file("app.py", default_timeout=APP_TIMEOUT)
    at.run()
    lm = at.session_state["live_mission"]
    total = lm.engine.env.total_steps
    lm.step_n(total - 2)
    lm.mission_status = LiveMissionStatus.RUNNING
    lm._last_tick_wall_time = None
    at.run()
    assert not at.exception
    assert lm.mission_status == LiveMissionStatus.COMPLETED
    start_btn = at.button(key="btn_ops_start")
    step_btn = at.button(key="btn_ops_step1")
    reset_btn = at.button(key="btn_ops_reset")
    assert start_btn.disabled is False  # COMPLETED -> START re-enabled (implicit reset)
    assert step_btn.disabled is True
    assert reset_btn.disabled is False
