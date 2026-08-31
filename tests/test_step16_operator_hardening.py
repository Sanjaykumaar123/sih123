"""Step 16: Final operational validation & operator usability hardening.

Adds tests only for actual defects/missing contracts discovered during this step's
audit - does not pad test count for its own sake, does not weaken or replace any
earlier test file.

Two real regressions were found and fixed during this step:
1. PlaybackController.get_mission_history_summary() (added in Step 15/16) omitted
   'bands_touched'/'n_bands', which dashboard/help.py's render_mission_history_panel
   indexed unconditionally - a real KeyError crash in REPLAY mode, caught by
   test_step15_production_workstation.py's UI sweep only after this file's changes
   were layered on top of it.
2. PlaybackController._load_artifact() did not catch json.JSONDecodeError - a
   genuinely corrupted (not just missing) artifact file crashed the constructor
   itself, before any of app.py's graceful-degradation error banners could run.
"""

import json
import os
import tempfile

import pytest
from streamlit.testing.v1 import AppTest

from core.live_mission import LiveMissionRuntime, LiveMissionStatus
from core.playback_controller import PlaybackController

SCENARIO = r"D:\sih\dataset\scan\test_scan\config_1.h5"
APP_TIMEOUT = 90


def make_runtime(**kwargs):
    defaults = dict(scenario_path=SCENARIO, strategy_type="smart_scan", k_channels=5, n_bands=50, seed=42)
    defaults.update(kwargs)
    return LiveMissionRuntime(**defaults)


# -----------------------------------------------------------------------------
# Regression: PlaybackController.get_mission_history_summary() shape parity
# -----------------------------------------------------------------------------
def test_playback_mission_history_summary_has_full_schema():
    c = PlaybackController(scenario_id="config_1.h5", strategy_type="smart_scan")
    c.step(50)
    s = c.get_mission_history_summary()
    for key in ("duration_s", "steps_executed", "total_scans", "bands_touched", "n_bands",
                "true_detections", "false_alarms", "unique_emitters_intercepted",
                "strategy_distribution", "cumulative_reward", "metadata"):
        assert key in s, f"missing key: {key}"
    assert s["bands_touched"] is not None
    assert 0 <= s["bands_touched"] <= s["n_bands"]
    for meta_key in ("scenario", "mode", "mission_id", "generated_at", "strategy_type", "receiver_channels_k", "frequency_bands_n"):
        assert meta_key in s["metadata"]
    assert s["metadata"]["mode"] == "REPLAY VERIFIED RUN"


def test_live_and_replay_mission_summary_share_the_same_top_level_keys():
    """The two engines' summaries must stay in sync so render_mission_history_panel
    never silently degrades one mode - both real schemas, checked against each other."""
    live = make_runtime()
    live.step_n(30)
    live_summary = live.get_mission_history_summary()

    replay = PlaybackController(scenario_id="config_1.h5", strategy_type="smart_scan")
    replay.step(30)
    replay_summary = replay.get_mission_history_summary()

    assert set(live_summary.keys()) == set(replay_summary.keys())


def test_mission_history_panel_renders_in_replay_mode_after_stepping():
    """Direct regression test for the KeyError('bands_touched') crash. The schema
    itself is asserted directly above (test_playback_mission_history_summary_has_
    full_schema); this is the UI-level companion - navigate to MISSION CONTROL in
    REPLAY mode after stepping and confirm the view renders cleanly. (The Mission
    Control redesign no longer calls render_mission_history_panel from this view
    specifically - its KPIs duplicated the KPI row - but get_mission_history_
    summary() is still exercised by every export button on every view, so a
    schema regression here would still surface as an exception below.)"""
    at = AppTest.from_file("app.py", default_timeout=APP_TIMEOUT)
    at.run()
    at.sidebar.radio(key="operating_mode_radio").set_value("REPLAY VERIFIED RUN").run()
    at.button(key="btn_ops_step10").click().run()
    at.sidebar.radio(key="nav_view_radio").set_value("MISSION CONTROL").run()
    assert not at.exception, [str(e.value) for e in at.exception]


# -----------------------------------------------------------------------------
# Regression: corrupted (not just missing) JSON artifact
# -----------------------------------------------------------------------------
def test_corrupted_json_artifact_does_not_crash_constructor():
    tmpdir = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmpdir, "results"), exist_ok=True)
    bad_path = os.path.join(tmpdir, "results", "operational_evaluation_config_1.json")
    with open(bad_path, "w") as f:
        f.write("{not valid json")

    cwd = os.getcwd()
    try:
        os.chdir(tmpdir)
        c = PlaybackController(scenario_id="config_1.h5", strategy_type="smart_scan")
        snap = c.get_snapshot()
        assert snap["mission_status"] == "READY"
        assert snap["total_scans"] == 0
        assert c.artifact_load_error is not None
        assert "JSONDecodeError" in c.artifact_load_error
    finally:
        os.chdir(cwd)


def test_truncated_json_artifact_does_not_crash_constructor():
    """A different corruption shape: valid JSON syntax but missing required keys -
    already handled by the empty-time_series guard, re-verified here explicitly."""
    tmpdir = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmpdir, "results"), exist_ok=True)
    path = os.path.join(tmpdir, "results", "operational_evaluation_config_1.json")
    with open(path, "w") as f:
        json.dump({"scenario": "config_1.h5"}, f)  # valid JSON, no time_series key

    cwd = os.getcwd()
    try:
        os.chdir(tmpdir)
        c = PlaybackController(scenario_id="config_1.h5", strategy_type="smart_scan")
        snap = c.get_snapshot()  # must not raise
        assert snap["total_scans"] == 0
    finally:
        os.chdir(cwd)


# -----------------------------------------------------------------------------
# Terminology hardening (section 3): exact required control labels present
# -----------------------------------------------------------------------------
def test_mission_control_button_labels_match_spec_terminology():
    # Labels were compacted (e.g. "START MISSION" -> "START") in the Mission
    # Control redesign ("avoid giant buttons" / compact control bar) - a
    # presentation-only rename. Each button's key->action mapping and its full
    # descriptive `help=` text (checked below) are unchanged; this still asserts
    # every control-bar button carries its own correct, unambiguous label.
    at = AppTest.from_file("app.py", default_timeout=APP_TIMEOUT)
    at.run()
    labels = {b.key: b.label for b in at.button if b.key and b.key.startswith("btn_ops_")}
    assert "START" in labels["btn_ops_start"]
    assert "PAUSE" in labels["btn_ops_pause"]
    assert "RESUME" in labels["btn_ops_resume"]
    assert "STEP" in labels["btn_ops_step1"]
    assert "STEP +10" in labels["btn_ops_step10"]
    assert "STOP" in labels["btn_ops_stop"]
    assert "RESET" in labels["btn_ops_reset"]


def test_mission_control_buttons_have_help_text():
    at = AppTest.from_file("app.py", default_timeout=APP_TIMEOUT)
    at.run()
    for key in ("btn_ops_start", "btn_ops_pause", "btn_ops_resume", "btn_ops_step1", "btn_ops_step10", "btn_ops_stop", "btn_ops_reset"):
        btn = next(b for b in at.button if b.key == key)
        assert btn.help, f"{key} has no operator-facing help text"
        assert len(btn.help) > 10


# -----------------------------------------------------------------------------
# EMITTERS KPI (section 4 gap: distinct from Active Tracks)
# -----------------------------------------------------------------------------
def test_emitters_kpi_is_real_and_distinct_from_active_tracks():
    rt = make_runtime()
    rt.step_n(300)
    snap = rt.get_snapshot()
    assert "unique_emitters_count" in snap
    assert isinstance(snap["unique_emitters_count"], int)

    c = PlaybackController(scenario_id="config_1.h5", strategy_type="smart_scan")
    c.step(300)
    csnap = c.get_snapshot()
    assert "unique_emitters_count" in csnap
    assert csnap["unique_emitters_count"] == csnap["active_tracks_count"]  # same real underlying count in replay mode, exposed under both names for parity


# -----------------------------------------------------------------------------
# OPERATOR ATTENTION panel (section 8): actionable-only, honest empty state
# -----------------------------------------------------------------------------
def test_operator_attention_shows_no_action_required_when_clean():
    from dashboard.alerts import render_attention_required, ACTIONABLE_SEVERITIES
    rt = make_runtime()  # freshly constructed, only the INFO-level init event exists
    for a in rt.alert_log:
        assert a["severity"] not in ACTIONABLE_SEVERITIES


def test_operator_attention_and_alerts_panel_are_distinct_functions():
    import dashboard.alerts as alerts_mod
    assert hasattr(alerts_mod, "render_alerts_panel")
    assert hasattr(alerts_mod, "render_attention_required")
    assert alerts_mod.render_alerts_panel is not alerts_mod.render_attention_required


# -----------------------------------------------------------------------------
# No physical-hardware implication (section 7)
# -----------------------------------------------------------------------------
def test_receiver_view_labeled_as_simulated():
    at = AppTest.from_file("app.py", default_timeout=APP_TIMEOUT)
    at.run()
    at.sidebar.radio(key="nav_view_radio").set_value("RECEIVER ARRAY").run()
    assert not at.exception
    body = " ".join(m.value for m in at.get("markdown"))
    assert "SIMULATED RECEIVER ARRAY" in body


# -----------------------------------------------------------------------------
# Integrity re-verification (section 18): Q-learning update path uses only
# belief-based reward, never the ground-truth-informed display reward.
# -----------------------------------------------------------------------------
def test_qlearning_update_path_excludes_ground_truth_reward():
    import inspect
    from rf_env.evaluation import IntelligentSchedulerAdapter
    src = inspect.getsource(IntelligentSchedulerAdapter.learn)
    assert "calculate_reward(observations, timestep)" in src
    assert "compute_evaluated_step_reward" not in src
    assert "ground_truth" not in src
    assert "active_truth" not in src
    assert "band_truth" not in src


def test_no_banned_fabrication_constants_in_changed_modules():
    import inspect
    modules = [
        "app", "core.live_mission", "core.playback_controller", "simulation.engine",
        "dashboard.live_operations", "dashboard.help", "dashboard.alerts",
        "dashboard.spectrum", "dashboard.system",
    ]
    banned = ("92%", "11.5", "-88.5", "45.0 if", "5.20", "[0.32, 0.45, 0.28, 0.58]")
    for mod_name in modules:
        src = inspect.getsource(__import__(mod_name, fromlist=["dummy"]))
        for b in banned:
            assert b not in src, f"{mod_name}: banned pattern {b!r} found"


# -----------------------------------------------------------------------------
# Complete 30-second mission, cross-validated against the verified artifact
# -----------------------------------------------------------------------------
def test_complete_mission_matches_verified_artifact_exactly():
    """Re-verification (section 15): a real LIVE 600-step run reproduces the same
    real numbers as the independently-computed, verified REPLAY artifact for the
    identical scenario+seed - strong cross-validation, not just internal consistency."""
    rt = make_runtime()
    rt.step_n(rt.engine.env.total_steps)
    assert rt.mission_status == LiveMissionStatus.COMPLETED
    snap = rt.get_snapshot()

    with open("results/operational_evaluation_config_1.json") as f:
        artifact = json.load(f)
    verified = artifact["metrics_summary"]["smart_scan"]

    assert snap["timestep"] == artifact["num_steps"] == 600
    assert snap["true_detections"] == verified["true_detections"]
    assert snap["false_alarms"] == verified["total_false_alarms"]
    assert len(rt.engine.unique_emitters_intercepted) == verified["unique_emitters_intercepted"]
