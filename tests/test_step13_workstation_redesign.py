"""Step 13: Production-Grade Operator Workstation UI/UX redesign — verification tests.

Covers: navigation across all 8 views (Step 16 added RECEIVER ARRAY's rename from
RECEIVERS and the new ALERTS view), mission lifecycle controls, step progression,
receiver/spectrum/strategy/metric updates, scenario switching, operator help content,
absence of fabricated telemetry, and the K=5-of-N=50 visibility constraint. Does not
touch or re-verify the underlying Smart Scan algorithm, datasets, or operational
evaluation JSON artifacts - those are exercised unchanged by tests/test_stage*.py.
"""

import pytest
from streamlit.testing.v1 import AppTest

from core.playback_controller import PlaybackController
from dashboard.help import GLOSSARY

APP_TIMEOUT = 60
NAV_VIEWS = ["MISSION CONTROL", "SPECTRUM", "COGNITIVE ENGINE", "RECEIVER ARRAY", "TRACKS", "ALERTS", "ANALYTICS", "SYSTEM"]


# -----------------------------------------------------------------------------
# PlaybackController: mission lifecycle (unit-level - avoids the real auto-advance
# loop that only terminates in a live browser, not a single scripted run)
# -----------------------------------------------------------------------------
def test_mission_lifecycle_start_pause_resume_step_reset():
    c = PlaybackController(scenario_id="config_1.h5", strategy_type="smart_scan")
    assert c.get_snapshot()["mission_status"] == "READY"

    c.start()
    assert c.running is True
    assert c.get_snapshot()["mission_status"] == "RUNNING"

    c.pause()
    assert c.paused is True
    assert c.get_snapshot()["mission_status"] == "PAUSED"

    c.resume()
    assert c.running is True

    c.pause()
    step_before = c.current_step
    c.step(num_steps=1)
    assert c.current_step == step_before + 1

    c.step(num_steps=10)
    assert c.current_step == step_before + 11

    c.reset()
    assert c.current_step == 0
    assert c.running is False
    assert c.get_snapshot()["mission_status"] == "READY"


def test_scenario_switch_reloads_artifact():
    c = PlaybackController(scenario_id="config_1.h5", strategy_type="smart_scan")
    c.step(20)
    c.set_scenario(scenario_id="config_2.h5", strategy_type="smart_scan")
    assert c.scenario_name == "config_2.h5"
    assert c.current_step == 0  # set_scenario resets playback


# -----------------------------------------------------------------------------
# No fabricated telemetry (this replay artifact does not record per-pulse
# SNR/amplitude/AoA/pulse-width, per-band Q-values, or a full 50-band score table)
# -----------------------------------------------------------------------------
def test_no_fabricated_channel_telemetry():
    c = PlaybackController(scenario_id="config_1.h5", strategy_type="smart_scan")
    c.step(50)
    snap = c.get_snapshot()
    for ch in snap["channel_telemetry"]:
        assert ch["snr_db"] is None
        assert ch["amplitude_dbm"] is None
        assert ch["aoa_deg"] is None
        assert ch["pulse_width_us"] is None


def test_no_fabricated_q_values():
    c = PlaybackController(scenario_id="config_1.h5", strategy_type="smart_scan")
    c.step(50)
    snap = c.get_snapshot()
    assert snap["meta_q_values"] is None
    assert "not exposed" in snap["q_value_note"].lower()


def test_no_fabricated_band_score_components():
    c = PlaybackController(scenario_id="config_1.h5", strategy_type="smart_scan")
    c.step(50)
    snap = c.get_snapshot()
    assert snap["band_scores_available_for_all_bands"] is False
    for row in snap["band_scores_table"]:
        assert row["P(Active)"] is None
        assert row["Uncertainty"] is None
        assert row["Temporal Score"] is None
        # Final Score IS real (logged per-step in the artifact for selected bands)
        assert row["Final Score"] is None or isinstance(row["Final Score"], float)


def test_no_fabricated_track_confidence():
    """Replay-mode 'tracks' must be real emitter_interceptions rows, never a
    made-up per-track confidence percentage."""
    c = PlaybackController(scenario_id="config_1.h5", strategy_type="smart_scan")
    c.step(50)
    for rec in c.get_snapshot()["tracks"]:
        assert "Confidence" not in rec
        assert "emitter_id" in rec


def test_open_loop_strategy_not_mislabeled_balanced():
    """Open-loop has no meta-strategy at all; it must not silently default to BALANCED."""
    c = PlaybackController(scenario_id="config_1.h5", strategy_type="open_loop")
    c.step(20)
    snap = c.get_snapshot()
    assert snap["current_strategy"] == "SEQUENTIAL_SWEEP"
    assert all(row["Final Score"] is None for row in snap["band_scores_table"])


# -----------------------------------------------------------------------------
# K=5-of-N=50 constraint held through the replay-driven UI layer
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("step", [0, 1, 50, 300, 599])
def test_k_of_n_visibility_constraint(step):
    c = PlaybackController(scenario_id="config_1.h5", strategy_type="smart_scan")
    c.step(step)
    snap = c.get_snapshot()
    assert snap["k_channels"] == 5
    assert snap["n_bands"] == 50
    assert len(snap["selected_bands"]) == 5
    assert len(snap["channel_telemetry"]) == 5


# -----------------------------------------------------------------------------
# Real per-step derived data (decision history / reward series / strategy mix)
# -----------------------------------------------------------------------------
def test_decision_history_matches_replay_data():
    c = PlaybackController(scenario_id="config_1.h5", strategy_type="smart_scan")
    c.step(30)
    hist = c.get_decision_history(window=10)
    assert len(hist) == 10
    assert hist[0]["Step"] == 30  # most recent first
    assert hist[-1]["Step"] == 21
    for row in hist:
        assert len(row["Selected Bands"].split()) == 5


def test_reward_timeseries_length_matches_current_step():
    c = PlaybackController(scenario_id="config_1.h5", strategy_type="smart_scan")
    c.step(99)
    series = c.get_reward_timeseries()
    assert len(series) == 100  # steps 0..99 inclusive


def test_strategy_distribution_sums_to_steps_taken():
    c = PlaybackController(scenario_id="config_1.h5", strategy_type="smart_scan")
    c.step(49)
    dist = c.get_strategy_distribution()
    assert sum(dist.values()) == 50  # steps 0..49 inclusive
    assert set(dist.keys()) <= {"EXPLORE", "EXPLOIT", "PREDICT", "BALANCED"}


# -----------------------------------------------------------------------------
# Full UI: navigation across all 7 operational views, no exceptions
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("view", NAV_VIEWS)
def test_nav_view_renders_without_exception(view):
    at = AppTest.from_file("app.py", default_timeout=APP_TIMEOUT)
    at.run()
    at.sidebar.radio(key="nav_view_radio").set_value(view).run()
    assert not at.exception, [str(e.value) for e in at.exception]


def test_initial_state_shows_ready_guidance():
    at = AppTest.from_file("app.py", default_timeout=APP_TIMEOUT)
    at.run()
    assert not at.exception
    body_text = " ".join(m.value for m in at.markdown) if hasattr(at, "markdown") else ""
    assert "SYSTEM READY" in body_text or any("SYSTEM READY" in str(el.value) for el in at.get("markdown"))


def test_step_button_advances_mission_and_updates_receivers():
    at = AppTest.from_file("app.py", default_timeout=APP_TIMEOUT)
    at.run()
    # Step 14 makes LIVE SIMULATION the default active mode; switch explicitly to
    # REPLAY VERIFIED RUN so this test exercises PlaybackController as originally
    # intended, regardless of whichever mode is the app's current default.
    at.sidebar.radio(key="operating_mode_radio").set_value("REPLAY VERIFIED RUN").run()
    controller = at.session_state["playback_controller"]
    before = controller.current_step
    at.button(key="btn_ops_step1").click().run()
    assert not at.exception
    assert controller.current_step == before + 1
    snap = controller.get_snapshot()
    assert len(snap["channel_telemetry"]) == 5


def test_scenario_switch_via_ui_updates_controller():
    at = AppTest.from_file("app.py", default_timeout=APP_TIMEOUT)
    at.run()
    at.sidebar.radio(key="operating_mode_radio").set_value("REPLAY VERIFIED RUN").run()
    at.sidebar.selectbox(key="sb_scen_select").set_value("config_3.h5").run()
    apply_btn = next(b for b in at.sidebar.button if "INITIALIZE" in (b.label or ""))
    apply_btn.click().run()
    assert not at.exception
    assert at.session_state["playback_controller"].scenario_name == "config_3.h5"


def test_operator_help_content_present_in_glossary():
    """Section 11: every concept the operator sees on-screen has a plain-language
    explanation available."""
    for term in ("Activity", "Uncertainty", "Temporal", "Exploration", "Exploitation", "Prediction", "Balanced", "Reward"):
        assert term in GLOSSARY
        assert len(GLOSSARY[term]) > 10


def test_reset_button_returns_to_ready():
    at = AppTest.from_file("app.py", default_timeout=APP_TIMEOUT)
    at.run()
    at.button(key="btn_ops_step10").click().run()
    at.button(key="btn_ops_reset").click().run()
    assert not at.exception
    controller = at.session_state["playback_controller"]
    assert controller.current_step == 0
    assert controller.get_snapshot()["mission_status"] == "READY"
