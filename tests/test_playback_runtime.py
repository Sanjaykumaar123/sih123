"""Comprehensive tests for PlaybackController and operational replay runtime."""

import json
import os
import pytest
from core.playback_controller import PlaybackController
from core.state import EngineStatus


def test_1_initial_state_step_zero():
    """Verify initial state has current_step == 0 and status == READY."""
    ctrl = PlaybackController("config_1.h5", speed=1.0, strategy_type="smart_scan")
    assert ctrl.current_step == 0
    assert ctrl.running is False
    assert ctrl.paused is False
    assert ctrl.status == EngineStatus.READY
    assert ctrl.total_timesteps == 600

    snap = ctrl.get_snapshot()
    assert snap["timestep"] == 0
    assert snap["total_scans"] == 5
    assert snap["true_detections"] == 0


def test_2_start_sets_running_true():
    """Verify START sets running == True and status == RUNNING."""
    ctrl = PlaybackController("config_1.h5", speed=1.0, strategy_type="smart_scan")
    ctrl.start()
    assert ctrl.running is True
    assert ctrl.paused is False
    assert ctrl.mission_started is True
    assert ctrl.status == EngineStatus.RUNNING


def test_3_step_increments_exactly_once():
    """Verify STEP advances exactly one step."""
    ctrl = PlaybackController("config_1.h5", speed=1.0, strategy_type="smart_scan")
    ctrl.step(1)
    assert ctrl.current_step == 1
    ctrl.step(1)
    assert ctrl.current_step == 2


def test_4_step_plus_10_increments_by_ten():
    """Verify STEP +10 advances exactly 10 steps."""
    ctrl = PlaybackController("config_1.h5", speed=1.0, strategy_type="smart_scan")
    ctrl.step(10)
    assert ctrl.current_step == 10
    assert ctrl.get_snapshot()["timestep"] == 10


def test_5_pause_preserves_current_step():
    """Verify PAUSE stops running and preserves current_step and metrics."""
    ctrl = PlaybackController("config_1.h5", speed=1.0, strategy_type="smart_scan")
    ctrl.start()
    ctrl.step(25)
    assert ctrl.current_step == 25
    snap25 = ctrl.get_snapshot()

    ctrl.pause()
    assert ctrl.running is False
    assert ctrl.paused is True
    assert ctrl.current_step == 25
    assert ctrl.status == EngineStatus.PAUSED
    assert ctrl.get_snapshot()["true_detections"] == snap25["true_detections"]


def test_6_resume_continues_from_current_step():
    """Verify RESUME continues from exact current_step."""
    ctrl = PlaybackController("config_1.h5", speed=1.0, strategy_type="smart_scan")
    ctrl.start()
    ctrl.step(25)
    ctrl.pause()
    assert ctrl.current_step == 25

    ctrl.resume()
    assert ctrl.running is True
    assert ctrl.paused is False
    ctrl.step(5)
    assert ctrl.current_step == 30


def test_7_reset_restores_step_zero():
    """Verify RESET returns current_step == 0 and status == READY."""
    ctrl = PlaybackController("config_1.h5", speed=1.0, strategy_type="smart_scan")
    ctrl.start()
    ctrl.step(50)
    assert ctrl.current_step == 50

    ctrl.reset()
    assert ctrl.current_step == 0
    assert ctrl.running is False
    assert ctrl.paused is False
    assert ctrl.mission_started is False
    assert ctrl.status == EngineStatus.READY


def test_8_completion_at_step_599():
    """Verify reaching step 599 marks mission_completed == True and stops running."""
    ctrl = PlaybackController("config_1.h5", speed=1.0, strategy_type="smart_scan")
    ctrl.start()
    ctrl.step(600)
    assert ctrl.current_step == 599
    assert ctrl.mission_completed is True
    assert ctrl.running is False
    assert ctrl.status == EngineStatus.COMPLETE


def test_9_selected_bands_match_artifact():
    """Verify receiver cards correspond to time_series[current_step]."""
    ctrl = PlaybackController("config_1.h5", speed=1.0, strategy_type="smart_scan")
    ctrl.step(10)
    snap = ctrl.get_snapshot()
    expected_bands = ctrl.time_series[10]["smart_scan_selected"]
    assert snap["selected_bands"] == expected_bands
    assert [ch["band"] for ch in snap["channel_telemetry"]] == expected_bands


def test_10_metrics_match_cumulative_artifact():
    """Verify displayed metrics match cumulative artifact values."""
    ctrl = PlaybackController("config_1.h5", speed=1.0, strategy_type="smart_scan")
    ctrl.step(599)
    snap = ctrl.get_snapshot()
    total_true_det_artifact = ctrl.artifact_data["metrics_summary"]["smart_scan"]["true_detections"]
    assert snap["true_detections"] == total_true_det_artifact


def test_11_scenario_switching_resets_state():
    """Verify switching from config_1 to config_2 resets runtime state."""
    ctrl = PlaybackController("config_1.h5", speed=1.0, strategy_type="smart_scan")
    ctrl.step(50)
    assert ctrl.current_step == 50

    ctrl.set_scenario("config_2.h5", strategy_type="open_loop")
    assert ctrl.current_step == 0
    assert ctrl.running is False
    assert "config_2" in ctrl.config_key
    assert ctrl.strategy_type == "open_loop"


def test_12_no_ground_truth_leakage_in_runtime():
    """Verify that scheduler interface and snapshot do not expose ground-truth emitter labels."""
    ctrl = PlaybackController("config_1.h5", speed=1.0, strategy_type="smart_scan")
    ctrl.step(20)
    snap = ctrl.get_snapshot()
    for ch in snap["channel_telemetry"]:
        assert "ground_truth_emitter_id" not in ch
        assert "emitter_label" not in ch
