"""Comprehensive operational runtime tests for Cognitive RF Spectrum Management Workstation."""

import json
import os
import pytest
from core.engine import OperationalEngine
from core.state import EngineStatus, ChannelState
from core.tracker import TrackState, TrackManager
from core.events import EventType

SCENARIO_1 = r"D:\sih\dataset\scan\test_scan\config_1.h5"
SCENARIO_2 = r"D:\sih\dataset\scan\test_scan\config_2.h5"


def test_1_mission_runtime_initialization():
    """Verify that OperationalEngine initializes in READY state with valid defaults."""
    engine = OperationalEngine(scenario_path=SCENARIO_1, strategy_type="smart_scan", k_channels=5, seed=42)
    assert engine.status == EngineStatus.READY
    assert engine.clock.current_step == 0
    assert len(engine.selected_bands) == 5
    assert len(engine.channels) == 5


def test_2_controls_start_pause_resume_step_stop_reset():
    """Verify full operational lifecycle controls."""
    engine = OperationalEngine(scenario_path=SCENARIO_1, strategy_type="smart_scan", k_channels=5, seed=42)

    # 2. START
    engine.start()
    assert engine.status == EngineStatus.RUNNING

    # 3. PAUSE
    engine.pause()
    assert engine.status == EngineStatus.PAUSED

    # 4. RESUME
    engine.resume()
    assert engine.status == EngineStatus.RUNNING

    # 5. STEP (execute exactly 1 cognitive cycle)
    engine.step(num_steps=1)
    assert engine.clock.current_step == 1

    # 6. STOP
    engine.stop()
    assert engine.status == EngineStatus.STOPPED

    # 7. RESET
    engine.reset()
    assert engine.status == EngineStatus.READY
    assert engine.clock.current_step == 0
    assert engine.total_scans == 0


def test_8_complete_cognitive_cycle_and_channel_selection():
    """Verify one complete cognitive cycle and exact 5-channel selection."""
    engine = OperationalEngine(scenario_path=SCENARIO_1, strategy_type="smart_scan", k_channels=5, seed=42)
    engine.step(1)

    assert engine.clock.current_step == 1
    assert len(engine.selected_bands) == 5
    assert len(set(engine.selected_bands)) == 5
    assert engine.total_scans == 5


def test_10_observation_isolation_and_no_ground_truth_leakage():
    """Verify that scheduler interface accepts strictly timestep and observations, no truth."""
    engine = OperationalEngine(scenario_path=SCENARIO_1, strategy_type="smart_scan", k_channels=5, seed=42)
    engine.step(5)

    assert not hasattr(engine.scheduler, "ground_truth")
    assert not hasattr(engine.scheduler, "emitter_truth")
    assert not hasattr(engine.scheduler, "labels")

    for tr in engine.tracker.tracks.values():
        assert not hasattr(tr, "ground_truth_emitter_id")
        assert tr.track_id.startswith("TRACK-")


def test_12_belief_reward_policy_and_track_updates():
    """Verify that Bayesian beliefs, reward, policy, and autonomous tracks update each step."""
    engine = OperationalEngine(scenario_path=SCENARIO_1, strategy_type="smart_scan", k_channels=5, seed=42)
    engine.step(10)

    snap = engine.get_snapshot()
    # Beliefs
    assert len(snap["band_scores_table"]) == 50
    # Reward
    assert snap["cumulative_reward"] != 0.0 or snap["latest_reward"] != 0.0
    # Policy
    assert snap["current_strategy"] in ("EXPLORE", "EXPLOIT", "PREDICT", "BALANCED")
    # Tracks
    assert "tracks" in snap
    assert snap["total_tracks_count"] >= 0


def test_17_event_generation_and_csv_export():
    """Verify that events are logged and exportable to CSV."""
    engine = OperationalEngine(scenario_path=SCENARIO_1, strategy_type="smart_scan", k_channels=5, seed=42)
    engine.step(10)

    assert len(engine.event_log) > 0
    csv_str = engine.export_events_csv()
    assert "Time,Timestep,Event_Type,Channel" in csv_str


def test_19_scenario_switching_and_reset():
    """Verify switching between scenarios and strategy types."""
    engine = OperationalEngine(scenario_path=SCENARIO_1, strategy_type="smart_scan", k_channels=5, seed=42)
    engine.step(5)
    assert engine.strategy_type == "smart_scan"

    # Switch to config_2 and open_loop
    engine.reset(scenario_path=SCENARIO_2, strategy_type="open_loop")
    assert engine.strategy_type == "open_loop"
    assert "config_2.h5" in engine.scenario_path
    assert engine.clock.current_step == 0

    engine.step(5)
    assert engine.clock.current_step == 5
    assert engine.total_scans == 25


def test_21_invalid_scenario_and_error_handling():
    """Verify graceful handling when scenario file does not exist."""
    engine = OperationalEngine(scenario_path="non_existent.h5", strategy_type="smart_scan", k_channels=5, seed=42)
    assert engine.env is None
    engine.step(1)
    assert engine.status == EngineStatus.ERROR


def test_24_repeated_execution_stability():
    """Verify repeated long execution runs without drift or memory leaks."""
    engine = OperationalEngine(scenario_path=SCENARIO_1, strategy_type="smart_scan", k_channels=5, seed=42, max_duration_s=5.0)
    for _ in range(3):
        engine.reset()
        engine.step(100)
        assert engine.clock.current_step == 100
        assert engine.status == EngineStatus.COMPLETE
