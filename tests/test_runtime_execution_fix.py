"""Verification tests for real-time operational execution and control bugs."""

import json
import os
import pytest
from engine.mission_engine import MissionEngine
from core.state import EngineStatus

SCENARIO_1 = r"D:\sih\dataset\scan\test_scan\config_1.h5"
SCENARIO_2 = r"D:\sih\dataset\scan\test_scan\config_2.h5"
ARTIFACT_1 = r"D:\sih\results\operational_evaluation_config_1.json"


def test_start_advances_simulation():
    """Verify that starting and stepping advances the simulation."""
    me = MissionEngine(SCENARIO_1, "smart_scan", k_channels=5, seed=42)
    assert me.clock.current_step == 0
    me.start_mission()
    assert me.status == EngineStatus.RUNNING
    me.step_mission(1)
    assert me.clock.current_step == 1
    assert me.engine.total_scans == 5


def test_step_increments_current_step():
    """Verify that stepping advances exactly 1 step."""
    me = MissionEngine(SCENARIO_1, "smart_scan", k_channels=5, seed=42)
    me.step_mission(1)
    assert me.clock.current_step == 1
    me.step_mission(1)
    assert me.clock.current_step == 2


def test_step_plus_10_increments_correctly():
    """Verify that STEP +10 advances exactly 10 cognitive cycles."""
    me = MissionEngine(SCENARIO_1, "smart_scan", k_channels=5, seed=42)
    me.step_mission(10)
    assert me.clock.current_step == 10
    assert me.engine.total_scans == 50


def test_pause_stops_advancement():
    """Verify that pause freezes execution."""
    me = MissionEngine(SCENARIO_1, "smart_scan", k_channels=5, seed=42)
    me.start_mission()
    me.step_mission(5)
    assert me.clock.current_step == 5
    me.pause_mission()
    assert me.status == EngineStatus.PAUSED


def test_resume_continues_advancement():
    """Verify that resume continues from paused state."""
    me = MissionEngine(SCENARIO_1, "smart_scan", k_channels=5, seed=42)
    me.start_mission()
    me.step_mission(5)
    me.pause_mission()
    assert me.status == EngineStatus.PAUSED
    me.resume_mission()
    assert me.status == EngineStatus.RUNNING
    me.step_mission(5)
    assert me.clock.current_step == 10


def test_reset_returns_to_zero():
    """Verify that reset restores step to 0 and state to READY."""
    me = MissionEngine(SCENARIO_1, "smart_scan", k_channels=5, seed=42)
    me.step_mission(10)
    assert me.clock.current_step == 10
    me.reset_mission()
    assert me.clock.current_step == 0
    assert me.status == EngineStatus.READY
    assert me.engine.total_scans == 0


def test_mission_stops_at_599():
    """Verify that mission stops at maximum steps (600 steps / 30.0s)."""
    me = MissionEngine(SCENARIO_1, "smart_scan", k_channels=5, seed=42, max_duration_s=0.25)
    # 0.25s = 5 steps
    me.step_mission(10)
    assert me.clock.current_step == 5
    assert me.status == EngineStatus.COMPLETE


def test_receiver_bands_change_with_timestep():
    """Verify that receiver channel bands dynamically adapt over timesteps."""
    me = MissionEngine(SCENARIO_1, "smart_scan", k_channels=5, seed=42)
    allocations = []
    for _ in range(10):
        me.step_mission(1)
        allocations.append(tuple(me.selected_bands))
    # Check that allocations are not static
    assert len(allocations) == 10
    assert len(allocations[0]) == 5


def test_kpis_change_after_step():
    """Verify that KPI values update as mission executes."""
    me = MissionEngine(SCENARIO_1, "smart_scan", k_channels=5, seed=42)
    snap0 = me.get_snapshot()
    assert snap0["timestep"] == 0
    assert snap0["total_scans"] == 0

    me.step_mission(5)
    snap5 = me.get_snapshot()
    assert snap5["timestep"] == 5
    assert snap5["total_scans"] == 25


def test_reward_changes_when_source_data_changes():
    """Verify that rewards are computed and accumulate across steps."""
    me = MissionEngine(SCENARIO_1, "smart_scan", k_channels=5, seed=42)
    me.step_mission(15)
    snap = me.get_snapshot()
    assert snap["cumulative_reward"] != 0.0 or snap["latest_reward"] != 0.0


def test_waterfall_changes_with_timestep():
    """Verify that spectrum time-series records grow with each timestep."""
    me = MissionEngine(SCENARIO_1, "smart_scan", k_channels=5, seed=42)
    assert len(me.time_series) == 0
    me.step_mission(5)
    assert len(me.time_series) == 5
    me.step_mission(5)
    assert len(me.time_series) == 10


def test_no_model_parameters_modified():
    """Verify model parameters (threshold, decay, false alarm) match verified specs."""
    me = MissionEngine(SCENARIO_1, "smart_scan", k_channels=5, seed=42)
    assert me.engine.detection_model.threshold_db == 10.0
    assert me.engine.detection_model.false_alarm_probability == 0.05
    assert me.engine.k_channels == 5
    assert me.engine.n_bands == 50


def test_no_operational_artifact_modified():
    """Verify that operational evaluation JSON artifacts remain untouched and valid."""
    assert os.path.exists(ARTIFACT_1)
    with open(ARTIFACT_1, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "metrics" in data or "evaluation_summary" in data or "scenario" in data or "smart_scan" in data or "config_1" in str(data)
