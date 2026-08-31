"""Comprehensive verification tests for Step 13: Production-Grade Operational Cognitive RF Application."""

import json
import os
import pytest
from engine.mission_engine import MissionEngine
from engine.execution_loop import ExecutionWorker
from core.state import EngineStatus, ChannelState, MissionState
from core.data_source import TSRDSignalSource, ReplaySignalSource, HardwareSignalSource
from core.events import EventType, EventSeverity

SCENARIO_1 = r"D:\sih\dataset\scan\test_scan\config_1.h5"
SCENARIO_2 = r"D:\sih\dataset\scan\test_scan\config_2.h5"


def test_mission_engine_initialization_and_signal_source():
    """Verify MissionEngine initializes with TSRDSignalSource in READY state."""
    m_engine = MissionEngine(scenario_path=SCENARIO_1, strategy_type="smart_scan", k_channels=5, seed=42)
    assert m_engine.status == EngineStatus.READY
    assert m_engine.clock.current_step == 0
    assert len(m_engine.selected_bands) == 5
    assert len(m_engine.channel_telemetry) == 5

    # Test Signal Sources
    src_tsrd = TSRDSignalSource(SCENARIO_1)
    assert src_tsrd.total_steps > 0
    assert "SIMULATION" in src_tsrd.source_type

    src_replay = ReplaySignalSource(SCENARIO_1)
    assert src_replay.total_steps > 0
    assert "REPLAY" in src_replay.source_type

    src_hw = HardwareSignalSource()
    assert src_hw.total_steps == 0
    assert "HARDWARE" in src_hw.source_type


def test_mission_controls_lifecycle():
    """Verify START, PAUSE, RESUME, STEP, STOP, RESET controls."""
    m_engine = MissionEngine(scenario_path=SCENARIO_1, strategy_type="smart_scan", k_channels=5, seed=42)

    # 1. START
    m_engine.start_mission()
    assert m_engine.status == EngineStatus.RUNNING

    # 2. PAUSE
    m_engine.pause_mission()
    assert m_engine.status == EngineStatus.PAUSED

    # 3. STEP (+1 cycle)
    m_engine.step_mission(1)
    assert m_engine.clock.current_step == 1

    # 4. RESUME
    m_engine.resume_mission()
    assert m_engine.status == EngineStatus.RUNNING

    # 5. STOP
    m_engine.stop_mission()
    assert m_engine.status == EngineStatus.STOPPED

    # 6. RESET
    m_engine.reset_mission()
    assert m_engine.status == EngineStatus.READY
    assert m_engine.clock.current_step == 0


def test_15_step_operational_dataflow_and_no_leakage():
    """Verify the 15-step operational pipeline and zero ground-truth leakage."""
    m_engine = MissionEngine(scenario_path=SCENARIO_1, strategy_type="smart_scan", k_channels=5, seed=42)
    m_engine.step_mission(15)

    snap = m_engine.get_snapshot()
    assert snap["timestep"] == 15
    assert snap["simulated_time_s"] == 15 * 0.05
    assert len(snap["selected_bands"]) == 5
    assert len(snap["receiver_channels"]) == 5
    assert len(snap["band_scores_table"]) == 50
    assert snap["total_scans"] == 75

    # Ground truth isolation
    assert not hasattr(m_engine.engine.scheduler, "ground_truth")
    assert not hasattr(m_engine.engine.scheduler, "labels")
    for tr in m_engine.tracker.tracks.values():
        assert not hasattr(tr, "ground_truth_emitter_id")
        assert tr.track_id.startswith("TRACK-")


def test_execution_worker_lifecycle():
    """Verify ExecutionWorker threaded loop start, stop, and clean termination."""
    m_engine = MissionEngine(scenario_path=SCENARIO_1, strategy_type="smart_scan", k_channels=5, seed=42)
    worker = ExecutionWorker(m_engine)

    started = worker.start()
    assert started is True

    # Duplicate worker start should return False
    dup_started = worker.start()
    assert dup_started is False

    # Let worker step a few times
    import time
    time.sleep(0.15)
    assert m_engine.clock.current_step > 0

    worker.stop()
    assert m_engine.status in (EngineStatus.PAUSED, EngineStatus.STOPPED, EngineStatus.COMPLETE)


def test_scenario_switching_and_record_saving():
    """Verify switching scenario to config_2 and saving structured mission records."""
    m_engine = MissionEngine(scenario_path=SCENARIO_1, strategy_type="smart_scan", k_channels=5, seed=42)
    m_engine.step_mission(10)

    # Save mission record
    rec_path = m_engine.save_mission_record()
    assert rec_path is not None
    assert os.path.exists(rec_path)

    # Switch to config_2 and open_loop
    m_engine.initialize_mission(scenario_path=SCENARIO_2, strategy_type="open_loop")
    assert "config_2.h5" in m_engine.engine.scenario_path
    assert m_engine.engine.strategy_type == "open_loop"
    assert m_engine.clock.current_step == 0

    m_engine.step_mission(5)
    assert m_engine.clock.current_step == 5
    assert m_engine.engine.total_scans == 25
