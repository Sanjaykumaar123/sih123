"""Unit and integration tests for MissionEngine service."""

import json
import pytest
from services.mission_engine import MissionEngine, MissionStatus
from core.tracker import TrackState

SCENARIO_PATH = r"D:\sih\dataset\scan\test_scan\config_1.h5"


def test_mission_engine_initialization_and_lifecycle():
    """Verify MissionEngine lifecycle: IDLE -> RUNNING -> PAUSED -> RUNNING -> STOPPED -> IDLE."""
    m_engine = MissionEngine(
        scenario_path=SCENARIO_PATH,
        strategy_type="smart_scan",
        k_channels=5,
        n_bands=50,
        seed=42,
        max_duration_s=10.0,
    )
    assert m_engine.status in (MissionStatus.IDLE, MissionStatus.READY)
    assert m_engine.max_steps == 200

    m_engine.start_mission()
    assert m_engine.status == MissionStatus.RUNNING

    m_engine.pause_mission()
    assert m_engine.status == MissionStatus.PAUSED

    m_engine.resume_mission()
    assert m_engine.status == MissionStatus.RUNNING

    m_engine.stop_mission()
    assert m_engine.status == MissionStatus.STOPPED

    m_engine.reset_mission()
    assert m_engine.status in (MissionStatus.IDLE, MissionStatus.READY)
    assert m_engine.clock.current_step == 0


def test_mission_engine_duration_limit_and_completion():
    """Verify that MissionEngine stops and transitions to COMPLETE when duration is reached."""
    m_engine = MissionEngine(
        scenario_path=SCENARIO_PATH,
        strategy_type="smart_scan",
        k_channels=5,
        n_bands=50,
        seed=42,
        max_duration_s=1.0,  # 20 steps
    )
    assert m_engine.max_steps == 20

    m_engine.step_mission(num_steps=25)
    assert m_engine.clock.current_step == 20
    assert m_engine.status == MissionStatus.COMPLETE

    snap = m_engine.get_snapshot()
    assert snap["progress_pct"] == 100.0
    assert snap["mission_status"] == MissionStatus.COMPLETE


def test_mission_engine_tracker_and_export():
    """Verify that MissionEngine updates signal tracks and exports machine-readable reports."""
    m_engine = MissionEngine(
        scenario_path=SCENARIO_PATH,
        strategy_type="smart_scan",
        k_channels=5,
        seed=42,
        max_duration_s=5.0,
    )
    m_engine.step_mission(num_steps=15)

    assert hasattr(m_engine, "tracker")
    snap = m_engine.get_snapshot()
    assert "tracks" in snap
    assert "band_scores_table" in snap

    # Test Exports
    rep = m_engine.export_report_json()
    assert "mission_metadata" in rep
    assert "performance_metrics" in rep

    csv_ev = m_engine.export_events_csv()
    assert "Time,Timestep,Channel" in csv_ev

    csv_dec = m_engine.export_decisions_csv()
    assert "Time,Timestep,Strategy" in csv_dec
