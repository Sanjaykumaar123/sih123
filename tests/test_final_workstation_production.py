"""Comprehensive unit and integration tests for Final Productionization Phase."""

import json
import os
import pytest
from services.mission_engine import MissionEngine, MissionStatus
from simulation.engine import SimulationEngine, SimulationStatus
from core.tracker import TrackState, TrackManager
from core.events import EventType

SCENARIO_1 = r"D:\sih\dataset\scan\test_scan\config_1.h5"
SCENARIO_2 = r"D:\sih\dataset\scan\test_scan\config_2.h5"


def test_mission_lifecycle_and_transition_guardrails():
    """Verify strict mission lifecycle state transitions and prevention of invalid actions."""
    me = MissionEngine(scenario_path=SCENARIO_1, strategy_type="smart_scan", k_channels=5, seed=42)
    assert me.status in (MissionStatus.IDLE, MissionStatus.READY)

    # Cannot pause when IDLE/READY
    me.pause_mission()
    assert me.status in (MissionStatus.IDLE, MissionStatus.READY)

    # Cannot resume when IDLE/READY
    me.resume_mission()
    assert me.status in (MissionStatus.IDLE, MissionStatus.READY)

    # Start mission -> RUNNING
    me.start_mission()
    assert me.status == MissionStatus.RUNNING

    # Pause mission -> PAUSED
    me.pause_mission()
    assert me.status == MissionStatus.PAUSED

    # Resume mission -> RUNNING
    me.resume_mission()
    assert me.status == MissionStatus.RUNNING

    # Stop mission -> STOPPED
    me.stop_mission()
    assert me.status == MissionStatus.STOPPED

    # Reset mission -> READY / IDLE
    me.reset_mission()
    assert me.status in (MissionStatus.IDLE, MissionStatus.READY)
    assert me.clock.current_step == 0


def test_mission_engine_15_step_closed_loop_dataflow():
    """Verify that MissionEngine advances clock, reads obs, updates beliefs, tracks, and produces snapshot."""
    me = MissionEngine(scenario_path=SCENARIO_1, strategy_type="smart_scan", k_channels=5, seed=42)
    me.step_mission(num_steps=15)

    snap = me.get_snapshot()
    assert snap["timestep"] == 15
    assert snap["simulated_time_s"] == 15 * 0.05
    assert len(snap["selected_bands"]) == 5
    assert len(snap["channel_telemetry"]) == 5
    assert len(snap["band_scores_table"]) == 50
    assert snap["total_scans"] == 75


def test_receiver_snapshot_physical_consistency():
    """Verify that 5 receiver channels provide physical frequencies, SNR, signal strength, and dwell times."""
    me = MissionEngine(scenario_path=SCENARIO_1, strategy_type="smart_scan", k_channels=5, seed=42)
    me.step_mission(5)

    snap = me.get_snapshot()
    ch_list = snap["channel_telemetry"]
    assert len(ch_list) == 5

    for idx, ch in enumerate(ch_list):
        assert ch["channel_idx"] == idx + 1
        assert ch["dwell_time_ms"] == 50.0
        assert ch["frequency_mhz"] >= 500.0
        assert ch["frequency_mhz"] <= 18000.0
        assert ch["status"] in ("MONITORING", "TRUE INTERCEPTION", "FALSE ALARM", "DETECT", "MISS", "IDLE")
        assert "scheduler_role" in ch


def test_ground_truth_isolation_operational_vs_validation():
    """Verify that scheduler and tracker operate without ground-truth access during runtime."""
    me = MissionEngine(scenario_path=SCENARIO_1, strategy_type="smart_scan", k_channels=5, seed=42)
    me.step_mission(10)

    # 1. Scheduler isolation
    assert not hasattr(me.engine.scheduler, "ground_truth")
    assert not hasattr(me.engine.scheduler, "labels")

    # 2. Tracker isolation
    for tr in me.tracker.tracks.values():
        assert not hasattr(tr, "ground_truth_emitter_id")
        assert not hasattr(tr, "true_label")
        assert tr.track_id.startswith("TRACK-")


def test_track_inspector_observable_history_and_csv_export():
    """Verify that SignalTrack records pulse history and exports valid CSV."""
    me = MissionEngine(scenario_path=SCENARIO_1, strategy_type="smart_scan", k_channels=5, seed=42)
    me.step_mission(30)

    snap = me.get_snapshot()
    if snap["tracks"]:
        t0_id = snap["tracks"][0]["Track ID"]
        t0 = me.tracker.tracks[t0_id]
        assert len(t0.observable_history) > 0
        p0 = t0.observable_history[0]
        assert "frequency_mhz" in p0
        assert "snr_db" in p0

    # Test Track CSV Export
    csv_str = me.export_tracks_csv()
    assert "Track_ID,Band,State,Estimated_Frequency_MHz" in csv_str


def test_decision_rationale_groundedness():
    """Verify that the decision engine produces grounded reasoning based on score components."""
    me = MissionEngine(scenario_path=SCENARIO_1, strategy_type="smart_scan", k_channels=5, seed=42)
    me.step_mission(8)

    snap = me.get_snapshot()
    table = snap["band_scores_table"]
    selected = snap["selected_bands"]

    sel_rows = [r for r in table if r["Band"] in selected]
    assert len(sel_rows) == 5
    for r in sel_rows:
        assert r["Reason"] != "—"
        assert len(r["Reason"]) > 5


def test_multi_scenario_full_run_and_reporting():
    """Verify full 600-step execution, JSON report export, and CSV exports."""
    me = MissionEngine(scenario_path=SCENARIO_1, strategy_type="smart_scan", k_channels=5, seed=42, max_duration_s=30.0)
    me.step_mission(600)

    assert me.status == MissionStatus.COMPLETE
    assert me.clock.current_step == 600

    rep = me.export_report_json()
    assert rep["mission_metadata"]["scenario"] == "config_1.h5"
    assert rep["mission_metadata"]["total_timesteps_executed"] == 600
    assert "performance_metrics" in rep
    assert "signal_tracks" in rep

    ev_csv = me.export_events_csv()
    assert "Time,Timestep,Channel" in ev_csv

    dec_csv = me.export_decisions_csv()
    assert "Time,Timestep,Strategy" in dec_csv


def test_error_handling_graceful_recovery():
    """Verify graceful handling of non-existent scenario files."""
    me = MissionEngine(scenario_path="non_existent_file.h5", strategy_type="smart_scan", k_channels=5, seed=42)
    assert me.engine.env is None

    # Step on missing environment should transition safely to ERROR without crash
    me.step_mission(1)
    assert me.engine.status == SimulationStatus.ERROR
