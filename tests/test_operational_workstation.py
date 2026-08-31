"""Stage 14: Verification tests for Production Operational Workstation and Real-Time Simulation Engine."""

import json
import os
import pytest
from simulation.engine import SimulationEngine, SimulationStatus
from dashboard import live_operations, spectrum, scheduler_view, events, system
from data.scenario_loader import get_validated_scenarios, discover_scenarios

SCENARIO_1 = r"D:\sih\dataset\scan\test_scan\config_1.h5"
SCENARIO_2 = r"D:\sih\dataset\scan\test_scan\config_2.h5"


def test_engine_full_lifecycle_state_machine():
    """Test full state machine: READY -> RUNNING -> PAUSED -> RUNNING -> COMPLETE / STOPPED -> RESET."""
    engine = SimulationEngine(
        scenario_path=SCENARIO_1,
        strategy_type="smart_scan",
        k_channels=5,
        n_bands=50,
        seed=42,
    )
    assert engine.status == SimulationStatus.READY

    engine.start()
    assert engine.status == SimulationStatus.RUNNING

    engine.pause()
    assert engine.status == SimulationStatus.PAUSED

    engine.start()
    assert engine.status == SimulationStatus.RUNNING

    engine.stop()
    assert engine.status == SimulationStatus.STOPPED

    engine.reset()
    assert engine.status == SimulationStatus.READY
    assert engine.clock.current_step == 0


def test_engine_discrete_stepping_and_channel_constraints():
    """Test step(+1) and step(+10), verifying that exactly K=5 channels are selected."""
    engine = SimulationEngine(scenario_path=SCENARIO_1, k_channels=5, n_bands=50, seed=42)
    
    # Step +1
    engine.step(num_steps=1)
    assert engine.clock.current_step == 1
    assert len(engine.selected_bands) == 5
    assert len(set(engine.selected_bands)) == 5
    assert engine.total_scans == 5

    # Step +10
    engine.step(num_steps=10)
    assert engine.clock.current_step == 11
    assert len(engine.selected_bands) == 5
    assert engine.total_scans == 55


def test_scenario_and_strategy_switching():
    """Test hot-switching between scenarios (config_1 -> config_2) and strategies (smart_scan -> open_loop)."""
    engine = SimulationEngine(scenario_path=SCENARIO_1, strategy_type="smart_scan", k_channels=5, seed=42)
    engine.step(5)
    assert engine.strategy_type == "smart_scan"

    # Switch to config_2 and open_loop
    engine.reset(scenario_path=SCENARIO_2, strategy_type="open_loop")
    assert engine.strategy_type == "open_loop"
    assert "config_2.h5" in engine.scenario_path
    assert engine.clock.current_step == 0
    assert engine.total_scans == 0

    engine.step(5)
    assert engine.clock.current_step == 5
    assert engine.total_scans == 25


def test_zero_ground_truth_leakage_in_scheduler():
    """Verify that scheduler interface accepts strictly timestep and observations, no truth."""
    engine = SimulationEngine(scenario_path=SCENARIO_1, strategy_type="smart_scan", k_channels=5, seed=42)
    
    # Check that select_bands only takes timestep
    import inspect
    sig_select = inspect.signature(engine.scheduler.select_bands)
    params_select = list(sig_select.parameters.keys())
    assert params_select == ["timestep"] or params_select == ["t"]

    # Step engine and ensure no ground-truth attributes leaked onto scheduler
    engine.step(5)
    assert not hasattr(engine.scheduler, "ground_truth")
    assert not hasattr(engine.scheduler, "emitter_ids")
    assert not hasattr(engine.scheduler, "active_truth")


def test_50_band_ranking_table_integrity():
    """Verify that get_snapshot provides a complete, sorted 50-band ranking table."""
    engine = SimulationEngine(scenario_path=SCENARIO_1, strategy_type="smart_scan", k_channels=5, seed=42)
    engine.step(10)
    
    snap = engine.get_snapshot()
    tbl = snap["band_scores_table"]
    assert len(tbl) == 50

    # Ranks must be 1 to 50
    ranks = [r["Rank"] for r in tbl]
    assert ranks == list(range(1, 51))

    # Scores must be sorted in descending order
    scores = [r["Final Score"] for r in tbl]
    assert scores == sorted(scores, reverse=True)

    # Exactly 5 bands must be marked as selected
    selected_count = sum(1 for r in tbl if "✓ SELECTED" in r["Selected"])
    assert selected_count == 5


def test_machine_readable_exports():
    """Test JSON mission report export and CSV event telemetry export."""
    engine = SimulationEngine(scenario_path=SCENARIO_1, strategy_type="smart_scan", k_channels=5, seed=42)
    engine.step(15)

    # 1. JSON Mission Report
    rep = engine.export_mission_report()
    assert "mission_metadata" in rep
    assert "performance_metrics" in rep
    assert "decision_history" in rep
    assert "detection_events" in rep
    assert rep["mission_metadata"]["total_timesteps_executed"] == 15
    assert rep["performance_metrics"]["total_channel_scans"] == 75

    # Validate JSON serializability
    json_str = json.dumps(rep)
    assert len(json_str) > 100

    # 2. CSV Events
    csv_events = engine.export_events_csv()
    assert "Time,Timestep,Channel,Band,Event_Type" in csv_events

    # 3. CSV Decisions
    csv_decisions = engine.export_decisions_csv()
    assert "Time,Timestep,Strategy,Selected_Bands,Hits" in csv_decisions
    assert "BALANCED" in csv_decisions or "EXPLORE" in csv_decisions or "EXPLOIT" in csv_decisions or "PREDICT" in csv_decisions


def test_temporal_prediction_panel_state():
    """Verify temporal prediction state reporting."""
    engine = SimulationEngine(scenario_path=SCENARIO_1, strategy_type="smart_scan", k_channels=5, seed=42)
    engine.step(25)

    assert hasattr(engine.scheduler, "temporal")
    t_state = engine.scheduler.temporal.get_state()
    assert len(t_state) == 50
    for p in t_state:
        assert hasattr(p, "band_id")
        assert hasattr(p, "periodicity_score")
        assert hasattr(p, "prediction_confidence")


def test_error_handling_invalid_scenario():
    """Verify graceful handling when scenario file does not exist."""
    engine = SimulationEngine(scenario_path="non_existent_path.h5", strategy_type="smart_scan", k_channels=5, seed=42)
    assert engine.env is None
    engine.step(1)
    assert engine.status == SimulationStatus.ERROR
