"""Integration tests for all operational workstation views and real-time execution."""

import pytest
from simulation.engine import SimulationEngine, SimulationStatus
from dashboard import (
    live_operations,
    receiver_panel,
    decision_panel,
    spectrum,
    tracks,
    performance,
    event_console,
    system,
)

SCENARIO_PATH = r"D:\sih\dataset\scan\test_scan\config_1.h5"


def test_workstation_operational_loop_and_views_render():
    """Verify that all operational workstation views render correctly on live stepped engine state."""
    engine = SimulationEngine(
        scenario_path=SCENARIO_PATH,
        strategy_type="smart_scan",
        k_channels=5,
        n_bands=50,
        seed=42,
    )

    # Step engine 10 times to populate tracks, decisions, time-series, and telemetry
    engine.step(num_steps=10)
    assert engine.clock.current_step == 10
    assert len(engine.selected_bands) == 5
    assert engine.total_scans == 50

    try:
        live_operations.render_live_operations(engine)
        receiver_panel.render_receiver_panel(engine)
        decision_panel.render_decision_panel(engine)
        spectrum.render_live_spectrum_map(engine)
        tracks.render_tracks_view(engine)
        performance.render_performance_monitor(engine)
        event_console.render_event_console(engine)
        system.render_scenario_lab(engine)
    except Exception as e:
        pytest.fail(f"Workstation rendering raised an exception: {str(e)}")


def test_workstation_state_snapshot_consistency():
    """Verify that engine snapshot contains all required operational keys."""
    engine = SimulationEngine(scenario_path=SCENARIO_PATH, strategy_type="smart_scan", k_channels=5, seed=42)
    engine.step(5)
    snap = engine.get_snapshot()

    required_keys = [
        "status",
        "timestep",
        "total_timesteps",
        "simulated_time_s",
        "strategy_type",
        "current_strategy",
        "meta_q_values",
        "selected_bands",
        "channel_telemetry",
        "total_scans",
        "true_detections",
        "false_alarms",
        "sensor_pd",
        "pfa",
        "cumulative_reward",
        "band_scores_table",
        "tracks",
        "active_tracks_count",
        "total_tracks_count",
        "recent_events",
        "recent_decisions",
        "time_series",
        "band_scan_counts",
    ]
    for k in required_keys:
        assert k in snap, f"Missing key '{k}' in engine snapshot"
