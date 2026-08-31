"""Tests for Stage 13: Production-Grade Real-Time Simulation Engine & Architecture."""

import os
import pytest
from simulation.engine import SimulationEngine, SimulationStatus
from simulation.clock import SimulationClock
from simulation.runner import run_full_simulation

SCENARIO_PATH = r"D:\sih\dataset\scan\test_scan\config_1.h5"


def test_simulation_clock_ticks_and_paces():
    """Verify simulation clock calculations and pacing."""
    clock = SimulationClock(step_duration_s=0.05, speed_multiplier=2.0)
    assert clock.current_step == 0
    assert clock.simulated_time_s == 0.0

    clock.tick()
    assert clock.current_step == 1
    assert clock.simulated_time_s == pytest.approx(0.05)

    clock.set_speed(5.0)
    assert clock.speed_multiplier == 5.0

    clock.reset()
    assert clock.current_step == 0


def test_simulation_engine_initialization_and_properties():
    """Verify engine initialization with K=5, N=50."""
    engine = SimulationEngine(
        scenario_path=SCENARIO_PATH,
        strategy_type="smart_scan",
        k_channels=5,
        n_bands=50,
        seed=42,
    )
    assert engine.status in (SimulationStatus.READY, SimulationStatus.STOPPED)
    assert engine.k_channels == 5
    assert engine.n_bands == 50
    assert len(engine.selected_bands) == 5
    assert engine.clock.current_step == 0


def test_simulation_engine_state_transitions():
    """Verify START, PAUSE, STOP, RESET state transitions."""
    engine = SimulationEngine(
        scenario_path=SCENARIO_PATH,
        strategy_type="smart_scan",
        k_channels=5,
        n_bands=50,
        seed=42,
    )
    assert engine.status in (SimulationStatus.READY, SimulationStatus.STOPPED)

    engine.start()
    assert engine.status == SimulationStatus.RUNNING

    engine.pause()
    assert engine.status == SimulationStatus.PAUSED

    engine.stop()
    assert engine.status == SimulationStatus.STOPPED

    engine.step(num_steps=5)
    assert engine.clock.current_step == 5
    assert engine.total_scans == 25  # 5 steps * 5 channels

    engine.reset()
    assert engine.clock.current_step == 0
    assert engine.total_scans == 0
    assert engine.status in (SimulationStatus.READY, SimulationStatus.STOPPED)


def test_simulation_engine_step_advances_closed_loop():
    """Verify that stepping the engine performs observation, reward, and decision updates."""
    engine = SimulationEngine(
        scenario_path=SCENARIO_PATH,
        strategy_type="smart_scan",
        k_channels=5,
        n_bands=50,
        seed=42,
    )

    engine.step(num_steps=10)
    assert engine.clock.current_step == 10
    assert engine.total_scans == 50
    assert len(engine.selected_bands) == 5
    assert len(engine.decision_history) == 10
    assert len(engine.time_series) == 10

    snap = engine.get_snapshot()
    assert snap["timestep"] == 10
    assert snap["simulated_time_s"] == pytest.approx(0.50)
    assert "band_scores_table" in snap
    assert len(snap["band_scores_table"]) == 50


def test_simulation_engine_zero_ground_truth_leakage():
    """Verify that the scheduler decision function receives no ground truth parameters."""
    engine = SimulationEngine(
        scenario_path=SCENARIO_PATH,
        strategy_type="smart_scan",
        k_channels=5,
        n_bands=50,
        seed=42,
    )

    # Step engine
    engine.step(num_steps=5)

    # Inspect scheduler internal state
    scheduler = engine.scheduler
    assert not hasattr(scheduler, "ground_truth")
    assert not hasattr(scheduler, "transmitters")
    assert not hasattr(scheduler, "emitter_identities")


def test_simulation_engine_open_loop_baseline():
    """Verify open-loop sequential sweep operates deterministically."""
    engine = SimulationEngine(
        scenario_path=SCENARIO_PATH,
        strategy_type="open_loop",
        k_channels=5,
        n_bands=50,
        seed=42,
    )
    assert engine.strategy_type == "open_loop"

    engine.step(num_steps=2)
    assert engine.selected_bands == ["F06", "F07", "F08", "F09", "F10"]


def test_simulation_runner_headless_execution():
    """Verify run_full_simulation runs without crashing."""
    result = run_full_simulation(
        scenario_path=SCENARIO_PATH,
        strategy_type="smart_scan",
        k_channels=5,
        n_bands=50,
        seed=42,
        max_steps=20,
    )
    assert result["timestep"] == 20
    assert result["total_scans"] == 100
    assert result["simulated_time_s"] == pytest.approx(1.0)
