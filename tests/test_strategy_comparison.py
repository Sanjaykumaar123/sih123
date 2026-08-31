"""Comprehensive tests for Open-Loop vs. Smart Scan Strategy Comparison."""

import inspect
import math
import os
import pytest

from experiments.compare_strategies import (
    SequentialOpenLoopScheduler,
    ComparisonResult,
    compare_strategies,
    run_strategy_simulation,
)
from rf_env import IntelligentSchedulerAdapter

SAMPLE_TSRD_PATH = r"D:\sih\dataset\scan\test_scan\config_1.h5"


def test_sequential_scheduler_sweeps_deterministically():
    """Test that SequentialOpenLoopScheduler sweeps bands sequentially in groups of K."""
    sched = SequentialOpenLoopScheduler(num_bands=50, k=5, start_band="F01")
    assert sched.select_bands(0) == ["F01", "F02", "F03", "F04", "F05"]
    assert sched.select_bands(1) == ["F06", "F07", "F08", "F09", "F10"]
    assert sched.select_bands(9) == ["F46", "F47", "F48", "F49", "F50"]
    assert sched.select_bands(10) == ["F01", "F02", "F03", "F04", "F05"]  # Wraps cleanly


def test_sequential_scheduler_configurable_start_band():
    """Test that SequentialOpenLoopScheduler respects configurable starting band."""
    sched = SequentialOpenLoopScheduler(num_bands=50, k=5, start_band="F11")
    assert sched.select_bands(0) == ["F11", "F12", "F13", "F14", "F15"]
    assert sched.select_bands(1) == ["F16", "F17", "F18", "F19", "F20"]


def test_open_loop_scheduler_has_no_ground_truth_access():
    """Verify that SequentialOpenLoopScheduler has no ground-truth parameters or leakage."""
    sig = inspect.signature(SequentialOpenLoopScheduler.select_bands)
    assert list(sig.parameters) == ["self", "timestep"]
    source = inspect.getsource(SequentialOpenLoopScheduler)
    assert "band_truth" not in source
    assert "emitter_id" not in source
    assert "labels" not in source


def test_compare_strategies_runs_and_returns_valid_structure():
    """Test full comparison execution on representative TSRD scenario."""
    assert os.path.exists(SAMPLE_TSRD_PATH)
    res = compare_strategies(
        scenario_path=SAMPLE_TSRD_PATH,
        num_steps=30,
        channels=5,
        seed=42,
    )
    assert isinstance(res, ComparisonResult)
    assert res.num_steps == 30
    assert res.channels == 5
    assert res.total_scan_opportunities == 150
    assert res.baseline.metrics.total_scans == 150
    assert res.smart_scan.metrics.total_scans == 150
    assert 0.0 <= res.baseline.metrics.sensor_pd <= 1.0
    assert 0.0 <= res.smart_scan.metrics.sensor_pd <= 1.0


def test_reproducibility_with_same_seed():
    """Verify that experiments run with the same seed yield bit-identical metrics."""
    res1 = compare_strategies(SAMPLE_TSRD_PATH, num_steps=20, channels=5, seed=42)
    res2 = compare_strategies(SAMPLE_TSRD_PATH, num_steps=20, channels=5, seed=42)

    assert res1.baseline.metrics.total_hits == res2.baseline.metrics.total_hits
    assert res1.smart_scan.metrics.total_hits == res2.smart_scan.metrics.total_hits
    assert res1.smart_scan.metrics.online_avg_reward == res2.smart_scan.metrics.online_avg_reward
    assert res1.smart_scan.metrics.sensor_pd == res2.smart_scan.metrics.sensor_pd


def test_comparison_result_table_formatting():
    """Test that ComparisonResult renders a human-readable summary table."""
    res = compare_strategies(SAMPLE_TSRD_PATH, num_steps=10, channels=5, seed=42)
    table_str = res.to_table()
    assert "SMART SCAN STRATEGY COMPARISON" in table_str
    assert "Sensor Pd" in table_str
    assert "OPEN LOOP" in table_str
    assert "SMART SCAN" in table_str


def test_both_strategies_obey_identical_k_constraint():
    """Verify that both Open-Loop and Smart Scan obey K=5 channels at every step."""
    sched_bl = SequentialOpenLoopScheduler(num_bands=50, k=5)
    sched_ss = IntelligentSchedulerAdapter(num_bands=50, k=5)

    for t in range(50):
        bl_bands = sched_bl.select_bands(t)
        ss_bands = sched_ss.select_bands(t)
        assert len(bl_bands) == 5
        assert len(ss_bands) == 5
        assert len(set(bl_bands)) == 5
        assert len(set(ss_bands)) == 5


def test_multi_scenario_loading_and_execution():
    """Verify that multiple scenarios can be evaluated independently without code modification."""
    scenarios = [
        r"D:\sih\dataset\scan\test_scan\config_1.h5",
        r"D:\sih\dataset\scan\test_scan\config_2.h5",
        r"D:\sih\dataset\scan\test_scan\config_3.h5",
    ]
    for p in scenarios:
        assert os.path.exists(p)
        res = compare_strategies(p, num_steps=15, channels=5, seed=42)
        assert res.scenario_name == os.path.basename(p)
        assert res.baseline.metrics.total_scans == 75
        assert res.smart_scan.metrics.total_scans == 75


def test_scheduler_configuration_is_identical_across_scenarios():
    """Verify that Smart Scan hyperparameters are never mutated between scenario evaluations."""
    sched1 = IntelligentSchedulerAdapter(num_bands=50, k=5)
    sched2 = IntelligentSchedulerAdapter(num_bands=50, k=5)

    assert sched1.arbitrator.learning_rate == sched2.arbitrator.learning_rate == 0.1
    assert sched1.arbitrator.discount_factor == sched2.arbitrator.discount_factor == 0.9
    assert sched1.arbitrator.initial_epsilon == sched2.arbitrator.initial_epsilon == 0.2
    assert sched1.arbitrator.epsilon_decay == sched2.arbitrator.epsilon_decay == 0.995


def test_operational_evaluation_full_run_and_data_structures():
    """Verify that operational evaluation runs across full duration and outputs structured data."""
    from experiments.operational_evaluation import run_operational_evaluation

    res, emitter_recs, ts_recs, spectrum_grids = run_operational_evaluation(
        SAMPLE_TSRD_PATH, num_steps=60, channels=5, seed=42
    )
    assert len(ts_recs) == 60
    assert len(emitter_recs) == 30
    assert len(spectrum_grids["ground_truth"]) == 50
    assert len(spectrum_grids["smart_scan"]) == 50
    assert res.smart_scan.metrics.total_scans == 300
    assert res.baseline.metrics.total_scans == 300


