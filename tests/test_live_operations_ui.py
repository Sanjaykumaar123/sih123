"""UI Integration tests for Live Operations Workstation modules."""

import pytest
from simulation.engine import SimulationEngine
from dashboard import live_operations, spectrum, scheduler_view, events, system
from data.scenario_loader import get_validated_scenarios

RESULTS_DIR = r"D:\sih\results"
SCAN_DIR = r"D:\sih\dataset\scan\test_scan"
SCENARIO_PATH = r"D:\sih\dataset\scan\test_scan\config_1.h5"


def test_live_operations_components_render():
    """Verify that all live operation views render without exceptions."""
    engine = SimulationEngine(
        scenario_path=SCENARIO_PATH,
        strategy_type="smart_scan",
        k_channels=5,
        n_bands=50,
        seed=42,
    )

    # Step a few times to generate live data
    engine.step(num_steps=5)

    try:
        live_operations.render_live_operations(engine)
        spectrum.render_live_spectrum_map(engine)
        scheduler_view.render_scheduler_view(engine)
        events.render_event_log(engine)
        system.render_scenario_lab(engine)
    except Exception as e:
        pytest.fail(f"Live operations component failed to render: {str(e)}")


def test_system_benchmark_suite_renders():
    """Verify that the benchmark suite in system.py renders cleanly."""
    validated = get_validated_scenarios(results_dir=RESULTS_DIR, scan_dir=SCAN_DIR)
    try:
        system.render_benchmark_suite(validated)
    except Exception as e:
        pytest.fail(f"Benchmark suite failed to render: {str(e)}")
