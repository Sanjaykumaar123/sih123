"""Tests for multi-scenario loader, aggregation calculations, and validation dashboard."""

import json
import math
import os
import tempfile
import plotly.graph_objects as go
import pytest

from dashboard.scenario_loader import (
    discover_scenarios,
    get_validated_scenarios,
    validate_operational_artifact,
)
from dashboard.multiscenario import (
    build_multiscenario_comparison_table,
    build_validation_status_table,
    calculate_aggregate_statistics,
    plot_multiscenario_detections,
    plot_multiscenario_emitters,
    plot_multiscenario_latency,
)

RESULTS_DIR = r"D:\sih\results"
SCAN_DIR = r"D:\sih\dataset\scan\test_scan"


def test_scenario_discovery_finds_available_scenarios():
    """Verify that scenario discovery engine finds H5 files and operational JSONs."""
    scenarios = discover_scenarios(results_dir=RESULTS_DIR, scan_dir=SCAN_DIR)
    assert len(scenarios) > 0
    assert "config_1" in scenarios
    assert scenarios["config_1"].status == "VALIDATED"
    assert scenarios["config_1"].data_present is True
    assert scenarios["config_1"].evaluation_present is True


def test_validate_operational_artifact_on_valid_file():
    """Verify schema validation on config_1 operational artifact."""
    j_path = os.path.join(RESULTS_DIR, "operational_evaluation_config_1.json")
    is_valid, err, data = validate_operational_artifact(j_path)
    assert is_valid is True
    assert err is None
    assert data is not None
    assert data["scenario"] == "config_1.h5"
    assert data["num_steps"] == 600


def test_validate_operational_artifact_rejects_corrupted_file():
    """Verify that corrupt or incomplete artifacts are safely rejected."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"broken": True}, f)
        temp_path = f.name

    try:
        is_valid, err, data = validate_operational_artifact(temp_path)
        assert is_valid is False
        assert "Missing required key" in err
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_get_validated_scenarios_loads_data():
    """Verify that get_validated_scenarios returns loaded dictionary of valid scenarios."""
    validated = get_validated_scenarios(results_dir=RESULTS_DIR, scan_dir=SCAN_DIR)
    assert len(validated) >= 1
    assert "config_1" in validated
    assert "metrics_summary" in validated["config_1"]


def test_calculate_aggregate_statistics_math():
    """Verify statistical aggregations (mean, median, min, max, totals) across validated scenarios."""
    validated = get_validated_scenarios(results_dir=RESULTS_DIR, scan_dir=SCAN_DIR)
    stats = calculate_aggregate_statistics(validated)

    assert stats["insufficient_data"] is False
    assert stats["num_scenarios"] == len(validated)
    assert stats["total_ss_true_detections"] >= 45
    assert stats["total_ol_true_detections"] >= 31

    # Check metrics structure
    for k in ["true_detections", "unique_emitters", "sensor_pd", "scenario_coverage", "pfa"]:
        assert k in stats["metrics"]
        assert "smart_scan" in stats["metrics"][k]
        assert "open_loop" in stats["metrics"][k]
        assert "mean" in stats["metrics"][k]["smart_scan"]
        assert "median" in stats["metrics"][k]["smart_scan"]
        assert not math.isnan(stats["metrics"][k]["smart_scan"]["mean"])

    # Consistency metrics (strictly verified against 5 operational JSON artifacts)
    assert stats["consistency"]["detection_advantage"] == (4, 5)
    assert stats["consistency"]["emitter_strict_advantage"] == (1, 5)
    assert stats["consistency"]["emitter_equal_or_better"] == (2, 5)
    assert stats["consistency"]["latency_advantage"] == (3, 5)


def test_build_multiscenario_comparison_table():
    """Verify generation of comparison table rows."""
    validated = get_validated_scenarios(results_dir=RESULTS_DIR, scan_dir=SCAN_DIR)
    rows = build_multiscenario_comparison_table(validated)

    assert len(rows) == len(validated)
    row_1 = next(r for r in rows if "config_1" in r["Scenario"])
    assert row_1["SS True Detections"] == 45
    assert row_1["OL True Detections"] == 31
    assert row_1["SS Emitters"] == 18
    assert row_1["OL Emitters"] == 14


def test_build_validation_status_table():
    """Verify validation status table contains all discovered scenarios."""
    all_discovered = discover_scenarios(results_dir=RESULTS_DIR, scan_dir=SCAN_DIR)
    status_rows = build_validation_status_table(all_discovered)

    assert len(status_rows) == len(all_discovered)
    valid_rows = [r for r in status_rows if r["STATUS"] == "VALIDATED"]
    assert len(valid_rows) >= 1


def test_multiscenario_plotly_figures():
    """Verify Plotly figure generation for multi-scenario benchmark charts."""
    validated = get_validated_scenarios(results_dir=RESULTS_DIR, scan_dir=SCAN_DIR)
    
    fig_det = plot_multiscenario_detections(validated)
    assert isinstance(fig_det, go.Figure)
    assert len(fig_det.data) == 2

    fig_emit = plot_multiscenario_emitters(validated)
    assert isinstance(fig_emit, go.Figure)
    assert len(fig_emit.data) == 2

    fig_lat = plot_multiscenario_latency(validated)
    assert isinstance(fig_lat, go.Figure)
    assert len(fig_lat.data) == 2


def test_empty_scenarios_aggregate_handling():
    """Verify safe handling when no scenarios are provided."""
    stats = calculate_aggregate_statistics({})
    assert stats["insufficient_data"] is True
    assert stats["num_scenarios"] == 0
