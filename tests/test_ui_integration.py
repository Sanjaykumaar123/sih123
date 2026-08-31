"""UI and Data Integration Tests for Cognitive RF Spectrum Management Dashboard."""

import json
import os
import plotly.graph_objects as go
import pytest

from dashboard import visualizations as viz

OPERATIONAL_JSON = r"D:\sih\results\operational_evaluation_config_1.json"


def test_operational_json_exists_and_has_required_schema():
    """Verify that operational evaluation result JSON exists with correct schema."""
    assert os.path.exists(OPERATIONAL_JSON), f"File missing: {OPERATIONAL_JSON}"
    with open(OPERATIONAL_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["scenario"] == "config_1.h5"
    assert data["num_steps"] == 600
    assert data["channels"] == 5
    assert len(data["time_series"]) == 600
    assert len(data["emitter_interceptions"]) == 30
    assert "metrics_summary" in data
    assert "spectrum_grids" in data


def test_spectrum_activity_map_renders_valid_figure():
    """Verify that spectrum activity map renders valid Plotly figure without errors."""
    with open(OPERATIONAL_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    fig = viz.spectrum_activity_map(
        time_series=data["time_series"],
        current_t=50,
        window_steps=50,
        strategy_view="smart_scan",
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_cumulative_detections_and_latency_charts():
    """Verify that cumulative and latency charts render valid Plotly figures."""
    with open(OPERATIONAL_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    cum_fig = viz.cumulative_detections_chart(data["time_series"], current_t=100)
    assert isinstance(cum_fig, go.Figure)

    lat_fig = viz.emitter_latency_bar_chart(data["emitter_interceptions"])
    assert isinstance(lat_fig, go.Figure)


def test_band_frequency_range_calculation():
    """Test frequency band range calculation for boundary bands."""
    f_min, f_max, f_center = viz.get_band_freq_range("F01")
    assert f_min == 500.0
    assert f_max == 850.0
    assert f_center == 675.0

    f_min_50, f_max_50, f_center_50 = viz.get_band_freq_range("F50")
    assert f_min_50 == 17650.0
    assert f_max_50 == 18000.0
    assert f_center_50 == 17825.0
