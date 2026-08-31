"""Tests for Stage 12: Jury Demonstration Mode."""

import json
import os
import streamlit as st
import pytest

from dashboard import jury_demo
from dashboard.scenario_loader import get_validated_scenarios
from dashboard.multiscenario import calculate_aggregate_statistics

RESULTS_DIR = r"D:\sih\results"
SCAN_DIR = r"D:\sih\dataset\scan\test_scan"


def test_jury_demo_module_imports_and_has_render():
    """Verify jury_demo module exists and exposes render_jury_demo."""
    assert hasattr(jury_demo, "render_jury_demo")


def test_jury_demo_renders_all_six_stages():
    """Verify that each of the 6 guided stages renders without raising exceptions."""
    validated = get_validated_scenarios(results_dir=RESULTS_DIR, scan_dir=SCAN_DIR)
    assert len(validated) >= 1

    for stage_num in range(1, 7):
        st.session_state.jury_stage = stage_num
        st.session_state.jury_auto_play = False
        try:
            jury_demo.render_jury_demo(validated)
        except Exception as e:
            pytest.fail(f"Jury demo failed to render stage {stage_num}: {str(e)}")


def test_jury_demo_benchmark_matches_verified_artifacts():
    """Verify that aggregate values used in Jury Demo match the underlying 5 scenario artifacts."""
    validated = get_validated_scenarios(results_dir=RESULTS_DIR, scan_dir=SCAN_DIR)
    stats = calculate_aggregate_statistics(validated)

    assert stats["total_ss_true_detections"] == 162
    assert stats["total_ol_true_detections"] == 127
    assert stats["consistency"]["detection_advantage"] == (4, 5)
    assert stats["consistency"]["emitter_equal_or_better"] == (2, 5)
    assert stats["consistency"]["latency_advantage"] == (3, 5)
