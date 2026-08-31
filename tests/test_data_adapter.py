"""Comprehensive test suite for the TSRD Data Adapter.

Validates:
- On-demand HDF5 loading (D:\sih\dataset\scan\test_scan\config_1.h5)
- Frequency mapping (500-18000 MHz -> 50 bands F01-F50)
- Time binning (microseconds -> seconds -> discrete steps)
- Ground-truth isolation in TruthManager
- Zero-leakage observation boundary in Receiver
- Seamless interoperability with Cognitive Smart Scan agents
"""

import inspect
import os
import numpy as np
import pytest

from data_adapter import (
    BandInfo,
    FrequencyMapper,
    PDWProcessor,
    TSRDLoader,
    TSRDRawData,
    TruthManager,
    TSRDEnvironment,
    EnvironmentSource,
    create_environment,
)
from rf_env import (
    Receiver,
    Observation,
    BeliefEngine,
    TemporalEngine,
    BandScoringEngine,
    QLearningArbitrator,
    IntelligentSchedulerAdapter,
    EvaluationMetrics,
    RFEnvironment,
)

SAMPLE_TSRD_PATH = r"D:\sih\dataset\scan\test_scan\config_1.h5"


def test_tsrd_loader_loads_config_1():
    """Task 2 & 10: Test that TSRDLoader opens representative file on-demand."""
    assert os.path.exists(SAMPLE_TSRD_PATH), f"Representative file not found at {SAMPLE_TSRD_PATH}"
    raw_data = TSRDLoader.load_file(SAMPLE_TSRD_PATH)
    assert isinstance(raw_data, TSRDRawData)
    assert raw_data.num_pulses == 50013
    assert raw_data.num_emitter_classes == 30
    assert len(raw_data.transmitter_metadata) == 36
    assert raw_data.receiver_metadata.collection_time_s == 30.0
    assert raw_data.receiver_metadata.sensitivity_dbm == -110.0


def test_tsrd_loader_features_decoded():
    """Task 2 & 10: Test feature names are properly decoded from HDF5 attributes."""
    raw_data = TSRDLoader.load_file(SAMPLE_TSRD_PATH)
    expected_feats = ["ToA", "Frequency", "PulseWidth", "AoA", "Amplitude"]
    assert raw_data.features == expected_feats
    assert raw_data.pulse_data.shape == (50013, 5)
    assert raw_data.labels.shape == (50013,)


def test_tsrd_loader_toa_monotonicity():
    """Task 2: Test that pulse sequence is strictly sorted by Time of Arrival."""
    raw_data = TSRDLoader.load_file(SAMPLE_TSRD_PATH)
    toa_col = raw_data.pulse_data[:, 0]
    assert np.all(toa_col[:-1] <= toa_col[1:]), "Pulse ToA values must be monotonically increasing."


def test_frequency_mapper_bounds_and_bands():
    """Task 3: Test deterministic 50-band frequency mapping over 500-18000 MHz."""
    mapper = FrequencyMapper(f_min_mhz=500.0, f_max_mhz=18000.0, num_bands=50)
    assert len(mapper.all_band_ids) == 50
    assert mapper.all_band_ids[0] == "F01"
    assert mapper.all_band_ids[-1] == "F50"
    assert mapper.band_width == pytest.approx(350.0)

    # Test exact boundaries
    b01 = mapper.get_band_info("F01")
    assert b01.freq_min_mhz == pytest.approx(500.0)
    assert b01.freq_max_mhz == pytest.approx(850.0)
    assert b01.freq_center_mhz == pytest.approx(675.0)

    b50 = mapper.get_band_info("F50")
    assert b50.freq_min_mhz == pytest.approx(17650.0)
    assert b50.freq_max_mhz == pytest.approx(18000.0)
    assert b50.freq_center_mhz == pytest.approx(17825.0)


def test_frequency_mapping_values():
    """Task 3: Test specific frequency mapping points."""
    mapper = FrequencyMapper(500.0, 18000.0, 50)
    assert mapper.freq_to_band_id(500.0) == "F01"
    assert mapper.freq_to_band_id(849.9) == "F01"
    assert mapper.freq_to_band_id(850.0) == "F02"
    assert mapper.freq_to_band_id(18000.0) == "F50"
    # Frequencies below min or above max clamp safely
    assert mapper.freq_to_band_id(100.0) == "F01"
    assert mapper.freq_to_band_id(25000.0) == "F50"


def test_pdw_processor_time_binning():
    """Task 4: Test conversion of ToA microseconds to seconds and discrete time binning."""
    raw_data = TSRDLoader.load_file(SAMPLE_TSRD_PATH)
    processor = PDWProcessor(raw_data, step_duration_s=0.05)
    assert processor.total_steps == 600
    assert processor.step_duration_s == 0.05

    # Check a populated time step
    total_binned_pulses = 0
    for t in range(processor.total_steps):
        step_act = processor.get_step_activity(t)
        for b_act in step_act.band_activities.values():
            total_binned_pulses += b_act.pulse_count
            assert b_act.max_amplitude_dbm >= b_act.mean_amplitude_dbm
            assert b_act.pulse_count > 0
            assert b_act.snr_db >= 0.0
    assert total_binned_pulses == 50013


def test_truth_manager_preserves_ground_truth():
    """Task 5: Test that TruthManager accurately tracks active emitter classes and total pulses."""
    raw_data = TSRDLoader.load_file(SAMPLE_TSRD_PATH)
    processor = PDWProcessor(raw_data, step_duration_s=0.05)
    truth_mgr = TruthManager(raw_data, processor)

    all_emitters = truth_mgr.get_all_emitter_ids()
    assert len(all_emitters) == 30
    assert len(raw_data.transmitter_metadata) == 36

    total_pulses = sum(truth_mgr.get_emitter_pulse_count(e) for e in all_emitters)
    assert total_pulses == 50013


def test_tsrd_environment_interface_parity():
    """Task 6 & 7: Test TSRDEnvironment matches RFEnvironment interface and steps seamlessly."""
    env = TSRDEnvironment(SAMPLE_TSRD_PATH, step_duration_s=0.05)
    assert len(env.bands) == 50
    assert env.total_steps == 600
    assert env.timestep == -1

    # Take steps
    for t in range(10):
        env.step()
        assert env.timestep == t
        truth_f01 = env.band_truth("F01")
        assert hasattr(truth_f01, "active")
        assert hasattr(truth_f01, "signal_strength")
        assert hasattr(truth_f01, "snr")
        assert hasattr(truth_f01, "emitter_id")
        assert hasattr(truth_f01, "emitter_type")

    snapshot = env.full_ground_truth_snapshot()
    assert len(snapshot) == 50


def test_receiver_constraint_on_tsrd_environment():
    """Task 7 & 8: Test Receiver observes exactly K=5 selected bands from TSRDEnvironment."""
    env = TSRDEnvironment(SAMPLE_TSRD_PATH, step_duration_s=0.05)
    receiver = Receiver(env, k=5)

    env.step()
    selected = ["F01", "F10", "F20", "F30", "F40"]
    obs = receiver.observe(selected)

    assert len(obs) == 5
    assert set(obs.keys()) == set(selected)
    for b_id, o in obs.items():
        assert isinstance(o, Observation)
        assert o.band_id == b_id
        assert isinstance(o.hit, bool)
        assert isinstance(o.signal_strength, float)
        assert isinstance(o.snr, float)


def test_zero_ground_truth_leakage_to_observation():
    """Task 5 & 8: Verify Observation object contains ZERO emitter labels or ground-truth details."""
    env = TSRDEnvironment(SAMPLE_TSRD_PATH, step_duration_s=0.05)
    receiver = Receiver(env, k=5)

    env.step()
    obs = receiver.observe(["F01", "F02", "F03", "F04", "F05"])
    for o in obs.values():
        assert not hasattr(o, "emitter_id")
        assert not hasattr(o, "emitter_type")
        assert not hasattr(o, "ground_truth_emitter_ids")


def test_closed_loop_cognitive_scheduler_on_tsrd():
    """Task 9: Run full IntelligentSchedulerAdapter (Belief + Temporal + Scoring + Q-Learning) on TSRD."""
    env = TSRDEnvironment(SAMPLE_TSRD_PATH, step_duration_s=0.05)
    receiver = Receiver(env, k=5)
    scheduler = IntelligentSchedulerAdapter(num_bands=50, k=5)
    metrics = EvaluationMetrics(redundancy_window=3)

    for t in range(50):
        env.step()
        selected_bands = scheduler.select_bands(t)
        assert len(selected_bands) == 5
        observations = receiver.observe(selected_bands)
        env.notify_scan_results(observations)
        metrics.observe_step(env, selected_bands, observations)
        scheduler.learn(observations, t)

    summary = metrics.summary()
    assert summary["total_scans"] == 50 * 5
    assert 0.0 <= summary["interception_rate"] <= 1.0
    assert 0.0 <= summary["pd"] <= 1.0


def test_environment_source_factory():
    """Task 9: Test create_environment factory for both SYNTHETIC and TSRD."""
    synthetic_env = create_environment(
        EnvironmentSource.SYNTHETIC,
        {"num_bands": 50, "receiver_channels": 5, "emitters": []}
    )
    assert isinstance(synthetic_env, RFEnvironment)
    assert len(synthetic_env.bands) == 50

    tsrd_env = create_environment(
        EnvironmentSource.TSRD,
        SAMPLE_TSRD_PATH,
        step_duration_s=0.05,
    )
    assert isinstance(tsrd_env, TSRDEnvironment)
    assert len(tsrd_env.bands) == 50
