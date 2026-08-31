"""Unit and integration tests for autonomous internal signal track manager."""

from core.tracker import TrackManager, TrackState
from rf_env.receiver import Observation


def test_track_creation_and_initialization():
    """Verify that first detection creates a NEW track with 50% confidence."""
    tm = TrackManager()
    obs = {"F27": Observation(timestep=1, band_id="F27", hit=True, signal_strength=-65.0, snr=18.0)}
    ch_telemetry = [{
        "band": "F27",
        "frequency_mhz": 9650.0,
        "amplitude_dbm": -65.0,
        "snr_db": 18.0,
        "pulse_width_us": 3.5,
        "aoa_deg": 42.0,
    }]

    events = tm.update(obs, ch_telemetry, timestep=1, simulated_time_s=0.05)
    assert len(tm.tracks) == 1
    assert "TRACK-001" in tm.tracks
    t1 = tm.tracks["TRACK-001"]
    assert t1.state == TrackState.NEW
    assert t1.hit_count == 1
    assert t1.confidence_pct == 50.0
    assert len(events) == 1


def test_track_confirmation_on_repeated_hits():
    """Verify that multiple hits promote track to CONFIRMED with higher confidence."""
    tm = TrackManager(confirmation_hits=2)
    obs1 = {"F27": Observation(timestep=1, band_id="F27", hit=True, signal_strength=-65.0, snr=18.0)}
    obs2 = {"F27": Observation(timestep=5, band_id="F27", hit=True, signal_strength=-64.0, snr=19.0)}
    ch_telemetry = [{
        "band": "F27",
        "frequency_mhz": 9650.0,
        "amplitude_dbm": -65.0,
        "snr_db": 18.0,
        "pulse_width_us": 3.5,
        "aoa_deg": 42.0,
    }]

    # Step 1
    tm.update(obs1, ch_telemetry, timestep=1, simulated_time_s=0.05)
    assert tm.tracks["TRACK-001"].state == TrackState.NEW

    # Step 2 (Hit again after 4 steps)
    tm.update(obs2, ch_telemetry, timestep=5, simulated_time_s=0.25)
    t1 = tm.tracks["TRACK-001"]
    assert t1.state == TrackState.CONFIRMED
    assert t1.hit_count == 2
    assert t1.confidence_pct >= 66.0
    assert t1.estimated_pri_timesteps == 4.0
    assert t1.expected_next_recurrence_s == 0.25 + (4.0 * 0.05)


def test_track_degradation_to_lost_and_expired():
    """Verify state transitions: CONFIRMED -> LOST (>8 steps) -> EXPIRED (>25 steps)."""
    tm = TrackManager(lost_threshold_steps=8, expire_threshold_steps=25)
    obs1 = {"F27": Observation(timestep=1, band_id="F27", hit=True, signal_strength=-65.0, snr=18.0)}
    obs2 = {"F27": Observation(timestep=2, band_id="F27", hit=True, signal_strength=-65.0, snr=18.0)}
    ch_telemetry = [{
        "band": "F27",
        "frequency_mhz": 9650.0,
        "amplitude_dbm": -65.0,
        "snr_db": 18.0,
        "pulse_width_us": 3.5,
        "aoa_deg": 42.0,
    }]

    # Create and confirm
    tm.update(obs1, ch_telemetry, timestep=1, simulated_time_s=0.05)
    tm.update(obs2, ch_telemetry, timestep=2, simulated_time_s=0.10)
    assert tm.tracks["TRACK-001"].state == TrackState.CONFIRMED

    # 8 steps without hits -> LOST
    tm.update({}, [], timestep=10, simulated_time_s=0.50)
    assert tm.tracks["TRACK-001"].state == TrackState.LOST

    # 26 steps without hits -> EXPIRED
    tm.update({}, [], timestep=28, simulated_time_s=1.40)
    assert tm.tracks["TRACK-001"].state == TrackState.EXPIRED


def test_zero_ground_truth_leakage_in_tracker():
    """Verify that track manager operates purely without ground-truth labels."""
    tm = TrackManager()
    assert not hasattr(tm, "ground_truth")
    assert not hasattr(tm, "emitter_truth")
    assert not hasattr(tm, "dataset_labels")


def test_track_manager_summary_table():
    """Verify get_tracks_summary formatting."""
    tm = TrackManager()
    obs = {"F10": Observation(timestep=1, band_id="F10", hit=True, signal_strength=-60.0, snr=20.0)}
    ch_telemetry = [{
        "band": "F10",
        "frequency_mhz": 3800.0,
        "amplitude_dbm": -60.0,
        "snr_db": 20.0,
        "pulse_width_us": 4.0,
        "aoa_deg": 90.0,
    }]

    tm.update(obs, ch_telemetry, timestep=1, simulated_time_s=0.05)
    summary = tm.get_tracks_summary()
    assert len(summary) == 1
    row = summary[0]
    assert row["Track ID"] == "TRACK-001"
    assert row["Band"] == "F10"
    assert row["State"] == "NEW"
    assert "3800.0" in row["Frequency (MHz)"]
