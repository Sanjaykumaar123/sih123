import inspect

import rf_env.temporal as temporal_module
from rf_env import TemporalEngine
from rf_env.receiver import Observation


def make_obs(band_id, hit, timestep, snr=15.0):
    return Observation(timestep=timestep, band_id=band_id, hit=hit,
                        signal_strength=-60.0 if hit else 0.0,
                        snr=snr if hit else 0.0,
                        detection_probability=0.9 if hit else 0.1)


def feed_hits(engine, band_id, hit_timesteps, last_timestep):
    """Feed HIT observations at the given timesteps, MISS everywhere else
    up to last_timestep -- mimics a real scan history."""
    hit_set = set(hit_timesteps)
    for t in range(last_timestep + 1):
        engine.update({band_id: make_obs(band_id, t in hit_set, t)}, t)


def test_empty_history_behaves_correctly():
    engine = TemporalEngine(50)
    p = engine.get_prediction("F01")
    assert p.number_of_hits == 0
    assert p.periodicity_score == 0.0
    assert p.estimated_period is None
    assert p.predicted_next_active_time is None
    assert p.prediction_confidence == 0.0
    assert p.behaviour_type == "insufficient_data"
    assert p.last_hit_timestep is None
    assert p.time_since_last_hit == float("inf")


def test_insufficient_hits_produce_no_prediction():
    engine = TemporalEngine(50)
    feed_hits(engine, "F01", [5, 15], last_timestep=15)  # only 2 hits, min=3
    p = engine.get_prediction("F01")
    assert p.predicted_next_active_time is None
    assert p.prediction_confidence == 0.0
    assert p.behaviour_type == "insufficient_data"


def test_evenly_spaced_hits_give_high_periodicity():
    engine = TemporalEngine(50)
    feed_hits(engine, "F01", [5, 15, 25, 35], last_timestep=35)
    p = engine.get_prediction("F01")
    assert p.periodicity_score > 0.95


def test_irregular_hits_give_lower_periodicity():
    engine = TemporalEngine(50)
    feed_hits(engine, "F02", [5, 15, 25, 35], last_timestep=35)
    even = engine.get_prediction("F02")

    engine2 = TemporalEngine(50)
    feed_hits(engine2, "F02", [5, 12, 30, 31], last_timestep=31)
    irregular = engine2.get_prediction("F02")

    assert irregular.periodicity_score < even.periodicity_score


def test_estimated_period_is_approximately_correct():
    engine = TemporalEngine(50)
    feed_hits(engine, "F01", [5, 15, 25, 35], last_timestep=35)
    p = engine.get_prediction("F01")
    assert p.estimated_period == 10.0


def test_next_active_prediction_is_approximately_correct():
    engine = TemporalEngine(50)
    feed_hits(engine, "F01", [5, 15, 25, 35], last_timestep=35)
    p = engine.get_prediction("F01")
    assert p.predicted_next_active_time == 45.0


def test_confidence_is_bounded():
    engine = TemporalEngine(50)
    feed_hits(engine, "F01", [5, 15, 25, 35], last_timestep=35)
    p = engine.get_prediction("F01")
    assert 0.0 <= p.prediction_confidence <= 1.0


def test_unselected_bands_get_no_fake_observations():
    engine = TemporalEngine(50)
    engine.update({"F01": make_obs("F01", True, 0)}, 0)
    p = engine.get_prediction("F02")
    assert p.number_of_hits == 0
    assert p.last_hit_timestep is None
    assert p.time_since_last_hit == float("inf")


def test_last_hit_updates_only_on_actual_hit():
    engine = TemporalEngine(50)
    engine.update({"F01": make_obs("F01", True, 0)}, 0)
    engine.update({"F01": make_obs("F01", False, 1)}, 1)
    p = engine.get_prediction("F01")
    assert p.last_hit_timestep == 0


def test_time_since_last_hit_increases():
    engine = TemporalEngine(50)
    engine.update({"F01": make_obs("F01", True, 0)}, 0)
    assert engine.get_prediction("F01").time_since_last_hit == 0
    engine.update({}, 5)
    assert engine.get_prediction("F01").time_since_last_hit == 5


def test_history_respects_max_length():
    engine = TemporalEngine(50, {"history_length": 10})
    for t in range(30):
        engine.update({"F01": make_obs("F01", t % 2 == 0, t)}, t)
    assert len(engine._history["F01"]) == 10


def test_reset_clears_temporal_state():
    engine = TemporalEngine(50)
    feed_hits(engine, "F01", [5, 15, 25, 35], last_timestep=35)
    engine.reset()
    p = engine.get_prediction("F01")
    assert p.number_of_hits == 0
    assert p.last_hit_timestep is None
    assert p.behaviour_type == "insufficient_data"


def test_temporal_engine_has_no_ground_truth_access():
    # Public API only accepts observations/timestep -- no environment arg.
    sig = inspect.signature(TemporalEngine.update)
    assert list(sig.parameters) == ["self", "observations", "current_timestep"]
    # No actual import of / call into environment/ground-truth internals
    # (checked as code patterns, not prose, so the module's own docstring
    # explaining this boundary doesn't trip the check).
    source = inspect.getsource(temporal_module)
    assert "from .environment import" not in source
    assert "band_truth(" not in source
    assert ".emitter_id" not in source
    assert ".emitter_type" not in source
