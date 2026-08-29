import inspect

import pytest

import rf_env.belief as belief_module
from rf_env import BeliefEngine
from rf_env.receiver import Observation


def make_obs(band_id, hit, timestep):
    return Observation(timestep=timestep, band_id=band_id, hit=hit,
                        signal_strength=-60.0 if hit else 0.0,
                        snr=15.0 if hit else 0.0,
                        detection_probability=0.9 if hit else 0.1)


def test_neutral_prior_gives_half_probability():
    engine = BeliefEngine(50)
    belief = engine.get_belief("F01")
    assert belief.activity_probability == 0.5


def test_repeated_hits_increase_probability():
    engine = BeliefEngine(50, {"decay_gamma": 1.0})
    p_before = engine.get_belief("F01").activity_probability
    for t in range(5):
        engine.update({"F01": make_obs("F01", True, t)})
    p_after = engine.get_belief("F01").activity_probability
    assert p_after > p_before


def test_repeated_misses_decrease_probability():
    engine = BeliefEngine(50, {"decay_gamma": 1.0})
    p_before = engine.get_belief("F01").activity_probability
    for t in range(5):
        engine.update({"F01": make_obs("F01", False, t)})
    p_after = engine.get_belief("F01").activity_probability
    assert p_after < p_before


def test_hit_updates_only_the_selected_band():
    engine = BeliefEngine(50, {"decay_gamma": 1.0})
    engine.update({"F01": make_obs("F01", True, 0)})
    selected = engine.get_belief("F01")
    assert (selected.alpha, selected.beta) == (2.0, 1.0)


def test_unselected_bands_remain_unchanged():
    engine = BeliefEngine(50, {"decay_gamma": 1.0})
    engine.update({"F01": make_obs("F01", True, 0)})
    untouched = engine.get_belief("F02")
    assert (untouched.alpha, untouched.beta) == (1.0, 1.0)
    assert untouched.last_observed is None


def test_uncertainty_decreases_with_evidence():
    engine = BeliefEngine(50, {"decay_gamma": 1.0})
    u_before = engine.get_belief("F01").uncertainty
    for t in range(10):
        engine.update({"F01": make_obs("F01", True, t)})
    u_after = engine.get_belief("F01").uncertainty
    assert u_after < u_before


def test_decay_reduces_influence_of_old_evidence():
    engine = BeliefEngine(50, {"decay_gamma": 0.9})
    for t in range(10):
        engine.update({"F01": make_obs("F01", True, t)})
    p_strong = engine.get_belief("F01").activity_probability
    # Stop observing F01; only decay acts on it for many steps.
    for t in range(10, 200):
        engine.update({})
    p_decayed = engine.get_belief("F01").activity_probability
    assert p_decayed < p_strong
    assert p_decayed == pytest.approx(0.5, abs=0.05)


def test_staleness_increases_when_not_observed():
    engine = BeliefEngine(50)
    engine.update({"F01": make_obs("F01", True, 0)})
    assert engine.get_belief("F01").staleness == 0
    for _ in range(3):
        engine.update({})  # nothing observed; internal clock still ticks
    b = engine.get_belief("F01")
    assert b.last_observed == 0
    assert b.staleness == 3
    never = engine.get_belief("F02")
    assert never.last_observed is None
    assert never.staleness == float("inf")


def test_reset_returns_initial_state():
    engine = BeliefEngine(50)
    engine.update({"F01": make_obs("F01", True, 0)})
    engine.reset()
    b = engine.get_belief("F01")
    assert (b.alpha, b.beta) == (1.0, 1.0)
    assert b.activity_probability == 0.5
    assert b.last_observed is None
    assert b.staleness == float("inf")


def test_engine_has_no_ground_truth_access():
    sig = inspect.signature(BeliefEngine.update)
    assert list(sig.parameters) == ["self", "observations"]
    source = inspect.getsource(belief_module)
    assert "RFEnvironment" not in source
    assert "band_truth" not in source
    assert "emitter_id" not in source
    assert "emitter_type" not in source
