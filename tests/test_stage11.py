"""Stage 11: integration-level edge-case / stress tests.

These exercise combinations not explicitly covered at the integration
level by Stages 1-10's own test files (extreme K, empty observations,
all-hit/all-miss detection configs, multiple evasion events through the
REAL RFEnvironment+Receiver+notify_scan_results loop). Existing per-stage
tests are not modified or weakened.
"""

import math

from rf_env import (RFEnvironment, Receiver, DetectionModel, BeliefEngine,
                     TemporalEngine, BandScoringEngine, QLearningArbitrator)
from dashboard.simulation_runner import SimulationRunner

BASE_EMITTERS = [
    {"id": "E1", "type": "static", "band": "F10", "signal_strength": -60.0,
     "snr": 15.0, "active_prob": 1.0},
]


def make_config(**overrides):
    cfg = {
        "num_bands": 50, "receiver_channels": 5, "random_seed": 1, "max_timesteps": 200,
        "emitters": BASE_EMITTERS,
        "detection": {"threshold_db": 10.0, "snr_scale": 3.0,
                      "false_alarm_probability": 0.05, "seed": 1},
        "belief": {"prior_alpha": 1.0, "prior_beta": 1.0, "decay_gamma": 0.98},
        "temporal": {"history_length": 50, "min_hits_for_prediction": 3,
                     "periodicity_threshold": 0.7, "stable_interval_max": 1.5},
        "scoring": {"staleness_scale": 10.0, "balanced_weights": {
            "exploration": 0.30, "exploitation": 0.40, "prediction": 0.30}},
        "ml_arbitrator": {"learning_rate": 0.1, "discount_factor": 0.9, "epsilon": 0.2,
                           "epsilon_decay": 0.995, "min_epsilon": 0.05,
                           "redundancy_window": 3, "redundant_scan_penalty": 0.20, "seed": 1},
        "adaptive_evasion": {"enabled": True, "hit_threshold": 3, "observation_window": 10,
                              "evasive_duration": 8, "seed": 1},
    }
    cfg.update(overrides)
    return cfg


def _all_finite(scores) -> bool:
    for s in scores:
        for v in (s.exploration_score, s.exploitation_score, s.prediction_score, s.balanced_score):
            if not math.isfinite(v):
                return False
    return True


def test_k_equals_1():
    config = make_config(receiver_channels=1)
    env = RFEnvironment(config)
    receiver = Receiver(env, k=1)
    env.step()
    obs = receiver.observe(["F10"])
    assert len(obs) == 1


def test_k_equals_num_bands():
    config = make_config(receiver_channels=50)
    env = RFEnvironment(config)
    receiver = Receiver(env, k=50)
    env.step()
    all_bands = env.bands
    obs = receiver.observe(all_bands)
    assert len(obs) == 50


def test_no_detections_config_stays_finite():
    # No emitters at all + zero false-alarm rate -> every scan is a genuine miss.
    config = make_config(emitters=[], detection={
        "threshold_db": 10.0, "snr_scale": 3.0, "false_alarm_probability": 0.0, "seed": 1})
    env = RFEnvironment(config)
    detection_model = DetectionModel(threshold_db=10.0, snr_scale=3.0,
                                      false_alarm_probability=0.0, seed=1)
    receiver = Receiver(env, k=5, detection_model=detection_model)
    belief = BeliefEngine(50, config.get("belief"))
    for t in range(20):
        env.step()
        obs = receiver.observe(["F01", "F02", "F03", "F04", "F05"])
        assert all(not o.hit for o in obs.values())
        belief.update(obs)
    b = belief.get_belief("F01")
    assert math.isfinite(b.activity_probability)
    assert b.activity_probability < 0.5  # repeated misses pull it below the neutral prior


def test_all_hits_config_stays_finite():
    # Very high SNR, low threshold -> P_d approx 1 every scan.
    config = make_config(emitters=[
        {"id": "E1", "type": "static", "band": "F10", "signal_strength": -50.0,
         "snr": 30.0, "active_prob": 1.0}])
    env = RFEnvironment(config)
    detection_model = DetectionModel(threshold_db=0.0, snr_scale=1.0,
                                      false_alarm_probability=0.0, seed=1)
    receiver = Receiver(env, k=5, detection_model=detection_model)
    belief = BeliefEngine(50, config.get("belief"))
    for t in range(20):
        env.step()
        obs = receiver.observe(["F10", "F20", "F30", "F40", "F50"])
        belief.update(obs)
    b = belief.get_belief("F10")
    assert math.isfinite(b.activity_probability)
    assert b.activity_probability > 0.9


def test_empty_observations_do_not_crash():
    belief = BeliefEngine(50)
    temporal = TemporalEngine(50)
    scoring = BandScoringEngine(50)
    arbitrator = QLearningArbitrator()

    belief.update({})
    temporal.update({}, 0)
    scoring.update(belief.get_state(), temporal.get_state(), 0)
    reward = arbitrator.calculate_reward({}, 0)
    assert reward == 0.0
    scores = scoring.get_scores()
    assert len(scores) == 50
    assert _all_finite(scores)


def test_never_observed_band_through_full_pipeline():
    belief = BeliefEngine(50)
    temporal = TemporalEngine(50)
    scoring = BandScoringEngine(50)
    scoring.update(belief.get_state(), temporal.get_state(), 0)
    s = scoring.score_band("F01")
    assert math.isfinite(s.exploration_score)
    assert s.prediction_score == 0.0  # no temporal prediction exists yet


def test_insufficient_temporal_history_reported_and_scored_zero():
    temporal = TemporalEngine(50, {"min_hits_for_prediction": 3})
    p = temporal.get_prediction("F01")
    assert p.behaviour_type == "insufficient_data"
    scoring = BandScoringEngine(50)
    belief = BeliefEngine(50)
    scoring.update(belief.get_state(), temporal.get_state(), 0)
    assert scoring.score_band("F01").prediction_score == 0.0


def test_multiple_evasion_events_through_full_environment_loop():
    config = make_config(
        emitters=[{"id": "E4", "type": "adaptive_evasive", "band": "F30",
                   "signal_strength": -50.0, "snr": 30.0}],
        adaptive_evasion={"enabled": True, "hit_threshold": 2, "observation_window": 10,
                           "evasive_duration": 3, "seed": 5},
    )
    env = RFEnvironment(config)
    detection_model = DetectionModel(threshold_db=0.0, snr_scale=1.0,
                                      false_alarm_probability=0.0, seed=1)
    receiver = Receiver(env, k=5, detection_model=detection_model)
    e4 = next(e for e in env.emitters if e.emitter_id == "E4")
    filler = ["F01", "F02", "F03", "F04", "F05", "F06"]
    for t in range(150):
        env.step()
        selected = list(dict.fromkeys([e4.current_band] + filler))[:5]
        obs = receiver.observe(selected)
        env.notify_scan_results(obs)
    assert e4.evasion_count >= 2


def test_scoring_finite_cold_and_after_heavy_observation():
    runner = SimulationRunner(seed=3)
    runner.run(80)
    scores = runner.scores()
    assert _all_finite(scores)
    assert all(0.0 <= s.balanced_score <= 1.0 for s in scores)


def test_receiver_k_constraint_respected_across_run():
    runner = SimulationRunner(seed=4)
    for _ in range(30):
        record = runner.step()
        assert len(record["selected_bands"]) == runner.k
        assert len(set(record["selected_bands"])) == runner.k
