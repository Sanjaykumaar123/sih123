import inspect

import rf_env.evaluation as evaluation_module
from rf_env import (RoundRobinScheduler, RandomKScheduler, EvaluationMetrics,
                     RewardTracker, QLearningArbitrator, RFEnvironment,
                     Receiver, run_single_experiment)
from rf_env.receiver import Observation

MINI_CONFIG = {
    "num_bands": 50, "receiver_channels": 5, "random_seed": 1, "max_timesteps": 500,
    "emitters": [
        {"id": "E1", "type": "static", "band": "F10", "signal_strength": -60.0,
         "snr": 15.0, "active_prob": 1.0},
    ],
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
}


def make_obs(band_id, hit, timestep):
    return Observation(timestep=timestep, band_id=band_id, hit=hit,
                        signal_strength=-60.0 if hit else 0.0,
                        snr=15.0 if hit else 0.0,
                        detection_probability=0.9 if hit else 0.1)


def test_round_robin_selects_exactly_k_bands():
    sched = RoundRobinScheduler(num_bands=50, k=5)
    for t in range(20):
        bands = sched.select_bands(t)
        assert len(bands) == 5
        assert len(set(bands)) == 5


def test_round_robin_expected_sequence():
    sched = RoundRobinScheduler(num_bands=50, k=5)
    assert sched.select_bands(0) == ["F01", "F02", "F03", "F04", "F05"]
    assert sched.select_bands(1) == ["F06", "F07", "F08", "F09", "F10"]
    assert sched.select_bands(9) == ["F46", "F47", "F48", "F49", "F50"]
    assert sched.select_bands(10) == sched.select_bands(0)  # restarts


def test_random_k_selects_exactly_k_unique_bands():
    sched = RandomKScheduler(num_bands=50, k=5, seed=7)
    for t in range(20):
        bands = sched.select_bands(t)
        assert len(bands) == 5
        assert len(set(bands)) == 5


def test_random_k_reproducible_with_same_seed():
    a = RandomKScheduler(num_bands=50, k=5, seed=42)
    b = RandomKScheduler(num_bands=50, k=5, seed=42)
    seq_a = [a.select_bands(t) for t in range(10)]
    seq_b = [b.select_bands(t) for t in range(10)]
    assert seq_a == seq_b


def test_random_k_differs_with_different_seeds():
    a = RandomKScheduler(num_bands=50, k=5, seed=1)
    b = RandomKScheduler(num_bands=50, k=5, seed=2)
    seq_a = [a.select_bands(t) for t in range(10)]
    seq_b = [b.select_bands(t) for t in range(10)]
    assert seq_a != seq_b


def test_baselines_cannot_access_ground_truth():
    # Constructors take no environment/ground-truth reference at all.
    rr_sig = inspect.signature(RoundRobinScheduler.__init__)
    rk_sig = inspect.signature(RandomKScheduler.__init__)
    assert "env" not in rr_sig.parameters and "environment" not in rr_sig.parameters
    assert "env" not in rk_sig.parameters and "environment" not in rk_sig.parameters
    source = inspect.getsource(evaluation_module)
    # band_truth/GroundTruthLogger may appear in EvaluationMetrics/runner
    # (explicitly permitted, post-hoc only) -- but never inside the
    # scheduler classes' own select_bands methods.
    rr_src = inspect.getsource(RoundRobinScheduler)
    rk_src = inspect.getsource(RandomKScheduler)
    for src in (rr_src, rk_src):
        assert "band_truth" not in src
        assert "env." not in src
        assert "GroundTruthLogger" not in src


def test_pd_hand_computable_example():
    metrics = EvaluationMetrics()
    config = dict(MINI_CONFIG)
    env = RFEnvironment(config)
    env.step()  # F10 (E1) active this and every step (active_prob=1.0)
    # 1 true detection, 1 miss on the same active band, 3 quiet scans
    observations = {
        "F10": make_obs("F10", True, 0),
        "F20": make_obs("F20", False, 0),
        "F30": make_obs("F30", False, 0),
        "F40": make_obs("F40", False, 0),
        "F50": make_obs("F50", False, 0),
    }
    metrics.observe_step(env, list(observations.keys()), observations)
    s = metrics.summary()
    assert s["pd"] == 1.0  # 1 true detection / 1 opportunity (F10 only)
    assert s["total_hits"] == 1
    assert s["total_scans"] == 5


def test_pfa_hand_computable_example():
    metrics = EvaluationMetrics()
    config = dict(MINI_CONFIG)
    env = RFEnvironment(config)
    env.step()
    observations = {
        "F20": make_obs("F20", True, 0),   # false alarm: F20 has no emitter
        "F30": make_obs("F30", False, 0),
        "F40": make_obs("F40", False, 0),
        "F50": make_obs("F50", False, 0),
    }
    metrics.observe_step(env, list(observations.keys()), observations)
    s = metrics.summary()
    assert s["pfa"] == 0.25  # 1 false detection / 4 quiet scans


def test_reward_calculation_consistent_with_arbitrator():
    arb = QLearningArbitrator({"redundancy_window": 3, "redundant_scan_penalty": 0.20})
    tracker = RewardTracker(redundancy_window=3, redundant_scan_penalty=0.20)
    obs_sequence = [
        {"F01": make_obs("F01", False, 0)},
        {"F01": make_obs("F01", False, 2)},   # redundant miss
        {"F01": make_obs("F01", True, 5)},    # hit, not redundant
    ]
    for t, obs in [(0, obs_sequence[0]), (2, obs_sequence[1]), (5, obs_sequence[2])]:
        r_arb = arb.calculate_reward(obs, t)
        r_track = tracker.compute(obs, t)
        assert r_arb == r_track


def test_evaluation_runs_without_crashing():
    def make_rr(num_bands, k, seed):
        return RoundRobinScheduler(num_bands, k)
    result = run_single_experiment("round_robin", make_rr, MINI_CONFIG, num_steps=30, seed=1)
    assert result["scheduler"] == "round_robin"
    assert result["steps"] == 30
    assert result["total_scans"] == 30 * 5


def test_all_schedulers_obey_same_k():
    def make_rr(n, k, seed):
        return RoundRobinScheduler(n, k)

    def make_rk(n, k, seed):
        return RandomKScheduler(n, k, seed=seed)

    for factory, name in [(make_rr, "rr"), (make_rk, "rk")]:
        result = run_single_experiment(name, factory, MINI_CONFIG, num_steps=10, seed=1)
        assert result["total_scans"] == 10 * MINI_CONFIG["receiver_channels"]


def test_ground_truth_used_only_inside_evaluation():
    # Receiver's own observation contract is untouched by Stage 8.
    env = RFEnvironment(dict(MINI_CONFIG))
    receiver = Receiver(env, k=5)
    env.step()
    obs = receiver.observe(["F10", "F20", "F30", "F40", "F50"])
    assert not hasattr(obs["F10"], "emitter_id")
    assert not hasattr(obs["F10"], "emitter_type")
