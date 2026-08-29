import inspect

import numpy as np
import pytest

import rf_env.arbitrator as arbitrator_module
from rf_env import QLearningArbitrator, Strategy, BandBelief
from rf_env.receiver import Observation


def make_belief(band_id, uncertainty=1.0 / 12.0):
    return BandBelief(band_id=band_id, alpha=1.0, beta=1.0,
                       activity_probability=0.5, uncertainty=uncertainty,
                       last_observed=None, staleness=float("inf"),
                       hit_count=0, miss_count=0)


def make_obs(band_id, hit, timestep):
    return Observation(timestep=timestep, band_id=band_id, hit=hit,
                        signal_strength=-60.0 if hit else 0.0,
                        snr=15.0 if hit else 0.0,
                        detection_probability=0.9 if hit else 0.1)


BELIEF_STATE = [make_belief(f"F{i:02d}") for i in range(1, 51)]


def test_q_table_shape_is_27_states_by_4_actions():
    arb = QLearningArbitrator()
    assert arb.q_table.shape == (3, 3, 3, 4)
    assert arb.q_table.size == 108


def test_initial_q_values_are_zero():
    arb = QLearningArbitrator()
    assert np.all(arb.q_table == 0.0)


def test_valid_states_map_to_valid_entries():
    arb = QLearningArbitrator()
    state = arb.get_state(BELIEF_STATE)
    assert len(state) == 3
    assert all(s in (0, 1, 2) for s in state)
    q_values = arb.get_q_values(state)
    assert q_values.shape == (4,)


def test_all_four_actions_are_valid():
    assert sorted(int(a) for a in Strategy) == [0, 1, 2, 3]
    assert set(arbitrator_module._ACTION_TO_STRATEGY_NAME.values()) == {
        "exploration", "exploitation", "prediction", "balanced"}


def test_invalid_action_rejected():
    arb = QLearningArbitrator()
    state = (0, 0, 0)
    with pytest.raises(ValueError):
        arb.update(state, 4, 1.0, state)
    with pytest.raises(ValueError):
        arb.update(state, -1, 1.0, state)


def test_update_changes_q_value_on_nonzero_reward():
    arb = QLearningArbitrator()
    state = (1, 1, 1)
    before = arb.get_q_values(state)[0]
    arb.update(state, 0, 1.0, state)
    after = arb.get_q_values(state)[0]
    assert after != before


def test_q_learning_equation_correct():
    arb = QLearningArbitrator({"learning_rate": 0.1, "discount_factor": 0.9})
    state, next_state, action, reward = (0, 0, 0), (1, 1, 1), 2, 1.0
    arb.q_table[next_state] = [0.5, 0.2, 0.9, 0.1]
    expected = 0.0 + 0.1 * (reward + 0.9 * 0.9 - 0.0)
    arb.update(state, action, reward, next_state)
    assert arb.get_q_values(state)[action] == pytest.approx(expected)


def test_negative_reward_decreases_q_value():
    arb = QLearningArbitrator()
    state = (2, 2, 2)
    arb.update(state, 1, -1.0, state)
    assert arb.get_q_values(state)[1] < 0.0


def test_epsilon_greedy_can_explore():
    arb = QLearningArbitrator({"epsilon": 1.0, "seed": 1})
    arb.q_table[(0, 0, 0)] = [10.0, 0.0, 0.0, 0.0]  # action 0 clearly best
    actions = {arb.choose_action((0, 0, 0)) for _ in range(200)}
    assert len(actions) > 1  # pure random must surface more than one action


def test_epsilon_greedy_can_exploit():
    arb = QLearningArbitrator({"epsilon": 0.0, "seed": 1})
    arb.q_table[(0, 0, 0)] = [0.1, 0.9, 0.2, 0.05]
    actions = {arb.choose_action((0, 0, 0)) for _ in range(50)}
    assert actions == {1}


def test_epsilon_bounded_within_range():
    arb = QLearningArbitrator({"epsilon": 0.20, "min_epsilon": 0.05, "epsilon_decay": 0.9})
    state = (0, 0, 0)
    for _ in range(500):
        arb.update(state, 0, 0.0, state)
        assert arb.min_epsilon <= arb.epsilon <= arb.initial_epsilon


def test_epsilon_decay_reduces_epsilon():
    arb = QLearningArbitrator({"epsilon": 0.20, "epsilon_decay": 0.9, "min_epsilon": 0.01})
    before = arb.epsilon
    arb.update((0, 0, 0), 0, 0.0, (0, 0, 0))
    assert arb.epsilon < before


def test_same_seed_is_reproducible():
    arb_a = QLearningArbitrator({"epsilon": 1.0, "seed": 42})
    arb_b = QLearningArbitrator({"epsilon": 1.0, "seed": 42})
    seq_a = [arb_a.choose_action((0, 0, 0)) for _ in range(30)]
    seq_b = [arb_b.choose_action((0, 0, 0)) for _ in range(30)]
    assert seq_a == seq_b


def test_different_seeds_can_differ():
    arb_a = QLearningArbitrator({"epsilon": 1.0, "seed": 1})
    arb_b = QLearningArbitrator({"epsilon": 1.0, "seed": 2})
    seq_a = [arb_a.choose_action((0, 0, 0)) for _ in range(30)]
    seq_b = [arb_b.choose_action((0, 0, 0)) for _ in range(30)]
    assert seq_a != seq_b


def test_reward_counts_new_hits():
    arb = QLearningArbitrator()
    obs = {"F01": make_obs("F01", True, 0), "F02": make_obs("F02", False, 0)}
    reward = arb.calculate_reward(obs, 0)
    assert reward == 1.0  # one hit, one first-time (non-redundant) miss


def test_redundant_scan_receives_penalty():
    arb = QLearningArbitrator({"redundancy_window": 3, "redundant_scan_penalty": 0.20})
    arb.calculate_reward({"F01": make_obs("F01", False, 0)}, 0)  # first scan, no penalty
    reward = arb.calculate_reward({"F01": make_obs("F01", False, 2)}, 2)  # miss again, within window
    assert reward == pytest.approx(-0.20)


def test_normal_miss_gets_no_penalty():
    arb = QLearningArbitrator()
    reward = arb.calculate_reward({"F01": make_obs("F01", False, 0)}, 0)
    assert reward == 0.0


def test_strategy_reward_history_updates():
    arb = QLearningArbitrator()
    state = (0, 0, 0)
    arb.update(state, Strategy.EXPLOIT, 1.0, state)
    stats = arb.get_strategy_statistics()
    assert stats["exploitation"]["selection_count"] == 1
    assert stats["exploitation"]["average_recent_reward"] == pytest.approx(1.0)
    assert stats["exploration"]["selection_count"] == 0


def test_arbitrator_has_no_ground_truth_access():
    sig = inspect.signature(QLearningArbitrator.get_state)
    assert list(sig.parameters) == ["self", "belief_state"]
    source = inspect.getsource(arbitrator_module)
    assert "from .environment import" not in source
    assert "from .receiver import" not in source
    assert "band_truth(" not in source
    assert "GroundTruthLogger" not in source
    assert ".emitter_id" not in source
    assert ".emitter_type" not in source
