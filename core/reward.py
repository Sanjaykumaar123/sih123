"""Reward computation for reinforcement learning and benchmark evaluation."""

from typing import Dict, List, Optional
from rf_env.receiver import Observation
from rf_env.evaluation import RewardTracker

__all__ = ["RewardTracker", "compute_evaluated_step_reward"]


def compute_evaluated_step_reward(
    selected_bands: List[str],
    observations: Dict[str, Observation],
    active_bands_truth: set,
    last_scan_times: Dict[str, int],
    timestep: int,
    true_reward: float = 2.0,
    fa_penalty: float = 0.5,
    redundant_miss_penalty: float = 0.20,
    redundancy_window: int = 3,
) -> float:
    """Compute post-hoc evaluated benchmark reward (+2.0 True, -0.5 FA, -0.2 Redundant Miss)."""
    step_eval_reward = 0.0
    for b in selected_bands:
        obs = observations.get(b)
        is_active = b in active_bands_truth
        last_t = last_scan_times.get(b)
        is_redundant = (last_t is not None) and ((timestep - last_t) <= redundancy_window)
        last_scan_times[b] = timestep

        if obs and obs.hit:
            if is_active:
                step_eval_reward += true_reward
            else:
                step_eval_reward -= fa_penalty
        else:
            if is_redundant:
                step_eval_reward -= redundant_miss_penalty

    return float(step_eval_reward)
