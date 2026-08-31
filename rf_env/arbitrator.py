"""Q-Learning ML Strategy Arbitrator."""

from enum import IntEnum
from typing import Dict, List, Optional, Tuple
import numpy as np

from .belief import BandBelief


class Strategy(IntEnum):
    EXPLORE = 0
    EXPLOIT = 1
    PREDICT = 2
    BALANCED = 3


_ACTION_TO_STRATEGY_NAME = {
    Strategy.EXPLORE: "exploration",
    Strategy.EXPLOIT: "exploitation",
    Strategy.PREDICT: "prediction",
    Strategy.BALANCED: "balanced",
}

_STRATEGY_NAME_TO_ACTION = {v: k for k, v in _ACTION_TO_STRATEGY_NAME.items()}


class QLearningArbitrator:
    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.learning_rate = float(cfg.get("learning_rate", 0.1))
        self.discount_factor = float(cfg.get("discount_factor", 0.9))
        self.initial_epsilon = float(cfg.get("epsilon", 0.20))
        self.epsilon = self.initial_epsilon
        self.epsilon_decay = float(cfg.get("epsilon_decay", 0.995))
        self.min_epsilon = float(cfg.get("min_epsilon", 0.05))

        self.redundancy_window = int(cfg.get("redundancy_window", 3))
        self.redundant_scan_penalty = float(cfg.get("redundant_scan_penalty", 0.20))

        self.seed = cfg.get("seed", 42)
        self.rng = np.random.RandomState(self.seed) if self.seed is not None else np.random.RandomState()

        self.q_table = np.zeros((3, 3, 3, 4), dtype=np.float64)
        self._last_scan_times: Dict[str, int] = {}
        self._strategy_stats: Dict[str, Dict] = {
            name: {"selection_count": 0, "rewards": []}
            for name in _ACTION_TO_STRATEGY_NAME.values()
        }

    def reset(self) -> None:
        self.epsilon = self.initial_epsilon
        self._last_scan_times = {}
        self.q_table.fill(0.0)
        self._strategy_stats = {
            name: {"selection_count": 0, "rewards": []}
            for name in _ACTION_TO_STRATEGY_NAME.values()
        }

    def get_state(self, belief_state: List[BandBelief]) -> Tuple[int, int, int]:
        # Discretize belief state into (perf_level, uncertainty_level, detection_level) in {0, 1, 2}^3
        if not belief_state:
            return (0, 0, 0)

        # 1. Performance / Hit Rate bucket
        total_hits = sum(b.hit_count for b in belief_state)
        total_obs = sum(b.hit_count + b.miss_count for b in belief_state)
        hit_rate = (total_hits / total_obs) if total_obs > 0 else 0.0
        if hit_rate < 0.10:
            perf_bin = 0
        elif hit_rate < 0.25:
            perf_bin = 1
        else:
            perf_bin = 2

        # 2. Uncertainty bucket
        mean_unc = float(np.mean([b.uncertainty for b in belief_state]))
        if mean_unc < 0.02:
            unc_bin = 0
        elif mean_unc < 0.05:
            unc_bin = 1
        else:
            unc_bin = 2

        # 3. Staleness / Activity bucket
        mean_act = float(np.mean([b.activity_probability for b in belief_state]))
        if mean_act < 0.40:
            act_bin = 0
        elif mean_act < 0.60:
            act_bin = 1
        else:
            act_bin = 2

        return (perf_bin, unc_bin, act_bin)

    def choose_action(self, state: Tuple[int, int, int]) -> int:
        if self.rng.uniform(0.0, 1.0) < self.epsilon:
            return int(self.rng.choice(4))
        q_vals = self.q_table[state]
        max_v = np.max(q_vals)
        best_actions = np.where(q_vals == max_v)[0]
        return int(self.rng.choice(best_actions))

    def select_strategy(self, state: Tuple[int, int, int]) -> str:
        action = self.choose_action(state)
        return _ACTION_TO_STRATEGY_NAME[Strategy(action)]

    def calculate_reward(self, observations: dict, timestep: int) -> float:
        if not observations:
            return 0.0

        new_hits = 0.0
        redundant_misses = 0.0

        for band_id, obs in observations.items():
            if obs.hit:
                new_hits += 1.0
                self._last_scan_times[band_id] = timestep
            else:
                last_time = self._last_scan_times.get(band_id)
                if last_time is not None and (timestep - last_time) <= self.redundancy_window:
                    redundant_misses += 1.0
                self._last_scan_times[band_id] = timestep

        return float(new_hits - self.redundant_scan_penalty * redundant_misses)

    def update(self, state: Tuple[int, int, int], action: int, reward: float,
               next_state: Tuple[int, int, int]) -> None:
        action_int = int(action)
        if action_int not in (0, 1, 2, 3):
            raise ValueError(f"Invalid action: {action}. Must be 0, 1, 2, or 3.")

        current_q = self.q_table[state][action_int]
        max_next_q = float(np.max(self.q_table[next_state]))
        td_target = reward + self.discount_factor * max_next_q
        self.q_table[state][action_int] += self.learning_rate * (td_target - current_q)

        # Decay epsilon
        self.epsilon = float(max(self.min_epsilon, self.epsilon * self.epsilon_decay))

        # Record strategy statistics
        strat_name = _ACTION_TO_STRATEGY_NAME[Strategy(action_int)]
        self._strategy_stats[strat_name]["selection_count"] += 1
        self._strategy_stats[strat_name]["rewards"].append(reward)

    def get_q_values(self, state: Tuple[int, int, int]) -> np.ndarray:
        return self.q_table[state].copy()

    def get_strategy_statistics(self) -> Dict[str, Dict]:
        result = {}
        for name, stats in self._strategy_stats.items():
            rewards = stats["rewards"]
            recent = rewards[-50:] if rewards else [0.0]
            result[name] = {
                "selection_count": stats["selection_count"],
                "average_recent_reward": float(np.mean(recent)),
            }
        return result
