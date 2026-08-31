"""Evaluation module containing baseline schedulers, metrics tracking, and experiment harnesses."""

from typing import Callable, Dict, List, Optional
import numpy as np

from .belief import BeliefEngine
from .temporal import TemporalEngine
from .scoring import BandScoringEngine
from .arbitrator import QLearningArbitrator, _ACTION_TO_STRATEGY_NAME, Strategy
from .environment import RFEnvironment
from .receiver import Receiver, Observation
from .detection import DetectionModel


class RoundRobinScheduler:
    def __init__(self, num_bands: int, k: int):
        self.num_bands = int(num_bands)
        self.k = int(k)
        self.bands = [f"F{i:02d}" for i in range(1, self.num_bands + 1)]

    def select_bands(self, t: int) -> List[str]:
        total_groups = (self.num_bands + self.k - 1) // self.k
        group_idx = t % total_groups
        start_idx = group_idx * self.k
        end_idx = min(self.num_bands, start_idx + self.k)
        selected = self.bands[start_idx:end_idx]
        # If group is smaller than k (e.g. boundary wrap), pad from start
        if len(selected) < self.k:
            selected.extend(self.bands[: self.k - len(selected)])
        return selected


class RandomKScheduler:
    def __init__(self, num_bands: int, k: int, seed: int = 42):
        self.num_bands = int(num_bands)
        self.k = int(k)
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.bands = [f"F{i:02d}" for i in range(1, self.num_bands + 1)]

    def select_bands(self, t: int) -> List[str]:
        chosen_indices = self.rng.choice(self.num_bands, size=self.k, replace=False)
        return [self.bands[i] for i in chosen_indices]


class IntelligentSchedulerAdapter:
    def __init__(self, num_bands: int, k: int, belief_config: Optional[dict] = None,
                 temporal_config: Optional[dict] = None, scoring_config: Optional[dict] = None,
                 arbitrator_config: Optional[dict] = None):
        self.num_bands = int(num_bands)
        self.k = int(k)
        self.belief = BeliefEngine(num_bands, belief_config)
        self.temporal = TemporalEngine(num_bands, temporal_config)
        self.scoring = BandScoringEngine(num_bands, scoring_config)
        self.arbitrator = QLearningArbitrator(arbitrator_config)

        self.last_strategy: str = "balanced"
        self._last_state: tuple = (0, 0, 0)
        self._last_action: int = 3
        self._last_predictions: Dict[str, Optional[float]] = {}

    def select_bands(self, timestep: int) -> List[str]:
        belief_state = self.belief.get_state()
        temporal_state = self.temporal.get_state()

        # Update scoring engine with current states
        self.scoring.update(belief_state, temporal_state, timestep)

        # Snapshot temporal predictions for accuracy evaluation
        self._last_predictions = {
            t.band_id: t.predicted_next_active_time for t in temporal_state
        }

        # Choose strategy via Q-learning arbitrator
        self._last_state = self.arbitrator.get_state(belief_state)
        self._last_action = self.arbitrator.choose_action(self._last_state)
        self.last_strategy = _ACTION_TO_STRATEGY_NAME[Strategy(self._last_action)]

        # Select top-k bands
        return self.scoring.top_k(self.last_strategy, self.k)

    def check_predictions(self, observations: Dict[str, Observation], timestep: int) -> List[tuple]:
        results = []
        for band_id, obs in observations.items():
            pred_time = self._last_predictions.get(band_id)
            if pred_time is not None:
                # Prediction was made for this band
                is_correct = bool(obs.hit and abs(pred_time - timestep) <= 2.0)
                results.append((band_id, is_correct))
        return results

    def learn(self, observations: Dict[str, Observation], timestep: int) -> float:
        # Update engines
        self.belief.update(observations)
        self.temporal.update(observations, timestep)

        # Calculate reward and update Q-table
        reward = self.arbitrator.calculate_reward(observations, timestep)
        next_state = self.arbitrator.get_state(self.belief.get_state())
        self.arbitrator.update(self._last_state, self._last_action, reward, next_state)
        return reward


class RewardTracker:
    def __init__(self, redundancy_window: int = 3, redundant_scan_penalty: float = 0.20):
        self.redundancy_window = int(redundancy_window)
        self.redundant_scan_penalty = float(redundant_scan_penalty)
        self._last_scan_times: Dict[str, int] = {}

    def compute(self, observations: Dict[str, Observation], timestep: int) -> float:
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


class EvaluationMetrics:
    def __init__(self, redundancy_window: int = 3):
        self.redundancy_window = int(redundancy_window)
        self.total_scans: int = 0
        self.total_hits: int = 0
        self.true_detections: int = 0
        self.detection_opportunities: int = 0
        self.false_detections: int = 0
        self.false_alarms: int = 0
        self.quiet_scans: int = 0
        self.redundant_scans: int = 0
        self._last_scan_times: Dict[str, int] = {}
        self.rewards: List[float] = []
        self.prediction_checks: List[bool] = []
        self._first_intercept_times: Dict[str, int] = {}

    def observe_step(self, env: RFEnvironment, selected_bands: List[str],
                     observations: Dict[str, Observation]) -> None:
        t = env.timestep
        for b in selected_bands:
            self.total_scans += 1
            truth = env.band_truth(b)
            obs = observations.get(b)
            hit = obs.hit if obs is not None else False

            # Redundancy tracking
            last_t = self._last_scan_times.get(b)
            if last_t is not None and (t - last_t) <= self.redundancy_window:
                self.redundant_scans += 1
            self._last_scan_times[b] = t

            if truth.active:
                self.detection_opportunities += 1
                if hit:
                    self.total_hits += 1
                    self.true_detections += 1
                    if truth.emitter_id and truth.emitter_id not in self._first_intercept_times:
                        self._first_intercept_times[truth.emitter_id] = t
            else:
                self.quiet_scans += 1
                if hit:
                    self.total_hits += 1
                    self.false_alarms += 1
                    self.false_detections += 1

    def record_prediction_check(self, correct: bool) -> None:
        self.prediction_checks.append(bool(correct))

    def record_reward(self, reward: float) -> None:
        self.rewards.append(float(reward))

    def summary(self) -> dict:
        pd = (self.true_detections / self.detection_opportunities) if self.detection_opportunities > 0 else 0.0
        pfa = (self.false_alarms / self.quiet_scans) if self.quiet_scans > 0 else 0.0
        interception_rate = (self.total_hits / self.total_scans) if self.total_scans > 0 else 0.0
        avg_reward = float(np.mean(self.rewards)) if self.rewards else 0.0
        redundant_rate = (self.redundant_scans / self.total_scans) if self.total_scans > 0 else 0.0
        pred_acc = (sum(self.prediction_checks) / len(self.prediction_checks)) if self.prediction_checks else 0.0
        avg_intercept = float(np.mean(list(self._first_intercept_times.values()))) if self._first_intercept_times else float("nan")

        return {
            "pd": float(pd),
            "pfa": float(pfa),
            "interception_rate": float(interception_rate),
            "avg_reward": avg_reward,
            "redundant_scan_rate": float(redundant_rate),
            "prediction_accuracy": float(pred_acc),
            "avg_intercept_time": avg_intercept,
            "total_hits": self.total_hits,
            "total_scans": self.total_scans,
            "true_detections": self.true_detections,
            "detection_opportunities": self.detection_opportunities,
            "false_alarms": self.false_alarms,
            "quiet_scans": self.quiet_scans,
        }


class EvasionReacquisitionTracker:
    def __init__(self, target_emitter):
        self.target = target_emitter
        self.evasion_events: List[Dict] = []
        self._in_evasion: bool = False
        self._evasion_start_t: int = 0
        self._evasion_end_t: int = 0

    def observe_step(self, timestep: int, hit_on_target: bool) -> None:
        if self.target is None:
            return
        if self.target.is_evasive and not self._in_evasion:
            self._in_evasion = True
            self._evasion_start_t = timestep
        elif not self.target.is_evasive and self._in_evasion:
            self._in_evasion = False
            self._evasion_end_t = timestep
            self.evasion_events.append({
                "start_t": self._evasion_start_t,
                "end_t": self._evasion_end_t,
                "reacquired_t": None,
            })

        if hit_on_target and self.evasion_events and self.evasion_events[-1]["reacquired_t"] is None:
            self.evasion_events[-1]["reacquired_t"] = timestep

    def summary(self) -> dict:
        reacq_times = []
        for ev in self.evasion_events:
            if ev["reacquired_t"] is not None and ev["end_t"] is not None:
                reacq_times.append(ev["reacquired_t"] - ev["end_t"])
        return {
            "evasion_events": len(self.evasion_events),
            "average_reacquisition_time": float(np.mean(reacq_times)) if reacq_times else float("nan"),
        }


def run_single_experiment(name: str, scheduler_factory: Callable, config: dict,
                          num_steps: int = 200, seed: int = 42) -> dict:
    run_config = dict(config)
    run_config["random_seed"] = seed
    env = RFEnvironment(run_config)
    det_cfg = run_config.get("detection", {})
    detection_model = DetectionModel(
        threshold_db=det_cfg.get("threshold_db", 10.0),
        snr_scale=det_cfg.get("snr_scale", 3.0),
        false_alarm_probability=det_cfg.get("false_alarm_probability", 0.05),
        seed=det_cfg.get("seed", seed),
    )
    k = int(run_config.get("receiver_channels", 5))
    receiver = Receiver(env, k=k, detection_model=detection_model)
    scheduler = scheduler_factory(run_config.get("num_bands", 50), k, seed=seed)
    metrics = EvaluationMetrics(redundancy_window=run_config.get("ml_arbitrator", {}).get("redundancy_window", 3))

    for t in range(num_steps):
        env.step()
        selected = scheduler.select_bands(t)
        observations = receiver.observe(selected)
        env.notify_scan_results(observations)
        metrics.observe_step(env, selected, observations)
        if hasattr(scheduler, "learn"):
            scheduler.learn(observations, t)

    s = metrics.summary()
    s["scheduler"] = name
    s["steps"] = num_steps
    s["seed"] = seed
    return s


def aggregate_results(results_list: List[dict]) -> dict:
    if not results_list:
        return {}
    keys = ["pd", "pfa", "interception_rate", "avg_reward", "redundant_scan_rate", "avg_intercept_time"]
    agg = {}
    for k in keys:
        vals = [r[k] for r in results_list if k in r and not np.isnan(r[k])]
        if vals:
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))
        else:
            agg[f"{k}_mean"] = float("nan")
            agg[f"{k}_std"] = float("nan")
    return agg
