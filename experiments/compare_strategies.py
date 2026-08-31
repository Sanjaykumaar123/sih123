"""Scientific Experiment Runner: Conventional Open-Loop vs. Intelligent Smart Scan.

Compares deterministic sequential sweeping against the adaptive cognitive scheduler
under strictly identical scenario conditions, receiver constraints, and random seeds.
"""

import argparse
from dataclasses import asdict, dataclass
import math
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

from data_adapter import TSRDEnvironment
from rf_env import (
    DetectionModel,
    EvaluationMetrics,
    IntelligentSchedulerAdapter,
    Receiver,
    RewardTracker,
)


class SequentialOpenLoopScheduler:
    """Conventional open-loop sequential frequency sweeping scheduler.

    Sweeps through all N bands sequentially in chunks of K channels without adaptation,
    belief modeling, temporal analysis, or reinforcement learning.
    """

    def __init__(self, num_bands: int = 50, k: int = 5, start_band: str = "F01"):
        self.num_bands = int(num_bands)
        self.k = int(k)
        self.all_bands = [f"F{i:02d}" for i in range(1, self.num_bands + 1)]

        # Find starting offset
        if start_band in self.all_bands:
            self.start_idx = self.all_bands.index(start_band)
        else:
            self.start_idx = 0

    def select_bands(self, timestep: int) -> List[str]:
        """Deterministically select K consecutive bands in open-loop rotation."""
        offset = (self.start_idx + timestep * self.k) % self.num_bands
        selected = []
        for i in range(self.k):
            band_idx = (offset + i) % self.num_bands
            selected.append(self.all_bands[band_idx])
        return selected


@dataclass
class StrategyMetrics:
    strategy_name: str
    total_scans: int
    total_hits: int
    total_misses: int
    total_false_alarms: int
    true_detections: int
    detection_opportunities: int
    active_bands_encountered: int
    unique_emitters_present: int
    unique_emitters_intercepted: int
    sensor_pd: float
    scenario_coverage: float
    interception_rate: float
    true_interception_rate: float
    pfa: float
    redundant_scan_rate: float
    online_avg_reward: float
    evaluated_reward: float
    avg_intercept_time: float
    total_env_active_events: int


@dataclass
class BaselineResult:
    strategy_name: str
    start_band: str
    metrics: StrategyMetrics


@dataclass
class SmartScanResult:
    strategy_name: str
    metrics: StrategyMetrics


@dataclass
class ComparisonResult:
    scenario_path: str
    scenario_name: str
    num_steps: int
    channels: int
    total_scan_opportunities: int
    seed: int
    baseline: BaselineResult
    smart_scan: SmartScanResult
    absolute_improvement: Dict[str, Optional[float]]
    percentage_improvement: Dict[str, Optional[float]]
    metric_directions: Dict[str, str]

    def to_table(self) -> str:
        """Render a formatted comparison table."""
        b = self.baseline.metrics
        s = self.smart_scan.metrics

        def fmt_val(v: Any, is_pct: bool = False) -> str:
            if isinstance(v, (int, np.integer)):
                return f"{v}"
            if isinstance(v, (float, np.floating)):
                if math.isnan(v):
                    return "n/a"
                if is_pct:
                    return f"{v * 100:.2f}%"
                return f"{v:.4f}"
            return str(v)

        def fmt_imp(metric_key: str) -> str:
            pct = self.percentage_improvement.get(metric_key)
            diff = self.absolute_improvement.get(metric_key)
            direction = self.metric_directions.get(metric_key, "HIGHER_IS_BETTER")
            val_b = getattr(b, metric_key, None)

            if val_b == 0 and diff is not None and diff > 0:
                return f"+{diff:.2f} (From zero baseline)"
            if pct is None or math.isnan(pct):
                if diff is not None and not math.isnan(diff):
                    sign = "+" if diff > 0 else ""
                    return f"{sign}{diff:.2f} (abs)"
                return "n/a"

            sign = "+" if pct > 0 else ""
            better = " (Better)" if ((pct > 0 and direction == "HIGHER_IS_BETTER") or (pct < 0 and direction == "LOWER_IS_BETTER")) else ""
            return f"{sign}{pct:.2f}%{better}"

        lines = [
            "=" * 82,
            "SMART SCAN STRATEGY COMPARISON (SIH BENCHMARK)",
            "=" * 82,
            f"Scenario:                    {self.scenario_name}",
            f"Duration / Timesteps:        {self.num_steps * 0.05:.2f}s ({self.num_steps} steps)",
            f"Receiver Channels (K):       {self.channels} of 50 bands",
            f"Total Scan Opportunities:    {self.total_scan_opportunities}",
            f"Random Seed:                 {self.seed}",
            f"Total Spectrum Active Events:{b.total_env_active_events}",
            f"Total Emitters in Scenario:  {b.unique_emitters_present}",
            "-" * 82,
            f"{'Metric':<34}{'OPEN LOOP':<16}{'SMART SCAN':<16}{'Improvement':<16}",
            "-" * 82,
            f"{'True Detections':<34}{b.true_detections:<16}{s.true_detections:<16}{fmt_imp('true_detections')}",
            f"{'Unique Emitters Intercepted':<34}{b.unique_emitters_intercepted:<16}{s.unique_emitters_intercepted:<16}{fmt_imp('unique_emitters_intercepted')}",
            f"{'Sensor Pd (on scanned bands)':<34}{fmt_val(b.sensor_pd, True):<16}{fmt_val(s.sensor_pd, True):<16}{fmt_imp('sensor_pd')}",
            f"{'Scenario Detection Coverage':<34}{fmt_val(b.scenario_coverage, True):<16}{fmt_val(s.scenario_coverage, True):<16}{fmt_imp('scenario_coverage')}",
            f"{'Interception Rate (Hit Yield)':<34}{fmt_val(b.interception_rate, True):<16}{fmt_val(s.interception_rate, True):<16}{fmt_imp('interception_rate')}",
            f"{'True Interception Rate':<34}{fmt_val(b.true_interception_rate, True):<16}{fmt_val(s.true_interception_rate, True):<16}{fmt_imp('true_interception_rate')}",
            f"{'False Alarm Rate (Pfa)':<34}{fmt_val(b.pfa, True):<16}{fmt_val(s.pfa, True):<16}{fmt_imp('pfa')}",
            f"{'Average Intercept Time':<34}{fmt_val(b.avg_intercept_time):<16}{fmt_val(s.avg_intercept_time):<16}{fmt_imp('avg_intercept_time')}",
            f"{'Redundant Scan Rate':<34}{fmt_val(b.redundant_scan_rate, True):<16}{fmt_val(s.redundant_scan_rate, True):<16}{fmt_imp('redundant_scan_rate')}",
            f"{'Evaluated Benchmark Reward':<34}{fmt_val(b.evaluated_reward):<16}{fmt_val(s.evaluated_reward):<16}{fmt_imp('evaluated_reward')}",
            f"{'Online Agent Reward':<34}{fmt_val(b.online_avg_reward):<16}{fmt_val(s.online_avg_reward):<16}{fmt_imp('online_avg_reward')}",
            f"{'Total Hits (True + FA)':<34}{b.total_hits:<16}{s.total_hits:<16}{fmt_imp('total_hits')}",
            f"{'Total False Alarms':<34}{b.total_false_alarms:<16}{s.total_false_alarms:<16}{fmt_imp('total_false_alarms')}",
            f"{'Active Bands Encountered':<34}{b.active_bands_encountered:<16}{s.active_bands_encountered:<16}{fmt_imp('active_bands_encountered')}",
            "-" * 82,
        ]
        return "\n".join(lines)


def run_strategy_simulation(
    strategy_type: str,
    scheduler: Any,
    scenario_path: str,
    num_steps: int = 50,
    channels: int = 5,
    seed: int = 42,
    step_duration_s: float = 0.05,
) -> StrategyMetrics:
    """Execute a single strategy through the standard TSRD simulation loop."""
    env = TSRDEnvironment(scenario_path, step_duration_s=step_duration_s, num_bands=50)
    detection_model = DetectionModel(
        threshold_db=10.0,
        snr_scale=3.0,
        false_alarm_probability=0.05,
        seed=seed,
    )
    receiver = Receiver(env, k=channels, detection_model=detection_model)
    metrics_tracker = EvaluationMetrics(redundancy_window=3)
    reward_tracker = RewardTracker(redundancy_window=3)

    total_env_active_events = 0
    intercepted_emitters: set = set()
    evaluated_rewards: List[float] = []

    last_scan_times: Dict[str, int] = {}

    for t in range(num_steps):
        env.step()

        # Count active band events in the full RF environment
        active_in_env = [b for b in env.bands if env.band_truth(b).active]
        total_env_active_events += len(active_in_env)

        # 1. Action Decision (no ground truth accessed)
        selected_bands = scheduler.select_bands(t)

        # 2. Channel Observation
        observations = receiver.observe(selected_bands)
        env.notify_scan_results(observations)

        # 3. Evaluation Accounting
        metrics_tracker.observe_step(env, selected_bands, observations)

        # Evaluated Benchmark Reward:
        # +2.0 for true interception, -0.5 for false alarm, -0.20 for redundant miss
        step_eval_reward = 0.0
        for b in selected_bands:
            obs = observations.get(b)
            truth = env.band_truth(b)
            is_redundant = (b in last_scan_times) and ((t - last_scan_times[b]) <= 3)
            last_scan_times[b] = t

            if obs and obs.hit:
                if truth.active:
                    step_eval_reward += 2.0  # True interception reward
                    if truth.emitter_id:
                        intercepted_emitters.add(truth.emitter_id)
                else:
                    step_eval_reward -= 0.5  # False alarm penalty
            else:
                if is_redundant:
                    step_eval_reward -= 0.20  # Redundant miss penalty

        evaluated_rewards.append(step_eval_reward)

        # 4. Learning feedback (Smart Scan only)
        if hasattr(scheduler, "learn"):
            online_r = scheduler.learn(observations, t)
        else:
            online_r = reward_tracker.compute(observations, t)
        metrics_tracker.record_reward(online_r)

    summary = metrics_tracker.summary()
    total_scans = num_steps * channels
    total_hits = summary["total_hits"]
    total_misses = total_scans - total_hits
    false_alarms = metrics_tracker.false_alarms
    true_detections = summary["true_detections"]
    active_enc = summary["detection_opportunities"]

    # Spectrum scenario coverage = True Detections / Total Active Band Events in Environment
    scenario_coverage = (true_detections / total_env_active_events) if total_env_active_events > 0 else 0.0
    true_interception_rate = true_detections / float(total_scans) if total_scans > 0 else 0.0

    all_emitters_in_scenario = len(env.truth_manager.get_all_emitter_ids())

    return StrategyMetrics(
        strategy_name=strategy_type,
        total_scans=total_scans,
        total_hits=total_hits,
        total_misses=total_misses,
        total_false_alarms=false_alarms,
        true_detections=true_detections,
        detection_opportunities=active_enc,
        active_bands_encountered=active_enc,
        unique_emitters_present=all_emitters_in_scenario,
        unique_emitters_intercepted=len(intercepted_emitters),
        sensor_pd=summary["pd"],
        scenario_coverage=scenario_coverage,
        interception_rate=summary["interception_rate"],
        true_interception_rate=true_interception_rate,
        pfa=summary["pfa"],
        redundant_scan_rate=summary["redundant_scan_rate"],
        online_avg_reward=summary["avg_reward"],
        evaluated_reward=float(np.mean(evaluated_rewards)),
        avg_intercept_time=summary["avg_intercept_time"],
        total_env_active_events=total_env_active_events,
    )


def compare_strategies(
    scenario_path: str,
    num_steps: int = 50,
    channels: int = 5,
    seed: int = 42,
    step_duration_s: float = 0.05,
    start_band: str = "F01",
) -> ComparisonResult:
    """Execute fair comparative benchmark between Open-Loop and Smart Scan."""
    if not os.path.exists(scenario_path):
        raise FileNotFoundError(f"Scenario file not found: {scenario_path}")

    # 1. Run Baseline (Open-Loop Sequential Sweeper)
    open_loop_scheduler = SequentialOpenLoopScheduler(
        num_bands=50, k=channels, start_band=start_band
    )
    baseline_metrics = run_strategy_simulation(
        strategy_type="Conventional Open-Loop",
        scheduler=open_loop_scheduler,
        scenario_path=scenario_path,
        num_steps=num_steps,
        channels=channels,
        seed=seed,
        step_duration_s=step_duration_s,
    )
    baseline_result = BaselineResult(
        strategy_name="Conventional Open-Loop",
        start_band=start_band,
        metrics=baseline_metrics,
    )

    # 2. Run Smart Scan (Cognitive Q-Learning + Bayesian + Temporal Scheduler)
    smart_scan_scheduler = IntelligentSchedulerAdapter(
        num_bands=50, k=channels
    )
    smart_scan_metrics = run_strategy_simulation(
        strategy_type="Intelligent Smart Scan",
        scheduler=smart_scan_scheduler,
        scenario_path=scenario_path,
        num_steps=num_steps,
        channels=channels,
        seed=seed,
        step_duration_s=step_duration_s,
    )
    smart_scan_result = SmartScanResult(
        strategy_name="Intelligent Smart Scan",
        metrics=smart_scan_metrics,
    )

    # 3. Calculate Differences & Percentage Improvements
    metric_keys = [
        ("true_detections", "HIGHER_IS_BETTER"),
        ("unique_emitters_intercepted", "HIGHER_IS_BETTER"),
        ("sensor_pd", "HIGHER_IS_BETTER"),
        ("scenario_coverage", "HIGHER_IS_BETTER"),
        ("interception_rate", "HIGHER_IS_BETTER"),
        ("true_interception_rate", "HIGHER_IS_BETTER"),
        ("pfa", "LOWER_IS_BETTER"),
        ("avg_intercept_time", "LOWER_IS_BETTER"),
        ("redundant_scan_rate", "LOWER_IS_BETTER"),
        ("evaluated_reward", "HIGHER_IS_BETTER"),
        ("online_avg_reward", "HIGHER_IS_BETTER"),
        ("total_hits", "HIGHER_IS_BETTER"),
        ("total_false_alarms", "LOWER_IS_BETTER"),
        ("active_bands_encountered", "HIGHER_IS_BETTER"),
    ]

    abs_imp: Dict[str, Optional[float]] = {}
    pct_imp: Dict[str, Optional[float]] = {}
    directions: Dict[str, str] = {}

    b_dict = asdict(baseline_metrics)
    s_dict = asdict(smart_scan_metrics)

    for k, direction in metric_keys:
        directions[k] = direction
        v_b = b_dict.get(k)
        v_s = s_dict.get(k)

        if isinstance(v_b, (int, float)) and isinstance(v_s, (int, float)):
            if not math.isnan(v_b) and not math.isnan(v_s):
                diff = float(v_s - v_b)
                abs_imp[k] = diff
                if v_b != 0:
                    pct_imp[k] = float((v_s - v_b) / abs(v_b)) * 100.0
                else:
                    pct_imp[k] = float("nan")
            else:
                abs_imp[k] = float("nan")
                pct_imp[k] = float("nan")
        else:
            abs_imp[k] = None
            pct_imp[k] = None

    return ComparisonResult(
        scenario_path=os.path.abspath(scenario_path),
        scenario_name=os.path.basename(scenario_path),
        num_steps=num_steps,
        channels=channels,
        total_scan_opportunities=num_steps * channels,
        seed=seed,
        baseline=baseline_result,
        smart_scan=smart_scan_result,
        absolute_improvement=abs_imp,
        percentage_improvement=pct_imp,
        metric_directions=directions,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Cognitive Smart Scan vs. Conventional Open-Loop Scanning on TSRD Scenarios."
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=r"D:\sih\dataset\scan\test_scan\config_1.h5",
        help="Path to TSRD HDF5 scenario file (e.g. config_1.h5)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of simulation decision steps (default: 50)",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=5,
        help="Number of receiver channels K (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for repeatable evaluation (default: 42)",
    )
    parser.add_argument(
        "--start-band",
        type=str,
        default="F01",
        help="Starting band for open-loop sweep (default: F01)",
    )
    parser.add_argument(
        "--step-duration",
        type=float,
        default=0.05,
        help="Duration of discrete timestep in seconds (default: 0.05)",
    )

    args = parser.parse_args()

    result = compare_strategies(
        scenario_path=args.scenario,
        num_steps=args.steps,
        channels=args.channels,
        seed=args.seed,
        step_duration_s=args.step_duration,
        start_band=args.start_band,
    )

    print(result.to_table())


if __name__ == "__main__":
    main()
