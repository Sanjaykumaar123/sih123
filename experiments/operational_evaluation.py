"""Full 30-Second Operational Evaluation Runner (SIH Benchmark).

Executes a complete 600-step (30.0s) evaluation on TSRD radar scenarios, generating:
1. Detailed per-emitter interception records & acquisition latency
2. Full 600-step time-series for dashboard visualization
3. 50-band x 600-step 2D time-frequency activity matrices
4. Cognitive decision trace (Q-learning state, strategy mode, band scores, rewards)
"""

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from data_adapter import TSRDEnvironment
from experiments.compare_strategies import (
    ComparisonResult,
    SequentialOpenLoopScheduler,
    compare_strategies,
)
from rf_env import (
    DetectionModel,
    EvaluationMetrics,
    IntelligentSchedulerAdapter,
    Receiver,
    RewardTracker,
)


@dataclass
class EmitterInterceptRecord:
    emitter_id: str
    first_activity_step: Optional[int]
    first_activity_time_s: Optional[float]
    first_intercept_step_ol: Optional[int]
    first_intercept_time_s_ol: Optional[float]
    intercept_latency_steps_ol: Optional[int]
    intercept_latency_s_ol: Optional[float]
    first_intercept_step_ss: Optional[int]
    first_intercept_time_s_ss: Optional[float]
    intercept_latency_steps_ss: Optional[int]
    intercept_latency_s_ss: Optional[float]


@dataclass
class TimeStepRecord:
    timestep: int
    simulated_time_s: float
    env_active_bands: List[str]
    open_loop_selected: List[str]
    open_loop_hits: List[str]
    open_loop_true_detections: List[str]
    open_loop_false_alarms: List[str]
    smart_scan_selected: List[str]
    smart_scan_hits: List[str]
    smart_scan_true_detections: List[str]
    smart_scan_false_alarms: List[str]
    smart_scan_strategy: str
    smart_scan_online_reward: float
    smart_scan_eval_reward: float
    smart_scan_band_scores: Dict[str, float]


def run_operational_evaluation(
    scenario_path: str,
    num_steps: int = 600,
    channels: int = 5,
    seed: int = 42,
    step_duration_s: float = 0.05,
    output_json_path: Optional[str] = None,
) -> Tuple[ComparisonResult, List[EmitterInterceptRecord], List[TimeStepRecord], Dict[str, Any]]:
    """Execute complete 600-step operational evaluation with time-series and emitter tracking."""
    if not os.path.exists(scenario_path):
        raise FileNotFoundError(f"Scenario not found: {scenario_path}")

    env = TSRDEnvironment(scenario_path, step_duration_s=step_duration_s, num_bands=50)
    all_bands = env.bands
    num_bands = len(all_bands)

    # 1. First Pass: Compute Ground-Truth First Activity per Emitter
    emitter_first_active_step: Dict[str, int] = {}
    for t in range(num_steps):
        step_act = env.processor.get_step_activity(t)
        for b, b_act in step_act.band_activities.items():
            if b_act.is_detectable:
                for e_num in b_act.ground_truth_emitter_ids:
                    e_id = f"TSRD_E{e_num:02d}"
                    if e_id not in emitter_first_active_step:
                        emitter_first_active_step[e_id] = t

    all_emitters_in_scenario = sorted([f"TSRD_E{e:02d}" for e in env.truth_manager.get_all_emitter_ids()])

    # 2. Run Open-Loop Simulation
    env_ol = TSRDEnvironment(scenario_path, step_duration_s=step_duration_s, num_bands=50)
    det_ol = DetectionModel(threshold_db=10.0, snr_scale=3.0, false_alarm_probability=0.05, seed=seed)
    rec_ol = Receiver(env_ol, k=channels, detection_model=det_ol)
    sched_ol = SequentialOpenLoopScheduler(num_bands=num_bands, k=channels, start_band="F01")

    ol_step_data = []
    ol_first_intercept: Dict[str, int] = {}

    for t in range(num_steps):
        env_ol.step()
        selected = sched_ol.select_bands(t)
        obs = rec_ol.observe(selected)
        env_ol.notify_scan_results(obs)

        hits = [b for b, o in obs.items() if o.hit]
        true_dets = [b for b in hits if env_ol.band_truth(b).active]
        fa_hits = [b for b in hits if not env_ol.band_truth(b).active]

        for b in true_dets:
            e_id = env_ol.band_truth(b).emitter_id
            if e_id and e_id not in ol_first_intercept:
                ol_first_intercept[e_id] = t

        ol_step_data.append({
            "selected": selected,
            "hits": hits,
            "true_dets": true_dets,
            "false_alarms": fa_hits,
        })

    # 3. Run Smart Scan Simulation
    env_ss = TSRDEnvironment(scenario_path, step_duration_s=step_duration_s, num_bands=50)
    det_ss = DetectionModel(threshold_db=10.0, snr_scale=3.0, false_alarm_probability=0.05, seed=seed)
    rec_ss = Receiver(env_ss, k=channels, detection_model=det_ss)
    sched_ss = IntelligentSchedulerAdapter(num_bands=num_bands, k=channels)

    ss_step_data = []
    ss_first_intercept: Dict[str, int] = {}
    time_series_records: List[TimeStepRecord] = []

    last_ss_scan_times: Dict[str, int] = {}

    for t in range(num_steps):
        env_ss.step()
        active_in_env = [b for b in env_ss.bands if env_ss.band_truth(b).active]

        # Get scores before selection
        strategy_name = sched_ss.last_strategy
        selected = sched_ss.select_bands(t)
        obs = rec_ss.observe(selected)
        env_ss.notify_scan_results(obs)
        online_r = sched_ss.learn(obs, t)

        hits = [b for b, o in obs.items() if o.hit]
        true_dets = [b for b in hits if env_ss.band_truth(b).active]
        fa_hits = [b for b in hits if not env_ss.band_truth(b).active]

        for b in true_dets:
            e_id = env_ss.band_truth(b).emitter_id
            if e_id and e_id not in ss_first_intercept:
                ss_first_intercept[e_id] = t

        # Evaluated reward
        eval_r = 0.0
        for b in selected:
            o = obs.get(b)
            truth = env_ss.band_truth(b)
            is_redundant = (b in last_ss_scan_times) and ((t - last_ss_scan_times[b]) <= 3)
            last_ss_scan_times[b] = t
            if o and o.hit:
                if truth.active:
                    eval_r += 2.0
                else:
                    eval_r -= 0.5
            else:
                if is_redundant:
                    eval_r -= 0.20

        # Snapshot top band scores
        score_snap = {}
        strat_lower = strategy_name.lower()
        for b in selected:
            bs = sched_ss.scoring.score_band(b)
            if "exploit" in strat_lower:
                score_snap[b] = float(bs.exploitation_score)
            elif "predict" in strat_lower:
                score_snap[b] = float(bs.prediction_score)
            elif "explore" in strat_lower:
                score_snap[b] = float(bs.exploration_score)
            else:
                score_snap[b] = float(bs.balanced_score)

        ol_info = ol_step_data[t]
        ts_rec = TimeStepRecord(
            timestep=t,
            simulated_time_s=float(t * step_duration_s),
            env_active_bands=active_in_env,
            open_loop_selected=ol_info["selected"],
            open_loop_hits=ol_info["hits"],
            open_loop_true_detections=ol_info["true_dets"],
            open_loop_false_alarms=ol_info["false_alarms"],
            smart_scan_selected=selected,
            smart_scan_hits=hits,
            smart_scan_true_detections=true_dets,
            smart_scan_false_alarms=fa_hits,
            smart_scan_strategy=strategy_name,
            smart_scan_online_reward=float(online_r),
            smart_scan_eval_reward=float(eval_r),
            smart_scan_band_scores=score_snap,
        )
        time_series_records.append(ts_rec)

    # 4. Build Emitter Interception Table
    emitter_records: List[EmitterInterceptRecord] = []
    for e_id in all_emitters_in_scenario:
        first_act_step = emitter_first_active_step.get(e_id)
        first_act_time = (first_act_step * step_duration_s) if first_act_step is not None else None

        ol_step = ol_first_intercept.get(e_id)
        ol_time = (ol_step * step_duration_s) if ol_step is not None else None
        ol_lat_steps = (ol_step - first_act_step) if (ol_step is not None and first_act_step is not None) else None
        ol_lat_s = (ol_lat_steps * step_duration_s) if ol_lat_steps is not None else None

        ss_step = ss_first_intercept.get(e_id)
        ss_time = (ss_step * step_duration_s) if ss_step is not None else None
        ss_lat_steps = (ss_step - first_act_step) if (ss_step is not None and first_act_step is not None) else None
        ss_lat_s = (ss_lat_steps * step_duration_s) if ss_lat_steps is not None else None

        emitter_records.append(
            EmitterInterceptRecord(
                emitter_id=e_id,
                first_activity_step=first_act_step,
                first_activity_time_s=first_act_time,
                first_intercept_step_ol=ol_step,
                first_intercept_time_s_ol=ol_time,
                intercept_latency_steps_ol=ol_lat_steps,
                intercept_latency_s_ol=ol_lat_s,
                first_intercept_step_ss=ss_step,
                first_intercept_time_s_ss=ss_time,
                intercept_latency_steps_ss=ss_lat_steps,
                intercept_latency_s_ss=ss_lat_s,
            )
        )

    # 5. Standard Comparison Object
    comp_result = compare_strategies(
        scenario_path=scenario_path,
        num_steps=num_steps,
        channels=channels,
        seed=seed,
        step_duration_s=step_duration_s,
    )

    # 6. Build Spectrum Activity Matrices for Visualization
    gt_matrix = np.zeros((num_bands, num_steps), dtype=int)
    ol_scan_matrix = np.zeros((num_bands, num_steps), dtype=int)
    ss_scan_matrix = np.zeros((num_bands, num_steps), dtype=int)

    for t in range(num_steps):
        for b in time_series_records[t].env_active_bands:
            b_idx = int(b.replace("F", "")) - 1
            gt_matrix[b_idx, t] = 1
        for b in time_series_records[t].open_loop_selected:
            b_idx = int(b.replace("F", "")) - 1
            ol_scan_matrix[b_idx, t] = 2 if b in time_series_records[t].open_loop_true_detections else (1 if b in time_series_records[t].open_loop_hits else 0)
        for b in time_series_records[t].smart_scan_selected:
            b_idx = int(b.replace("F", "")) - 1
            ss_scan_matrix[b_idx, t] = 2 if b in time_series_records[t].smart_scan_true_detections else (1 if b in time_series_records[t].smart_scan_hits else 0)

    spectrum_grids = {
        "ground_truth": gt_matrix.tolist(),
        "open_loop": ol_scan_matrix.tolist(),
        "smart_scan": ss_scan_matrix.tolist(),
    }

    # 7. Persist Results JSON
    if output_json_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
        dump_data = {
            "scenario": os.path.basename(scenario_path),
            "num_steps": num_steps,
            "channels": channels,
            "seed": seed,
            "metrics_comparison": comp_result.to_table(),
            "metrics_summary": {
                "baseline": asdict(comp_result.baseline.metrics),
                "smart_scan": asdict(comp_result.smart_scan.metrics),
                "absolute_improvement": comp_result.absolute_improvement,
                "percentage_improvement": comp_result.percentage_improvement,
            },
            "emitter_interceptions": [asdict(r) for r in emitter_records],
            "time_series": [asdict(r) for r in time_series_records],
            "spectrum_grids": spectrum_grids,
        }
        with open(output_json_path, "w") as f:
            json.dump(dump_data, f, indent=2)

    return comp_result, emitter_records, time_series_records, spectrum_grids


def main():
    parser = argparse.ArgumentParser(
        description="Full 30-Second TSRD Operational Evaluation Runner."
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=r"D:\sih\dataset\scan\test_scan\config_1.h5",
        help="Path to TSRD HDF5 scenario",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=600,
        help="Number of simulation steps (default: 600 for full 30s)",
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
        help="Random seed for repeatability (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=r"D:\sih\results\operational_evaluation_config_1.json",
        help="Path to save evaluation output JSON",
    )

    args = parser.parse_args()

    comp_res, emitter_recs, ts_recs, _ = run_operational_evaluation(
        scenario_path=args.scenario,
        num_steps=args.steps,
        channels=args.channels,
        seed=args.seed,
        output_json_path=args.output,
    )

    print(comp_res.to_table())
    print("\n" + "=" * 82)
    print("EMITTER INTERCEPTION & LATENCY BREAKDOWN (SAMPLE)")
    print("=" * 82)
    print(f"{'Emitter ID':<12}{'First Active (s)':<18}{'Open Loop First (s)':<22}{'Smart Scan First (s)':<22}{'Smart Scan Latency'}")
    print("-" * 82)
    for r in emitter_recs[:15]:
        fa_s = f"{r.first_activity_time_s:.2f}s" if r.first_activity_time_s is not None else "n/a"
        ol_s = f"{r.first_intercept_time_s_ol:.2f}s" if r.first_intercept_time_s_ol is not None else "MISSED"
        ss_s = f"{r.first_intercept_time_s_ss:.2f}s" if r.first_intercept_time_s_ss is not None else "MISSED"
        lat_s = f"{r.intercept_latency_s_ss:.2f}s" if r.intercept_latency_s_ss is not None else "n/a"
        print(f"{r.emitter_id:<12}{fa_s:<18}{ol_s:<22}{ss_s:<22}{lat_s}")

    print("-" * 82)
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
