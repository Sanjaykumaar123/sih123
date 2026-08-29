"""Stage 8 demo: fair baseline comparison and evaluation.

Run: python demo_stage8.py

Runs THREE schedulers (Intelligent = Stage 5+6, Round Robin, Random-K)
under IDENTICAL environment conditions per seed (same emitters, same
detection model, same adaptive-evasive emitter -- only band-selection
differs), across config.yaml's evaluation.seeds, and reports mean/std
per metric. Ground truth is read only by the evaluation layer, strictly
after Receiver.observe() -- never by any scheduler. Results are also
saved to results/stage8_results.json and results/stage8_summary.csv.

This does NOT modify Stage 1-7 logic, the Q-learning algorithm, Stage 5
scoring formulas, or Stage 7 evasion behaviour -- it only calls their
existing public APIs.
"""

import csv
import json
import os

from rf_env import (RoundRobinScheduler, RandomKScheduler,
                     IntelligentSchedulerAdapter, run_single_experiment,
                     aggregate_results)
from rf_env.config import load_config


def make_round_robin(num_bands, k, seed):
    return RoundRobinScheduler(num_bands, k)


def make_random_k(num_bands, k, seed):
    return RandomKScheduler(num_bands, k, seed=seed)


def make_intelligent(config):
    def factory(num_bands, k, seed):
        arb_cfg = dict(config.get("ml_arbitrator") or {})
        arb_cfg["seed"] = seed
        return IntelligentSchedulerAdapter(
            num_bands, k, config.get("belief"), config.get("temporal"),
            config.get("scoring"), arb_cfg,
        )
    return factory


def fmt(v):
    if isinstance(v, (int, float)):
        return f"{v:.3f}"
    return "n/a" if v == "insufficient_data" else str(v)


def main():
    config = load_config("config.yaml")
    eval_cfg = config.get("evaluation", {})
    num_steps = eval_cfg.get("num_steps", 2000)
    seeds = eval_cfg.get("seeds", [100, 200, 300, 400, 500])

    schedulers = {
        "Intelligent": make_intelligent(config),
        "Round Robin": make_round_robin,
        "Random-K": make_random_k,
    }

    all_results = []
    aggregates = {}
    for name, factory in schedulers.items():
        per_seed = [run_single_experiment(name, factory, config, num_steps, seed)
                    for seed in seeds]
        all_results.extend(per_seed)
        aggregates[name] = aggregate_results(per_seed)

    print(f"Fair comparison: {num_steps} steps x {len(seeds)} seeds {seeds}")
    print("(n/a = insufficient_data, e.g. no evasion events occurred to measure)\n")

    print("-" * 78)
    print(f"{'Scheduler':<14}{'Pd':>10}{'Pfa':>10}{'Intercept%':>12}{'Reward':>10}{'Redundant%':>12}")
    print("-" * 78)
    for name, agg in aggregates.items():
        pd_ = f"{fmt(agg['pd_mean'])}+/-{fmt(agg['pd_std'])}" if agg["pd_mean"] != "insufficient_data" else "n/a"
        pfa = f"{fmt(agg['pfa_mean'])}" if agg["pfa_mean"] != "insufficient_data" else "n/a"
        icr = f"{fmt(agg['interception_rate_mean'])}" if agg["interception_rate_mean"] != "insufficient_data" else "n/a"
        rew = f"{fmt(agg['avg_reward_mean'])}" if agg["avg_reward_mean"] != "insufficient_data" else "n/a"
        red = f"{fmt(agg['redundant_scan_rate_mean'])}" if agg["redundant_scan_rate_mean"] != "insufficient_data" else "n/a"
        print(f"{name:<14}{pd_:>10}{pfa:>10}{icr:>12}{rew:>10}{red:>12}")
    print("-" * 78)

    print("\nAdaptive Evasion Performance")
    print("-" * 78)
    print(f"{'Scheduler':<14}{'Evasions':>10}{'Reacquired':>12}{'ReacqTime':>12}")
    print("-" * 78)
    for name, agg in aggregates.items():
        evasions = fmt(agg.get("evasion_events_mean", "insufficient_data"))
        reacq_n = fmt(agg.get("reacquired_count_mean", "insufficient_data"))
        reacq_t = fmt(agg.get("reacquisition_time_mean", "insufficient_data"))
        print(f"{name:<14}{evasions:>10}{reacq_n:>12}{reacq_t:>12}")
    print("-" * 78)

    print("\nPer-scheduler detail (mean +/- std over seeds):")
    for name, agg in aggregates.items():
        print(f"\n  {name}:")
        for key in sorted(agg):
            if key.endswith("_mean"):
                base = key[:-5]
                std_key = base + "_std"
                print(f"    {base:<24} {fmt(agg[key])} +/- {fmt(agg.get(std_key))}")

    os.makedirs("results", exist_ok=True)
    with open("results/stage8_results.json", "w", encoding="utf-8") as f:
        json.dump({"per_seed": all_results, "aggregates": aggregates}, f, indent=2, default=str)

    with open("results/stage8_summary.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = ["scheduler"] + sorted({k for agg in aggregates.values() for k in agg if k != "scheduler"})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, agg in aggregates.items():
            row = {"scheduler": name}
            row.update({k: agg.get(k, "") for k in fieldnames if k != "scheduler"})
            writer.writerow(row)

    print("\nSaved results/stage8_results.json and results/stage8_summary.csv")


if __name__ == "__main__":
    main()
