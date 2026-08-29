"""Stage 10 terminal demo: the full integrated pipeline, no Streamlit.

Run: python demo_stage10.py

Uses the exact same SimulationRunner as app.py (dashboard/simulation_runner.py)
-- the real Stage 1-8 closed loop, no second scheduler, no fabricated
numbers. If no evasion event occurs within STEPS, that is reported
honestly rather than manufactured.
"""

import json
import os
import pickle

from dashboard.simulation_runner import SimulationRunner

STEPS = 500
STRATEGY_LABELS = ["EXPLORE", "EXPLOIT", "PREDICT", "BALANCED"]


def main():
    print("=" * 72)
    print("SMART SCAN STRATEGY FOR ELECTRONIC WARFARE -- Stage 10 terminal demo")
    print("=" * 72)

    runner = SimulationRunner(seed=42)
    print(f"\n[1] INITIALIZED: {runner.config['num_bands']} bands, "
          f"K={runner.k} receiver channels, seed={runner.seed}")

    print(f"\n[2-7] Running {STEPS} steps: band selection -> receiver observation -> "
          f"Bayesian belief update -> temporal prediction -> strategy scoring -> "
          f"Q-learning decision ...")
    evasion_reported = False
    reacquired_reported = False
    record = None
    for _ in range(STEPS):
        record = runner.step()
        if record["just_triggered_evasion"] and not evasion_reported:
            print(f"\n[8] t={record['t']}: *** ADAPTIVE EVASION DETECTED *** "
                  f"(evasion_count={record['evasion_count']}) -- E4 changed its band pattern.")
            evasion_reported = True
        if (evasion_reported and not reacquired_reported and not record["evasive"]
                and record["observations"].get(runner.e4.current_band)
                and record["observations"][runner.e4.current_band].hit):
            print(f"[9] t={record['t']}: RE-ACQUIRED -- E4 detected again on {runner.e4.current_band}.")
            reacquired_reported = True

    if not evasion_reported:
        print("\n[8-9] No evasion event was triggered within this run's step budget "
              "-- reporting honestly, not manufactured.")

    print(f"\n[2-7 example] last step t={runner.t}: selected={record['selected_bands']}, "
          f"strategy={record['strategy']}, reward={record['reward']:.2f}")
    state, q = runner.current_q_state_and_values()
    q_named = dict(zip(STRATEGY_LABELS, [round(float(x), 3) for x in q]))
    print(f"    Q-learning state={state} (perf/uncertainty/detection), Q-values={q_named}")

    print("\n[10] FINAL METRICS (this run)")
    for key, value in runner.metrics_summary().items():
        print(f"    {key:<22} {value}")
    print(f"    evasion_summary        {runner.evasion_summary()}")

    print("\n[11] BASELINE COMPARISON (Stage 8, results/stage8_results.json)")
    if os.path.exists("results/stage8_results.json"):
        with open("results/stage8_results.json", encoding="utf-8") as f:
            stage8 = json.load(f)
        for name, agg in stage8["aggregates"].items():
            print(f"    {name:<14} Pd={agg.get('pd_mean')} "
                  f"Interception={agg.get('interception_rate_mean')} Reward={agg.get('avg_reward_mean')}")
    else:
        print("    (missing -- run `python demo_stage8.py` first)")

    print("\n[12] PREDICTIVE ML (Stage 9, results/stage9_results.json + stage9_predictor.pkl)")
    if os.path.exists("results/stage9_results.json") and os.path.exists("results/stage9_predictor.pkl"):
        with open("results/stage9_results.json", encoding="utf-8") as f:
            stage9 = json.load(f)
        with open("results/stage9_predictor.pkl", "rb") as f:
            predictor = pickle.load(f)
        test_m = stage9["test_metrics"]
        print(f"    Held-out TEST R2: intercept_time="
              f"{test_m['intercept_time']['random_forest']['r2']}, interception_rate="
              f"{test_m['interception_rate']['random_forest']['r2'] if isinstance(test_m['interception_rate'], dict) else 'n/a'}")
        band_id, features = next(iter(runner.last_features.items()))
        pred = predictor.predict(features)
        print(f"    Live prediction for {band_id}: {pred}")
    else:
        print("    (missing -- run `python demo_stage9.py` first)")

    print("\nDone.")


if __name__ == "__main__":
    main()
