"""Stage 9 demo: predictive performance module (Random Forest).

Run: python demo_stage9.py

Generates training/validation/test samples from CLEARLY SEPARATED seed
ranges (no run is ever both trained on and tested on), trains two
RandomForestRegressors (expected intercept time, expected interception
rate) plus mean-baseline predictors, reports MAE/RMSE/R2 on the held-out
TEST split, feature importance, and predicted-vs-actual examples for the
configured emitters' home bands.

This module is an analysis/decision-support layer only -- it does not
control Stage 6's Q-learning policy, and does not model real future radar
behaviour; it estimates interception time/rate from scheduler-visible
features (Stage 3/4/5 outputs), evaluated here against ground truth ONLY
for scoring after the fact.
"""

import json
import os
import pickle
import time

from rf_env import generate_training_samples, PredictiveModelTrainer, FEATURE_NAMES
from rf_env.config import load_config

TRAIN_SEEDS = list(range(100, 110))   # 10 seeds
VAL_SEEDS = list(range(400, 405))     # 5 seeds
TEST_SEEDS = list(range(450, 455))    # 5 seeds
RUN_LENGTH = 600
TARGET_BANDS = ["F10", "F20", "F30"]  # static / periodic / adaptive-evasive homes


def print_metric_table(title, metrics):
    print(f"\n{title}")
    print("-" * 55)
    print(f"{'Model':<16}{'MAE':>10}{'RMSE':>10}{'R2':>10}")
    print("-" * 55)
    for label, key in [("Mean baseline", "mean_baseline"), ("Random Forest", "random_forest")]:
        m = metrics[key]
        r2 = m["r2"] if m["r2"] == "insufficient_data" else f"{m['r2']:.4f}"
        print(f"{label:<16}{m['mae']:>10.4f}{m['rmse']:>10.4f}{r2:>10}")
    print("-" * 55)
    print(f"n_samples = {metrics['n_samples']}")


def main():
    config = load_config("config.yaml")
    horizon = config.get("predictive", {}).get("prediction_horizon", 100)

    print(f"Generating samples: train seeds={TRAIN_SEEDS[0]}-{TRAIN_SEEDS[-1]} "
          f"({len(TRAIN_SEEDS)}), val seeds={VAL_SEEDS[0]}-{VAL_SEEDS[-1]} "
          f"({len(VAL_SEEDS)}), test seeds={TEST_SEEDS[0]}-{TEST_SEEDS[-1]} "
          f"({len(TEST_SEEDS)}); run_length={RUN_LENGTH}, horizon={horizon}")

    t0 = time.time()
    train_samples = generate_training_samples(config, TRAIN_SEEDS, RUN_LENGTH, horizon)
    val_samples = generate_training_samples(config, VAL_SEEDS, RUN_LENGTH, horizon)
    test_samples = generate_training_samples(config, TEST_SEEDS, RUN_LENGTH, horizon)
    gen_time = time.time() - t0
    print(f"Samples: train={len(train_samples)} val={len(val_samples)} "
          f"test={len(test_samples)} (generated in {gen_time:.1f}s)")

    trainer = PredictiveModelTrainer(config)
    t0 = time.time()
    trainer.train(train_samples)
    train_time = time.time() - t0
    print(f"Trained both RandomForestRegressors in {train_time:.2f}s")

    val_metrics = trainer.evaluate(val_samples)
    test_metrics = trainer.evaluate(test_samples)

    print_metric_table("INTERCEPT TIME -- validation split", val_metrics["intercept_time"])
    print_metric_table("INTERCEPT TIME -- TEST split (held out)", test_metrics["intercept_time"])
    if test_metrics["interception_rate"] != "insufficient_data":
        print_metric_table("INTERCEPTION RATE -- validation split", val_metrics["interception_rate"])
        print_metric_table("INTERCEPTION RATE -- TEST split (held out)", test_metrics["interception_rate"])
    else:
        print("\nINTERCEPTION RATE: insufficient_data (no rate-labeled test samples)")

    print("\nFeature importance -- Model A (intercept time):")
    for name, importance in trainer.feature_importance("time")[:10]:
        print(f"  {name:<38}{importance:.4f}")

    if trainer.model_rate is not None:
        print("\nFeature importance -- Model B (interception rate):")
        for name, importance in trainer.feature_importance("rate")[:10]:
            print(f"  {name:<38}{importance:.4f}")

    # 3 evenly-spaced examples per target band (not just the first
    # occurrence) so these aren't a cherry-picked easy/hard case.
    def spaced_examples(band_id, need_rate=False):
        rows = [s for s in test_samples if s["band_id"] == band_id
                and (not need_rate or s["interception_rate"] is not None)]
        if not rows:
            return []
        idxs = sorted({0, len(rows) // 2, len(rows) - 1})
        return [rows[i] for i in idxs]

    predictor = trainer.to_predictor()
    print("\n" + "-" * 64)
    print(f"{'Target':<8}{'Pred Time':>12}{'Actual Time':>13}{'Time Error':>12}")
    print("-" * 64)
    for band_id in TARGET_BANDS:
        for s in spaced_examples(band_id):
            pred = predictor.predict(s["features"])
            err = abs(pred["predicted_intercept_time"] - s["intercept_time"])
            print(f"{band_id:<8}{pred['predicted_intercept_time']:>12.2f}"
                  f"{s['intercept_time']:>13}{err:>12.2f}")
    print("-" * 64)

    print("\n" + "-" * 64)
    print(f"{'Target':<8}{'Pred Rate':>12}{'Actual Rate':>13}{'Abs Error':>12}")
    print("-" * 64)
    for band_id in TARGET_BANDS:
        for s in spaced_examples(band_id, need_rate=True):
            pred = predictor.predict(s["features"])
            err = abs(pred["predicted_interception_rate"] - s["interception_rate"])
            print(f"{band_id:<8}{pred['predicted_interception_rate']:>12.2f}"
                  f"{s['interception_rate']:>13.2f}{err:>12.2f}")
    print("-" * 64)

    cold_sample = [0.5, 1.0 / 12.0, 100.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 3.0, 0.5, 0.5, 0.5, 0.0, 0.5]
    cold_pred = predictor.predict(cold_sample)
    print(f"\nCold-start example prediction: {cold_pred}")

    # Persist results + trained predictor so Stage 10's dashboard/demo can
    # load them instead of retraining (mirrors demo_stage8.py's existing
    # results/ saving pattern; no change to rf_env/predictor.py itself).
    os.makedirs("results", exist_ok=True)
    artifact = {
        "feature_names": FEATURE_NAMES,
        "prediction_horizon": horizon,
        "train_seeds": TRAIN_SEEDS, "val_seeds": VAL_SEEDS, "test_seeds": TEST_SEEDS,
        "run_length": RUN_LENGTH,
        "n_train_samples": len(train_samples), "n_val_samples": len(val_samples),
        "n_test_samples": len(test_samples),
        "generation_seconds": round(gen_time, 2), "training_seconds": round(train_time, 2),
        "val_metrics": val_metrics, "test_metrics": test_metrics,
        "feature_importance_time": trainer.feature_importance("time"),
        "feature_importance_rate": (trainer.feature_importance("rate")
                                     if trainer.model_rate is not None else []),
    }
    with open("results/stage9_results.json", "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, default=str)
    with open("results/stage9_predictor.pkl", "wb") as f:
        pickle.dump(predictor, f)
    print("\nSaved results/stage9_results.json and results/stage9_predictor.pkl")


if __name__ == "__main__":
    main()
