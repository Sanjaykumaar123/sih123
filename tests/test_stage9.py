import inspect
import math

import numpy as np
import pytest

import rf_env.predictor as predictor_module
from rf_env import (BandBelief, TemporalPrediction, BandScore, FeatureExtractor,
                     PredictiveModelTrainer, generate_training_samples, FEATURE_NAMES)
from rf_env.predictor import _regression_metrics
import demo_stage9 as stage9_demo

MINI_CONFIG = {
    "num_bands": 50, "receiver_channels": 5, "random_seed": 1, "max_timesteps": 200,
    "emitters": [
        {"id": "E1", "type": "static", "band": "F10", "signal_strength": -60.0,
         "snr": 15.0, "active_prob": 1.0},
        {"id": "E4", "type": "adaptive_evasive", "band": "F30",
         "signal_strength": -63.0, "snr": 10.0},
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
    "adaptive_evasion": {"enabled": True, "hit_threshold": 3, "observation_window": 10,
                          "evasive_duration": 8, "seed": 1},
    "predictive": {"enabled": True, "n_estimators": 20, "max_depth": 5,
                   "min_samples_leaf": 2, "prediction_horizon": 20, "seed": 1},
}

_COLD_BELIEF = BandBelief(band_id="F01", alpha=1.0, beta=1.0, activity_probability=0.5,
                           uncertainty=1.0 / 12.0, last_observed=None, staleness=float("inf"),
                           hit_count=0, miss_count=0)
_COLD_TEMPORAL = TemporalPrediction(band_id="F01", periodicity_score=0.0, estimated_period=None,
                                     predicted_next_active_time=None, prediction_confidence=0.0,
                                     last_hit_timestep=None, time_since_last_hit=float("inf"),
                                     behaviour_type="insufficient_data", number_of_hits=0)
_COLD_SCORE = BandScore(band_id="F01", exploration_score=1.0, exploitation_score=0.5,
                         prediction_score=0.0, balanced_score=0.5)


def test_feature_vector_is_fixed_size():
    extractor = FeatureExtractor(prediction_horizon=20)
    features = extractor.extract("F01", _COLD_BELIEF, _COLD_TEMPORAL, _COLD_SCORE, 0, "balanced")
    assert len(features) == len(FEATURE_NAMES) == 19


def test_feature_order_is_deterministic():
    extractor = FeatureExtractor(prediction_horizon=20)
    a = extractor.extract("F01", _COLD_BELIEF, _COLD_TEMPORAL, _COLD_SCORE, 5, "exploitation")
    b = extractor.extract("F01", _COLD_BELIEF, _COLD_TEMPORAL, _COLD_SCORE, 5, "exploitation")
    assert a == b


def test_feature_extractor_has_no_ground_truth_access():
    source = inspect.getsource(predictor_module.FeatureExtractor)
    assert "band_truth" not in source
    assert "GroundTruthLogger" not in source
    assert "emitter_id" not in source
    assert "emitter_type" not in source
    assert "RFEnvironment" not in source
    sig = inspect.signature(predictor_module.FeatureExtractor.extract)
    assert "env" not in sig.parameters and "environment" not in sig.parameters


def test_cold_start_features_are_valid():
    extractor = FeatureExtractor(prediction_horizon=20)
    features = extractor.extract("F01", _COLD_BELIEF, _COLD_TEMPORAL, _COLD_SCORE, 0, "balanced")
    assert all(math.isfinite(x) for x in features)
    assert features[FEATURE_NAMES.index("belief_activity_probability")] == 0.5
    assert features[FEATURE_NAMES.index("number_of_observations")] == 0.0
    # staleness/distance defaults capped at the horizon, not left as inf
    assert features[FEATURE_NAMES.index("staleness")] == 20.0
    assert features[FEATURE_NAMES.index("predicted_next_active_time_distance")] == 20.0


def test_training_data_generation_works():
    samples = generate_training_samples(MINI_CONFIG, seeds=[1], run_length=60, horizon=20)
    assert len(samples) > 0
    row = samples[0]
    assert len(row["features"]) == len(FEATURE_NAMES)
    assert "intercept_time" in row and "interception_rate" in row


def test_train_and_test_seeds_are_disjoint():
    assert set(stage9_demo.TRAIN_SEEDS).isdisjoint(stage9_demo.VAL_SEEDS)
    assert set(stage9_demo.TRAIN_SEEDS).isdisjoint(stage9_demo.TEST_SEEDS)
    assert set(stage9_demo.VAL_SEEDS).isdisjoint(stage9_demo.TEST_SEEDS)


def test_random_forest_models_train_successfully():
    samples = generate_training_samples(MINI_CONFIG, seeds=[1, 2], run_length=60, horizon=20)
    trainer = PredictiveModelTrainer(MINI_CONFIG)
    trainer.train(samples)
    assert trainer.model_time is not None
    # a trained model can predict without raising
    trainer.model_time.predict(np.array([samples[0]["features"]]))


def test_predictor_returns_valid_output():
    samples = generate_training_samples(MINI_CONFIG, seeds=[1, 2], run_length=60, horizon=20)
    trainer = PredictiveModelTrainer(MINI_CONFIG)
    trainer.train(samples)
    predictor = trainer.to_predictor()
    result = predictor.predict(samples[0]["features"])
    for key in ("predicted_intercept_time", "predicted_interception_rate",
                "prediction_interval_available", "prediction_quality"):
        assert key in result


def test_predicted_values_within_sensible_bounds():
    samples = generate_training_samples(MINI_CONFIG, seeds=[1, 2], run_length=60, horizon=20)
    trainer = PredictiveModelTrainer(MINI_CONFIG)
    trainer.train(samples)
    predictor = trainer.to_predictor()
    for s in samples[:20]:
        result = predictor.predict(s["features"])
        assert 0.0 <= result["predicted_intercept_time"] <= trainer.horizon + 1e-6
        assert 0.0 <= result["predicted_interception_rate"] <= 1.0


def test_interception_rate_prediction_is_bounded_0_1():
    samples = generate_training_samples(MINI_CONFIG, seeds=[1, 2], run_length=60, horizon=20)
    trainer = PredictiveModelTrainer(MINI_CONFIG)
    trainer.train(samples)
    predictor = trainer.to_predictor()
    # an extreme/out-of-distribution feature row should still clip to [0,1]
    extreme = [1.0] * len(FEATURE_NAMES)
    result = predictor.predict(extreme)
    assert 0.0 <= result["predicted_interception_rate"] <= 1.0


def test_baseline_predictor_works():
    samples = generate_training_samples(MINI_CONFIG, seeds=[1, 2], run_length=60, horizon=20)
    trainer = PredictiveModelTrainer(MINI_CONFIG)
    trainer.train(samples)
    assert isinstance(trainer.baseline_time_mean, float)
    metrics = trainer.evaluate(samples)
    assert "mean_baseline" in metrics["intercept_time"]


def test_mae_rmse_calculations_work():
    y_true = [10.0, 20.0, 30.0]
    y_pred = [12.0, 18.0, 33.0]
    m = _regression_metrics(y_true, y_pred)
    assert m["mae"] == pytest.approx((2 + 2 + 3) / 3, abs=1e-4)
    assert m["rmse"] > 0


def test_feature_importance_is_generated():
    samples = generate_training_samples(MINI_CONFIG, seeds=[1, 2], run_length=60, horizon=20)
    trainer = PredictiveModelTrainer(MINI_CONFIG)
    trainer.train(samples)
    importances = trainer.feature_importance("time")
    assert len(importances) == len(FEATURE_NAMES)
    values = [v for _, v in importances]
    assert values == sorted(values, reverse=True)
    assert abs(sum(values) - 1.0) < 1e-6
