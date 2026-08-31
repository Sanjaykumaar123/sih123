"""Predictive ML module (Random Forest Regressors for Interception Time and Rate)."""

import math
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .belief import BandBelief
from .temporal import TemporalPrediction
from .scoring import BandScore
from .receiver import Observation
from .environment import RFEnvironment
from .evaluation import IntelligentSchedulerAdapter
from .detection import DetectionModel
from .receiver import Receiver

FEATURE_NAMES = [
    "belief_activity_probability",
    "belief_uncertainty",
    "staleness",
    "hit_count",
    "miss_count",
    "number_of_observations",
    "temporal_periodicity_score",
    "estimated_period",
    "predicted_next_active_time_distance",
    "prediction_confidence",
    "time_since_last_hit",
    "number_of_hits",
    "score_exploration",
    "score_exploitation",
    "score_prediction",
    "score_balanced",
    "strategy_is_explore",
    "strategy_is_exploit",
    "strategy_is_predict",
]


class FeatureExtractor:
    def __init__(self, prediction_horizon: int = 100):
        self.horizon = float(prediction_horizon)
        self._recent_snr: Dict[str, List[float]] = {}
        self._obs_count: Dict[str, int] = {}

    def observe(self, observations: Dict[str, Observation]) -> None:
        for band_id, obs in observations.items():
            self._obs_count[band_id] = self._obs_count.get(band_id, 0) + 1
            if obs.hit:
                self._recent_snr.setdefault(band_id, []).append(obs.snr)

    def extract(self, band_id: str, belief: BandBelief, temporal: TemporalPrediction,
                score: BandScore, timestep: int, strategy: str) -> List[float]:
        # Handle inf staleness / distance with horizon cap
        stale_val = self.horizon if math.isinf(belief.staleness) else min(self.horizon, float(belief.staleness))
        time_since_hit = self.horizon if math.isinf(temporal.time_since_last_hit) else min(self.horizon, float(temporal.time_since_last_hit))

        if temporal.predicted_next_active_time is not None:
            dist_val = min(self.horizon, abs(float(temporal.predicted_next_active_time) - float(timestep)))
        else:
            dist_val = self.horizon

        est_period = float(temporal.estimated_period) if temporal.estimated_period is not None else 0.0
        n_obs = float(belief.hit_count + belief.miss_count)

        strat_lower = strategy.lower()
        is_explore = 1.0 if "explore" in strat_lower else 0.0
        is_exploit = 1.0 if "exploit" in strat_lower else 0.0
        is_predict = 1.0 if "predict" in strat_lower else 0.0

        features = [
            float(belief.activity_probability),
            float(belief.uncertainty),
            stale_val,
            float(belief.hit_count),
            float(belief.miss_count),
            n_obs,
            float(temporal.periodicity_score),
            est_period,
            dist_val,
            float(temporal.prediction_confidence),
            time_since_hit,
            float(temporal.number_of_hits),
            float(score.exploration_score),
            float(score.exploitation_score),
            float(score.prediction_score),
            float(score.balanced_score),
            is_explore,
            is_exploit,
            is_predict,
        ]
        return features


def _regression_metrics(y_true, y_pred) -> Dict[str, float]:
    yt = np.array(y_true, dtype=np.float64)
    yp = np.array(y_pred, dtype=np.float64)
    mae = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    r2 = float(r2_score(yt, yp)) if len(yt) > 1 and np.var(yt) > 1e-9 else 0.0
    return {"mae": mae, "rmse": rmse, "r2": r2}


class Predictor:
    def __init__(self, model_time, model_rate, baseline_time, baseline_rate, horizon: float = 100.0):
        self.model_time = model_time
        self.model_rate = model_rate
        self.baseline_time = float(baseline_time)
        self.baseline_rate = float(baseline_rate)
        self.horizon = float(horizon)

    def predict(self, features: List[float]) -> dict:
        feat_arr = np.array([features], dtype=np.float64)
        if self.model_time is not None and self.model_rate is not None:
            pred_t = float(np.clip(self.model_time.predict(feat_arr)[0], 0.0, self.horizon))
            pred_r = float(np.clip(self.model_rate.predict(feat_arr)[0], 0.0, 1.0))
            quality = "high"
        else:
            pred_t = self.baseline_time
            pred_r = self.baseline_rate
            quality = "cold_start"

        return {
            "predicted_intercept_time": pred_t,
            "predicted_interception_rate": pred_r,
            "prediction_interval_available": True,
            "prediction_quality": quality,
        }


class PredictiveModelTrainer:
    def __init__(self, config: dict):
        cfg = config.get("predictive", {})
        self.n_estimators = int(cfg.get("n_estimators", 50))
        self.max_depth = int(cfg.get("max_depth", 8))
        self.min_samples_leaf = int(cfg.get("min_samples_leaf", 2))
        self.horizon = float(cfg.get("prediction_horizon", 100))
        self.seed = int(cfg.get("seed", 999))

        self.model_time: Optional[RandomForestRegressor] = None
        self.model_rate: Optional[RandomForestRegressor] = None
        self.baseline_time_mean: float = 0.0
        self.baseline_rate_mean: float = 0.0

    def train(self, samples: List[Dict]) -> None:
        if not samples:
            return
        X = np.array([s["features"] for s in samples], dtype=np.float64)
        y_time = np.array([s["intercept_time"] for s in samples], dtype=np.float64)
        y_rate = np.array([s["interception_rate"] for s in samples], dtype=np.float64)

        self.baseline_time_mean = float(np.mean(y_time))
        self.baseline_rate_mean = float(np.mean(y_rate))

        self.model_time = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.seed,
        )
        self.model_time.fit(X, y_time)

        self.model_rate = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.seed + 1,
        )
        self.model_rate.fit(X, y_rate)

    def evaluate(self, test_samples: List[Dict]) -> dict:
        if not test_samples:
            return {
                "intercept_time": {"random_forest": {"mae": 0.0, "rmse": 0.0, "r2": 0.0}, "mean_baseline": {"mae": 0.0, "rmse": 0.0, "r2": 0.0}, "n_samples": 0},
                "interception_rate": {"random_forest": {"mae": 0.0, "rmse": 0.0, "r2": 0.0}, "mean_baseline": {"mae": 0.0, "rmse": 0.0, "r2": 0.0}, "n_samples": 0}
            }
        X = np.array([s["features"] for s in test_samples], dtype=np.float64)
        y_time = np.array([s["intercept_time"] for s in test_samples], dtype=np.float64)
        y_rate = np.array([s["interception_rate"] for s in test_samples], dtype=np.float64)

        pred_time = self.model_time.predict(X)
        pred_rate = np.clip(self.model_rate.predict(X), 0.0, 1.0)

        time_rf = _regression_metrics(y_time, pred_time)
        base_time_pred = np.full_like(y_time, self.baseline_time_mean)
        time_base = _regression_metrics(y_time, base_time_pred)

        rate_rf = _regression_metrics(y_rate, pred_rate)
        base_rate_pred = np.full_like(y_rate, self.baseline_rate_mean)
        rate_base = _regression_metrics(y_rate, base_rate_pred)

        return {
            "intercept_time": {
                "random_forest": time_rf,
                "mean_baseline": time_base,
                "n_samples": len(test_samples),
            },
            "interception_rate": {
                "random_forest": rate_rf,
                "mean_baseline": rate_base,
                "n_samples": len(test_samples),
            }
        }

    def feature_importance(self, target_type: str = "time") -> List[Tuple[str, float]]:
        model = self.model_time if target_type == "time" else self.model_rate
        if model is None:
            return [(name, 1.0 / len(FEATURE_NAMES)) for name in FEATURE_NAMES]
        importances = model.feature_importances_
        sorted_pairs = sorted(zip(FEATURE_NAMES, importances), key=lambda p: p[1], reverse=True)
        return [(name, float(val)) for name, val in sorted_pairs]

    def to_predictor(self) -> Predictor:
        return Predictor(
            self.model_time, self.model_rate,
            self.baseline_time_mean, self.baseline_rate_mean,
            self.horizon
        )


def generate_training_samples(config: dict, seeds: List[int], run_length: int = 200,
                              horizon: int = 50) -> List[Dict]:
    samples = []
    num_bands = int(config.get("num_bands", 50))
    k = int(config.get("receiver_channels", 5))

    for seed in seeds:
        run_cfg = dict(config)
        run_cfg["random_seed"] = seed
        env = RFEnvironment(run_cfg)
        det_cfg = run_cfg.get("detection", {})
        det_model = DetectionModel(
            threshold_db=det_cfg.get("threshold_db", 10.0),
            snr_scale=det_cfg.get("snr_scale", 3.0),
            false_alarm_probability=det_cfg.get("false_alarm_probability", 0.05),
            seed=seed,
        )
        receiver = Receiver(env, k=k, detection_model=det_model)
        adapter = IntelligentSchedulerAdapter(
            num_bands, k, run_cfg.get("belief"), run_cfg.get("temporal"),
            run_cfg.get("scoring"), run_cfg.get("ml_arbitrator")
        )
        extractor = FeatureExtractor(prediction_horizon=horizon)

        # Log steps
        step_features = []  # (t, band_id, feature_vector)
        step_band_activity = []  # list of dict: t -> {band_id: is_active}

        for t in range(run_length):
            env.step()
            truth_map = {b: env.band_truth(b).active for b in env.bands}
            step_band_activity.append(truth_map)

            selected = adapter.select_bands(t)
            strategy = adapter.last_strategy

            belief_state = {b.band_id: b for b in adapter.belief.get_state()}
            temporal_state = {t_p.band_id: t_p for t_p in adapter.temporal.get_state()}
            score_state = {s.band_id: s for s in adapter.scoring.get_scores()}

            for band_id in selected:
                feat = extractor.extract(
                    band_id, belief_state[band_id], temporal_state[band_id],
                    score_state[band_id], t, strategy
                )
                step_features.append((t, band_id, feat))

            observations = receiver.observe(selected)
            env.notify_scan_results(observations)
            extractor.observe(observations)
            adapter.learn(observations, t)

        # Label samples by looking forward into completed run
        for t, band_id, feat in step_features:
            if t + horizon >= run_length:
                continue
            # Look forward up to horizon steps
            active_window = [step_band_activity[t + h][band_id] for h in range(1, horizon + 1)]
            hits_in_window = sum(active_window)
            rate = hits_in_window / float(horizon)
            first_hit = next((h for h, act in enumerate(active_window, 1) if act), horizon)

            samples.append({
                "features": feat,
                "intercept_time": float(first_hit),
                "interception_rate": float(rate),
                "band_id": band_id,
                "timestep": t,
            })

    return samples
