"""Stage 10: thin integration wrapper around the REAL Stage 1-8 pipeline.

Creates NO new scheduler and NO new ML -- reuses IntelligentSchedulerAdapter
(Stage 8), EvaluationMetrics/EvasionReacquisitionTracker (Stage 8) and
env.notify_scan_results (Stage 7) exactly as demo_stage7.py/demo_stage8.py
already do. This module only wires the existing public APIs together and
keeps a small rolling history for the UI to display.

INFORMATION BOUNDARY: `_tick()` calls `adapter.select_bands(t)` and
`adapter.learn(observations, t)` using ONLY Receiver observations --
ground truth (`env.band_truth`) is read only by EvaluationMetrics /
EvasionReacquisitionTracker, both strictly AFTER `receiver.observe()`, and
is stored here only in `self.last_ground_truth_debug` for an explicitly
labelled debug view -- never passed to the adapter.
"""

from __future__ import annotations

from rf_env import (RFEnvironment, Receiver, DetectionModel,
                     IntelligentSchedulerAdapter, EvaluationMetrics,
                     EvasionReacquisitionTracker, FeatureExtractor)
from rf_env.config import load_config

_HISTORY_LIMIT = 400
_BELIEF_CHART_LIMIT = 200


class SimulationRunner:
    def __init__(self, config_path: str = "config.yaml", seed: int | None = None):
        self.config = load_config(config_path)
        self.seed = seed if seed is not None else self.config.get("random_seed", 42)
        self.reset(self.seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed
        run_config = dict(self.config)
        run_config["random_seed"] = self.seed
        if "adaptive_evasion" in self.config:
            run_config["adaptive_evasion"] = {**self.config["adaptive_evasion"], "seed": self.seed}

        self.env = RFEnvironment(run_config)
        det_cfg = self.config["detection"]
        self.detection_model = DetectionModel(
            threshold_db=det_cfg["threshold_db"], snr_scale=det_cfg["snr_scale"],
            false_alarm_probability=det_cfg["false_alarm_probability"], seed=self.seed)
        self.k = self.config["receiver_channels"]
        self.receiver = Receiver(self.env, k=self.k, detection_model=self.detection_model)

        arb_cfg = dict(self.config.get("ml_arbitrator") or {})
        arb_cfg["seed"] = self.seed
        self.adapter = IntelligentSchedulerAdapter(
            self.config["num_bands"], self.k, self.config.get("belief"),
            self.config.get("temporal"), self.config.get("scoring"), arb_cfg)

        self.e4 = next((e for e in self.env.emitters if e.emitter_id == "E4"), None)
        self.metrics_tracker = EvaluationMetrics(redundancy_window=arb_cfg.get("redundancy_window", 3))
        self.evasion_tracker = EvasionReacquisitionTracker(self.e4) if self.e4 is not None else None
        # Reuses Stage 9's own FeatureExtractor verbatim, purely so the
        # Predictive ML view has live features to feed the loaded model --
        # not a new feature/algorithm.
        horizon = self.config.get("predictive", {}).get("prediction_horizon", 100)
        self.feature_extractor = FeatureExtractor(prediction_horizon=horizon)

        self.t = -1
        self.history = []            # recent per-step summaries (bounded)
        self.belief_history = {}     # band_id -> [(t, P_active), ...] (bounded)
        self.reward_history = []     # [(t, reward), ...] (bounded)
        self.last_ground_truth_debug = None  # ONLY for the labelled debug view
        self.last_features = {}      # band_id -> feature vector, most recent step

    def step(self) -> dict:
        self.t += 1
        self.env.step()

        selected_bands = self.adapter.select_bands(self.t)
        strategy = self.adapter.last_strategy
        # Recomputed via the public API -- belief hasn't changed since
        # select_bands() used this same value internally (learn() is what
        # updates it), so this is exactly the state the decision was made on.
        state = self.adapter.arbitrator.get_state(self.adapter.belief.get_state())

        was_evasive = self.e4.is_evasive if self.e4 is not None else False

        observations = self.receiver.observe(selected_bands)
        self.env.notify_scan_results(observations)

        # --- evaluation-layer-only ground truth, AFTER observe(), never
        # fed to adapter.select_bands()/learn() ---
        self.metrics_tracker.observe_step(self.env, selected_bands, observations)
        self.last_ground_truth_debug = {
            b: {"active": self.env.band_truth(b).active} for b in selected_bands
        }
        for _band_id, correct in self.adapter.check_predictions(observations, self.t):
            self.metrics_tracker.record_prediction_check(correct)

        # snapshot features for each selected band BEFORE this step's own
        # observation is folded into the extractor's recent-history tracker
        belief_state = {b.band_id: b for b in self.adapter.belief.get_state()}
        temporal_state = {b.band_id: b for b in self.adapter.temporal.get_state()}
        score_state = {s.band_id: s for s in self.adapter.scoring.get_scores()}
        self.last_features = {
            band_id: self.feature_extractor.extract(
                band_id, belief_state[band_id], temporal_state[band_id],
                score_state[band_id], self.t, strategy)
            for band_id in selected_bands
        }
        self.feature_extractor.observe(observations)

        reward = self.adapter.learn(observations, self.t)
        self.metrics_tracker.record_reward(reward)

        if self.evasion_tracker is not None:
            hit_on_e4 = bool(observations.get(self.e4.current_band)
                              and observations[self.e4.current_band].hit)
            self.evasion_tracker.observe_step(self.t, hit_on_e4)

        record = {
            "t": self.t, "selected_bands": selected_bands, "strategy": strategy,
            "observations": observations, "reward": reward, "state": state,
            "evasive": bool(self.e4.is_evasive) if self.e4 is not None else False,
            "evasion_count": self.e4.evasion_count if self.e4 is not None else 0,
            "just_triggered_evasion": bool(self.e4 is not None and self.e4.is_evasive and not was_evasive),
        }
        self.history.append(record)
        if len(self.history) > _HISTORY_LIMIT:
            self.history.pop(0)

        self.reward_history.append((self.t, reward))
        if len(self.reward_history) > _HISTORY_LIMIT:
            self.reward_history.pop(0)

        for band_id in selected_bands:
            b = self.adapter.belief.get_belief(band_id)
            hist = self.belief_history.setdefault(band_id, [])
            hist.append((self.t, b.activity_probability))
            if len(hist) > _BELIEF_CHART_LIMIT:
                hist.pop(0)

        return record

    def run(self, n_steps: int) -> None:
        for _ in range(n_steps):
            self.step()

    # ---- read-only snapshots for the UI, all from public Stage 3-6 APIs ----
    def belief_state(self):
        return self.adapter.belief.get_state()

    def temporal_state(self):
        return self.adapter.temporal.get_state()

    def scores(self):
        return self.adapter.scoring.get_scores()

    def current_q_state_and_values(self):
        state = self.adapter.arbitrator.get_state(self.adapter.belief.get_state())
        return state, self.adapter.arbitrator.get_q_values(state)

    def metrics_summary(self) -> dict:
        return self.metrics_tracker.summary()

    def evasion_summary(self) -> dict:
        return self.evasion_tracker.summary() if self.evasion_tracker is not None else {}
