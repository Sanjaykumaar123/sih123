"""Temporal behaviour analysis and prediction engine."""

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional
import numpy as np

from .receiver import Observation


@dataclass
class TemporalPrediction:
    band_id: str
    periodicity_score: float
    estimated_period: Optional[float]
    predicted_next_active_time: Optional[float]
    prediction_confidence: float
    last_hit_timestep: Optional[int]
    time_since_last_hit: float
    behaviour_type: str
    number_of_hits: int


class TemporalEngine:
    def __init__(self, num_bands: int, config: Optional[dict] = None):
        self.num_bands = int(num_bands)
        self.bands: List[str] = [f"F{i:02d}" for i in range(1, self.num_bands + 1)]
        cfg = config or {}
        self.history_length = int(cfg.get("history_length", 50))
        self.min_hits_for_prediction = int(cfg.get("min_hits_for_prediction", 3))
        self.periodicity_threshold = float(cfg.get("periodicity_threshold", 0.7))
        self.stable_interval_max = float(cfg.get("stable_interval_max", 1.5))

        self._current_timestep: int = -1
        self._history: Dict[str, Deque[Observation]] = {}
        self._hit_timesteps: Dict[str, List[int]] = {}
        self.reset()

    def reset(self) -> None:
        self._current_timestep = -1
        self._history = {b: deque(maxlen=self.history_length) for b in self.bands}
        self._hit_timesteps = {b: [] for b in self.bands}

    def update(self, observations: Dict[str, Observation], current_timestep: int) -> None:
        self._current_timestep = int(current_timestep)
        for band_id, obs in observations.items():
            if band_id in self._history:
                self._history[band_id].append(obs)
                if obs.hit:
                    self._hit_timesteps[band_id].append(obs.timestep)
                    if len(self._hit_timesteps[band_id]) > self.history_length:
                        self._hit_timesteps[band_id].pop(0)

    def get_prediction(self, band_id: str) -> TemporalPrediction:
        hits = self._hit_timesteps[band_id]
        n_hits = len(hits)
        last_hit = hits[-1] if n_hits > 0 else None
        if last_hit is None or self._current_timestep < 0:
            time_since_last = float("inf")
        else:
            time_since_last = float(max(0, self._current_timestep - last_hit))

        if n_hits < self.min_hits_for_prediction:
            return TemporalPrediction(
                band_id=band_id,
                periodicity_score=0.0,
                estimated_period=None,
                predicted_next_active_time=None,
                prediction_confidence=0.0,
                last_hit_timestep=last_hit,
                time_since_last_hit=time_since_last,
                behaviour_type="insufficient_data",
                number_of_hits=n_hits,
            )

        # Calculate inter-hit intervals
        intervals = np.diff(hits)
        mean_int = float(np.mean(intervals))
        std_int = float(np.std(intervals))

        if mean_int > 0:
            cov = std_int / mean_int
            periodicity_score = float(1.0 / (1.0 + cov))
        else:
            periodicity_score = 1.0

        predicted_next = float(last_hit + mean_int) if last_hit is not None else None
        # Confidence scaled by evidence factor
        evidence_factor = min(1.0, n_hits / float(self.min_hits_for_prediction + 2))
        confidence = float(np.clip(periodicity_score * evidence_factor, 0.0, 1.0))

        if periodicity_score >= self.periodicity_threshold:
            behaviour_type = "periodic"
        elif std_int <= self.stable_interval_max:
            behaviour_type = "stable"
        else:
            behaviour_type = "intermittent"

        return TemporalPrediction(
            band_id=band_id,
            periodicity_score=periodicity_score,
            estimated_period=mean_int,
            predicted_next_active_time=predicted_next,
            prediction_confidence=confidence,
            last_hit_timestep=last_hit,
            time_since_last_hit=time_since_last,
            behaviour_type=behaviour_type,
            number_of_hits=n_hits,
        )

    def get_state(self) -> List[TemporalPrediction]:
        return [self.get_prediction(b) for b in self.bands]
