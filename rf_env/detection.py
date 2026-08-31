"""Probabilistic detection physics model."""

from dataclasses import dataclass
import numpy as np


@dataclass
class DetectionResult:
    detected: bool
    snr: float
    detection_probability: float
    false_alarm: bool


class DetectionModel:
    def __init__(self, threshold_db: float = 10.0, snr_scale: float = 3.0,
                 false_alarm_probability: float = 0.05, seed: int | None = 42):
        self.threshold_db = float(threshold_db)
        self.snr_scale = float(snr_scale)
        self.false_alarm_probability = float(false_alarm_probability)
        self.seed = seed
        self.rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

    def probability_of_detection(self, snr: float) -> float:
        # Standard logistic sigmoid centered at threshold_db
        z = (snr - self.threshold_db) / self.snr_scale
        # Clip to prevent overflow
        z = np.clip(z, -50.0, 50.0)
        return float(1.0 / (1.0 + np.exp(-z)))

    def detect(self, present: bool, snr: float) -> DetectionResult:
        if present:
            p_d = self.probability_of_detection(snr)
            detected = bool(self.rng.uniform(0.0, 1.0) < p_d)
            return DetectionResult(
                detected=detected,
                snr=float(snr),
                detection_probability=p_d,
                false_alarm=False,
            )
        else:
            p_fa = self.false_alarm_probability
            detected = bool(self.rng.uniform(0.0, 1.0) < p_fa)
            return DetectionResult(
                detected=detected,
                snr=0.0,
                detection_probability=p_fa,
                false_alarm=detected,
            )
