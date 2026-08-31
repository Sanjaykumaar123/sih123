"""Receiver model enforcing the limited channel observation boundary."""

from dataclasses import dataclass
from typing import Dict, List, Optional
from .detection import DetectionModel


class ReceiverCapacityError(Exception):
    """Raised when the scheduler attempts to observe more bands than capacity K."""
    pass


@dataclass
class Observation:
    timestep: int
    band_id: str
    hit: bool
    signal_strength: float
    snr: float
    detection_probability: float = 0.0


class Receiver:
    def __init__(self, environment, k: int = 5,
                 detection_model: Optional[DetectionModel] = None):
        self.env = environment
        self.k = int(k)
        if detection_model is not None:
            self.detection_model = detection_model
        else:
            self.detection_model = DetectionModel(
                threshold_db=10.0, snr_scale=3.0, false_alarm_probability=0.05, seed=42
            )

    def observe(self, selected_bands: List[str]) -> Dict[str, Observation]:
        # Deduplicate while preserving order
        unique_bands = list(dict.fromkeys(selected_bands))
        if len(unique_bands) > self.k:
            raise ReceiverCapacityError(
                f"Cannot observe {len(unique_bands)} bands. Receiver capacity is K={self.k}."
            )

        observations: Dict[str, Observation] = {}
        for band_id in unique_bands:
            truth = self.env.band_truth(band_id)
            det_result = self.detection_model.detect(
                present=truth.active, snr=truth.snr
            )
            hit = det_result.detected
            # Only report signal strength and SNR if detected
            str_val = float(truth.signal_strength) if hit and truth.active else (-70.0 if hit else 0.0)
            snr_val = float(truth.snr) if hit and truth.active else (10.0 if hit else 0.0)

            observations[band_id] = Observation(
                timestep=self.env.timestep,
                band_id=band_id,
                hit=hit,
                signal_strength=str_val,
                snr=snr_val,
                detection_probability=det_result.detection_probability,
            )
        return observations
