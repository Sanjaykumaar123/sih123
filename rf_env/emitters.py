"""Ground-truth emitter models for the RF simulator.

Implements Static, Periodic, Frequency-Agile, and Adaptive-Evasive emitter behaviors.
"""

from typing import List, Optional
import numpy as np


class Emitter:
    def __init__(self, emitter_id: str, emitter_type: str, signal_strength: float, snr: float):
        self.emitter_id = emitter_id
        self.emitter_type = emitter_type
        self.signal_strength = float(signal_strength)
        self.snr = float(snr)
        self.current_band: str = "F01"
        self.active: bool = False

    def update(self, timestep: int) -> None:
        raise NotImplementedError


class StaticEmitter(Emitter):
    def __init__(self, emitter_id: str, band: str, signal_strength: float, snr: float,
                 active_prob: float = 1.0, rng: Optional[np.random.RandomState] = None):
        super().__init__(emitter_id, "static", signal_strength, snr)
        self.current_band = band
        self.active_prob = float(active_prob)
        self.rng = rng if rng is not None else np.random.RandomState()

    def update(self, timestep: int) -> None:
        if self.active_prob >= 1.0:
            self.active = True
        else:
            self.active = bool(self.rng.uniform(0.0, 1.0) < self.active_prob)


class PeriodicEmitter(Emitter):
    def __init__(self, emitter_id: str, band: str, period: int, duty_cycle: float,
                 signal_strength: float, snr: float):
        super().__init__(emitter_id, "periodic", signal_strength, snr)
        self.current_band = band
        self.period = int(period)
        self.duty_cycle = float(duty_cycle)
        self.on_steps = max(1, int(round(self.period * self.duty_cycle)))

    def update(self, timestep: int) -> None:
        phase = timestep % self.period
        self.active = phase < self.on_steps


class FrequencyAgileEmitter(Emitter):
    def __init__(self, emitter_id: str, all_bands: List[str], hop_interval: int,
                 pattern_length: int, signal_strength: float, snr: float,
                 rng: Optional[np.random.RandomState] = None):
        super().__init__(emitter_id, "frequency_agile", signal_strength, snr)
        self.all_bands = list(all_bands)
        self.hop_interval = int(hop_interval)
        self.pattern_length = int(pattern_length)
        self.rng = rng if rng is not None else np.random.RandomState()
        # Generate deterministic pseudo-random sequence of bands
        indices = self.rng.choice(len(self.all_bands), size=self.pattern_length, replace=True)
        self.pattern = [self.all_bands[i] for i in indices]
        self.current_band = self.pattern[0]
        self.active = True

    def update(self, timestep: int) -> None:
        hop_idx = (timestep // self.hop_interval) % self.pattern_length
        self.current_band = self.pattern[hop_idx]
        self.active = True


class AdaptiveEvasiveEmitter(Emitter):
    def __init__(self, emitter_id: str, all_bands: List[str], normal_band: str = "F30",
                 signal_strength: float = -63.0, snr: float = 10.0, hit_threshold: int = 3,
                 observation_window: int = 10, evasive_duration: int = 8, enabled: bool = True,
                 seed: Optional[int] = None, rng: Optional[np.random.RandomState] = None):
        super().__init__(emitter_id, "adaptive_evasive", signal_strength, snr)
        self.all_bands = list(all_bands)
        self.normal_band = normal_band
        self.current_band = normal_band
        self.active = True
        self.hit_threshold = int(hit_threshold)
        self.observation_window = int(observation_window)
        self.evasive_duration = int(evasive_duration)
        self.enabled = bool(enabled)
        self.seed = seed
        self.rng = rng if rng is not None else (np.random.RandomState(seed) if seed is not None else np.random.RandomState())

        self.recent_detection_times: List[int] = []
        self.is_evasive: bool = False
        self.evasion_count: int = 0
        self._evasive_remaining: int = 0
        self._evasive_pattern: List[str] = []

    def register_detection(self, detected: bool, timestep: int) -> None:
        if not self.enabled:
            return
        if self.is_evasive:
            # Already evading; don't stack triggers during evasion burst
            return
        if detected:
            self.recent_detection_times.append(timestep)
            # Filter detections to observation window
            cutoff = timestep - self.observation_window + 1
            self.recent_detection_times = [t for t in self.recent_detection_times if t >= cutoff]
            if len(self.recent_detection_times) >= self.hit_threshold:
                # Trigger evasion
                self.is_evasive = True
                self.evasion_count += 1
                self._evasive_remaining = self.evasive_duration
                # Pick unique random evasive path
                chosen_idx = self.rng.choice(len(self.all_bands), size=self.evasive_duration, replace=True)
                self._evasive_pattern = [self.all_bands[i] for i in chosen_idx]
                # Reset detection history immediately
                self.recent_detection_times = []
        else:
            cutoff = timestep - self.observation_window + 1
            self.recent_detection_times = [t for t in self.recent_detection_times if t >= cutoff]

    def update(self, timestep: int) -> None:
        self.active = True
        if self.is_evasive:
            if self._evasive_remaining <= 0:
                self.is_evasive = False
                self.normal_band = self.current_band
                self.current_band = self.normal_band
            else:
                step_idx = self.evasive_duration - self._evasive_remaining
                self.current_band = self._evasive_pattern[step_idx]
                self._evasive_remaining -= 1
                if self._evasive_remaining == 0:
                    self.normal_band = self.current_band
        else:
            self.current_band = self.normal_band
