"""RF Environment and Ground Truth Logger."""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from .emitters import (Emitter, StaticEmitter, PeriodicEmitter,
                       FrequencyAgileEmitter, AdaptiveEvasiveEmitter)


@dataclass
class BandTruth:
    band_id: str
    active: bool
    emitter_id: Optional[str]
    emitter_type: Optional[str]
    signal_strength: float
    snr: float


class GroundTruthLogger:
    def __init__(self):
        self.records: List[Dict] = []

    def log(self, timestep: int, emitter_id: str, emitter_type: str, band: str,
            active: bool, signal_strength: float, snr: float) -> None:
        self.records.append({
            "timestep": timestep,
            "emitter_id": emitter_id,
            "emitter_type": emitter_type,
            "band": band,
            "active": active,
            "signal_strength": signal_strength,
            "snr": snr,
        })


class RFEnvironment:
    def __init__(self, config: dict):
        self.config = dict(config)
        self.num_bands = int(self.config.get("num_bands", 50))
        self.bands: List[str] = [f"F{i:02d}" for i in range(1, self.num_bands + 1)]
        self.seed = self.config.get("random_seed", 42)
        self.rng = np.random.RandomState(self.seed) if self.seed is not None else np.random.RandomState()
        self.timestep: int = -1
        self.logger = GroundTruthLogger()
        self._band_truth_map: Dict[str, BandTruth] = {}
        self.emitters: List[Emitter] = self._build_emitters()

    def _build_emitters(self) -> List[Emitter]:
        emitters = []
        emitter_configs = self.config.get("emitters", [])
        evasion_global = self.config.get("adaptive_evasion", {})

        for cfg in emitter_configs:
            e_id = cfg["id"]
            e_type = cfg["type"]
            str_val = float(cfg.get("signal_strength", -60.0))
            snr_val = float(cfg.get("snr", 15.0))

            if e_type == "static":
                band = cfg.get("band", "F10")
                prob = float(cfg.get("active_prob", 1.0))
                e_seed = self.rng.randint(0, 1_000_000)
                emitters.append(StaticEmitter(e_id, band, str_val, snr_val, active_prob=prob,
                                              rng=np.random.RandomState(e_seed)))
            elif e_type == "periodic":
                band = cfg.get("band", "F20")
                period = int(cfg.get("period", 10))
                duty = float(cfg.get("duty_cycle", 0.1))
                emitters.append(PeriodicEmitter(e_id, band, period, duty, str_val, snr_val))
            elif e_type == "frequency_agile":
                hop = int(cfg.get("hop_interval", 2))
                pat = int(cfg.get("pattern_length", 20))
                e_seed = self.rng.randint(0, 1_000_000)
                emitters.append(FrequencyAgileEmitter(e_id, self.bands, hop, pat, str_val, snr_val,
                                                      rng=np.random.RandomState(e_seed)))
            elif e_type == "adaptive_evasive":
                band = cfg.get("band", "F30")
                hit_thresh = int(evasion_global.get("hit_threshold", cfg.get("hit_threshold", 3)))
                obs_win = int(evasion_global.get("observation_window", cfg.get("observation_window", 10)))
                ev_dur = int(evasion_global.get("evasive_duration", cfg.get("evasive_duration", 8)))
                enabled = bool(evasion_global.get("enabled", cfg.get("enabled", True)))
                e_seed = evasion_global.get("seed", cfg.get("seed", self.rng.randint(0, 1_000_000)))
                emitters.append(AdaptiveEvasiveEmitter(
                    e_id, self.bands, normal_band=band, signal_strength=str_val, snr=snr_val,
                    hit_threshold=hit_thresh, observation_window=obs_win, evasive_duration=ev_dur,
                    enabled=enabled, seed=e_seed
                ))
            else:
                raise ValueError(f"Unknown emitter type: {e_type}")
        return emitters

    def step(self) -> None:
        self.timestep += 1
        # Update all emitters
        for e in self.emitters:
            e.update(self.timestep)
            self.logger.log(
                timestep=self.timestep,
                emitter_id=e.emitter_id,
                emitter_type=e.emitter_type,
                band=e.current_band,
                active=e.active,
                signal_strength=e.signal_strength if e.active else 0.0,
                snr=e.snr if e.active else 0.0,
            )

        # Build current step band truth map
        self._band_truth_map = {}
        for b in self.bands:
            # Find active emitters on band b
            active_on_b = [e for e in self.emitters if e.current_band == b and e.active]
            if active_on_b:
                # If multiple, choose highest SNR
                best = max(active_on_b, key=lambda e: e.snr)
                self._band_truth_map[b] = BandTruth(
                    band_id=b,
                    active=True,
                    emitter_id=best.emitter_id,
                    emitter_type=best.emitter_type,
                    signal_strength=best.signal_strength,
                    snr=best.snr,
                )
            else:
                self._band_truth_map[b] = BandTruth(
                    band_id=b,
                    active=False,
                    emitter_id=None,
                    emitter_type=None,
                    signal_strength=0.0,
                    snr=0.0,
                )

    def band_truth(self, band_id: str) -> BandTruth:
        if not self._band_truth_map:
            # Cold start prior to first step
            return BandTruth(band_id=band_id, active=False, emitter_id=None, emitter_type=None,
                             signal_strength=0.0, snr=0.0)
        return self._band_truth_map.get(
            band_id,
            BandTruth(band_id=band_id, active=False, emitter_id=None, emitter_type=None,
                      signal_strength=0.0, snr=0.0)
        )

    def full_ground_truth_snapshot(self) -> List[Dict]:
        return [
            {
                "band_id": b,
                "active": self.band_truth(b).active,
                "emitter_id": self.band_truth(b).emitter_id,
                "emitter_type": self.band_truth(b).emitter_type,
                "signal_strength": self.band_truth(b).signal_strength,
                "snr": self.band_truth(b).snr,
            }
            for b in self.bands
        ]

    def notify_scan_results(self, observations: dict) -> None:
        """Route detection feedback to the relevant emitter object."""
        for e in self.emitters:
            if hasattr(e, "register_detection"):
                if e.current_band in observations:
                    obs = observations[e.current_band]
                    # The hit occurred on this emitter's active band
                    e.register_detection(bool(obs.hit and e.active), self.timestep)
                else:
                    e.register_detection(False, self.timestep)
