"""TSRD Environment and Unified Scenario Builder.

Provides TSRDEnvironment with identical interface to RFEnvironment,
enabling plug-and-play operation with Receiver, BeliefEngine, TemporalEngine,
BandScoringEngine, QLearningArbitrator, and EvaluationMetrics.
"""

from enum import Enum
import os
from typing import Any, Dict, List, Optional, Union

from rf_env.environment import BandTruth, RFEnvironment
from rf_env.receiver import Receiver

from .frequency_mapper import FrequencyMapper
from .pdw_processor import PDWProcessor
from .truth_manager import TruthManager
from .tsrd_loader import TSRDLoader, TSRDRawData


class EnvironmentSource(str, Enum):
    SYNTHETIC = "synthetic"
    TSRD = "tsrd"


class TSRDEnvironment:
    """RF Environment powered by Turing Synthetic Radar Dataset (TSRD) HDF5 scenario recordings.

    Adheres strictly to the RFEnvironment interface while drawing RF activity from
    high-fidelity synthetic pulse streams.
    """

    def __init__(self, file_path: str, step_duration_s: float = 0.05,
                 num_bands: int = 50, sensitivity_dbm: Optional[float] = None):
        self.file_path = os.path.abspath(file_path)
        self.num_bands = int(num_bands)
        self.step_duration_s = float(step_duration_s)

        # 1. Load HDF5 Scenario
        self.raw_data: TSRDRawData = TSRDLoader.load_file(self.file_path)

        # 2. Setup Frequency Mapper
        rec_meta = self.raw_data.receiver_metadata
        self.freq_mapper = FrequencyMapper(
            f_min_mhz=rec_meta.freq_range_mhz[0],
            f_max_mhz=rec_meta.freq_range_mhz[1],
            num_bands=self.num_bands,
        )
        self.bands: List[str] = self.freq_mapper.all_band_ids

        # 3. Setup PDW Processor
        self.processor = PDWProcessor(
            raw_data=self.raw_data,
            freq_mapper=self.freq_mapper,
            step_duration_s=self.step_duration_s,
            sensitivity_dbm=sensitivity_dbm,
        )
        self.total_steps = self.processor.total_steps

        # 4. Setup Truth Manager
        self.truth_manager = TruthManager(
            raw_data=self.raw_data,
            processor=self.processor,
        )

        self.timestep: int = -1
        self._band_truth_cache: Dict[str, BandTruth] = {}

    def reset(self, start_timestep: int = -1) -> None:
        """Reset the environment timestep."""
        self.timestep = int(start_timestep)
        self._band_truth_cache = {}

    def step(self) -> None:
        """Advance simulation by one time step."""
        self.timestep += 1
        step_act = self.processor.get_step_activity(self.timestep)

        # Build current step's BandTruth cache
        self._band_truth_cache = {}
        for b in self.bands:
            b_act = step_act.band_activities.get(b)
            if b_act is not None and b_act.is_detectable:
                primary_e_id = f"TSRD_E{b_act.ground_truth_emitter_ids[0]:02d}" if b_act.ground_truth_emitter_ids else "TSRD_E00"
                self._band_truth_cache[b] = BandTruth(
                    band_id=b,
                    active=True,
                    emitter_id=primary_e_id,
                    emitter_type="tsrd_radar",
                    signal_strength=b_act.max_amplitude_dbm,
                    snr=b_act.snr_db,
                )
            else:
                self._band_truth_cache[b] = BandTruth(
                    band_id=b,
                    active=False,
                    emitter_id=None,
                    emitter_type=None,
                    signal_strength=-120.0 if b_act else 0.0,
                    snr=0.0,
                )

    def band_truth(self, band_id: str) -> BandTruth:
        """Return hidden ground-truth RF state for evaluation."""
        if not self._band_truth_cache:
            return BandTruth(
                band_id=band_id,
                active=False,
                emitter_id=None,
                emitter_type=None,
                signal_strength=0.0,
                snr=0.0,
            )
        return self._band_truth_cache.get(
            band_id,
            BandTruth(
                band_id=band_id,
                active=False,
                emitter_id=None,
                emitter_type=None,
                signal_strength=0.0,
                snr=0.0,
            )
        )

    def full_ground_truth_snapshot(self) -> List[Dict]:
        """Return snapshot of all bands for dashboard/evaluation display."""
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
        """TSRD scenarios are recorded radar datasets; notifications do not mutate recorded pulses."""
        pass


def create_environment(source: Union[EnvironmentSource, str],
                       config_or_path: Union[dict, str],
                       **kwargs) -> Union[RFEnvironment, TSRDEnvironment]:
    """Factory creating either Synthetic or TSRD RF environments."""
    src = str(source).lower()
    if "tsrd" in src:
        if isinstance(config_or_path, dict):
            fpath = config_or_path.get("file_path", config_or_path.get("path"))
            step_dur = config_or_path.get("step_duration_s", kwargs.get("step_duration_s", 0.05))
            num_b = config_or_path.get("num_bands", kwargs.get("num_bands", 50))
            return TSRDEnvironment(fpath, step_duration_s=step_dur, num_bands=num_b, **kwargs)
        else:
            return TSRDEnvironment(str(config_or_path), **kwargs)
    elif "synth" in src:
        if isinstance(config_or_path, str):
            from rf_env.config import load_config
            cfg = load_config(config_or_path)
            return RFEnvironment(cfg)
        return RFEnvironment(config_or_path)
    else:
        raise ValueError(f"Unknown environment source: {source}. Expected 'synthetic' or 'tsrd'.")
