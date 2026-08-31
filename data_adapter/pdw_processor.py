"""PDW (Pulse Descriptor Word) stream processor and discrete time binner.

Converts continuous asynchronous pulse streams (ToA in microseconds, Frequency in MHz,
PulseWidth in microseconds, AoA in degrees, Amplitude in dBm) into discrete 50-band
time-binned RF activity frames.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from .frequency_mapper import FrequencyMapper
from .tsrd_loader import TSRDRawData


@dataclass
class BinnedBandActivity:
    band_id: str
    timestep: int
    pulse_count: int
    max_amplitude_dbm: float
    mean_amplitude_dbm: float
    mean_frequency_mhz: float
    mean_pulse_width_us: float
    mean_aoa_deg: float
    snr_db: float
    is_detectable: bool
    ground_truth_emitter_ids: List[int] = field(default_factory=list)


@dataclass
class TimeStepActivity:
    timestep: int
    time_start_s: float
    time_end_s: float
    active_bands: Set[str] = field(default_factory=set)
    band_activities: Dict[str, BinnedBandActivity] = field(default_factory=dict)


class PDWProcessor:
    """Processes raw TSRD pulse matrices into discrete, synchronized RF environment time steps."""

    def __init__(self, raw_data: TSRDRawData, freq_mapper: Optional[FrequencyMapper] = None,
                 step_duration_s: float = 0.05, sensitivity_dbm: Optional[float] = None):
        self.raw_data = raw_data
        rec_meta = raw_data.receiver_metadata
        self.sensitivity_dbm = float(sensitivity_dbm if sensitivity_dbm is not None else rec_meta.sensitivity_dbm)
        self.freq_mapper = freq_mapper or FrequencyMapper(
            f_min_mhz=rec_meta.freq_range_mhz[0],
            f_max_mhz=rec_meta.freq_range_mhz[1],
            num_bands=50
        )
        self.step_duration_s = float(step_duration_s)
        self.total_duration_s = float(raw_data.duration_s)
        self.total_steps = max(1, int(np.ceil(self.total_duration_s / self.step_duration_s)))

        # Precompute binned activities
        self._timestep_activities: List[TimeStepActivity] = []
        self._build_time_bins()

    def _build_time_bins(self) -> None:
        """Bin all pulses into (timestep, band_id) matrix buckets using vectorized operations."""
        self._timestep_activities = [
            TimeStepActivity(
                timestep=t,
                time_start_s=t * self.step_duration_s,
                time_end_s=(t + 1) * self.step_duration_s,
                active_bands=set(),
                band_activities={},
            )
            for t in range(self.total_steps)
        ]

        if self.raw_data.num_pulses == 0:
            return

        pulse_matrix = self.raw_data.pulse_data
        labels = self.raw_data.labels

        # Column indices
        # 0: ToA (microseconds)
        # 1: Frequency (MHz)
        # 2: PulseWidth (microseconds)
        # 3: AoA (degrees)
        # 4: Amplitude (dBm)
        toa_s = pulse_matrix[:, 0] * 1e-6
        freqs_mhz = pulse_matrix[:, 1]
        pws_us = pulse_matrix[:, 2]
        aoas_deg = pulse_matrix[:, 3]
        amps_dbm = pulse_matrix[:, 4]

        # Vectorized step assignment
        step_indices = np.clip(
            (toa_s / self.step_duration_s).astype(np.int32),
            0, self.total_steps - 1
        )

        # Vectorized band mapping
        f_min = self.freq_mapper.f_min_mhz
        bw = self.freq_mapper.band_width
        n_bands = self.freq_mapper.num_bands
        band_indices = np.clip(((freqs_mhz - f_min) / bw).astype(np.int32), 0, n_bands - 1)

        # Composite key for fast group by: key = step_idx * n_bands + band_idx
        composite_keys = step_indices * n_bands + band_indices
        sort_order = np.argsort(composite_keys)
        sorted_keys = composite_keys[sort_order]
        unique_keys, split_indices = np.unique(sorted_keys, return_index=True)
        key_groups = np.split(sort_order, split_indices[1:])

        for key, group_pulse_indices in zip(unique_keys, key_groups):
            t_idx = int(key // n_bands)
            b_idx = int(key % n_bands)
            band_id = self.freq_mapper.band_index_to_id(b_idx)

            g_amps = amps_dbm[group_pulse_indices]
            g_freqs = freqs_mhz[group_pulse_indices]
            g_pws = pws_us[group_pulse_indices]
            g_aoas = aoas_deg[group_pulse_indices]
            g_labels = labels[group_pulse_indices]

            max_amp = float(np.max(g_amps))
            mean_amp = float(np.mean(g_amps))
            mean_freq = float(np.mean(g_freqs))
            mean_pw = float(np.mean(g_pws))
            mean_aoa = float(np.mean(g_aoas))
            p_count = len(group_pulse_indices)

            # Detectability check against receiver sensitivity (-110 dBm)
            is_det = bool(max_amp >= self.sensitivity_dbm)
            snr = float(max(0.0, max_amp - self.sensitivity_dbm))

            unique_e_ids = sorted(list(set(int(x) for x in g_labels)))

            activity = BinnedBandActivity(
                band_id=band_id,
                timestep=t_idx,
                pulse_count=p_count,
                max_amplitude_dbm=max_amp,
                mean_amplitude_dbm=mean_amp,
                mean_frequency_mhz=mean_freq,
                mean_pulse_width_us=mean_pw,
                mean_aoa_deg=mean_aoa,
                snr_db=snr,
                is_detectable=is_det,
                ground_truth_emitter_ids=unique_e_ids,
            )

            step_act = self._timestep_activities[t_idx]
            step_act.band_activities[band_id] = activity
            if is_det:
                step_act.active_bands.add(band_id)

    def get_step_activity(self, timestep: int) -> TimeStepActivity:
        """Get the full activity snapshot for a timestep."""
        if 0 <= timestep < self.total_steps:
            return self._timestep_activities[timestep]
        # Return empty activity for out-of-range steps
        return TimeStepActivity(timestep=timestep, time_start_s=0.0, time_end_s=0.0, active_bands=set(), band_activities={})

    def get_band_activity(self, timestep: int, band_id: str) -> Optional[BinnedBandActivity]:
        """Get activity for a specific band at a timestep."""
        if 0 <= timestep < self.total_steps:
            return self._timestep_activities[timestep].band_activities.get(band_id)
        return None

    def get_active_bands(self, timestep: int) -> Set[str]:
        """Get the set of detectable active band IDs at a timestep."""
        if 0 <= timestep < self.total_steps:
            return self._timestep_activities[timestep].active_bands
        return set()
