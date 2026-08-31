"""Ground-truth manager for TSRD scenarios.

Isolates emitter labels and transmitter identities for post-hoc evaluation and verification,
preventing ground-truth leakage into online cognitive decision algorithms.
"""

from typing import Any, Dict, List, Optional, Set
from .pdw_processor import PDWProcessor
from .tsrd_loader import TSRDRawData


class TruthManager:
    """Manages ground-truth emitter information strictly for evaluation and metric computation."""

    def __init__(self, raw_data: TSRDRawData, processor: PDWProcessor):
        self.raw_data = raw_data
        self.processor = processor
        self.total_steps = processor.total_steps

        # Precompute emitter statistics
        self._emitter_pulse_counts: Dict[int, int] = {}
        self._emitter_first_seen_step: Dict[int, int] = {}
        self._emitter_last_seen_step: Dict[int, int] = {}
        self._emitter_active_steps: Dict[int, Set[int]] = {}
        self._emitter_active_bands: Dict[int, Set[str]] = {}

        self._build_truth_indexes()

    def _build_truth_indexes(self) -> None:
        labels = self.raw_data.labels
        for e_id in labels:
            e_int = int(e_id)
            self._emitter_pulse_counts[e_int] = self._emitter_pulse_counts.get(e_int, 0) + 1

        for t in range(self.total_steps):
            step_act = self.processor.get_step_activity(t)
            for band_id, b_act in step_act.band_activities.items():
                for e_int in b_act.ground_truth_emitter_ids:
                    if e_int not in self._emitter_first_seen_step:
                        self._emitter_first_seen_step[e_int] = t
                    self._emitter_last_seen_step[e_int] = t
                    self._emitter_active_steps.setdefault(e_int, set()).add(t)
                    self._emitter_active_bands.setdefault(e_int, set()).add(band_id)

    def get_all_emitter_ids(self) -> List[int]:
        """Return list of all unique emitter IDs in the scenario."""
        return sorted(list(self._emitter_pulse_counts.keys()))

    def get_emitter_pulse_count(self, emitter_id: int) -> int:
        """Return total pulse count emitted by the specified emitter."""
        return self._emitter_pulse_counts.get(int(emitter_id), 0)

    def get_active_emitters_at_step(self, timestep: int) -> List[int]:
        """Return list of emitter IDs with pulses present at the specified timestep."""
        step_act = self.processor.get_step_activity(timestep)
        active_emitters = set()
        for b_act in step_act.band_activities.values():
            active_emitters.update(b_act.ground_truth_emitter_ids)
        return sorted(list(active_emitters))

    def get_band_ground_truth(self, timestep: int, band_id: str) -> Dict[str, Any]:
        """Return ground truth activity for a specific band at a timestep."""
        b_act = self.processor.get_band_activity(timestep, band_id)
        if b_act is None:
            return {
                "band_id": band_id,
                "active": False,
                "pulse_count": 0,
                "emitter_ids": [],
                "max_amplitude_dbm": 0.0,
                "snr_db": 0.0,
                "is_detectable": False,
            }
        return {
            "band_id": band_id,
            "active": b_act.is_detectable,
            "pulse_count": b_act.pulse_count,
            "emitter_ids": list(b_act.ground_truth_emitter_ids),
            "max_amplitude_dbm": b_act.max_amplitude_dbm,
            "snr_db": b_act.snr_db,
            "is_detectable": b_act.is_detectable,
        }

    def get_emitter_metadata(self, emitter_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve transmitter metadata from HDF5 scenario metadata."""
        e_idx = int(emitter_id)
        if 0 <= e_idx < len(self.raw_data.transmitter_metadata):
            return self.raw_data.transmitter_metadata[e_idx]
        return None
