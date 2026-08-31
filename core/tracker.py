"""Stage 14: Internal Emitter/Signal Track Manager.

Clusters observable RF pulse detections into autonomous internal signal tracks
WITHOUT accessing ground-truth labels or emitter identities.
Strict ground-truth boundary isolation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from rf_env.receiver import Observation


def get_band_freq_range(band_id: str, n_bands: int = 50, f_min_mhz: float = 500.0, f_max_mhz: float = 18000.0) -> Tuple[float, float, float]:
    """Calculate frequency range for given band ID without dashboard dependency."""
    try:
        idx = int(band_id.replace("F", "")) - 1
    except Exception:
        idx = 0
    band_width = (f_max_mhz - f_min_mhz) / n_bands
    f_low = f_min_mhz + idx * band_width
    f_high = f_low + band_width
    f_center = (f_low + f_high) / 2.0
    return f_low, f_high, f_center


class TrackState:
    NEW = "NEW"
    TENTATIVE = "TENTATIVE"
    CONFIRMED = "CONFIRMED"
    ACTIVE = "ACTIVE"
    LOST = "LOST"
    EXPIRED = "EXPIRED"


@dataclass
class SignalTrack:
    track_id: str
    band_id: str
    state: str = TrackState.NEW
    estimated_frequency_mhz: float = 0.0
    estimated_pulse_width_us: float = 0.0
    estimated_aoa_deg: float = 0.0
    estimated_amplitude_dbm: float = -90.0
    estimated_snr_db: float = 0.0
    last_seen_time_s: float = 0.0
    last_seen_timestep: int = 0
    first_seen_timestep: int = 0
    hit_count: int = 1
    miss_count_since_last_hit: int = 0
    confidence_pct: float = 50.0
    observed_pri_timesteps: List[float] = field(default_factory=list)
    estimated_pri_timesteps: Optional[float] = None
    expected_next_recurrence_s: Optional[float] = None
    observable_history: List[Dict[str, Any]] = field(default_factory=list)


class TrackManager:
    """Manages creation, association, state transitions, and pruning of signal tracks."""

    def __init__(
        self,
        confirmation_hits: int = 2,
        lost_threshold_steps: int = 8,
        expire_threshold_steps: int = 25,
        freq_tolerance_mhz: float = 120.0,
    ):
        self.confirmation_hits = int(confirmation_hits)
        self.lost_threshold_steps = int(lost_threshold_steps)
        self.expire_threshold_steps = int(expire_threshold_steps)
        self.freq_tolerance_mhz = float(freq_tolerance_mhz)

        self.tracks: Dict[str, SignalTrack] = {}
        self._next_track_num: int = 1
        self.track_event_log: List[Dict[str, Any]] = []

    def reset(self) -> None:
        self.tracks.clear()
        self._next_track_num = 1
        self.track_event_log.clear()

    def update(
        self,
        observations: Dict[str, Observation],
        channel_telemetry: List[Dict[str, Any]],
        timestep: int,
        simulated_time_s: float,
    ) -> List[Dict[str, Any]]:
        """Update existing tracks or create new ones from current step observations."""
        events_generated = []

        # 1. Process active hits from observable channel telemetry
        for ch_data in channel_telemetry:
            b_name = ch_data.get("band", "")
            obs = observations.get(b_name)
            if not obs or not obs.hit:
                continue

            meas_freq = ch_data.get("frequency_mhz", 0.0)
            meas_amp = ch_data.get("amplitude_dbm", obs.signal_strength)
            meas_snr = ch_data.get("snr_db", obs.snr)
            meas_pw = ch_data.get("pulse_width_us", 5.0)
            meas_aoa = ch_data.get("aoa_deg", 45.0)

            # Fallback if telemetry was None
            if meas_amp is None:
                meas_amp = obs.signal_strength
            if meas_snr is None:
                meas_snr = obs.snr
            if meas_pw is None:
                meas_pw = 5.0
            if meas_aoa is None:
                meas_aoa = 45.0

            # Match against existing tracks on this band or adjacent frequency
            matched_track = self._find_best_match(b_name, meas_freq)

            if matched_track:
                # Update existing track
                prev_state = matched_track.state
                delta_t = timestep - matched_track.last_seen_timestep
                if delta_t > 0:
                    matched_track.observed_pri_timesteps.append(float(delta_t))
                    if len(matched_track.observed_pri_timesteps) > 10:
                        matched_track.observed_pri_timesteps.pop(0)
                    matched_track.estimated_pri_timesteps = float(np.median(matched_track.observed_pri_timesteps))
                    matched_track.expected_next_recurrence_s = simulated_time_s + (matched_track.estimated_pri_timesteps * 0.05)

                matched_track.hit_count += 1
                matched_track.miss_count_since_last_hit = 0
                matched_track.last_seen_timestep = timestep
                matched_track.last_seen_time_s = simulated_time_s

                # Exponential smoothing of observable measurements
                alpha = 0.40
                matched_track.estimated_frequency_mhz = (1 - alpha) * matched_track.estimated_frequency_mhz + alpha * meas_freq
                matched_track.estimated_pulse_width_us = (1 - alpha) * matched_track.estimated_pulse_width_us + alpha * meas_pw
                matched_track.estimated_aoa_deg = (1 - alpha) * matched_track.estimated_aoa_deg + alpha * meas_aoa
                matched_track.estimated_amplitude_dbm = (1 - alpha) * matched_track.estimated_amplitude_dbm + alpha * meas_amp
                matched_track.estimated_snr_db = (1 - alpha) * matched_track.estimated_snr_db + alpha * meas_snr

                # Record observable measurement history (Zero ground truth)
                matched_track.observable_history.append({
                    "timestep": timestep,
                    "time_s": simulated_time_s,
                    "frequency_mhz": meas_freq,
                    "pulse_width_us": meas_pw,
                    "aoa_deg": meas_aoa,
                    "amplitude_dbm": meas_amp,
                    "snr_db": meas_snr,
                })
                if len(matched_track.observable_history) > 100:
                    matched_track.observable_history.pop(0)

                # State Transitions
                if matched_track.hit_count >= self.confirmation_hits:
                    matched_track.state = TrackState.CONFIRMED
                    matched_track.confidence_pct = min(98.0, 50.0 + matched_track.hit_count * 8.0)
                else:
                    matched_track.state = TrackState.TENTATIVE
                    matched_track.confidence_pct = 60.0

                if prev_state != matched_track.state:
                    ev = {
                        "time_s": f"{simulated_time_s:.2f}s",
                        "timestep": timestep,
                        "track_id": matched_track.track_id,
                        "band": b_name,
                        "event": f"STATUS: {prev_state} → {matched_track.state}",
                        "confidence": f"{matched_track.confidence_pct:.0f}%",
                    }
                    self.track_event_log.insert(0, ev)
                    events_generated.append(ev)
            else:
                # Create NEW Track
                new_track_id = f"TRACK-{self._next_track_num:03d}"
                self._next_track_num += 1

                new_track = SignalTrack(
                    track_id=new_track_id,
                    band_id=b_name,
                    state=TrackState.NEW,
                    estimated_frequency_mhz=meas_freq,
                    estimated_pulse_width_us=meas_pw,
                    estimated_aoa_deg=meas_aoa,
                    estimated_amplitude_dbm=meas_amp,
                    estimated_snr_db=meas_snr,
                    last_seen_time_s=simulated_time_s,
                    last_seen_timestep=timestep,
                    first_seen_timestep=timestep,
                    hit_count=1,
                    miss_count_since_last_hit=0,
                    confidence_pct=50.0,
                    observable_history=[{
                        "timestep": timestep,
                        "time_s": simulated_time_s,
                        "frequency_mhz": meas_freq,
                        "pulse_width_us": meas_pw,
                        "aoa_deg": meas_aoa,
                        "amplitude_dbm": meas_amp,
                        "snr_db": meas_snr,
                    }],
                )
                self.tracks[new_track_id] = new_track
                ev = {
                    "time_s": f"{simulated_time_s:.2f}s",
                    "timestep": timestep,
                    "track_id": new_track_id,
                    "band": b_name,
                    "event": "NEW SIGNAL DETECTED (INIT)",
                    "confidence": "50%",
                }
                self.track_event_log.insert(0, ev)
                events_generated.append(ev)

        # 2. Update track degradation for unobserved tracks
        for t_id, track in list(self.tracks.items()):
            if track.state == TrackState.EXPIRED:
                continue

            steps_unseen = timestep - track.last_seen_timestep
            if steps_unseen > 0:
                track.miss_count_since_last_hit = steps_unseen

                if steps_unseen >= self.expire_threshold_steps:
                    track.state = TrackState.EXPIRED
                    track.confidence_pct = max(0.0, track.confidence_pct - 20.0)
                    ev = {
                        "time_s": f"{simulated_time_s:.2f}s",
                        "timestep": timestep,
                        "track_id": t_id,
                        "band": track.band_id,
                        "event": "TRACK EXPIRED (STALE > 25 STEPS)",
                        "confidence": f"{track.confidence_pct:.0f}%",
                    }
                    self.track_event_log.insert(0, ev)
                    events_generated.append(ev)
                elif steps_unseen >= self.lost_threshold_steps and track.state != TrackState.LOST:
                    track.state = TrackState.LOST
                    track.confidence_pct = max(20.0, track.confidence_pct - 15.0)
                    ev = {
                        "time_s": f"{simulated_time_s:.2f}s",
                        "timestep": timestep,
                        "track_id": t_id,
                        "band": track.band_id,
                        "event": "TRACK LOST (STALE > 8 STEPS)",
                        "confidence": f"{track.confidence_pct:.0f}%",
                    }
                    self.track_event_log.insert(0, ev)
                    events_generated.append(ev)

        if len(self.track_event_log) > 200:
            self.track_event_log = self.track_event_log[:200]

        return events_generated

    def _find_best_match(self, band_id: str, freq_mhz: float) -> Optional[SignalTrack]:
        """Associate pulse detection with closest existing active track in same or adjacent band."""
        best_track = None
        min_freq_diff = float("inf")

        for track in self.tracks.values():
            if track.state == TrackState.EXPIRED:
                continue
            if track.band_id == band_id:
                diff = abs(track.estimated_frequency_mhz - freq_mhz)
                if diff < self.freq_tolerance_mhz and diff < min_freq_diff:
                    min_freq_diff = diff
                    best_track = track

        return best_track

    def get_tracks_summary(self) -> List[Dict[str, Any]]:
        """Return structured track table for UI rendering."""
        out = []
        for track in sorted(self.tracks.values(), key=lambda t: t.hit_count, reverse=True):
            pri_str = f"{track.estimated_pri_timesteps * 0.05 * 1000:.1f} ms" if track.estimated_pri_timesteps else "Estimating..."
            recur_str = f"{track.expected_next_recurrence_s:.2f}s" if track.expected_next_recurrence_s else "N/A"

            out.append({
                "Track ID": track.track_id,
                "Band": track.band_id,
                "State": track.state,
                "Frequency (MHz)": f"{track.estimated_frequency_mhz:.1f}",
                "AoA (Deg)": f"{track.estimated_aoa_deg:.1f}°",
                "Pulse Width (µs)": f"{track.estimated_pulse_width_us:.2f}",
                "Amplitude (dBm)": f"{track.estimated_amplitude_dbm:.1f}",
                "SNR (dB)": f"{track.estimated_snr_db:.1f}",
                "Hits": track.hit_count,
                "Confidence": f"{track.confidence_pct:.0f}%",
                "Estimated PRI": pri_str,
                "Next Recurrence": recur_str,
                "Last Seen": f"{track.last_seen_time_s:.2f}s",
            })
        return out

    def export_tracks_csv(self) -> str:
        """Export all current signal tracks as CSV format."""
        import csv
        import io
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "Track_ID", "Band", "State", "Estimated_Frequency_MHz",
            "Estimated_AoA_Deg", "Estimated_Pulse_Width_us", "Estimated_Amplitude_dBm",
            "Estimated_SNR_dB", "Hit_Count", "Confidence_Pct", "Estimated_PRI_ms",
            "Next_Recurrence_s", "First_Seen_Step", "Last_Seen_Step", "Last_Seen_Time_s"
        ])
        for track in self.tracks.values():
            pri_ms = (track.estimated_pri_timesteps * 0.05 * 1000.0) if track.estimated_pri_timesteps else ""
            writer.writerow([
                track.track_id,
                track.band_id,
                track.state,
                f"{track.estimated_frequency_mhz:.1f}",
                f"{track.estimated_aoa_deg:.1f}",
                f"{track.estimated_pulse_width_us:.2f}",
                f"{track.estimated_amplitude_dbm:.1f}",
                f"{track.estimated_snr_db:.1f}",
                track.hit_count,
                f"{track.confidence_pct:.1f}",
                pri_ms,
                f"{track.expected_next_recurrence_s:.2f}" if track.expected_next_recurrence_s else "",
                track.first_seen_timestep,
                track.last_seen_timestep,
                f"{track.last_seen_time_s:.2f}",
            ])
        return output.getvalue()
