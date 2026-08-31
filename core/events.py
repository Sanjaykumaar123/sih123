"""Structured event definitions, event types, and telemetry formatting."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


class EventType:
    SCAN = "SCAN"
    OBSERVATION = "OBSERVATION"
    DETECTION = "DETECTION"
    INTERCEPTION = "INTERCEPTION"
    FALSE_ALARM = "FALSE_ALARM"
    MISS = "MISS"
    QUIET = "QUIET"
    TRACK_CREATED = "TRACK_CREATED"
    TRACK_CONFIRMED = "TRACK_CONFIRMED"
    TRACK_UPDATED = "TRACK_UPDATED"
    TRACK_LOST = "TRACK_LOST"
    TRACK_EXPIRED = "TRACK_EXPIRED"
    STRATEGY_CHANGE = "STRATEGY_CHANGE"
    MISSION_STATE_CHANGE = "MISSION_STATE_CHANGE"


class EventSeverity:
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ALERT = "ALERT"


@dataclass
class TelemetryEvent:
    time_s: str
    timestep: int
    event_type: str
    channel: str = "SYS"
    band: str = "N/A"
    frequency_mhz: float = 0.0
    pulse_width_us: Optional[float] = None
    aoa_deg: Optional[float] = None
    amplitude_dbm: Optional[float] = None
    snr_db: Optional[float] = None
    track_id: Optional[str] = None
    severity: str = EventSeverity.INFO
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_s": self.time_s,
            "timestep": self.timestep,
            "event_type": self.event_type,
            "channel": self.channel,
            "band": self.band,
            "frequency_mhz": f"{self.frequency_mhz:.1f} MHz" if self.frequency_mhz > 0 else "N/A",
            "pulse_width_us": f"{self.pulse_width_us:.2f} µs" if self.pulse_width_us is not None else "N/A",
            "aoa_deg": f"{self.aoa_deg:.1f}°" if self.aoa_deg is not None else "N/A",
            "amplitude_dbm": f"{self.amplitude_dbm:.1f} dBm" if self.amplitude_dbm is not None else "N/A",
            "snr_db": f"{self.snr_db:.1f} dB" if self.snr_db is not None else "N/A",
            "track_id": self.track_id or "N/A",
            "severity": self.severity,
            "message": self.message,
        }
