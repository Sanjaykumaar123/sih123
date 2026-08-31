"""Operational Workstation State Models and Lifecycle Definitions."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


class EngineStatus:
    IDLE = "IDLE"
    READY = "READY"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"


class ChannelState:
    IDLE = "IDLE"
    TUNING = "TUNING"
    SCANNING = "SCANNING"
    SIGNAL_DETECTED = "SIGNAL DETECTED"
    TRACKING = "TRACKING"
    FALSE_ALARM = "FALSE ALARM"
    QUIET = "QUIET"
    ERROR = "ERROR"


class TrackStatus:
    NEW = "NEW"
    TENTATIVE = "TENTATIVE"
    CONFIRMED = "CONFIRMED"
    ACTIVE = "ACTIVE"
    LOST = "LOST"
    EXPIRED = "EXPIRED"


class StrategyMode:
    EXPLORE = "EXPLORE"
    EXPLOIT = "EXPLOIT"
    PREDICT = "PREDICT"
    BALANCED = "BALANCED"
    SEQUENTIAL_SWEEP = "SEQUENTIAL_SWEEP"


@dataclass
class ChannelTelemetry:
    channel_idx: int
    band: str
    frequency_mhz: float
    frequency_range_ghz: str
    dwell_time_ms: float = 50.0
    state: str = ChannelState.IDLE
    scheduler_role: str = "PRIMARY"
    amplitude_dbm: Optional[float] = None
    snr_db: Optional[float] = None
    aoa_deg: Optional[float] = None
    pulse_width_us: Optional[float] = None
    target_track_id: Optional[str] = None
    last_update_time_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel_idx": self.channel_idx,
            "band": self.band,
            "frequency_mhz": self.frequency_mhz,
            "frequency_range_ghz": self.frequency_range_ghz,
            "dwell_time_ms": self.dwell_time_ms,
            "status": self.state,
            "state": self.state,
            "scheduler_role": self.scheduler_role,
            "amplitude_dbm": self.amplitude_dbm,
            "snr_db": self.snr_db,
            "aoa_deg": self.aoa_deg,
            "pulse_width_us": self.pulse_width_us,
            "target_track_id": self.target_track_id or "N/A",
            "last_update_time_s": self.last_update_time_s,
        }


@dataclass
class SystemHealth:
    engine: str = "ONLINE"
    data_source: str = "ONLINE"
    receiver: str = "ONLINE"
    scheduler: str = "ONLINE"
    detector: str = "ONLINE"
    tracker: str = "ONLINE"
    ui: str = "ONLINE"
    last_cycle_latency_ms: float = 0.0
    average_cycle_latency_ms: float = 0.0
    engine_loop_rate_hz: float = 20.0
    total_cycles_executed: int = 0
    total_events_generated: int = 0
    is_realtime: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine": self.engine,
            "data_source": self.data_source,
            "receiver": self.receiver,
            "scheduler": self.scheduler,
            "detector": self.detector,
            "tracker": self.tracker,
            "ui": self.ui,
            "last_cycle_latency_ms": self.last_cycle_latency_ms,
            "average_cycle_latency_ms": self.average_cycle_latency_ms,
            "engine_loop_rate_hz": self.engine_loop_rate_hz,
            "total_cycles_executed": self.total_cycles_executed,
            "total_events_generated": self.total_events_generated,
            "is_realtime": self.is_realtime,
        }


@dataclass
class MissionState:
    """Authoritative real-time operational mission state object."""
    mission_id: str
    scenario_name: str
    mission_status: str
    current_step: int
    simulation_time_s: float
    max_duration_s: float
    max_steps: int
    speed: float
    selected_strategy: str
    k_channels: int
    n_bands: int
    selected_bands: List[str]
    receiver_channels: List[Dict[str, Any]]
    total_scans: int
    true_detections: int
    false_alarms: int
    quiet_scans: int
    sensor_pd: float
    pfa: float
    coverage: float
    interception_rate: float
    latest_reward: float
    cumulative_reward: float
    active_tracks_count: int
    total_tracks_count: int
    tracks: List[Dict[str, Any]]
    band_scores_table: List[Dict[str, Any]]
    recent_events: List[Dict[str, Any]]
    recent_decisions: List[Dict[str, Any]]
    time_series: List[Dict[str, Any]]
    health: Dict[str, Any]
    progress_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "scenario_name": self.scenario_name,
            "mission_status": self.mission_status,
            "status": self.mission_status,
            "current_step": self.current_step,
            "timestep": self.current_step,
            "simulation_time_s": self.simulation_time_s,
            "simulated_time_s": self.simulation_time_s,
            "max_duration_s": self.max_duration_s,
            "max_steps": self.max_steps,
            "total_timesteps": self.max_steps,
            "speed": self.speed,
            "speed_multiplier": self.speed,
            "selected_strategy": self.selected_strategy,
            "current_strategy": self.selected_strategy,
            "strategy_type": self.selected_strategy,
            "k_channels": self.k_channels,
            "n_bands": self.n_bands,
            "selected_bands": self.selected_bands,
            "receiver_channels": self.receiver_channels,
            "channel_telemetry": self.receiver_channels,
            "total_scans": self.total_scans,
            "true_detections": self.true_detections,
            "false_alarms": self.false_alarms,
            "quiet_scans": self.quiet_scans,
            "sensor_pd": self.sensor_pd,
            "pfa": self.pfa,
            "coverage": self.coverage,
            "interception_rate": self.interception_rate,
            "latest_reward": self.latest_reward,
            "cumulative_reward": self.cumulative_reward,
            "active_tracks_count": self.active_tracks_count,
            "total_tracks_count": self.total_tracks_count,
            "tracks": self.tracks,
            "band_scores_table": self.band_scores_table,
            "recent_events": self.recent_events,
            "recent_decisions": self.recent_decisions,
            "time_series": self.time_series,
            "health": self.health,
            "progress_pct": self.progress_pct,
        }
