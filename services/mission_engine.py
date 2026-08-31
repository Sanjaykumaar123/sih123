"""Mission Engine Service: Single source of truth for operational simulation execution."""

from __future__ import annotations
import math
import os
from typing import Any, Dict, List, Optional, Set, Tuple

from simulation.engine import SimulationEngine, SimulationStatus
from core.tracker import TrackManager, TrackState
from core.events import TelemetryEvent, EventType, EventSeverity


class MissionStatus:
    IDLE = "IDLE"
    INITIALIZING = "INITIALIZING"
    READY = "READY"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"


class InvalidMissionStateError(Exception):
    """Raised when an illegal mission lifecycle state transition is attempted."""
    pass


class MissionEngine:
    """Production Mission Engine orchestrating real-time RF sensing, scheduler, and telemetry."""

    def __init__(
        self,
        scenario_path: str = r"D:\sih\dataset\scan\test_scan\config_1.h5",
        strategy_type: str = "smart_scan",
        k_channels: int = 5,
        n_bands: int = 50,
        seed: int = 42,
        speed_multiplier: float = 1.0,
        max_duration_s: float = 30.0,
    ):
        self.scenario_path = os.path.abspath(scenario_path)
        self.strategy_type = strategy_type.lower()
        self.k_channels = int(k_channels)
        self.n_bands = int(n_bands)
        self.seed = int(seed)
        self.max_duration_s = float(max_duration_s)
        self.max_steps = int(round(self.max_duration_s / 0.05))

        self.engine = SimulationEngine(
            scenario_path=self.scenario_path,
            strategy_type=self.strategy_type,
            k_channels=self.k_channels,
            n_bands=self.n_bands,
            seed=self.seed,
            speed_multiplier=speed_multiplier,
        )

    @property
    def status(self) -> str:
        s = self.engine.status
        if s == SimulationStatus.READY:
            return MissionStatus.READY
        return s

    @property
    def clock(self):
        return self.engine.clock

    @property
    def tracker(self) -> TrackManager:
        return self.engine.tracker

    @property
    def selected_bands(self) -> List[str]:
        return self.engine.selected_bands

    @property
    def channel_telemetry(self) -> List[Dict[str, Any]]:
        return self.engine.channel_telemetry

    @property
    def event_log(self) -> List[Dict[str, Any]]:
        return self.engine.event_log

    @property
    def decision_history(self) -> List[Dict[str, Any]]:
        return self.engine.decision_history

    @property
    def time_series(self) -> List[Dict[str, Any]]:
        return self.engine.time_series

    def set_duration(self, duration_s: float) -> None:
        self.max_duration_s = float(duration_s)
        self.max_steps = int(round(self.max_duration_s / 0.05))

    def set_speed(self, multiplier: float) -> None:
        self.engine.clock.set_speed(multiplier)

    def initialize_mission(
        self,
        scenario_path: Optional[str] = None,
        strategy_type: Optional[str] = None,
        k_channels: Optional[int] = None,
        seed: Optional[int] = None,
        duration_s: Optional[float] = None,
    ) -> None:
        """Initialize or reconfigure mission parameters."""
        if duration_s is not None:
            self.max_duration_s = float(duration_s)
            self.max_steps = int(round(self.max_duration_s / 0.05))

        self.engine.reset(
            scenario_path=scenario_path,
            strategy_type=strategy_type,
            k_channels=k_channels,
            seed=seed,
        )

    def start_mission(self) -> None:
        """Start or resume mission execution."""
        if self.status in (MissionStatus.RUNNING, MissionStatus.COMPLETE):
            return
        self.engine.start()

    def pause_mission(self) -> None:
        """Pause running mission execution."""
        if self.status != MissionStatus.RUNNING:
            return
        self.engine.pause()

    def resume_mission(self) -> None:
        """Resume paused mission execution."""
        if self.status != MissionStatus.PAUSED:
            return
        self.engine.start()

    def stop_mission(self) -> None:
        """Stop current mission execution."""
        self.engine.stop()

    def reset_mission(
        self,
        scenario_path: Optional[str] = None,
        strategy_type: Optional[str] = None,
        k_channels: Optional[int] = None,
        seed: Optional[int] = None,
        duration_s: Optional[float] = None,
    ) -> None:
        """Reset mission to initial ready state."""
        self.initialize_mission(
            scenario_path=scenario_path,
            strategy_type=strategy_type,
            k_channels=k_channels,
            seed=seed,
            duration_s=duration_s,
        )

    def step_mission(self, num_steps: int = 1) -> None:
        """Step the mission up to max_steps."""
        if self.status in (MissionStatus.STOPPED, MissionStatus.COMPLETE):
            return

        current_step = self.engine.clock.current_step
        remaining = self.max_steps - current_step
        if remaining <= 0:
            self.engine.status = SimulationStatus.COMPLETE
            return

        steps_to_run = min(num_steps, remaining)
        self.engine.step(num_steps=steps_to_run)
        if self.engine.clock.current_step >= self.max_steps:
            self.engine.status = SimulationStatus.COMPLETE

    def get_snapshot(self) -> Dict[str, Any]:
        """Return unified read-only operational state snapshot."""
        snap = self.engine.get_snapshot()
        snap["mission_status"] = self.status
        snap["max_duration_s"] = self.max_duration_s
        snap["max_steps"] = self.max_steps
        snap["progress_pct"] = (self.engine.clock.current_step / self.max_steps * 100.0) if self.max_steps > 0 else 0.0
        return snap

    def export_report_json(self) -> Dict[str, Any]:
        return self.engine.export_mission_report()

    def export_events_csv(self) -> str:
        return self.engine.export_events_csv()

    def export_decisions_csv(self) -> str:
        return self.engine.export_decisions_csv()

    def export_tracks_csv(self) -> str:
        return self.engine.tracker.export_tracks_csv()
