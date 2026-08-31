"""Central Mission Engine executing live cognitive RF sensing missions."""

from __future__ import annotations
import csv
import io
import json
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from core.engine import OperationalEngine
from core.state import EngineStatus, ChannelState, ChannelTelemetry, SystemHealth, MissionState
from core.data_source import SignalSource, TSRDSignalSource, ReplaySignalSource
from engine.state_manager import StateManager

# Setup Logger
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/mission.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("MissionEngine")


class MissionEngine:
    """Central production mission engine orchestrating live RF sensing, cognitive scheduling, and telemetry."""

    def __init__(
        self,
        scenario_path: str = r"D:\sih\dataset\scan\test_scan\config_1.h5",
        strategy_type: str = "smart_scan",
        k_channels: int = 5,
        n_bands: int = 50,
        seed: int = 42,
        speed_multiplier: float = 1.0,
        max_duration_s: float = 30.0,
        learning_enabled: bool = True,
    ):
        self.scenario_path = os.path.abspath(scenario_path)
        self.strategy_type = strategy_type.lower()
        self.k_channels = int(k_channels)
        self.n_bands = int(n_bands)
        self.seed = int(seed)
        self.speed_multiplier = float(speed_multiplier)
        self.max_duration_s = float(max_duration_s)
        self.learning_enabled = bool(learning_enabled)

        self.mission_id = f"MSN-{int(time.time())}"
        self.engine = OperationalEngine(
            scenario_path=self.scenario_path,
            strategy_type=self.strategy_type,
            k_channels=self.k_channels,
            n_bands=self.n_bands,
            seed=self.seed,
            speed_multiplier=self.speed_multiplier,
            max_duration_s=self.max_duration_s,
        )

        self.state_manager = StateManager()
        self._sync_state()
        logger.info(f"Initialized mission {self.mission_id} on {os.path.basename(self.scenario_path)}")

    @property
    def status(self) -> str:
        return self.engine.status

    @property
    def clock(self):
        return self.engine.clock

    @property
    def tracker(self):
        return self.engine.tracker

    @property
    def selected_bands(self) -> List[str]:
        return self.engine.selected_bands

    @property
    def channel_telemetry(self) -> List[Dict[str, Any]]:
        return [ch.to_dict() for ch in self.engine.channels]

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
        self.engine.max_duration_s = float(duration_s)
        self.engine.max_steps = int(round(self.max_duration_s / 0.05))
        self._sync_state()

    def set_speed(self, multiplier: float) -> None:
        self.speed_multiplier = float(multiplier)
        self.engine.clock.set_speed(multiplier)
        self._sync_state()

    def set_learning(self, enabled: bool) -> None:
        self.learning_enabled = bool(enabled)
        logger.info(f"Learning mode set to: {self.learning_enabled}")

    def initialize_mission(
        self,
        scenario_path: Optional[str] = None,
        strategy_type: Optional[str] = None,
        k_channels: Optional[int] = None,
        seed: Optional[int] = None,
        duration_s: Optional[float] = None,
    ) -> None:
        """Initialize mission parameters and reset runtime."""
        self.mission_id = f"MSN-{int(time.time())}"
        if duration_s is not None:
            self.max_duration_s = float(duration_s)
        self.engine.reset(
            scenario_path=scenario_path,
            strategy_type=strategy_type,
            k_channels=k_channels,
            seed=seed,
            duration_s=self.max_duration_s,
        )
        self._sync_state()
        logger.info(f"Initialized mission {self.mission_id}")

    def start_mission(self) -> None:
        """Start or resume mission execution."""
        if self.status not in (EngineStatus.RUNNING, EngineStatus.COMPLETE):
            self.engine.start()
            self._sync_state()
            logger.info(f"Mission {self.mission_id} started")

    def pause_mission(self) -> None:
        """Pause mission execution."""
        if self.status == EngineStatus.RUNNING:
            self.engine.pause()
            self._sync_state()
            logger.info(f"Mission {self.mission_id} paused")

    def resume_mission(self) -> None:
        """Resume mission execution."""
        if self.status == EngineStatus.PAUSED:
            self.engine.resume()
            self._sync_state()
            logger.info(f"Mission {self.mission_id} resumed")

    def stop_mission(self) -> None:
        """Stop current mission execution."""
        self.engine.stop()
        self._sync_state()
        self.save_mission_record()
        logger.info(f"Mission {self.mission_id} stopped")

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
        """Step the live mission closed loop."""
        if self.status in (EngineStatus.STOPPED, EngineStatus.COMPLETE):
            return

        self.engine.step(num_steps=num_steps)
        self._sync_state()

        if self.engine.status == EngineStatus.COMPLETE:
            self.save_mission_record()
            logger.info(f"Mission {self.mission_id} completed naturally")

    def _sync_state(self) -> None:
        """Synchronize snapshot into state manager."""
        snap = self.engine.get_snapshot()
        sc_cov = (len([c for c, cnt in self.engine.band_scan_counts.items() if cnt > 0]) / self.n_bands) if self.n_bands > 0 else 0.0
        ir = (self.engine.true_detections / self.engine.total_scans * 100.0) if self.engine.total_scans > 0 else 0.0

        m_state = MissionState(
            mission_id=self.mission_id,
            scenario_name=os.path.basename(self.engine.scenario_path),
            mission_status=self.engine.status,
            current_step=self.engine.clock.current_step,
            simulation_time_s=self.engine.clock.simulated_time_s,
            max_duration_s=self.max_duration_s,
            max_steps=self.engine.max_steps,
            speed=self.speed_multiplier,
            selected_strategy=snap.get("current_strategy", "BALANCED"),
            k_channels=self.engine.k_channels,
            n_bands=self.engine.n_bands,
            selected_bands=list(self.engine.selected_bands),
            receiver_channels=[ch.to_dict() for ch in self.engine.channels],
            total_scans=self.engine.total_scans,
            true_detections=self.engine.true_detections,
            false_alarms=self.engine.false_alarms,
            quiet_scans=self.engine.quiet_scans,
            sensor_pd=snap.get("sensor_pd", 0.0),
            pfa=snap.get("pfa", 0.0),
            coverage=sc_cov,
            interception_rate=ir,
            latest_reward=self.engine.latest_reward,
            cumulative_reward=self.engine.cumulative_rewards,
            active_tracks_count=snap.get("active_tracks_count", 0),
            total_tracks_count=len(self.engine.tracker.tracks),
            tracks=snap.get("tracks", []),
            band_scores_table=snap.get("band_scores_table", []),
            recent_events=list(self.engine.event_log[:50]),
            recent_decisions=list(self.engine.decision_history[:25]),
            time_series=list(self.engine.time_series[-60:]),
            health=self.engine.health.to_dict(),
            progress_pct=snap.get("progress_pct", 0.0),
        )
        self.state_manager.set_state(m_state)

    def get_snapshot(self) -> Dict[str, Any]:
        """Return unified read-only operational state snapshot."""
        return self.state_manager.get_snapshot()

    def save_mission_record(self) -> Optional[str]:
        """Save structured mission record JSON in results/missions/."""
        try:
            os.makedirs("results/missions", exist_ok=True)
            f_path = os.path.join("results/missions", f"mission_{self.mission_id}.json")
            rep = self.export_report_json()
            with open(f_path, "w", encoding="utf-8") as f:
                json.dump(rep, f, indent=2)
            logger.info(f"Saved mission record: {f_path}")
            return f_path
        except Exception as e:
            logger.error(f"Failed to save mission record: {e}")
            return None

    def export_report_json(self) -> Dict[str, Any]:
        rep = self.engine.export_mission_report()
        rep["mission_metadata"]["mission_id"] = self.mission_id
        rep["mission_metadata"]["learning_enabled"] = self.learning_enabled
        return rep

    def export_events_csv(self) -> str:
        return self.engine.export_events_csv()

    def export_decisions_csv(self) -> str:
        return self.engine.export_decisions_csv()

    def export_tracks_csv(self) -> str:
        return self.engine.export_tracks_csv()
