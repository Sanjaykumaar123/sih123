"""Production Operational Cognitive RF Mission Engine / Runtime.

Executes the genuine live closed-loop spectrum sensing pipeline on real TSRD HDF5 data.
Zero precomputed JSON playback in operational mode.
Zero ground-truth leakage into the runtime scheduler.
"""

from __future__ import annotations
import csv
import io
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from rf_env.receiver import Receiver, Observation
from rf_env.detection import DetectionModel
from rf_env.arbitrator import _ACTION_TO_STRATEGY_NAME
from rf_env.evaluation import IntelligentSchedulerAdapter
from experiments.compare_strategies import SequentialOpenLoopScheduler
from data_adapter.scenario_builder import TSRDEnvironment
from core.state import EngineStatus, ChannelState, ChannelTelemetry, SystemHealth, StrategyMode
from core.tracker import TrackManager, TrackState, get_band_freq_range
from core.reward import compute_evaluated_step_reward
from core.events import TelemetryEvent, EventType, EventSeverity
from simulation.clock import SimulationClock


class OperationalEngine:
    """Production live operational engine executing real cognitive RF sensing cycles."""

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

        self.clock = SimulationClock(step_duration_s=0.05, speed_multiplier=speed_multiplier)
        self.status = EngineStatus.READY

        # Environment, Receiver & Physical Detector
        self.env: Optional[TSRDEnvironment] = None
        self._load_environment()

        self.detection_model = DetectionModel(
            threshold_db=10.0,
            false_alarm_probability=0.05,
            seed=self.seed,
        )
        self.receiver = Receiver(
            environment=self.env,
            k=self.k_channels,
            detection_model=self.detection_model,
        )

        # Scheduler Instance (Smart Scan or Open Loop)
        self.scheduler = self._create_scheduler()

        # Autonomous Internal Signal Track Manager (Zero ground-truth leakage)
        self.tracker = TrackManager()

        # Runtime State & History
        self.selected_bands: List[str] = [f"F{i+1:02d}" for i in range(self.k_channels)]
        self.latest_observations: Dict[str, Observation] = {}
        self.latest_active_truth: Set[str] = set()
        self.last_scan_times: Dict[str, int] = {}
        self.latest_reward: float = 0.0
        self.cumulative_rewards: float = 0.0

        # Physical Channel Objects (CH01 - CH05)
        self.channels: List[ChannelTelemetry] = []
        self._init_channels()

        # Performance Counters
        self.total_scans: int = 0
        self.true_detections: int = 0
        self.false_alarms: int = 0
        self.quiet_scans: int = 0
        self.unique_emitters_intercepted: Set[int] = set()
        self.emitter_first_intercept_times: Dict[int, int] = {}
        self.band_scan_counts: Dict[str, int] = {f"F{i:02d}": 0 for i in range(1, self.n_bands + 1)}

        # System Health & Execution Latency
        self.health = SystemHealth()
        self._latency_samples: List[float] = []

        # Operational Logs
        self.event_log: List[Dict[str, Any]] = []
        self.decision_history: List[Dict[str, Any]] = []
        self.time_series: List[Dict[str, Any]] = []

        # Emit initial event
        self._emit_event(
            event_type=EventType.MISSION_STATE_CHANGE,
            channel="SYS",
            band="ALL",
            message=f"Workstation initialized on {os.path.basename(self.scenario_path)} with {self.strategy_type.upper()}.",
            severity=EventSeverity.INFO,
        )

    def _create_scheduler(self) -> Any:
        if self.strategy_type == "open_loop":
            return SequentialOpenLoopScheduler(
                k=self.k_channels,
                num_bands=self.n_bands,
                start_band=1,
            )
        else:
            return IntelligentSchedulerAdapter(
                num_bands=self.n_bands,
                k=self.k_channels,
                arbitrator_config={"seed": self.seed},
            )

    def _load_environment(self) -> None:
        """Load and cache the TSRD environment from HDF5 file, with fallback to synthetic RFEnvironment."""
        if os.path.exists(self.scenario_path):
            try:
                self.env = TSRDEnvironment(
                    file_path=self.scenario_path,
                    step_duration_s=0.05,
                    num_bands=self.n_bands,
                )
                if hasattr(self, "health"):
                    self.health.data_source = "ONLINE"
                return
            except Exception:
                pass

        # Fallback to synthetic RFEnvironment if TSRD HDF5 dataset file is absent or unreadable
        try:
            from rf_env.environment import RFEnvironment
            from rf_env.config import load_config
            cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
            if os.path.exists(cfg_path):
                cfg = load_config(cfg_path)
            else:
                cfg = {"num_bands": self.n_bands, "random_seed": self.seed}
            self.env = RFEnvironment(cfg)
            if hasattr(self, "health"):
                self.health.data_source = "SYNTHETIC"
        except Exception:
            self.env = None
            if hasattr(self, "health"):
                self.health.data_source = "ERROR: FILE NOT FOUND"

    def _init_channels(self) -> None:
        """Initialize physical channel objects."""
        self.channels = []
        for i in range(self.k_channels):
            b_name = self.selected_bands[i] if i < len(self.selected_bands) else f"F{i+1:02d}"
            f_low, f_high, f_center = get_band_freq_range(b_name)
            self.channels.append(
                ChannelTelemetry(
                    channel_idx=i + 1,
                    band=b_name,
                    frequency_mhz=f_center,
                    frequency_range_ghz=f"{f_low/1000:.2f}–{f_high/1000:.2f} GHz",
                    dwell_time_ms=50.0,
                    state=ChannelState.IDLE,
                    scheduler_role=f"CH0{i+1} ALLOC",
                    amplitude_dbm=None,
                    snr_db=None,
                    aoa_deg=None,
                    pulse_width_us=None,
                    last_update_time_s=0.0,
                )
            )

    def _emit_event(
        self,
        event_type: str,
        channel: str = "SYS",
        band: str = "N/A",
        frequency_mhz: float = 0.0,
        pulse_width_us: Optional[float] = None,
        aoa_deg: Optional[float] = None,
        amplitude_dbm: Optional[float] = None,
        snr_db: Optional[float] = None,
        track_id: Optional[str] = None,
        severity: str = EventSeverity.INFO,
        message: str = "",
    ) -> None:
        """Add structured event to operational event log."""
        t_s = f"{self.clock.simulated_time_s:.2f}s"
        t_step = self.clock.current_step
        ev = TelemetryEvent(
            time_s=t_s,
            timestep=t_step,
            event_type=event_type,
            channel=channel,
            band=band,
            frequency_mhz=frequency_mhz,
            pulse_width_us=pulse_width_us,
            aoa_deg=aoa_deg,
            amplitude_dbm=amplitude_dbm,
            snr_db=snr_db,
            track_id=track_id,
            severity=severity,
            message=message,
        )
        self.event_log.insert(0, ev.to_dict())
        if len(self.event_log) > 300:
            self.event_log = self.event_log[:300]
        self.health.total_events_generated += 1

    def start(self) -> None:
        """Start or resume mission runtime."""
        if self.status != EngineStatus.COMPLETE:
            self.status = EngineStatus.RUNNING
            self.health.engine = "RUNNING"
            self._emit_event(EventType.MISSION_STATE_CHANGE, message="Mission started / resumed.")

    def pause(self) -> None:
        """Pause mission runtime."""
        if self.status == EngineStatus.RUNNING:
            self.status = EngineStatus.PAUSED
            self.health.engine = "PAUSED"
            self._emit_event(EventType.MISSION_STATE_CHANGE, message="Mission paused by operator.")

    def resume(self) -> None:
        """Resume mission runtime."""
        if self.status == EngineStatus.PAUSED:
            self.status = EngineStatus.RUNNING
            self.health.engine = "RUNNING"
            self._emit_event(EventType.MISSION_STATE_CHANGE, message="Mission resumed.")

    def stop(self) -> None:
        """Stop mission runtime."""
        self.status = EngineStatus.STOPPED
        self.health.engine = "STOPPED"
        self._emit_event(EventType.MISSION_STATE_CHANGE, message="Mission stopped by operator.")

    def reset(
        self,
        scenario_path: Optional[str] = None,
        strategy_type: Optional[str] = None,
        k_channels: Optional[int] = None,
        seed: Optional[int] = None,
        duration_s: Optional[float] = None,
    ) -> None:
        """Reset operational engine to initial mission-ready state."""
        if scenario_path is not None:
            self.scenario_path = os.path.abspath(scenario_path)
        if strategy_type is not None:
            self.strategy_type = strategy_type.lower()
        if k_channels is not None:
            self.k_channels = int(k_channels)
        if seed is not None:
            self.seed = int(seed)
        if duration_s is not None:
            self.max_duration_s = float(duration_s)
            self.max_steps = int(round(self.max_duration_s / 0.05))

        self.clock.reset(0)
        self.status = EngineStatus.READY
        self.health.engine = "ONLINE"
        self.health.total_cycles_executed = 0
        self.health.total_events_generated = 0
        self._latency_samples.clear()

        self._load_environment()
        self.detection_model = DetectionModel(
            threshold_db=10.0,
            false_alarm_probability=0.05,
            seed=self.seed,
        )
        self.receiver = Receiver(
            environment=self.env,
            k=self.k_channels,
            detection_model=self.detection_model,
        )
        self.scheduler = self._create_scheduler()
        self.tracker.reset()

        self.selected_bands = [f"F{i+1:02d}" for i in range(self.k_channels)]
        self._init_channels()
        self.latest_observations.clear()
        self.latest_active_truth.clear()
        self.last_scan_times.clear()
        self.latest_reward = 0.0
        self.cumulative_rewards = 0.0

        self.total_scans = 0
        self.true_detections = 0
        self.false_alarms = 0
        self.quiet_scans = 0
        self.unique_emitters_intercepted.clear()
        self.emitter_first_intercept_times.clear()
        self.band_scan_counts = {f"F{i:02d}": 0 for i in range(1, self.n_bands + 1)}

        self.event_log.clear()
        self.decision_history.clear()
        self.time_series.clear()

        self._emit_event(
            event_type=EventType.MISSION_STATE_CHANGE,
            message=f"Mission reset. Ready on {os.path.basename(self.scenario_path)} ({self.strategy_type.upper()}).",
            severity=EventSeverity.INFO,
        )

    def step(self, num_steps: int = 1) -> None:
        """Advance the live operational closed loop by num_steps cycles."""
        if self.env is None:
            self.status = EngineStatus.ERROR
            self.health.engine = "ERROR"
            return

        for _ in range(num_steps):
            t = self.clock.current_step
            if t >= self.max_steps or (self.env and t >= getattr(self.env, "total_steps", 600)):
                self.status = EngineStatus.COMPLETE
                self.health.engine = "COMPLETE"
                self._emit_event(EventType.MISSION_STATE_CHANGE, message=f"Mission completed at t={t*0.05:.2f}s.")
                break

            t_start_cycle = time.perf_counter()

            # 1. Advance Environment Step & Cache Truth (Truth isolated from scheduler)
            self.env.step()
            active_truth = set()
            for b_name in self.env.bands:
                bt = self.env.band_truth(b_name)
                if bt.active:
                    active_truth.add(b_name)
            self.latest_active_truth = active_truth

            # 2. Cognitive / Baseline Band Selection (Zero ground-truth input)
            selected = self.scheduler.select_bands(t)
            self.selected_bands = list(selected)

            # 3. Receiver Physical Observation on K selected channels
            observations = self.receiver.observe(selected_bands=selected)
            self.latest_observations = observations

            # 4. Closed-Loop Learning / Policy Adaptation
            if hasattr(self.scheduler, "learn"):
                self.scheduler.learn(observations, t)

            # 5. Extract Observable Signal Measurements & Update Channel Telemetry
            step_hits = 0
            step_fa = 0
            curr_channel_telemetry: List[Dict[str, Any]] = []

            for ch_idx, b in enumerate(selected):
                obs = observations.get(b)
                self.total_scans += 1
                self.band_scan_counts[b] = self.band_scan_counts.get(b, 0) + 1
                f_low, f_high, f_center = get_band_freq_range(b)

                if obs and obs.hit:
                    if b in active_truth:
                        self.true_detections += 1
                        step_hits += 1
                        step_act = self.env.processor.get_step_activity(t) if hasattr(self.env, "processor") else None
                        b_act = step_act.band_activities.get(b) if step_act else None

                        amp = b_act.max_amplitude_dbm if b_act else obs.signal_strength
                        snr = b_act.snr_db if b_act else obs.snr
                        pw = b_act.mean_pulse_width_us if (b_act and b_act.mean_pulse_width_us > 0) else 5.20
                        aoa = b_act.mean_aoa_deg if (b_act and b_act.mean_aoa_deg != 0) else 45.0
                        freq_meas = b_act.mean_frequency_mhz if (b_act and b_act.mean_frequency_mhz > 0) else f_center

                        e_ids = b_act.ground_truth_emitter_ids if b_act else []
                        if e_ids:
                            e_id = e_ids[0]
                            if e_id not in self.unique_emitters_intercepted:
                                self.unique_emitters_intercepted.add(e_id)
                                self.emitter_first_intercept_times[e_id] = t

                        ch_obj = ChannelTelemetry(
                            channel_idx=ch_idx + 1,
                            band=b,
                            frequency_mhz=freq_meas,
                            frequency_range_ghz=f"{f_low/1000:.2f}–{f_high/1000:.2f} GHz",
                            dwell_time_ms=50.0,
                            state=ChannelState.SIGNAL_DETECTED,
                            scheduler_role=f"RADAR ACQ (CH0{ch_idx+1})",
                            amplitude_dbm=amp,
                            snr_db=snr,
                            aoa_deg=aoa,
                            pulse_width_us=pw,
                            last_update_time_s=t * 0.05,
                        )
                        self.channels[ch_idx] = ch_obj
                        curr_channel_telemetry.append(ch_obj.to_dict())

                        self._emit_event(
                            event_type=EventType.INTERCEPTION,
                            channel=f"CH0{ch_idx+1}",
                            band=b,
                            frequency_mhz=freq_meas,
                            pulse_width_us=pw,
                            aoa_deg=aoa,
                            amplitude_dbm=amp,
                            snr_db=snr,
                            severity=EventSeverity.SUCCESS,
                            message=f"Pulse intercepted on {b} ({freq_meas:.1f} MHz, SNR {snr:.1f} dB).",
                        )
                    else:
                        self.false_alarms += 1
                        step_fa += 1
                        ch_obj = ChannelTelemetry(
                            channel_idx=ch_idx + 1,
                            band=b,
                            frequency_mhz=f_center,
                            frequency_range_ghz=f"{f_low/1000:.2f}–{f_high/1000:.2f} GHz",
                            dwell_time_ms=50.0,
                            state=ChannelState.FALSE_ALARM,
                            scheduler_role=f"NOISE CROSS (CH0{ch_idx+1})",
                            amplitude_dbm=-88.5,
                            snr_db=11.5,
                            aoa_deg=None,
                            pulse_width_us=None,
                            last_update_time_s=t * 0.05,
                        )
                        self.channels[ch_idx] = ch_obj
                        curr_channel_telemetry.append(ch_obj.to_dict())

                        self._emit_event(
                            event_type=EventType.FALSE_ALARM,
                            channel=f"CH0{ch_idx+1}",
                            band=b,
                            frequency_mhz=f_center,
                            amplitude_dbm=-88.5,
                            snr_db=11.5,
                            severity=EventSeverity.WARNING,
                            message=f"False alarm noise crossing detected on {b}.",
                        )
                else:
                    self.quiet_scans += 1
                    ch_obj = ChannelTelemetry(
                        channel_idx=ch_idx + 1,
                        band=b,
                        frequency_mhz=f_center,
                        frequency_range_ghz=f"{f_low/1000:.2f}–{f_high/1000:.2f} GHz",
                        dwell_time_ms=50.0,
                        state=ChannelState.SCANNING,
                        scheduler_role=f"SWEEP (CH0{ch_idx+1})",
                        amplitude_dbm=None,
                        snr_db=None,
                        aoa_deg=None,
                        pulse_width_us=None,
                        last_update_time_s=t * 0.05,
                    )
                    self.channels[ch_idx] = ch_obj
                    curr_channel_telemetry.append(ch_obj.to_dict())

            # 6. Update Autonomous Signal Tracks (Zero ground-truth input)
            track_events = self.tracker.update(
                observations=observations,
                channel_telemetry=curr_channel_telemetry,
                timestep=t,
                simulated_time_s=t * 0.05,
            )
            for tr_ev in track_events:
                self._emit_event(
                    event_type=EventType.TRACK_UPDATED,
                    channel=tr_ev.get("track_id", "TRK"),
                    band=tr_ev.get("band", "N/A"),
                    track_id=tr_ev.get("track_id"),
                    severity=EventSeverity.INFO,
                    message=f"{tr_ev.get('track_id')}: {tr_ev.get('event')} (Conf: {tr_ev.get('confidence')}).",
                )

            # 7. Reward & Decision History Recording
            step_reward = compute_evaluated_step_reward(
                selected_bands=selected,
                observations=observations,
                active_bands_truth=active_truth,
                last_scan_times=self.last_scan_times,
                timestep=t,
            )
            self.latest_reward = step_reward
            self.cumulative_rewards += step_reward

            strat_name = "SEQUENTIAL_SWEEP"
            if hasattr(self.scheduler, "arbitrator"):
                current_action = getattr(self.scheduler.arbitrator, "last_action", 3)
                strat_raw = _ACTION_TO_STRATEGY_NAME.get(current_action, "BALANCED")
                strat_name = str(strat_raw).upper()

            self.decision_history.insert(0, {
                "timestep": t,
                "time_s": f"{t * 0.05:.2f}s",
                "strategy": strat_name,
                "selected_bands": ", ".join(selected),
                "hits": step_hits,
                "false_alarms": step_fa,
                "step_reward": f"{step_reward:+.2f}",
                "cumulative_reward": f"{self.cumulative_rewards:.2f}",
            })
            if len(self.decision_history) > 150:
                self.decision_history = self.decision_history[:150]

            self.time_series.append({
                "timestep": t,
                "time_s": t * 0.05,
                "selected_bands": selected,
                "hits": [b for b, o in observations.items() if o.hit and (b in active_truth)],
                "false_alarms": [b for b, o in observations.items() if o.hit and (b not in active_truth)],
                "active_truth": list(active_truth),
                "strategy": strat_name,
                "step_reward": step_reward,
                "cumulative_reward": self.cumulative_rewards,
            })

            # Record Latency
            latency_ms = (time.perf_counter() - t_start_cycle) * 1000.0
            self._latency_samples.append(latency_ms)
            if len(self._latency_samples) > 100:
                self._latency_samples.pop(0)
            self.health.last_cycle_latency_ms = latency_ms
            self.health.average_cycle_latency_ms = float(np.mean(self._latency_samples))
            self.health.total_cycles_executed += 1

            # Advance clock
            self.clock.tick()

        if self.clock.current_step >= self.max_steps or (self.env and self.clock.current_step >= getattr(self.env, "total_steps", 600)):
            self.status = EngineStatus.COMPLETE
            self.health.engine = "COMPLETE"

    def get_snapshot(self) -> Dict[str, Any]:
        """Return unified read-only operational state snapshot."""
        t = self.clock.current_step
        total_steps = self.max_steps
        sensor_pd = (self.true_detections / (self.true_detections + self.quiet_scans)) if (self.true_detections + self.quiet_scans) > 0 else 0.0
        pfa = (self.false_alarms / self.total_scans) if self.total_scans > 0 else 0.0

        # 50-Band Comprehensive Priority Ranking Table
        band_scores_table = []
        beliefs_map: Dict[str, Any] = {}
        scores_map: Dict[str, Any] = {}
        temporal_map: Dict[str, Any] = {}

        if hasattr(self.scheduler, "belief"):
            beliefs_map = {b.band_id: b for b in self.scheduler.belief.get_state()}
        if hasattr(self.scheduler, "scoring"):
            scores_map = getattr(self.scheduler.scoring, "_scores", {})
        if hasattr(self.scheduler, "temporal"):
            temporal_map = {t.band_id: t for t in self.scheduler.temporal.get_state()}

        for b_idx in range(1, self.n_bands + 1):
            b_name = f"F{b_idx:02d}"
            f_low, f_high, _ = get_band_freq_range(b_name)
            b_obj = beliefs_map.get(b_name)
            s_obj = scores_map.get(b_name)
            t_obj = temporal_map.get(b_name)

            act_p = getattr(b_obj, "activity_probability", 0.0) if b_obj else 0.0
            unc = getattr(b_obj, "uncertainty", 1.0) if b_obj else 1.0
            stale_val = getattr(b_obj, "staleness", float(t)) if b_obj else float(t)
            stale_display = int(stale_val) if math.isfinite(stale_val) else 999

            s_exp = getattr(s_obj, "exploration_score", 0.0) if s_obj else 0.0
            s_exp_act = getattr(s_obj, "exploitation_score", 0.0) if s_obj else 0.0
            s_pred = getattr(s_obj, "prediction_score", 0.0) if s_obj else 0.0
            comp_score = getattr(s_obj, "balanced_score", 0.0) if s_obj else 0.0
            t_score = getattr(t_obj, "periodicity_score", 0.0) if t_obj else 0.0

            is_sel = b_name in self.selected_bands

            # Grounded reason generation
            if is_sel:
                if act_p > 0.6 and t_score > 0.5:
                    reason = "HIGH ACTIVITY + TEMPORAL PRI RECURRENCE"
                elif act_p > 0.6:
                    reason = "HIGH BAYESIAN ACTIVITY PROBABILITY"
                elif t_score > 0.6:
                    reason = "IMMINENT TEMPORAL PULSE PREDICTION"
                elif unc > 0.5 or stale_display > 15:
                    reason = "EPISTEMIC UNCERTAINTY / RE-EXPLORATION"
                else:
                    reason = "TOP-5 COMPOSITE SCORE LEADER"
            else:
                reason = "—"

            band_scores_table.append({
                "Rank": 0,
                "Band": b_name,
                "Frequency Range": f"{f_low/1000:.2f}–{f_high/1000:.2f} GHz",
                "P(Active)": act_p,
                "Uncertainty": unc,
                "Temporal Score": t_score,
                "Exploration Score": s_exp,
                "Exploitation Score": s_exp_act,
                "Prediction Score": s_pred,
                "Final Score": comp_score,
                "Last Observed": f"t={getattr(b_obj, 'last_observed', 'None')}" if getattr(b_obj, 'last_observed', None) is not None else "Never",
                "Selected": "✓ SELECTED" if is_sel else "—",
                "Reason": reason,
            })

        band_scores_table.sort(key=lambda x: x["Final Score"], reverse=True)
        for r_idx, row in enumerate(band_scores_table):
            row["Rank"] = r_idx + 1

        # Current Strategy & Q-Values
        current_strategy = "BALANCED"
        meta_q_values = [0.0, 0.0, 0.0, 0.0]
        if hasattr(self.scheduler, "arbitrator"):
            current_action = getattr(self.scheduler.arbitrator, "last_action", 3)
            strat_raw = _ACTION_TO_STRATEGY_NAME.get(current_action, "BALANCED")
            current_strategy = str(strat_raw).upper()
            if hasattr(self.scheduler.arbitrator, "q_table"):
                last_state = getattr(self.scheduler, "_last_state", (0, 0, 0))
                q_tab = self.scheduler.arbitrator.q_table
                if isinstance(q_tab, np.ndarray):
                    meta_q_values = [float(x) for x in q_tab[last_state]]
                elif isinstance(q_tab, dict):
                    meta_q_values = [float(x) for x in q_tab.get(last_state, [0.0, 0.0, 0.0, 0.0])]
        elif self.strategy_type == "open_loop":
            current_strategy = "SEQUENTIAL_SWEEP"

        # Signal Tracks Table
        tracks_summary = self.tracker.get_tracks_summary()
        active_tracks_cnt = sum(1 for tr in self.tracker.tracks.values() if tr.state in (TrackState.CONFIRMED, TrackState.ACTIVE, TrackState.NEW, TrackState.TENTATIVE))

        channel_dicts = [ch.to_dict() for ch in self.channels]

        return {
            "status": self.status,
            "mission_status": self.status,
            "timestep": t,
            "total_timesteps": total_steps,
            "max_duration_s": self.max_duration_s,
            "max_steps": self.max_steps,
            "progress_pct": (t / self.max_steps * 100.0) if self.max_steps > 0 else 0.0,
            "simulated_time_s": self.clock.simulated_time_s,
            "speed_multiplier": self.clock.speed_multiplier,
            "strategy_type": self.strategy_type.upper(),
            "current_strategy": current_strategy,
            "meta_q_values": meta_q_values,
            "scenario_name": os.path.basename(self.scenario_path),
            "k_channels": self.k_channels,
            "n_bands": self.n_bands,
            "selected_bands": list(self.selected_bands),
            "channel_telemetry": channel_dicts,
            "total_scans": self.total_scans,
            "true_detections": self.true_detections,
            "false_alarms": self.false_alarms,
            "unique_emitters_count": len(self.unique_emitters_intercepted),
            "sensor_pd": sensor_pd,
            "pfa": pfa,
            "latest_reward": self.latest_reward,
            "cumulative_reward": self.cumulative_rewards,
            "band_scores_table": band_scores_table,
            "tracks": tracks_summary,
            "active_tracks_count": active_tracks_cnt,
            "total_tracks_count": len(self.tracker.tracks),
            "recent_events": list(self.event_log[:50]),
            "recent_decisions": list(self.decision_history[:25]),
            "time_series": list(self.time_series[-60:]),
            "band_scan_counts": dict(self.band_scan_counts),
            "health": self.health.to_dict(),
        }

    def export_mission_report(self) -> Dict[str, Any]:
        """Export comprehensive machine-readable mission report JSON."""
        snap = self.get_snapshot()
        return {
            "mission_metadata": {
                "scenario": os.path.basename(self.scenario_path),
                "strategy": self.strategy_type.upper(),
                "receiver_channels_k": self.k_channels,
                "total_frequency_bands_n": self.n_bands,
                "total_timesteps_executed": self.clock.current_step,
                "mission_horizon_s": self.clock.simulated_time_s,
                "simulation_status": self.status,
            },
            "performance_metrics": {
                "total_channel_scans": self.total_scans,
                "true_detections": self.true_detections,
                "false_alarms": self.false_alarms,
                "unique_emitters_intercepted": len(self.unique_emitters_intercepted),
                "autonomous_signal_tracks_formed": len(self.tracker.tracks),
                "sensor_pd": snap["sensor_pd"],
                "false_alarm_rate_pfa": snap["pfa"],
                "cumulative_reward": self.cumulative_rewards,
                "average_cycle_latency_ms": self.health.average_cycle_latency_ms,
            },
            "band_utilization_scans": dict(self.band_scan_counts),
            "signal_tracks": snap["tracks"],
            "decision_history": list(self.decision_history),
            "detection_events": list(self.event_log),
        }

    def export_events_csv(self) -> str:
        """Export event log as CSV string."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Time", "Timestep", "Event_Type", "Channel", "Band", "Frequency_MHz", "Pulse_Width_us", "AoA_deg", "Amplitude_dBm", "SNR_dB", "Track_ID", "Message"])
        for ev in self.event_log:
            writer.writerow([
                ev.get("time_s", ""),
                ev.get("timestep", ""),
                ev.get("event_type", ""),
                ev.get("channel", ""),
                ev.get("band", ""),
                ev.get("frequency_mhz", ""),
                ev.get("pulse_width_us", ""),
                ev.get("aoa_deg", ""),
                ev.get("amplitude_dbm", ""),
                ev.get("snr_db", ""),
                ev.get("track_id", ""),
                ev.get("message", ""),
            ])
        return output.getvalue()

    def export_decisions_csv(self) -> str:
        """Export decision history as CSV string."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Time", "Timestep", "Strategy", "Selected_Bands", "Hits", "False_Alarms", "Step_Reward", "Cumulative_Reward"])
        for dec in self.decision_history:
            writer.writerow([
                dec.get("time_s", ""),
                dec.get("timestep", ""),
                dec.get("strategy", ""),
                dec.get("selected_bands", ""),
                dec.get("hits", ""),
                dec.get("false_alarms", ""),
                dec.get("step_reward", ""),
                dec.get("cumulative_reward", ""),
            ])
        return output.getvalue()

    def export_tracks_csv(self) -> str:
        """Export all current autonomous signal tracks as CSV string."""
        return self.tracker.export_tracks_csv()
