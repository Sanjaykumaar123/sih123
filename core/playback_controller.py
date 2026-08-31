"""Deterministic Operational Playback Controller.

Consumes the verified 600-step operational evaluation time-series artifacts
(results/operational_evaluation_config_1.json through config_5.json)
as the authoritative runtime stream for the interactive workstation.

Strictly zero model changes, zero dataset changes, and zero artifact changes.
"""

from __future__ import annotations
import csv
import io
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from core.tracker import get_band_freq_range
from core.state import EngineStatus, ChannelState, ChannelTelemetry, SystemHealth


class PlaybackController:
    """Deterministic, time-based operational playback controller for Streamlit workstation."""

    def __init__(
        self,
        scenario_id: str = "config_1",
        speed: float = 1.0,
        strategy_type: str = "smart_scan",
    ):
        self.scenario_id = scenario_id if scenario_id.endswith(".h5") or "config" in scenario_id else "config_1"
        self.scenario_name = os.path.basename(self.scenario_id)
        if not self.scenario_name.endswith(".h5"):
            self.scenario_name = f"{self.scenario_name}.h5"
        self.config_key = self.scenario_name.replace(".h5", "")

        self.strategy_type = strategy_type.lower()
        self.speed = float(speed)

        # Operational Playback State
        self.current_step: int = 0
        self.running: bool = False
        self.paused: bool = False
        self.mission_started: bool = False
        self.mission_completed: bool = False
        self.mission_id: str = f"MSN-{int(time.time())}"

        self.start_wall_time: float = time.perf_counter()
        self.last_update_time: float = time.perf_counter()
        self.ui_latency_ms: float = 0.0

        # Artifact Data Storage
        self.artifact_data: Dict[str, Any] = {}
        self.time_series: List[Dict[str, Any]] = []
        self.total_timesteps: int = 600
        self.max_duration_s: float = 30.0

        # Precomputed Timeline Lookups for O(1) step access
        self.cum_true_detections: List[int] = []
        self.cum_false_alarms: List[int] = []
        self.cum_eval_reward: List[float] = []
        self.cum_online_reward: List[float] = []

        # Load Scenario Artifact
        self._load_artifact()

    def _load_artifact(self) -> None:
        """Load and index operational evaluation JSON artifact."""
        artifact_path = os.path.join("results", f"operational_evaluation_{self.config_key}.json")
        if not os.path.exists(artifact_path):
            # Fallback to config_1 if specific config not found
            artifact_path = os.path.join("results", "operational_evaluation_config_1.json")

        self.artifact_load_error: Optional[str] = None
        if os.path.exists(artifact_path):
            try:
                with open(artifact_path, "r", encoding="utf-8") as f:
                    self.artifact_data = json.load(f)
                self.time_series = self.artifact_data.get("time_series", [])
                self.total_timesteps = len(self.time_series) if self.time_series else 600
                self.max_duration_s = self.total_timesteps * 0.05
            except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
                # Corrupted/unreadable artifact (section 12/20) - degrade to the same
                # honest empty state as "file not found", never crash the constructor.
                # app.py's "OPERATIONAL ARTIFACT UNAVAILABLE" banner (which checks
                # `not controller.time_series`) then surfaces this to the operator.
                self.artifact_load_error = f"{type(e).__name__}: {e}"
                self.artifact_data = {}
                self.time_series = []
                self.total_timesteps = 600
                self.max_duration_s = 30.0
        else:
            self.artifact_data = {}
            self.time_series = []
            self.total_timesteps = 600
            self.max_duration_s = 30.0

        # Precompute Cumulative Arrays
        self.cum_true_detections = [0] * len(self.time_series)
        self.cum_false_alarms = [0] * len(self.time_series)
        self.cum_eval_reward = [0.0] * len(self.time_series)
        self.cum_online_reward = [0.0] * len(self.time_series)

        running_td = 0
        running_fa = 0
        running_rew = 0.0
        running_on_rew = 0.0

        is_smart = "smart" in self.strategy_type
        for i, step_data in enumerate(self.time_series):
            if is_smart:
                td_list = step_data.get("smart_scan_true_detections", [])
                fa_list = step_data.get("smart_scan_false_alarms", [])
                rew = step_data.get("smart_scan_eval_reward", 0.0)
                on_rew = step_data.get("smart_scan_online_reward", 0.0)
            else:
                td_list = step_data.get("open_loop_true_detections", [])
                fa_list = step_data.get("open_loop_false_alarms", [])
                rew = step_data.get("open_loop_eval_reward", step_data.get("smart_scan_eval_reward", 0.0))
                on_rew = step_data.get("open_loop_online_reward", step_data.get("smart_scan_online_reward", 0.0))

            running_td += len(td_list) if isinstance(td_list, list) else int(td_list)
            running_fa += len(fa_list) if isinstance(fa_list, list) else int(fa_list)
            running_rew += rew
            running_on_rew += on_rew

            self.cum_true_detections[i] = running_td
            self.cum_false_alarms[i] = running_fa
            self.cum_eval_reward[i] = running_rew
            self.cum_online_reward[i] = running_on_rew

    @property
    def status(self) -> str:
        if self.mission_completed:
            return EngineStatus.COMPLETE
        if self.running:
            return EngineStatus.RUNNING
        if self.paused:
            return EngineStatus.PAUSED
        if self.current_step > 0:
            return EngineStatus.PAUSED
        return EngineStatus.READY

    @property
    def simulated_time_s(self) -> float:
        return self.current_step * 0.05

    @property
    def k_channels(self) -> int:
        """Real receiver channel count, from the artifact's own 'channels' field."""
        return int(self.artifact_data.get("channels", 5))

    @property
    def n_bands(self) -> int:
        """Fixed by architecture (F01-F50); not stored per-artifact."""
        return 50

    @property
    def seed(self) -> int:
        """Real random seed the artifact was generated with."""
        return int(self.artifact_data.get("seed", 42))

    def set_scenario(self, scenario_id: str, strategy_type: Optional[str] = None) -> None:
        """Switch active scenario and reset operational playback."""
        self.scenario_id = scenario_id
        self.scenario_name = os.path.basename(self.scenario_id)
        if not self.scenario_name.endswith(".h5"):
            self.scenario_name = f"{self.scenario_name}.h5"
        self.config_key = self.scenario_name.replace(".h5", "")

        if strategy_type is not None:
            self.strategy_type = strategy_type.lower()

        self._load_artifact()
        self.reset()

    def set_strategy(self, strategy_type: str) -> None:
        self.strategy_type = strategy_type.lower()
        self._load_artifact()

    def set_speed(self, speed: float) -> None:
        self.speed = max(0.5, float(speed))
        self.last_update_time = time.perf_counter()

    def start(self) -> None:
        """Start or restart the operational mission playback."""
        self.mission_started = True
        self.running = True
        self.paused = False
        self.mission_completed = False
        if self.current_step >= self.total_timesteps - 1:
            self.current_step = 0
        self.start_wall_time = time.perf_counter()
        self.last_update_time = time.perf_counter()

    def pause(self) -> None:
        """Pause mission playback preserving exact timestep and metrics."""
        self.running = False
        self.paused = True
        self.last_update_time = time.perf_counter()

    def resume(self) -> None:
        """Resume mission playback from exact paused step."""
        if not self.mission_completed and self.current_step < self.total_timesteps - 1:
            self.running = True
            self.paused = False
            self.last_update_time = time.perf_counter()

    def step(self, num_steps: int = 1) -> None:
        """Advance exactly num_steps operational cycles."""
        target = self.current_step + int(num_steps)
        if target >= self.total_timesteps - 1:
            self.current_step = self.total_timesteps - 1
            self.mission_completed = True
            self.running = False
        else:
            self.current_step = max(0, target)
        self.last_update_time = time.perf_counter()

    def stop(self) -> None:
        """Stop playback safely while keeping current view."""
        self.running = False
        self.paused = False
        self.last_update_time = time.perf_counter()

    def reset(self) -> None:
        """Reset mission state to step 0 and READY."""
        self.current_step = 0
        self.running = False
        self.paused = False
        self.mission_started = False
        self.mission_completed = False
        self.mission_id = f"MSN-{int(time.time())}"
        self.last_update_time = time.perf_counter()

    def advance_time_tick(self) -> bool:
        """Advance current_step based on elapsed wall-clock time and speed multiplier."""
        if not self.running or self.mission_completed:
            return False

        now = time.perf_counter()
        elapsed_s = now - self.last_update_time
        sim_elapsed_s = elapsed_s * self.speed
        steps_to_advance = int(math.floor(sim_elapsed_s / 0.05))

        if steps_to_advance >= 1:
            self.last_update_time = now - (sim_elapsed_s % 0.05) / self.speed
            self.step(steps_to_advance)
            return True
        return False

    def get_snapshot(self) -> Dict[str, Any]:
        """Return unified read-only operational telemetry snapshot for current_step."""
        if not self.time_series:
            # No verified artifact could be loaded (missing/corrupted JSON) - return an
            # honest, structurally-complete empty snapshot rather than indexing into an
            # empty list. app.py surfaces this as an actionable "artifact unavailable"
            # error; this guard is defense-in-depth against the same condition being
            # reached some other way (section 20).
            return {
                "scenario_name": self.scenario_name, "strategy_type": self.strategy_type.upper(),
                "mission_status": self.status, "status": self.status,
                "current_step": 0, "timestep": 0, "simulated_time_s": 0.0, "simulation_time_s": 0.0,
                "max_duration_s": self.max_duration_s, "max_steps": self.total_timesteps,
                "total_timesteps": self.total_timesteps, "speed": self.speed, "speed_multiplier": self.speed,
                "k_channels": self.k_channels, "n_bands": self.n_bands, "seed": self.seed,
                "selected_bands": [], "channel_telemetry": [], "receiver_channels": [],
                "total_scans": 0, "true_detections": 0, "false_alarms": 0,
                "step_true_detections": [], "step_false_alarms": [],
                "active_tracks_count": 0, "total_tracks_count": 0, "total_emitters_in_scenario": 0,
                "tracks": [], "sensor_pd": 0.0, "pfa": 0.0, "latest_reward": 0.0, "cumulative_reward": 0.0,
                "current_strategy": "N/A", "selected_strategy": "N/A",
                "meta_q_values": None, "q_value_note": "No operational artifact loaded.",
                "band_scores_table": [], "band_scores_available_for_all_bands": False,
                "band_scores_note": "No operational artifact loaded.",
                "time_series": [], "recent_events": [],
                "health": {"engine": "OFFLINE", "data_source": "NONE", "receiver": "OFFLINE", "scheduler": "OFFLINE",
                           "mode": "DETERMINISTIC REPLAY", "last_cycle_latency_ms": self.ui_latency_ms,
                           "total_cycles_executed": 0, "total_events_generated": 0},
            }
        t = min(max(0, self.current_step), self.total_timesteps - 1)
        t = min(t, len(self.time_series) - 1)
        sim_time = t * 0.05
        step_data = self.time_series[t] if t < len(self.time_series) else {}

        is_smart = "smart" in self.strategy_type
        sel_key = "smart_scan_selected" if is_smart else "open_loop_selected"
        td_key = "smart_scan_true_detections" if is_smart else "open_loop_true_detections"
        fa_key = "smart_scan_false_alarms" if is_smart else "open_loop_false_alarms"
        strat_key = "smart_scan_strategy" if is_smart else "open_loop_strategy"
        rew_key = "smart_scan_eval_reward" if is_smart else "open_loop_eval_reward"
        scores_key = "smart_scan_band_scores" if is_smart else "open_loop_band_scores"

        selected_bands = step_data.get(sel_key, [f"F{i+1:02d}" for i in range(5)])
        step_td = step_data.get(td_key, [])
        step_fa = step_data.get(fa_key, [])
        latest_rew = step_data.get(rew_key, 0.0)
        band_scores = step_data.get(scores_key, {})

        # Meta-strategy name. Only the smart-scan run logs a real per-step Q-learning
        # strategy choice (exploration/exploitation/prediction/balanced); the open-loop
        # baseline has no strategy concept at all - it is a fixed sequential sweep, so it
        # is labelled SEQUENTIAL_SWEEP rather than defaulted to a fake "BALANCED".
        if is_smart:
            raw_strat = str(step_data.get(strat_key, "")).strip().lower()
            strat_name = {
                "exploration": "EXPLORE",
                "exploitation": "EXPLOIT",
                "prediction": "PREDICT",
                "balanced": "BALANCED",
            }.get(raw_strat, raw_strat.upper() if raw_strat else "UNKNOWN")
        else:
            strat_name = "SEQUENTIAL_SWEEP"

        # Cumulative Metrics
        cum_td = self.cum_true_detections[t] if t < len(self.cum_true_detections) else 0
        cum_fa = self.cum_false_alarms[t] if t < len(self.cum_false_alarms) else 0
        cum_rew = self.cum_eval_reward[t] if t < len(self.cum_eval_reward) else 0.0
        total_scans = (t + 1) * 5
        sensor_pd = (cum_td / max(1, cum_td + (total_scans - cum_td - cum_fa))) if total_scans > 0 else 0.0
        pfa = (cum_fa / max(1, total_scans)) if total_scans > 0 else 0.0

        # Receiver Channel Objects (CH01 - CH05). Only real, artifact-backed fields are
        # populated: band, computed frequency range, dwell time, and hit/fa/quiet state.
        # This replay artifact does NOT record per-pulse SNR/amplitude/AoA/pulse-width,
        # so those fields are honestly None ("N/A" in the UI) rather than fabricated.
        channels: List[Dict[str, Any]] = []
        for i, b_name in enumerate(selected_bands):
            f_low, f_high, f_center = get_band_freq_range(b_name)
            is_hit = b_name in step_td
            is_fa = b_name in step_fa

            if is_hit:
                st_state = ChannelState.SIGNAL_DETECTED
                role_txt = f"RADAR INTERCEPTION (CH0{i+1})"
            elif is_fa:
                st_state = ChannelState.FALSE_ALARM
                role_txt = f"NOISE CROSSING (CH0{i+1})"
            else:
                st_state = ChannelState.SCANNING
                role_txt = f"SWEEP SEARCH (CH0{i+1})"

            channels.append({
                "channel_idx": i + 1,
                "band": b_name,
                "frequency_mhz": f_center,
                "frequency_range_ghz": f"{f_low/1000:.2f}–{f_high/1000:.2f} GHz",
                "dwell_time_ms": 50.0,
                "status": st_state,
                "state": st_state,
                "scheduler_role": role_txt,
                "amplitude_dbm": None,
                "snr_db": None,
                "aoa_deg": None,
                "pulse_width_us": None,
                "last_update_time_s": sim_time,
            })

        # Band scores: this replay artifact stores only the FINAL blended score for the
        # 5 bands actually selected each step (see data_adapter/README.md / experiments/
        # operational_evaluation.py). It does not retain per-band Activity/Uncertainty/
        # Temporal components, nor scores for the other 45 unselected bands. So the table
        # below only covers the 5 real, selected bands - never a fabricated 50-band ranking.
        reason_txt = {
            "EXPLORE": "EXPLORATION — searching less-known regions of the spectrum.",
            "EXPLOIT": "HIGH RECENT ACTIVITY — focusing on bands already showing useful activity.",
            "PREDICT": "TEMPORAL RECURRENCE — prioritizing bands likely to become active based on temporal evidence.",
            "BALANCED": "BALANCED POLICY — balances exploration, exploitation and temporal evidence.",
            "SEQUENTIAL_SWEEP": "FIXED SEQUENTIAL SWEEP — open-loop baseline, no learned strategy.",
        }.get(strat_name, "Strategy not resolved for this step.")

        band_scores_table: List[Dict[str, Any]] = []
        for b_id in selected_bands:
            f_low, f_high, _ = get_band_freq_range(b_id)
            score_val = band_scores.get(b_id)
            band_scores_table.append({
                "Rank": 0,
                "Band": b_id,
                "Frequency Range": f"{f_low/1000:.2f}–{f_high/1000:.2f} GHz",
                "P(Active)": None,
                "Uncertainty": None,
                "Temporal Score": None,
                "Final Score": round(score_val, 3) if score_val is not None else None,
                "Selected": "✓ SELECTED",
                "Reason": reason_txt,
            })
        band_scores_table.sort(key=lambda x: (x["Final Score"] is None, -(x["Final Score"] or 0)))
        for r_idx, row in enumerate(band_scores_table):
            row["Rank"] = r_idx + 1

        # Emitter interception records (real, ground-truth-derived, post-hoc only - see
        # get_emitter_interception_records()). Used here only for the live KPI counters;
        # this is NOT live per-pulse track clustering (that only exists in the separate
        # live SimulationEngine/TrackManager runtime path), so it is deliberately not
        # labelled "tracks" with a fabricated confidence value.
        emitter_records = self.get_emitter_interception_records()
        intercept_step_key = "first_intercept_step_ss" if is_smart else "first_intercept_step_ol"
        intercepted_so_far = [
            e for e in emitter_records
            if e.get(intercept_step_key) is not None and e[intercept_step_key] <= t
        ]
        active_tracks_list = intercepted_so_far

        # Sliced Time Series for Waterfall Rendering (up to current step)
        waterfall_slice: List[Dict[str, Any]] = []
        step_stride = 1 if t < 120 else (2 if t < 300 else 3)
        for past_idx in range(0, t + 1, step_stride):
            p_step = self.time_series[past_idx]
            waterfall_slice.append({
                "time_s": p_step.get("simulated_time_s", past_idx * 0.05),
                "timestep": past_idx,
                "selected_bands": p_step.get(sel_key, []),
                "hits": p_step.get(td_key, []),
                "false_alarms": p_step.get(fa_key, []),
                "active_truth": p_step.get("env_active_bands", []),
            })

        # Recent Operational Events
        recent_events: List[Dict[str, Any]] = []
        ev_start = max(0, t - 15)
        for ev_t in range(t, ev_start - 1, -1):
            ev_data = self.time_series[ev_t]
            ev_time_str = f"{ev_t * 0.05:.2f}s"
            ev_hits = ev_data.get(td_key, [])
            ev_fa = ev_data.get(fa_key, [])

            for h in ev_hits:
                recent_events.append({
                    "time_s": ev_time_str,
                    "timestep": ev_t,
                    "event_type": "INTERCEPTION",
                    "channel": "RADAR",
                    "band": h,
                    "severity": "SUCCESS",
                    "message": f"Pulse intercepted on {h} ({strat_name}).",
                })
            for fa in ev_fa:
                recent_events.append({
                    "time_s": ev_time_str,
                    "timestep": ev_t,
                    "event_type": "FALSE_ALARM",
                    "channel": "NOISE",
                    "band": fa,
                    "severity": "WARNING",
                    "message": f"Noise threshold crossing on {fa}.",
                })

        return {
            "scenario_name": self.scenario_name,
            "strategy_type": self.strategy_type.upper(),
            "mission_status": self.status,
            "status": self.status,
            "current_step": t,
            "timestep": t,
            "simulated_time_s": sim_time,
            "simulation_time_s": sim_time,
            "max_duration_s": self.max_duration_s,
            "max_steps": self.total_timesteps,
            "total_timesteps": self.total_timesteps,
            "speed": self.speed,
            "speed_multiplier": self.speed,
            "k_channels": self.k_channels,
            "n_bands": self.n_bands,
            "seed": self.seed,
            "selected_bands": selected_bands,
            "channel_telemetry": channels,
            "receiver_channels": channels,
            "total_scans": total_scans,
            "true_detections": cum_td,
            "false_alarms": cum_fa,
            "step_true_detections": step_td,
            "step_false_alarms": step_fa,
            "active_tracks_count": len(active_tracks_list),
            "total_tracks_count": len(active_tracks_list),
            "total_emitters_in_scenario": len(emitter_records),
            # Real count of distinct emitters intercepted so far this replay (same
            # underlying real intercept_step_key filtering as active_tracks_list above,
            # exposed under an unambiguous name matching SimulationEngine's real
            # unique_emitters_count field - section 4's cockpit EMITTERS card).
            "unique_emitters_count": len(intercepted_so_far),
            "tracks": active_tracks_list,
            "sensor_pd": sensor_pd,
            "pfa": pfa,
            "latest_reward": latest_rew,
            "cumulative_reward": cum_rew,
            "current_strategy": strat_name,
            "selected_strategy": strat_name,
            # This replay artifact logs which meta-strategy (explore/exploit/predict/
            # balanced) was chosen each step, but not the underlying Q(s,a) values that
            # produced that choice - those exist only inside the live arbitrator's
            # in-memory Q-table (see rf_env/arbitrator.py), which the replay artifact does
            # not capture. Rather than invent numbers, this is left None with an explicit
            # note; the live SimulationEngine path (see simulation/engine.py) exposes the
            # real per-state Q-values instead.
            "meta_q_values": None,
            "q_value_note": (
                "Per-band Q-values are not exposed by the current 4-action "
                "meta-strategy Q-table."
            ),
            "band_scores_table": band_scores_table,
            "band_scores_available_for_all_bands": False,
            "band_scores_note": (
                "This replay artifact stores only the final blended score for the 5 "
                "bands actually selected each step - not per-band Activity/Uncertainty/"
                "Temporal components, and not scores for the other 45 bands."
            ),
            "time_series": waterfall_slice,
            "recent_events": recent_events[:40],
            "health": {
                "engine": "ONLINE",
                "data_source": "TSRD ARTIFACT",
                "receiver": "ONLINE",
                "scheduler": "ONLINE",
                "mode": "DETERMINISTIC REPLAY",
                "last_cycle_latency_ms": self.ui_latency_ms,
                "total_cycles_executed": t,
                "total_events_generated": len(recent_events),
            },
        }

    def _safe_current_t(self) -> int:
        """Current step index, clamped against BOTH total_timesteps and the actual
        length of the loaded time_series - so every helper method below stays safe
        even when the artifact failed to load (time_series=[]), rather than each one
        needing its own bounds guard (an empty time_series - length mismatch with a
        stale total_timesteps was a real IndexError found during Step 15 testing)."""
        if not self.time_series:
            return -1  # signals "nothing to iterate"; callers use range(t + 1) etc.
        t = min(max(0, self.current_step), self.total_timesteps - 1)
        return min(t, len(self.time_series) - 1)

    def get_emitter_interception_records(self) -> List[Dict[str, Any]]:
        """Real, ground-truth-derived emitter interception records from the artifact's
        own `emitter_interceptions` list (see experiments/operational_evaluation.py).

        This is POST-HOC data only - it is never fed to the scheduler. It records each
        physical emitter's first activity and, per strategy, when it was first
        intercepted. It is NOT live per-pulse track clustering (that only happens in the
        separate live SimulationEngine/TrackManager runtime path); no confidence value
        is fabricated here.
        """
        return list(self.artifact_data.get("emitter_interceptions", []))

    def get_recent_band_activity(self, window: int = 30) -> List[Dict[str, Any]]:
        """Real per-band hit/false-alarm counts over the last `window` steps up to the
        current step, derived directly from the replay time series. No confidence or
        track-state values are invented - just observed counts.
        """
        t = self._safe_current_t()
        if t < 0:
            return []
        is_smart = "smart" in self.strategy_type
        td_key = "smart_scan_true_detections" if is_smart else "open_loop_true_detections"
        fa_key = "smart_scan_false_alarms" if is_smart else "open_loop_false_alarms"

        start = max(0, t - window + 1)
        counts: Dict[str, Dict[str, Any]] = {}
        for past_t in range(start, t + 1):
            p_data = self.time_series[past_t]
            for b in p_data.get(td_key, []):
                row = counts.setdefault(b, {"Band": b, "Hits": 0, "False Alarms": 0, "Last Seen Step": past_t})
                row["Hits"] += 1
                row["Last Seen Step"] = past_t
            for b in p_data.get(fa_key, []):
                row = counts.setdefault(b, {"Band": b, "Hits": 0, "False Alarms": 0, "Last Seen Step": past_t})
                row["False Alarms"] += 1
                row["Last Seen Step"] = past_t

        rows = sorted(counts.values(), key=lambda r: r["Hits"], reverse=True)
        return rows

    def get_reward_timeseries(self) -> List[Dict[str, Any]]:
        """Real cumulative reward curve up to the current step (precomputed in
        _load_artifact from actual per-step rewards - no synthetic smoothing)."""
        t = self._safe_current_t()
        if t < 0:
            return []
        out = []
        for i in range(t + 1):
            out.append({
                "timestep": i,
                "time_s": i * 0.05,
                "cumulative_reward": self.cum_eval_reward[i] if i < len(self.cum_eval_reward) else 0.0,
            })
        return out

    def get_strategy_distribution(self) -> Dict[str, int]:
        """Real count of each meta-strategy actually chosen so far this run (smart-scan
        only; the open-loop baseline has no strategy concept)."""
        t = self._safe_current_t()
        if t < 0:
            return {}
        display_map = {"exploration": "EXPLORE", "exploitation": "EXPLOIT", "prediction": "PREDICT", "balanced": "BALANCED"}
        counts: Dict[str, int] = {}
        for i in range(t + 1):
            raw = str(self.time_series[i].get("smart_scan_strategy", "")).strip().lower()
            name = display_map.get(raw)
            if name:
                counts[name] = counts.get(name, 0) + 1
        return counts

    def get_decision_history(self, window: int = 50) -> List[Dict[str, Any]]:
        """Real per-step decision trace (time, strategy, selected bands, detections,
        reward) built directly from the replay artifact - most recent first."""
        t = self._safe_current_t()
        if t < 0:
            return []
        is_smart = "smart" in self.strategy_type
        sel_key = "smart_scan_selected" if is_smart else "open_loop_selected"
        td_key = "smart_scan_true_detections" if is_smart else "open_loop_true_detections"
        fa_key = "smart_scan_false_alarms" if is_smart else "open_loop_false_alarms"
        strat_key = "smart_scan_strategy" if is_smart else "open_loop_strategy"
        rew_key = "smart_scan_eval_reward" if is_smart else "open_loop_eval_reward"
        display_map = {"exploration": "EXPLORE", "exploitation": "EXPLOIT", "prediction": "PREDICT", "balanced": "BALANCED"}

        start = max(0, t - window + 1)
        rows = []
        for i in range(t, start - 1, -1):
            d = self.time_series[i]
            raw_strat = str(d.get(strat_key, "")).strip().lower()
            strat = display_map.get(raw_strat, "SEQUENTIAL_SWEEP" if not is_smart else "UNKNOWN")
            rows.append({
                "Time (s)": f"{i * 0.05:.2f}",
                "Step": i,
                "Strategy": strat,
                "Selected Bands": " ".join(d.get(sel_key, [])),
                "Detections": len(d.get(td_key, [])),
                "False Alarms": len(d.get(fa_key, [])),
                "Reward": round(d.get(rew_key, 0.0), 2),
            })
        return rows

    def get_mission_history_summary(self) -> Dict[str, Any]:
        """REPLAY-mode equivalent of LiveMissionRuntime.get_mission_history_summary()
        (section 13) - same shape, so app.py's MISSION SUMMARY export carries real
        scenario/mode/duration/strategy/receiver-config metadata in both modes rather
        than falling back to a placeholder note."""
        import datetime as _dt
        snap = self.get_snapshot()
        t = self._safe_current_t()
        is_smart = "smart" in self.strategy_type
        sel_key = "smart_scan_selected" if is_smart else "open_loop_selected"
        bands_touched = len({b for i in range(t + 1) for b in self.time_series[i].get(sel_key, [])}) if t >= 0 else 0
        return {
            "metadata": {
                "scenario": snap.get("scenario_name"),
                "mode": "REPLAY VERIFIED RUN",
                "mission_id": self.mission_id,
                "mission_status": snap.get("mission_status"),
                "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
                "strategy_type": snap.get("strategy_type"),
                "receiver_channels_k": snap.get("k_channels"),
                "frequency_bands_n": snap.get("n_bands"),
            },
            "duration_s": snap.get("simulated_time_s", 0.0),
            "steps_executed": snap.get("timestep", 0),
            "total_scans": snap.get("total_scans", 0),
            "bands_touched": bands_touched,
            "n_bands": snap.get("n_bands", 50),
            "true_detections": snap.get("true_detections", 0),
            "false_alarms": snap.get("false_alarms", 0),
            "unique_emitters_intercepted": snap.get("unique_emitters_count"),
            "strategy_distribution": self.get_strategy_distribution(),
            "cumulative_reward": snap.get("cumulative_reward", 0.0),
        }

    def export_report_json(self) -> Dict[str, Any]:
        snap = self.get_snapshot()
        return {
            "mission_metadata": {
                "mission_id": self.mission_id,
                "scenario": self.scenario_name,
                "strategy": self.strategy_type.upper(),
                "timesteps_executed": snap["current_step"],
                "simulated_time_s": snap["simulated_time_s"],
                "status": snap["mission_status"],
                # Step 17 section 13: every export must unambiguously state which
                # runtime produced it - see core/live_mission.py's matching addition.
                "mode": "REPLAY VERIFIED RUN",
            },
            "performance_metrics": {
                "total_scans": snap["total_scans"],
                "true_detections": snap["true_detections"],
                "false_alarms": snap["false_alarms"],
                "active_tracks": snap["active_tracks_count"],
                "cumulative_reward": snap["cumulative_reward"],
                "sensor_pd": snap["sensor_pd"],
                "pfa": snap["pfa"],
            },
            "recent_events": snap["recent_events"],
        }

    def export_events_csv(self) -> str:
        snap = self.get_snapshot()
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Time", "Timestep", "Event_Type", "Channel", "Band", "Severity", "Message"])
        for ev in snap.get("recent_events", []):
            writer.writerow([
                ev.get("time_s", ""),
                ev.get("timestep", ""),
                ev.get("event_type", ""),
                ev.get("channel", ""),
                ev.get("band", ""),
                ev.get("severity", ""),
                ev.get("message", ""),
            ])
        return output.getvalue()

    def export_tracks_csv(self) -> str:
        snap = self.get_snapshot()
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Track_ID", "Band", "Frequency", "Status", "Last_Seen", "Confidence"])
        for tr in snap.get("tracks", []):
            writer.writerow([
                tr.get("Track ID", ""),
                tr.get("Band", ""),
                tr.get("Frequency", ""),
                tr.get("Status", ""),
                tr.get("Last Seen", ""),
                tr.get("Confidence", ""),
            ])
        return output.getvalue()
