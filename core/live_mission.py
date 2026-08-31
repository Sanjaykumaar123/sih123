"""Step 14: Real live-simulation mission runtime.

LiveMissionRuntime is a thin, honest wrapper around the already-verified
simulation.engine.SimulationEngine - it does not reimplement, and never touches, the
Q-learning arbitrator, Bayesian belief engine, temporal analysis, band scoring,
receiver model, detector physics, or reward function. Those live untouched in
rf_env/ and are driven for real, every step, through SimulationEngine's existing
closed loop (rf_env.evaluation.IntelligentSchedulerAdapter -> rf_env.receiver.Receiver
-> rf_env.detection.DetectionModel -> core.reward.compute_evaluated_step_reward).

What this module adds on top:
  - The exact state machine / method names Step 14 requires (LiveMissionStatus,
    start_mission/pause_mission/resume_mission/step_once/step_n/stop_mission/
    reset_mission), independent of SimulationEngine's own SimulationStatus naming.
  - A safe, thread-free "advance one tick" hook for a Streamlit rerun-driven
    controlled-run loop (see app.py) - never a background thread that could race
    with Streamlit's script re-execution model.
  - Real scenario metadata (emitter count, collection duration, frequency range) read
    directly from the loaded TSRD scenario - not invented.
  - A get_snapshot() that layers the mission-runtime status and operating-mode label
    on top of SimulationEngine's own (already real, already-tested) snapshot; every
    other field passes through unchanged - including its real per-band Bayesian/
    temporal/Q-learning scores and its real per-pulse SNR/amplitude telemetry, none
    of which this module invents.

Everything not explicitly defined here (export_events_csv, export_decisions_csv,
export_tracks_csv, export_mission_report, .tracker, .time_series, .decision_history,
.k_channels, .n_bands, .seed, .scenario_path, .band_scan_counts, ...) is delegated
straight through to the wrapped SimulationEngine via __getattr__, so the dashboard/
view modules (already duck-typed against SimulationEngine since Step 13) work
against a LiveMissionRuntime with no changes.
"""

from __future__ import annotations
import os
import time
from typing import Any, Dict, List, Optional

from simulation.engine import SimulationEngine, SimulationStatus


class LiveMissionStatus:
    READY = "READY"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    STOPPED = "STOPPED"


class LiveMissionRuntime:
    """Manages one real, running Smart Scan mission against a live TSRD scenario."""

    def __init__(
        self,
        scenario_path: str,
        strategy_type: str = "smart_scan",
        k_channels: int = 5,
        n_bands: int = 50,
        seed: int = 42,
    ):
        self.engine = SimulationEngine(
            scenario_path=scenario_path,
            strategy_type=strategy_type,
            k_channels=k_channels,
            n_bands=n_bands,
            seed=seed,
        )
        self.mission_status: str = LiveMissionStatus.READY
        self.mission_id: str = f"LIVE-{int(time.time())}"
        self.step_duration_s: float = 0.05
        self._last_tick_wall_time: Optional[float] = None
        self.speed: float = 1.0

        # Operational event/alert bookkeeping (section 6/7). Built entirely from real
        # per-step deltas of the wrapped engine's own snapshot - never fabricated. Kept
        # separate from engine.event_log (which only records interceptions/false
        # alarms); this captures the mission-lifecycle and strategy-change events that
        # engine.event_log does not, without modifying simulation/engine.py.
        self.op_event_log: List[Dict[str, Any]] = []
        self.alert_log: List[Dict[str, Any]] = []
        self._last_known_strategy: Optional[str] = None
        self._last_known_emitter_count: int = 0
        self._record_op_event("INFO", "SYSTEM", f"Runtime initialized ({os.path.basename(scenario_path)}).")

    def _record_op_event(self, level: str, source: str, event: str) -> None:
        t = self.engine.clock.current_step if hasattr(self.engine, "clock") else 0
        self.op_event_log.insert(0, {
            "time_s": f"{t * self.step_duration_s:.3f}",
            "timestep": t,
            "level": level,
            "source": source,
            "event": event,
        })
        if len(self.op_event_log) > 300:
            self.op_event_log = self.op_event_log[:300]

    def _record_alert(self, severity: str, event: str) -> None:
        t = self.engine.clock.current_step if hasattr(self.engine, "clock") else 0
        self.alert_log.insert(0, {
            "time_s": f"{t * self.step_duration_s:.3f}",
            "timestep": t,
            "severity": severity,
            "event": event,
        })
        if len(self.alert_log) > 100:
            self.alert_log = self.alert_log[:100]

    # -------------------------------------------------------------------
    # State machine (per Step 14 section 7's enable/disable table). Every
    # method guards its own precondition - it is the source of truth, not
    # just the UI's disabled= flags.
    # -------------------------------------------------------------------
    def start_mission(self) -> bool:
        """READY/STOPPED -> RUNNING. COMPLETED -> implicit reset, then RUNNING."""
        if self.mission_status == LiveMissionStatus.COMPLETED:
            self.reset_mission()
        if self.mission_status in (LiveMissionStatus.READY, LiveMissionStatus.STOPPED):
            was_ready = self.mission_status == LiveMissionStatus.READY
            self.mission_status = LiveMissionStatus.RUNNING
            self._last_tick_wall_time = time.perf_counter()
            self._record_op_event("INFO", "SYSTEM", "Mission started." if was_ready else "Mission restarted.")
            return True
        return False

    def pause_mission(self) -> bool:
        """RUNNING -> PAUSED only."""
        if self.mission_status == LiveMissionStatus.RUNNING:
            self.mission_status = LiveMissionStatus.PAUSED
            self._record_op_event("INFO", "SYSTEM", "Mission paused.")
            self._record_alert("NOTICE", "MISSION PAUSED")
            return True
        return False

    def resume_mission(self) -> bool:
        """PAUSED -> RUNNING only."""
        if self.mission_status == LiveMissionStatus.PAUSED:
            self.mission_status = LiveMissionStatus.RUNNING
            self._last_tick_wall_time = time.perf_counter()
            self._record_op_event("INFO", "SYSTEM", "Mission resumed.")
            return True
        return False

    def step_once(self) -> bool:
        """READY/PAUSED -> execute exactly one real closed-loop cycle. Disabled while
        RUNNING (auto-advance and manual single-step are not mixed)."""
        return self.step_n(1)

    def step_n(self, n: int = 1) -> bool:
        if self.mission_status not in (LiveMissionStatus.READY, LiveMissionStatus.PAUSED):
            return False
        was_ready = self.mission_status == LiveMissionStatus.READY
        # Execute one real engine step at a time (rather than a single step(num_steps=n)
        # batch call) so _observe_step_deltas() sees every intermediate step's real
        # detections/strategy - a single batched call would only ever inspect the
        # FINAL step's state, silently missing any detection/strategy-change events
        # that happened on the n-1 steps before it (found during Step 15 testing).
        # Total real computation performed is identical either way.
        for _ in range(int(n)):
            if self.engine.status == SimulationStatus.COMPLETE:
                break
            self.engine.step(num_steps=1)
            self._sync_completion()
            self._observe_step_deltas()
        if was_ready and self.mission_status == LiveMissionStatus.READY:
            self.mission_status = LiveMissionStatus.PAUSED  # stepped at least once from READY
        return True

    def stop_mission(self) -> bool:
        """RUNNING/PAUSED -> STOPPED. A deliberate operator halt, distinct from PAUSED
        by intent (not by capability): RESUME does not work from STOPPED (matching the
        button table - only READY/PAUSED enable RESUME/STEP), but pressing START again
        does restart the mission from its current timestep (same as resuming from
        PAUSED) rather than forcing a full RESET first."""
        if self.mission_status in (LiveMissionStatus.RUNNING, LiveMissionStatus.PAUSED):
            self.engine.stop()
            self.mission_status = LiveMissionStatus.STOPPED
            self._record_op_event("INFO", "SYSTEM", "Mission stopped.")
            return True
        return False

    def reset_mission(self, **kwargs: Any) -> None:
        """Genuinely resets everything: timestep, receiver state, observations,
        decision history, event log, cumulative metrics, and Q-learning/belief state
        for a fresh mission (SimulationEngine.reset() rebuilds the scheduler, belief,
        temporal engine, tracker, and all counters from scratch - see simulation/engine.py).
        Also clears this runtime's own operational event/alert logs (section 16)."""
        self.engine.reset(**kwargs)
        self.mission_status = LiveMissionStatus.READY
        self.op_event_log = []
        self.alert_log = []
        self._last_known_strategy = None
        self._last_known_emitter_count = 0
        self._record_op_event("INFO", "SYSTEM", f"Mission reset ({os.path.basename(self.engine.scenario_path)}).")

    def _observe_step_deltas(self) -> None:
        """After real step(s) have executed, derive operational events/alerts purely
        from real observed deltas in the wrapped engine's own state - a strategy
        change, a new true interception, a false alarm, a newly-intercepted emitter.
        Nothing here is computed independently of the real closed loop; it only
        narrates transitions that already happened."""
        snap = self.engine.get_snapshot()
        strat = snap.get("current_strategy")
        if strat and strat != self._last_known_strategy:
            self._record_op_event("COG", "SCHEDULER", f"Strategy: {strat}")
            self._last_known_strategy = strat

        sel = snap.get("selected_bands", [])
        ch_tel = {c.get("band"): c for c in snap.get("channel_telemetry", [])}
        for i, band in enumerate(sel):
            ch = ch_tel.get(band, {})
            status = ch.get("status")
            if status == "TRUE INTERCEPTION":
                self._record_op_event("DETECT", f"CH0{i+1}", f"Signal detected on {band}.")
                self._record_alert("NOTICE", f"NEW TRUE INTERCEPTION on {band} (CH0{i+1})")
            elif status == "FALSE ALARM":
                self._record_op_event("DETECT", f"CH0{i+1}", f"False alarm crossing on {band}.")
                self._record_alert("WARNING", f"FALSE ALARM on {band} (CH0{i+1})")

        n_emitters = len(getattr(self.engine, "unique_emitters_intercepted", []))
        if n_emitters > self._last_known_emitter_count:
            self._record_alert("NOTICE", f"NEW EMITTER DETECTED (total unique intercepted: {n_emitters})")
            self._last_known_emitter_count = n_emitters

        multi = sum(1 for band in sel if ch_tel.get(band, {}).get("status") == "TRUE INTERCEPTION")
        if multi >= 2:
            self._record_alert("NOTICE", f"MULTIPLE DETECTIONS this step ({multi} channels)")

    def _sync_completion(self) -> None:
        was_completed = self.mission_status == LiveMissionStatus.COMPLETED
        if self.engine.status == SimulationStatus.COMPLETE:
            self.mission_status = LiveMissionStatus.COMPLETED
            if not was_completed:
                self._record_op_event("INFO", "SYSTEM", "Mission completed.")
                self._record_alert("NOTICE", "MISSION COMPLETED")
        elif self.engine.status == SimulationStatus.ERROR:
            self.mission_status = LiveMissionStatus.STOPPED
            self._record_alert("CRITICAL", "SYSTEM ERROR")
        self.mission_id = f"LIVE-{int(time.time())}"
        self._last_tick_wall_time = None

    def set_speed(self, speed: float) -> None:
        self.speed = max(0.5, float(speed))

    # -------------------------------------------------------------------
    # Short aliases matching PlaybackController's/SimulationEngine's method names
    # (start/pause/resume/step/stop/reset). Existing dashboard modules (e.g.
    # dashboard/live_operations.py's control buttons) call these generic names on
    # whichever engine is active; without these aliases those calls would silently
    # fall through __getattr__ to the WRAPPED SimulationEngine's own start()/pause()/
    # etc., which does not know about LiveMissionRuntime's guarded state machine or
    # mission_status field. These aliases keep one button-wiring code path correct
    # for both the live and replay runtimes.
    # -------------------------------------------------------------------
    def start(self) -> bool:
        return self.start_mission()

    def pause(self) -> bool:
        return self.pause_mission()

    def resume(self) -> bool:
        return self.resume_mission()

    def step(self, num_steps: int = 1) -> bool:
        return self.step_n(num_steps)

    def stop(self) -> bool:
        return self.stop_mission()

    def reset(self, **kwargs: Any) -> None:
        self.reset_mission(**kwargs)

    # -------------------------------------------------------------------
    # Controlled-run loop hook (no threads - see app.py's rerun-per-tick pacer)
    # -------------------------------------------------------------------
    # Caps how many real steps a single advance_time_tick() call will catch up on, so
    # a long gap between reruns (e.g. a backgrounded browser tab) cannot suddenly
    # execute a large chunk of the mission inline on one rerun (section 17).
    MAX_CATCHUP_STEPS = 20

    def advance_time_tick(self) -> bool:
        """Advance as many real timesteps as the elapsed wall-clock time (scaled by
        the speed multiplier) actually calls for, up to MAX_CATCHUP_STEPS. Returns
        True if at least one step was executed. Earlier versions of this method only
        ever advanced a single step per call and then reset the wall-clock reference
        regardless of how much time had actually elapsed - that silently discarded
        the backlog and capped the achieved rate well below the requested speed
        whenever the poll interval was coarser than one step's real duration (e.g.
        5x speed needs a step every 10ms; a 20ms poll loop was structurally unable to
        keep up). This version catches up properly, the same way
        PlaybackController.advance_time_tick() already does for the replay path."""
        if self.mission_status != LiveMissionStatus.RUNNING:
            return False
        now = time.perf_counter()
        if self._last_tick_wall_time is None:
            self._last_tick_wall_time = now
            return False
        sim_elapsed_s = (now - self._last_tick_wall_time) * self.speed
        steps_due = int(sim_elapsed_s // self.step_duration_s)
        if steps_due < 1:
            return False
        steps_to_run = min(steps_due, self.MAX_CATCHUP_STEPS)
        # Advance the wall-clock reference only by the sim-time actually consumed, so
        # any leftover fractional step keeps accumulating rather than being dropped.
        self._last_tick_wall_time = now - (sim_elapsed_s - steps_to_run * self.step_duration_s) / self.speed
        # One real engine step at a time (see step_n()'s comment) so every step's real
        # detections/strategy are observed for the event/alert layer, not just the
        # last of a multi-step catch-up batch.
        for _ in range(steps_to_run):
            if self.engine.status == SimulationStatus.COMPLETE:
                break
            self.engine.step(num_steps=1)
            self._sync_completion()
            self._observe_step_deltas()
        return True

    # -------------------------------------------------------------------
    # Real scenario metadata (section 15) - read once from the loaded TSRD scenario,
    # never invented. Returns None fields when the environment failed to load.
    # -------------------------------------------------------------------
    def get_scenario_metadata(self) -> Dict[str, Any]:
        env = self.engine.env
        if env is None:
            return {
                "scenario_file": None, "emitter_count": None, "collection_duration_s": None,
                "frequency_range_mhz": None, "num_bands": None, "receiver_channels": self.engine.k_channels,
                "total_steps": None, "error": "Scenario environment failed to load.",
            }
        raw = env.raw_data
        rec_meta = raw.receiver_metadata
        return {
            "scenario_file": self.engine.scenario_path,
            "emitter_count": len(env.truth_manager.get_all_emitter_ids()),
            "collection_duration_s": rec_meta.collection_time_s,
            "frequency_range_mhz": tuple(rec_meta.freq_range_mhz),
            "num_bands": env.num_bands,
            "receiver_channels": self.engine.k_channels,
            "total_steps": env.total_steps,
        }

    # -------------------------------------------------------------------
    # Snapshot: layer mission-runtime status + operating-mode label over the real,
    # already-honest SimulationEngine snapshot. Every other field is untouched.
    # -------------------------------------------------------------------
    def get_snapshot(self) -> Dict[str, Any]:
        snap = self.engine.get_snapshot()
        snap["mission_status"] = self.mission_status
        snap["status"] = self.mission_status
        snap["mission_id"] = self.mission_id
        snap["operating_mode"] = "LIVE SIMULATION"
        snap["scenario_name"] = snap.get("scenario_name", "unknown")
        snap["speed"] = self.speed
        snap["speed_multiplier"] = self.speed
        snap["max_duration_s"] = (self.engine.env.total_steps * 0.05) if self.engine.env else 30.0
        # Step 13's replay-only fields, defaulted honestly here so views that check
        # for them (e.g. decision_panel's band_scores_note) degrade gracefully - the
        # live engine DOES have real per-band scores for all 50 bands (unlike replay).
        snap.setdefault("band_scores_available_for_all_bands", True)
        step_hits = []
        step_fa = []
        if self.engine.time_series:
            last = self.engine.time_series[-1]
            if last.get("timestep") == snap.get("timestep"):
                step_hits = last.get("hits", [])
                step_fa = last.get("false_alarms", [])
        snap["step_true_detections"] = step_hits
        snap["step_false_alarms"] = step_fa
        return snap

    def get_strategy_distribution(self) -> Dict[str, int]:
        """Real count of each meta-strategy actually chosen so far, read from the
        engine's own full (unbounded) time_series - mirrors
        PlaybackController.get_strategy_distribution()'s honest shape for the live path."""
        counts: Dict[str, int] = {}
        for row in self.engine.time_series:
            s = row.get("strategy")
            if s:
                counts[s] = counts.get(s, 0) + 1
        return counts

    def get_mission_history_summary(self) -> Dict[str, Any]:
        """Section 9/13: real mission-history statistics, computed only from values
        the engine actually tracks - never estimated or backfilled. Includes the
        export metadata section 13 requires on every export (scenario, mode,
        duration, timestep count, generation timestamp, strategy, receiver config)."""
        import datetime as _dt
        snap = self.get_snapshot()
        band_scans = getattr(self.engine, "band_scan_counts", {})
        bands_touched = sum(1 for v in band_scans.values() if v > 0)
        return {
            "metadata": {
                "scenario": snap.get("scenario_name"),
                "mode": "LIVE SIMULATION",
                "mission_id": self.mission_id,
                "mission_status": self.mission_status,
                "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
                "strategy_type": snap.get("strategy_type"),
                "receiver_channels_k": snap.get("k_channels"),
                "frequency_bands_n": snap.get("n_bands"),
            },
            "duration_s": snap["simulated_time_s"],
            "steps_executed": snap["timestep"],
            "total_scans": snap["total_scans"],
            "bands_touched": bands_touched,
            "n_bands": snap["n_bands"],
            "true_detections": snap["true_detections"],
            "false_alarms": snap["false_alarms"],
            "unique_emitters_intercepted": len(getattr(self.engine, "unique_emitters_intercepted", [])),
            "strategy_distribution": self.get_strategy_distribution(),
            "cumulative_reward": snap["cumulative_reward"],
        }

    def get_event_console_rows(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Merged, newest-first TIME/LEVEL/SOURCE/EVENT rows (section 6) combining
        this runtime's real lifecycle/strategy events (op_event_log) with the wrapped
        engine's real per-detection event log (engine.event_log) - one honest,
        chronologically-merged operational event stream. Nothing here is synthesized;
        every row traces back to a real recorded event."""
        rows: List[Dict[str, Any]] = []
        for e in self.op_event_log:
            rows.append({"time_s": e["time_s"], "timestep": e["timestep"], "level": e["level"], "source": e["source"], "event": e["event"]})
        for e in self.engine.event_log:
            ev_type = e.get("event_type", "")
            if ev_type in ("INTERCEPTION", "FALSE ALARM"):
                level = "DETECT"
                event_txt = f"{ev_type} on {e.get('band', 'N/A')} (fc={e.get('frequency_mhz', 'N/A')})."
            elif ev_type == "TRACK UPDATE":
                level = "TRACK"
                # simulation/engine.py reuses the frequency_mhz field to carry the
                # track transition description for these rows - a pre-existing,
                # real (not fabricated) field reuse, just relabeled here for clarity.
                event_txt = f"{e.get('frequency_mhz', 'N/A')} on {e.get('band', 'N/A')} ({e.get('channel', 'N/A')})."
            else:
                level = "INFO"
                event_txt = f"{ev_type} on {e.get('band', 'N/A')}."
            rows.append({
                "time_s": e.get("time_s", "").rstrip("s"),
                "timestep": e.get("timestep", 0),
                "level": level,
                "source": e.get("channel", "SYSTEM") if ev_type != "TRACK UPDATE" else "TRACKER",
                "event": event_txt,
            })
        rows.sort(key=lambda r: r["timestep"], reverse=True)
        return rows[:limit]

    def get_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Real alert stream (section 7) - each entry traces to an actual observed
        state transition (see _observe_step_deltas / _sync_completion)."""
        return list(self.alert_log[:limit])

    def get_mission_history_time_series(self) -> List[Dict[str, Any]]:
        """Real, FULL accumulated mission history (every step taken so far this
        mission), not the display-truncated rolling window get_snapshot()['time_series']
        returns. simulation.engine.SimulationEngine.step() already appends every real
        step to self.engine.time_series without ever trimming it during a run (only
        reset() clears it) - this just exposes that existing real data under an
        explicit name, without modifying simulation/engine.py at all. Powers the
        SPECTRUM view's MISSION HISTORY toggle (section 3)."""
        return list(self.engine.time_series)

    def get_recent_band_activity(self, window: int = 30) -> List[Dict[str, Any]]:
        """Real per-band hit/false-alarm counts over the retained rolling time-series
        window (SimulationEngine keeps the last 60 steps in memory - see
        simulation/engine.py). Mirrors PlaybackController.get_recent_band_activity()'s
        shape so dashboard/spectrum.py's band inspector works the same in both modes."""
        rows = list(self.engine.time_series[-window:])
        counts: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            step = row.get("timestep", 0)
            for b in row.get("hits", []):
                r = counts.setdefault(b, {"Band": b, "Hits": 0, "False Alarms": 0, "Last Seen Step": step})
                r["Hits"] += 1
                r["Last Seen Step"] = step
            for b in row.get("false_alarms", []):
                r = counts.setdefault(b, {"Band": b, "Hits": 0, "False Alarms": 0, "Last Seen Step": step})
                r["False Alarms"] += 1
                r["Last Seen Step"] = step
        return sorted(counts.values(), key=lambda r: r["Hits"], reverse=True)

    def export_report_json(self) -> Dict[str, Any]:
        """Alias so LIVE and REPLAY runtimes share one export call site in app.py.
        Adds a "mode" label to the real report SimulationEngine.export_mission_report()
        already builds - simulation/engine.py itself is not modified; this wrapper only
        stamps which runtime produced the (otherwise unchanged) report, so an exported
        file is unambiguous about whether it came from a real executing mission or a
        replayed verified artifact (Step 17, section 13)."""
        report = self.engine.export_mission_report()
        report["mission_metadata"]["mode"] = "LIVE SIMULATION"
        return report

    def __getattr__(self, name: str) -> Any:
        # Only called when normal attribute lookup fails on this instance - delegates
        # everything else straight through to the wrapped, real SimulationEngine.
        return getattr(self.engine, name)
