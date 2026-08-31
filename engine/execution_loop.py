"""Worker and execution loop controller for continuous real-time mission execution."""

import threading
import time
import logging
from typing import Optional
from engine.mission_engine import MissionEngine
from core.state import EngineStatus

logger = logging.getLogger("ExecutionLoop")


class ExecutionWorker:
    """Threaded worker for continuous non-blocking mission execution."""

    def __init__(self, mission_engine: MissionEngine):
        self.engine = mission_engine
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        """Start worker thread if not already running."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Execution worker is already active.")
            return False

        self._stop_event.clear()
        self.engine.start_mission()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Execution worker thread started.")
        return True

    def stop(self) -> None:
        """Signal worker thread to stop and wait for completion."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self.engine.pause_mission()
        logger.info("Execution worker thread stopped.")

    def _run_loop(self) -> None:
        """Internal execution loop advancing timesteps according to speed."""
        while not self._stop_event.is_set():
            if self.engine.status != EngineStatus.RUNNING:
                break

            self.engine.step_mission(1)
            spd = max(0.25, self.engine.speed_multiplier)
            sleep_time = max(0.005, 0.05 / spd)
            time.sleep(sleep_time)

            if self.engine.status == EngineStatus.COMPLETE:
                break
