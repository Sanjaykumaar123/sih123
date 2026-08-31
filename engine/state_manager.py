"""Thread-safe mission state manager."""

import threading
from typing import Any, Dict, Optional
from core.state import MissionState, EngineStatus


class StateManager:
    """Thread-safe manager for operational mission state."""

    def __init__(self, initial_state: Optional[MissionState] = None):
        self._lock = threading.Lock()
        self._state = initial_state

    def set_state(self, state: MissionState) -> None:
        with self._lock:
            self._state = state

    def get_state(self) -> Optional[MissionState]:
        with self._lock:
            return self._state

    def get_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            if self._state:
                return self._state.to_dict()
            return {"status": EngineStatus.IDLE}
