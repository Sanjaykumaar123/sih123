"""Real-time simulation clock and pacing controller."""

import time
from typing import Optional


class SimulationClock:
    """Manages simulated time, execution pacing, and real-time factor."""

    def __init__(self, step_duration_s: float = 0.05, speed_multiplier: float = 1.0):
        self.step_duration_s: float = float(step_duration_s)
        self.speed_multiplier: float = float(speed_multiplier)
        self._current_step: int = 0
        self._last_step_wall_time: Optional[float] = None
        self._wall_start_time: Optional[float] = None

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def simulated_time_s(self) -> float:
        return self._current_step * self.step_duration_s

    def set_speed(self, multiplier: float) -> None:
        """Set simulation speed multiplier (e.g. 0.5x, 1.0x, 2.0x, 5.0x, 10.0x)."""
        self.speed_multiplier = max(0.01, float(multiplier))

    def reset(self, initial_step: int = 0) -> None:
        """Reset the clock."""
        self._current_step = int(initial_step)
        self._last_step_wall_time = None
        self._wall_start_time = None

    def tick(self) -> None:
        """Advance simulation by one timestep."""
        now = time.time()
        if self._wall_start_time is None:
            self._wall_start_time = now
        self._last_step_wall_time = now
        self._current_step += 1

    def pace(self) -> None:
        """Apply throttle delay to match requested simulation speed."""
        if self.speed_multiplier <= 0:
            return
        target_delay = self.step_duration_s / self.speed_multiplier
        if self._last_step_wall_time is not None:
            elapsed = time.time() - self._last_step_wall_time
            sleep_time = target_delay - elapsed
            if sleep_time > 0.001:
                time.sleep(sleep_time)
