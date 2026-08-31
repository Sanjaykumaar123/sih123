"""Signal Source abstraction layer for TSRD simulation, replay, and future SDR hardware."""

from abc import ABC, abstractmethod
import os
from typing import Any, Dict, List, Optional, Set, Tuple

from data_adapter.scenario_builder import TSRDEnvironment
from data_adapter.pdw_processor import TimeStepActivity


class SignalSource(ABC):
    """Abstract base class for all RF signal sources."""

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize signal source connection or data file."""
        pass

    @abstractmethod
    def step(self) -> Optional[TimeStepActivity]:
        """Advance signal stream by one timestep and return pulse activity."""
        pass

    @property
    @abstractmethod
    def total_steps(self) -> int:
        """Total number of discrete timesteps available."""
        pass

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Name of source type (e.g. SIMULATION, REPLAY, HARDWARE)."""
        pass


class TSRDSignalSource(SignalSource):
    """Live operational signal source ingesting Turing Synthetic Radar Dataset HDF5 streams."""

    def __init__(self, file_path: str, step_duration_s: float = 0.05, num_bands: int = 50):
        self.file_path = os.path.abspath(file_path)
        self.step_duration_s = float(step_duration_s)
        self.num_bands = int(num_bands)
        self.env: Optional[TSRDEnvironment] = None
        self.initialize()

    def initialize(self) -> bool:
        if os.path.exists(self.file_path):
            self.env = TSRDEnvironment(
                file_path=self.file_path,
                step_duration_s=self.step_duration_s,
                num_bands=self.num_bands,
            )
            return True
        self.env = None
        return False

    def step(self) -> Optional[TimeStepActivity]:
        if self.env:
            return self.env.step()
        return None

    @property
    def total_steps(self) -> int:
        return getattr(self.env, "total_steps", 600) if self.env else 600

    @property
    def source_type(self) -> str:
        return "SIMULATION (TSRD HDF5)"


class ReplaySignalSource(SignalSource):
    """Recorded operational replay signal source with controlled playback timing."""

    def __init__(self, file_path: str, step_duration_s: float = 0.05, num_bands: int = 50):
        self.file_path = os.path.abspath(file_path)
        self.step_duration_s = float(step_duration_s)
        self.num_bands = int(num_bands)
        self.env: Optional[TSRDEnvironment] = None
        self.initialize()

    def initialize(self) -> bool:
        if os.path.exists(self.file_path):
            self.env = TSRDEnvironment(
                file_path=self.file_path,
                step_duration_s=self.step_duration_s,
                num_bands=self.num_bands,
            )
            return True
        self.env = None
        return False

    def step(self) -> Optional[TimeStepActivity]:
        if self.env:
            return self.env.step()
        return None

    @property
    def total_steps(self) -> int:
        return getattr(self.env, "total_steps", 600) if self.env else 600

    @property
    def source_type(self) -> str:
        return "REPLAY (RECORDED TSRD)"


class HardwareSignalSource(SignalSource):
    """Future SDR / Live RF Receiver hardware interface stub."""

    def __init__(self, device_uri: str = "sdr://127.0.0.1:5000"):
        self.device_uri = device_uri

    def initialize(self) -> bool:
        return False  # Hardware stream not connected in local simulation environment

    def step(self) -> Optional[TimeStepActivity]:
        return None

    @property
    def total_steps(self) -> int:
        return 0

    @property
    def source_type(self) -> str:
        return "LIVE HARDWARE (SDR)"
