"""Mission runtime aliases and state definitions."""

from core.engine import OperationalEngine
from core.state import EngineStatus, ChannelState, TrackStatus, StrategyMode, SystemHealth

MissionRuntime = OperationalEngine

__all__ = [
    "OperationalEngine",
    "MissionRuntime",
    "EngineStatus",
    "ChannelState",
    "TrackStatus",
    "StrategyMode",
    "SystemHealth",
]
