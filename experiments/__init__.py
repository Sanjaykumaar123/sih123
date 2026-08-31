"""Experiments module for Cognitive Smart Scan vs Baseline comparisons."""

__all__ = [
    "SequentialOpenLoopScheduler",
    "StrategyMetrics",
    "BaselineResult",
    "SmartScanResult",
    "ComparisonResult",
    "compare_strategies",
]


def __getattr__(name: str):
    if name in __all__:
        from . import compare_strategies as cs
        return getattr(cs, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

