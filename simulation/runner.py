"""Simulation runner for batch execution and benchmarking."""

from typing import Any, Dict, Optional
from simulation.engine import SimulationEngine, SimulationStatus


def run_full_simulation(
    scenario_path: str,
    strategy_type: str = "smart_scan",
    k_channels: int = 5,
    n_bands: int = 50,
    seed: int = 42,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute a complete simulation headlessly and return final snapshot."""
    engine = SimulationEngine(
        scenario_path=scenario_path,
        strategy_type=strategy_type,
        k_channels=k_channels,
        n_bands=n_bands,
        seed=seed,
    )
    total_steps = max_steps or (engine.env.total_timesteps if engine.env else 600)
    engine.step(num_steps=total_steps)
    return engine.get_snapshot()
