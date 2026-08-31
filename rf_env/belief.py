"""Bayesian cognitive belief engine (Beta-Bernoulli per band with decay)."""

from dataclasses import dataclass
from typing import Dict, List, Optional
from .receiver import Observation


@dataclass
class BandBelief:
    band_id: str
    alpha: float
    beta: float
    activity_probability: float
    uncertainty: float
    last_observed: Optional[int]
    staleness: float
    hit_count: int
    miss_count: int


class BeliefEngine:
    def __init__(self, num_bands: int, config: Optional[dict] = None):
        self.num_bands = int(num_bands)
        self.bands: List[str] = [f"F{i:02d}" for i in range(1, self.num_bands + 1)]
        cfg = config or {}
        self.prior_alpha = float(cfg.get("prior_alpha", 1.0))
        self.prior_beta = float(cfg.get("prior_beta", 1.0))
        self.decay_gamma = float(cfg.get("decay_gamma", 0.98))

        self._current_step: int = -1
        self._beliefs: Dict[str, Dict] = {}
        self.reset()

    def reset(self) -> None:
        self._current_step = -1
        self._beliefs = {}
        for b in self.bands:
            self._beliefs[b] = {
                "alpha": self.prior_alpha,
                "beta": self.prior_beta,
                "last_observed": None,
                "hit_count": 0,
                "miss_count": 0,
            }

    def update(self, observations: Dict[str, Observation]) -> None:
        self._current_step += 1
        observed_bands = set(observations.keys())

        for b in self.bands:
            entry = self._beliefs[b]
            if b in observed_bands:
                obs = observations[b]
                entry["last_observed"] = self._current_step
                if obs.hit:
                    entry["alpha"] += 1.0
                    entry["hit_count"] += 1
                else:
                    entry["beta"] += 1.0
                    entry["miss_count"] += 1
            else:
                # Decay unobserved bands towards prior
                entry["alpha"] = self.prior_alpha + self.decay_gamma * (entry["alpha"] - self.prior_alpha)
                entry["beta"] = self.prior_beta + self.decay_gamma * (entry["beta"] - self.prior_beta)

    def get_belief(self, band_id: str) -> BandBelief:
        entry = self._beliefs[band_id]
        a = entry["alpha"]
        b = entry["beta"]
        p = a / (a + b)
        var = (a * b) / (((a + b) ** 2) * (a + b + 1.0))
        last_obs = entry["last_observed"]
        if last_obs is None:
            staleness = float("inf")
        else:
            staleness = float(max(0, self._current_step - last_obs))

        return BandBelief(
            band_id=band_id,
            alpha=float(a),
            beta=float(b),
            activity_probability=float(p),
            uncertainty=float(var),
            last_observed=last_obs,
            staleness=staleness,
            hit_count=entry["hit_count"],
            miss_count=entry["miss_count"],
        )

    def get_state(self) -> List[BandBelief]:
        return [self.get_belief(b) for b in self.bands]
