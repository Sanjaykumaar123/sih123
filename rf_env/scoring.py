"""Band scoring engine implementing exploration, exploitation, prediction, and balanced strategies."""

from dataclasses import dataclass
import math
from typing import Dict, List, Optional
import numpy as np

from .belief import BandBelief
from .temporal import TemporalPrediction


@dataclass
class BandScore:
    band_id: str
    exploration_score: float
    exploitation_score: float
    prediction_score: float
    balanced_score: float


class BandScoringEngine:
    def __init__(self, num_bands: int, config: Optional[dict] = None):
        self.num_bands = int(num_bands)
        self.bands: List[str] = [f"F{i:02d}" for i in range(1, self.num_bands + 1)]
        cfg = config or {}
        self.staleness_scale = float(cfg.get("staleness_scale", 10.0))

        weights = cfg.get("balanced_weights", {
            "exploration": 0.30,
            "exploitation": 0.40,
            "prediction": 0.30,
        })
        self.w_explore = float(weights.get("exploration", 0.30))
        self.w_exploit = float(weights.get("exploitation", 0.40))
        self.w_predict = float(weights.get("prediction", 0.30))

        if not math.isclose(self.w_explore + self.w_exploit + self.w_predict, 1.0, abs_tol=1e-4):
            raise ValueError(f"Balanced weights must sum to 1.0, got {self.w_explore + self.w_exploit + self.w_predict}")

        self._current_timestep: int = 0
        self._scores: Dict[str, BandScore] = {}
        self.reset()

    def reset(self) -> None:
        self._current_timestep = 0
        self._scores = {}
        # Initialize default cold start scores
        for b in self.bands:
            self._scores[b] = BandScore(
                band_id=b,
                exploration_score=1.0,
                exploitation_score=0.5,
                prediction_score=0.0,
                balanced_score=float(self.w_explore * 1.0 + self.w_exploit * 0.5 + self.w_predict * 0.0),
            )

    def update(self, belief_state: List[BandBelief], temporal_state: List[TemporalPrediction],
               current_timestep: int) -> None:
        self._current_timestep = int(current_timestep)
        beliefs = {b.band_id: b for b in belief_state}
        temporals = {t.band_id: t for t in temporal_state}

        # Max uncertainty for batch normalization (cold start prior Beta(1,1) is 1/12 ≈ 0.08333)
        prior_var = 1.0 / 12.0
        max_unc = max([b.uncertainty for b in belief_state] + [prior_var])

        for b in self.bands:
            bel = beliefs[b]
            temp = temporals[b]

            # 1. Exploration Score = normalized_uncertainty * normalized_staleness
            norm_unc = min(1.0, max(0.0, bel.uncertainty / max_unc)) if max_unc > 0 else 0.0
            if math.isinf(bel.staleness):
                norm_stale = 1.0
            else:
                norm_stale = float(bel.staleness / (bel.staleness + self.staleness_scale))
            explore_score = float(np.clip(norm_unc * norm_stale, 0.0, 1.0))

            # 2. Exploitation Score = P(active)
            exploit_score = float(np.clip(bel.activity_probability, 0.0, 1.0))

            # 3. Prediction Score = proximity * confidence * periodicity
            if temp.predicted_next_active_time is not None and temp.prediction_confidence > 0:
                dist = abs(temp.predicted_next_active_time - self._current_timestep)
                period_scale = max(1.0, float(temp.estimated_period or 10.0))
                proximity = float(math.exp(-dist / period_scale))
                pred_score = float(np.clip(proximity * temp.prediction_confidence * temp.periodicity_score, 0.0, 1.0))
            else:
                pred_score = 0.0

            # 4. Balanced Score
            bal_score = float(np.clip(
                self.w_explore * explore_score +
                self.w_exploit * exploit_score +
                self.w_predict * pred_score,
                0.0, 1.0
            ))

            self._scores[b] = BandScore(
                band_id=b,
                exploration_score=explore_score,
                exploitation_score=exploit_score,
                prediction_score=pred_score,
                balanced_score=bal_score,
            )

    def score_band(self, band_id: str) -> BandScore:
        return self._scores[band_id]

    def get_scores(self) -> List[BandScore]:
        return [self._scores[b] for b in self.bands]

    def rank(self, strategy: str) -> List[str]:
        strat = strategy.lower()
        if strat in ("explore", "exploration"):
            key_fn = lambda b: self._scores[b].exploration_score
        elif strat in ("exploit", "exploitation"):
            key_fn = lambda b: self._scores[b].exploitation_score
        elif strat in ("predict", "prediction"):
            key_fn = lambda b: self._scores[b].prediction_score
        elif strat in ("balanced", "balance"):
            key_fn = lambda b: self._scores[b].balanced_score
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Sort descending by score, tie-break by band index
        return sorted(self.bands, key=key_fn, reverse=True)

    def top_k(self, strategy: str, k: int) -> List[str]:
        return self.rank(strategy)[:k]
