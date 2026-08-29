import inspect

import pytest

import rf_env.scoring as scoring_module
from rf_env import BandScoringEngine, BandBelief, TemporalPrediction

BANDS = [f"F{i:02d}" for i in range(1, 51)]
PRIOR_UNCERTAINTY = 1.0 / 12.0  # Beta(1,1) variance -- the cold-start value


def make_belief(band_id, activity_probability=0.5, uncertainty=PRIOR_UNCERTAINTY,
                 last_observed=None, staleness=float("inf")):
    return BandBelief(band_id=band_id, alpha=1.0, beta=1.0,
                       activity_probability=activity_probability,
                       uncertainty=uncertainty, last_observed=last_observed,
                       staleness=staleness, hit_count=0, miss_count=0)


def make_temporal(band_id, periodicity_score=0.0, estimated_period=None,
                   predicted_next_active_time=None, prediction_confidence=0.0,
                   behaviour_type="insufficient_data"):
    return TemporalPrediction(band_id=band_id, periodicity_score=periodicity_score,
                               estimated_period=estimated_period,
                               predicted_next_active_time=predicted_next_active_time,
                               prediction_confidence=prediction_confidence,
                               last_hit_timestep=None, time_since_last_hit=float("inf"),
                               behaviour_type=behaviour_type, number_of_hits=0)


def cold_start_states():
    """All 50 bands, never observed (the default/reset state)."""
    return [make_belief(b) for b in BANDS], [make_temporal(b) for b in BANDS]


def test_all_bands_receive_scores():
    engine = BandScoringEngine(50)
    belief_state, temporal_state = cold_start_states()
    engine.update(belief_state, temporal_state, 0)
    scores = engine.get_scores()
    assert {s.band_id for s in scores} == set(BANDS)


def test_exploration_scores_bounded():
    engine = BandScoringEngine(50)
    belief_state, temporal_state = cold_start_states()
    engine.update(belief_state, temporal_state, 0)
    assert all(0.0 <= s.exploration_score <= 1.0 for s in engine.get_scores())


def test_exploitation_scores_bounded():
    engine = BandScoringEngine(50)
    belief_state, temporal_state = cold_start_states()
    engine.update(belief_state, temporal_state, 0)
    assert all(0.0 <= s.exploitation_score <= 1.0 for s in engine.get_scores())


def test_prediction_scores_bounded():
    engine = BandScoringEngine(50)
    belief_state, temporal_state = cold_start_states()
    engine.update(belief_state, temporal_state, 0)
    assert all(0.0 <= s.prediction_score <= 1.0 for s in engine.get_scores())


def test_balanced_scores_bounded():
    engine = BandScoringEngine(50)
    belief_state, temporal_state = cold_start_states()
    engine.update(belief_state, temporal_state, 0)
    assert all(0.0 <= s.balanced_score <= 1.0 for s in engine.get_scores())


def test_high_uncertainty_increases_exploration():
    engine = BandScoringEngine(50)
    belief_state, temporal_state = cold_start_states()
    beliefs = {b.band_id: b for b in belief_state}
    beliefs["F01"] = make_belief("F01", uncertainty=PRIOR_UNCERTAINTY, staleness=5)
    beliefs["F02"] = make_belief("F02", uncertainty=0.01, staleness=5)  # more evidence
    engine.update(list(beliefs.values()), temporal_state, 10)
    assert engine.score_band("F01").exploration_score > engine.score_band("F02").exploration_score


def test_high_staleness_increases_exploration():
    engine = BandScoringEngine(50)
    belief_state, temporal_state = cold_start_states()
    beliefs = {b.band_id: b for b in belief_state}
    beliefs["F01"] = make_belief("F01", uncertainty=0.02, staleness=100)
    beliefs["F02"] = make_belief("F02", uncertainty=0.02, staleness=0)
    engine.update(list(beliefs.values()), temporal_state, 100)
    assert engine.score_band("F01").exploration_score > engine.score_band("F02").exploration_score


def test_high_activity_probability_increases_exploitation():
    engine = BandScoringEngine(50)
    belief_state, temporal_state = cold_start_states()
    beliefs = {b.band_id: b for b in belief_state}
    beliefs["F01"] = make_belief("F01", activity_probability=0.9, staleness=0)
    beliefs["F02"] = make_belief("F02", activity_probability=0.1, staleness=0)
    engine.update(list(beliefs.values()), temporal_state, 0)
    assert engine.score_band("F01").exploitation_score > engine.score_band("F02").exploitation_score


def test_imminent_prediction_scores_higher_than_distant():
    engine = BandScoringEngine(50)
    belief_state, temporal_state = cold_start_states()
    temporals = {t.band_id: t for t in temporal_state}
    temporals["F01"] = make_temporal("F01", periodicity_score=0.9, prediction_confidence=0.9,
                                      predicted_next_active_time=50, behaviour_type="periodic")
    temporals["F02"] = make_temporal("F02", periodicity_score=0.9, prediction_confidence=0.9,
                                      predicted_next_active_time=500, behaviour_type="periodic")
    engine.update(belief_state, list(temporals.values()), 50)
    assert engine.score_band("F01").prediction_score > engine.score_band("F02").prediction_score


def test_no_prediction_gives_zero_score():
    engine = BandScoringEngine(50)
    belief_state, temporal_state = cold_start_states()
    engine.update(belief_state, temporal_state, 0)
    assert engine.score_band("F01").prediction_score == 0.0


def test_never_observed_bands_remain_highly_discoverable():
    engine = BandScoringEngine(50)
    belief_state, temporal_state = cold_start_states()
    beliefs = {b.band_id: b for b in belief_state}
    # F05 has been heavily, recently scanned; everything else is untouched.
    beliefs["F05"] = make_belief("F05", activity_probability=0.9, uncertainty=0.005, staleness=0)
    engine.update(list(beliefs.values()), temporal_state, 100)
    top10_explore = engine.top_k("exploration", 10)
    assert "F05" not in top10_explore
    assert "F01" in top10_explore  # a never-observed band


def test_balanced_weights_sum_to_one_by_default():
    engine = BandScoringEngine(50)
    assert engine.w_explore + engine.w_exploit + engine.w_predict == pytest.approx(1.0)


def test_invalid_balanced_weights_rejected():
    with pytest.raises(ValueError):
        BandScoringEngine(50, {"balanced_weights": {
            "exploration": 0.5, "exploitation": 0.5, "prediction": 0.5}})


def test_changing_balanced_weights_changes_balanced_score():
    belief_state, temporal_state = cold_start_states()
    beliefs = {b.band_id: b for b in belief_state}
    beliefs["F01"] = make_belief("F01", activity_probability=0.9, uncertainty=0.001, staleness=0)

    engine_a = BandScoringEngine(50, {"balanced_weights": {
        "exploration": 0.0, "exploitation": 1.0, "prediction": 0.0}})
    engine_a.update(list(beliefs.values()), temporal_state, 0)

    engine_b = BandScoringEngine(50, {"balanced_weights": {
        "exploration": 1.0, "exploitation": 0.0, "prediction": 0.0}})
    engine_b.update(list(beliefs.values()), temporal_state, 0)

    assert engine_a.score_band("F01").balanced_score != engine_b.score_band("F01").balanced_score


def test_rank_returns_descending_order():
    engine = BandScoringEngine(50)
    belief_state, temporal_state = cold_start_states()
    beliefs = {b.band_id: b for b in belief_state}
    beliefs["F01"] = make_belief("F01", activity_probability=0.9, staleness=0)
    beliefs["F02"] = make_belief("F02", activity_probability=0.5, staleness=0)
    beliefs["F03"] = make_belief("F03", activity_probability=0.1, staleness=0)
    engine.update(list(beliefs.values()), temporal_state, 0)
    ranked = engine.rank("exploitation")
    scores = {s.band_id: s.exploitation_score for s in engine.get_scores()}
    ranked_scores = [scores[b] for b in ranked]
    assert ranked_scores == sorted(ranked_scores, reverse=True)


def test_top_k_returns_exactly_k_bands():
    engine = BandScoringEngine(50)
    belief_state, temporal_state = cold_start_states()
    engine.update(belief_state, temporal_state, 0)
    assert len(engine.top_k("balanced", 5)) == 5


def test_no_ground_truth_access():
    sig = inspect.signature(BandScoringEngine.update)
    assert list(sig.parameters) == ["self", "belief_state", "temporal_state", "current_timestep"]
    source = inspect.getsource(scoring_module)
    assert "from .environment import" not in source
    assert "from .receiver import" not in source
    assert "band_truth(" not in source
    assert ".observe(" not in source
    assert ".emitter_id" not in source
    assert ".emitter_type" not in source
