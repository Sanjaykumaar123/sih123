import inspect

from dashboard.simulation_runner import SimulationRunner
from dashboard import visualizations as viz


def test_dashboard_modules_import_successfully():
    import app  # noqa: F401 -- module-level import must not raise
    assert hasattr(viz, "spectrum_waterfall")


def test_simulation_initializes_successfully():
    runner = SimulationRunner(seed=1)
    assert runner.t == -1
    assert runner.k == runner.config["receiver_channels"]
    assert len(runner.env.bands) == runner.config["num_bands"]


def test_simulation_produces_receiver_observations():
    runner = SimulationRunner(seed=1)
    record = runner.step()
    assert len(record["selected_bands"]) == runner.k
    assert set(record["observations"].keys()) == set(record["selected_bands"])


def test_stage3_belief_outputs_accessible():
    runner = SimulationRunner(seed=1)
    runner.run(5)
    belief_state = runner.belief_state()
    assert len(belief_state) == runner.config["num_bands"]
    assert all(0.0 <= b.activity_probability <= 1.0 for b in belief_state)


def test_stage4_temporal_outputs_accessible():
    runner = SimulationRunner(seed=1)
    runner.run(5)
    temporal_state = runner.temporal_state()
    assert len(temporal_state) == runner.config["num_bands"]


def test_stage5_scores_accessible():
    runner = SimulationRunner(seed=1)
    runner.run(5)
    scores = runner.scores()
    assert len(scores) == runner.config["num_bands"]
    assert all(0.0 <= s.balanced_score <= 1.0 for s in scores)


def test_stage6_strategy_selection_accessible():
    runner = SimulationRunner(seed=1)
    record = runner.step()
    assert record["strategy"] in ("exploration", "exploitation", "prediction", "balanced")
    state, q_values = runner.current_q_state_and_values()
    assert len(state) == 3
    assert len(q_values) == 4


def test_stage7_event_state_can_be_surfaced():
    runner = SimulationRunner(seed=1)
    runner.run(5)
    assert runner.e4 is not None
    assert hasattr(runner.e4, "is_evasive")
    summary = runner.evasion_summary()
    assert "evasion_events" in summary


def test_stage8_results_can_be_loaded():
    import json
    import os
    assert os.path.exists("results/stage8_results.json")
    with open("results/stage8_results.json", encoding="utf-8") as f:
        data = json.load(f)
    assert "aggregates" in data


def test_stage9_results_and_model_can_be_loaded():
    import json
    import os
    import pickle
    assert os.path.exists("results/stage9_results.json")
    assert os.path.exists("results/stage9_predictor.pkl")
    with open("results/stage9_results.json", encoding="utf-8") as f:
        results = json.load(f)
    assert "test_metrics" in results
    with open("results/stage9_predictor.pkl", "rb") as f:
        predictor = pickle.load(f)
    assert hasattr(predictor, "predict")


def test_ground_truth_not_passed_into_scheduler_decisions():
    # select_bands()/learn() must accept no environment/ground-truth argument.
    from rf_env import IntelligentSchedulerAdapter
    sig_select = inspect.signature(IntelligentSchedulerAdapter.select_bands)
    sig_learn = inspect.signature(IntelligentSchedulerAdapter.learn)
    assert list(sig_select.parameters) == ["self", "timestep"]
    assert list(sig_learn.parameters) == ["self", "observations", "timestep"]
    # SimulationRunner reads ground truth only into last_ground_truth_debug,
    # via metrics_tracker/evasion_tracker -- never back into the adapter.
    source = inspect.getsource(SimulationRunner)
    assert "adapter.select_bands(self.t)" in source
    assert "adapter.select_bands(self.t, " not in source  # never passed extra (truth) args


def test_simulation_reset_works():
    runner = SimulationRunner(seed=1)
    runner.run(10)
    assert runner.t == 9
    runner.reset(seed=2)
    assert runner.t == -1
    assert runner.seed == 2
    assert runner.history == []
