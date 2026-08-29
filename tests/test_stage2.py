from rf_env import RFEnvironment, Receiver, DetectionModel

from test_stage1 import BASE_CONFIG, make_env


def test_higher_snr_has_higher_detection_probability():
    model = DetectionModel(threshold_db=10.0, snr_scale=3.0)
    assert model.probability_of_detection(5.0) < model.probability_of_detection(20.0)


def test_detection_probability_monotonic_with_snr():
    model = DetectionModel(threshold_db=10.0, snr_scale=3.0)
    snrs = [-5, 0, 5, 10, 15, 20, 25]
    probs = [model.probability_of_detection(s) for s in snrs]
    assert probs == sorted(probs)


def test_present_detection_rate_matches_configured_probability():
    # snr == threshold_db -> P_d == 0.5 exactly (logistic midpoint).
    model = DetectionModel(threshold_db=10.0, snr_scale=3.0, seed=1)
    n = 2000
    hits = sum(model.detect(present=True, snr=10.0).detected for _ in range(n))
    assert abs(hits / n - 0.5) < 0.05


def test_false_alarm_rate_matches_configured_probability():
    model = DetectionModel(false_alarm_probability=0.1, seed=2)
    n = 2000
    results = [model.detect(present=False, snr=0.0) for _ in range(n)]
    rate = sum(r.detected for r in results) / n
    assert abs(rate - 0.1) < 0.03
    # every detection with nothing present must be flagged as a false alarm
    assert all(r.false_alarm == r.detected for r in results)


def test_fixed_seed_is_reproducible():
    model_a = DetectionModel(threshold_db=10.0, snr_scale=3.0, seed=7)
    model_b = DetectionModel(threshold_db=10.0, snr_scale=3.0, seed=7)
    seq_a = [model_a.detect(True, 12.0).detected for _ in range(50)]
    seq_b = [model_b.detect(True, 12.0).detected for _ in range(50)]
    assert seq_a == seq_b


def test_receiver_information_boundary_intact_with_detection():
    env = make_env()
    detection_model = DetectionModel(threshold_db=10.0, snr_scale=3.0,
                                      false_alarm_probability=0.05, seed=1)
    receiver = Receiver(env, k=5, detection_model=detection_model)
    env.step()
    selected = ["F10", "F20", "F30", "F40", "F50"]
    obs = receiver.observe(selected)
    assert set(obs.keys()) == set(selected)
    for band_id in selected:
        assert not hasattr(obs[band_id], "emitter_id")
        assert not hasattr(obs[band_id], "emitter_type")
        assert 0.0 <= obs[band_id].detection_probability <= 1.0
