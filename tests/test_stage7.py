import inspect

import rf_env.emitters as emitters_module
from rf_env import (RFEnvironment, Receiver, StaticEmitter, PeriodicEmitter,
                     FrequencyAgileEmitter, AdaptiveEvasiveEmitter)

ALL_BANDS = [f"F{i:02d}" for i in range(1, 51)]


def make_emitter(**overrides):
    kwargs = dict(emitter_id="E4", all_bands=ALL_BANDS, normal_band="F10",
                  signal_strength=-63.0, snr=10.0, hit_threshold=3,
                  observation_window=10, evasive_duration=8, enabled=True,
                  seed=789)
    kwargs.update(overrides)
    return AdaptiveEvasiveEmitter(**kwargs)


def test_behaves_normally_before_threshold():
    e = make_emitter(hit_threshold=10)  # threshold high enough to stay unmet
    for t in range(5):
        e.update(t)
        e.register_detection(True, t)
        assert e.current_band == "F10"
        assert not e.is_evasive


def test_detection_history_recorded():
    e = make_emitter(hit_threshold=10)
    e.update(0)
    e.register_detection(True, 0)
    e.update(1)
    e.register_detection(False, 1)
    e.update(2)
    e.register_detection(True, 2)
    assert e.recent_detection_times == [0, 2]


def test_threshold_not_triggered_prematurely():
    e = make_emitter(hit_threshold=3, observation_window=10)
    for t in range(2):
        e.update(t)
        e.register_detection(True, t)
    assert not e.is_evasive
    assert e.evasion_count == 0


def test_h_hits_within_w_triggers_evasion():
    e = make_emitter(hit_threshold=3, observation_window=10)
    for t in range(3):
        e.update(t)
        e.register_detection(True, t)
    assert e.is_evasive
    assert e.evasion_count == 1


def test_misses_do_not_count_as_detections():
    e = make_emitter(hit_threshold=3, observation_window=10)
    for t in range(5):
        e.update(t)
        e.register_detection(False, t)
    assert not e.is_evasive
    assert e.recent_detection_times == []


def test_hits_outside_window_do_not_trigger():
    # threshold=2: without window filtering, hit@0 + hit@10 would trigger.
    e = make_emitter(hit_threshold=2, observation_window=5)
    e.update(0)
    e.register_detection(True, 0)
    for t in range(1, 10):
        e.update(t)
        e.register_detection(False, t)
    e.update(10)
    e.register_detection(True, 10)
    assert not e.is_evasive  # the t=0 hit fell outside the window by t=10
    assert e.recent_detection_times == [10]


def test_evasion_changes_active_band():
    e = make_emitter(hit_threshold=3, observation_window=10, evasive_duration=8)
    for t in range(3):
        e.update(t)
        e.register_detection(True, t)
    band_before = "F10"
    e.update(3)  # first evasive step
    assert e.current_band != band_before


def test_evasion_lasts_exact_configured_duration():
    e = make_emitter(hit_threshold=3, observation_window=10, evasive_duration=8)
    for t in range(3):
        e.update(t)
        e.register_detection(True, t)
    evasive_steps = 0
    for t in range(3, 20):
        e.update(t)
        if e.is_evasive:
            evasive_steps += 1
    assert evasive_steps == 8


def test_evasion_state_exits_correctly():
    e = make_emitter(hit_threshold=3, observation_window=10, evasive_duration=8)
    for t in range(3):
        e.update(t)
        e.register_detection(True, t)
    for t in range(3, 12):
        e.update(t)
    assert not e.is_evasive
    settled_band = e.current_band
    e.update(12)
    assert e.current_band == settled_band  # stays put after evasion


def test_detection_history_resets_after_evasion():
    e = make_emitter(hit_threshold=3, observation_window=10, evasive_duration=8)
    for t in range(3):
        e.update(t)
        e.register_detection(True, t)
    assert e.recent_detection_times == []


def test_multiple_evasion_events_can_occur():
    e = make_emitter(hit_threshold=2, observation_window=10, evasive_duration=3)
    t = 0
    for _ in range(2):
        for _ in range(2):
            e.update(t)
            e.register_detection(True, t)
            t += 1
        for _ in range(4):  # let the burst finish
            e.update(t)
            t += 1
    assert e.evasion_count == 2


def test_reproducible_given_same_seed():
    e1 = make_emitter(hit_threshold=3, observation_window=10, evasive_duration=8, seed=111)
    e2 = make_emitter(hit_threshold=3, observation_window=10, evasive_duration=8, seed=111)
    bands1, bands2 = [], []
    for t in range(20):
        e1.update(t)
        e1.register_detection(t < 3, t)
        e2.update(t)
        e2.register_detection(t < 3, t)
        bands1.append(e1.current_band)
        bands2.append(e2.current_band)
    assert bands1 == bands2


def test_static_emitter_unchanged():
    import numpy as np
    e = StaticEmitter("E1", "F10", -60.0, 15.0, active_prob=1.0, rng=np.random.RandomState(0))
    for t in range(10):
        e.update(t)
        assert e.current_band == "F10"
        assert e.active


def test_periodic_emitter_unchanged():
    e = PeriodicEmitter("E2", "F20", period=10, duty_cycle=0.1,
                        signal_strength=-55.0, snr=18.0)
    for t in range(20):
        e.update(t)
        assert e.active == ((t % 10) < 1)


def test_frequency_agile_emitter_unchanged():
    import numpy as np
    e = FrequencyAgileEmitter("E3", ALL_BANDS, hop_interval=2, pattern_length=20,
                              signal_strength=-65.0, snr=12.0, rng=np.random.RandomState(7))
    bands = []
    for t in range(12):
        e.update(t)
        bands.append(e.current_band)
    assert len(set(bands)) > 1


def test_receiver_observation_contract_unchanged():
    config = {
        "num_bands": 50, "receiver_channels": 5, "random_seed": 42,
        "emitters": [{"id": "E4", "type": "adaptive_evasive", "band": "F30",
                      "signal_strength": -63.0, "snr": 10.0}],
    }
    env = RFEnvironment(config)
    receiver = Receiver(env, k=5)
    env.step()
    obs = receiver.observe(["F10", "F20", "F30", "F40", "F50"])
    assert set(obs.keys()) == {"F10", "F20", "F30", "F40", "F50"}
    assert not hasattr(obs["F30"], "emitter_id")


def test_adaptive_emitter_receives_no_q_table_info():
    sig = inspect.signature(AdaptiveEvasiveEmitter.register_detection)
    assert list(sig.parameters) == ["self", "detected", "timestep"]


def test_adaptive_emitter_receives_no_strategy_name():
    # Precise code-usage patterns, not prose -- the module docstring
    # legitimately explains this boundary using these words.
    source = inspect.getsource(emitters_module)
    assert "from .arbitrator import" not in source
    assert "from .scoring import" not in source
    assert "q_table" not in source
    assert "select_strategy" not in source


def test_no_ground_truth_leak_in_scheduler_facing_code():
    source = inspect.getsource(emitters_module)
    assert "from .environment import" not in source
    assert "from .receiver import" not in source
    assert "band_truth(" not in source
    assert "GroundTruthLogger" not in source
