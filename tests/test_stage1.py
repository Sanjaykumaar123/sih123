import pytest

from rf_env import RFEnvironment, Receiver, ReceiverCapacityError

BASE_CONFIG = {
    "num_bands": 50,
    "receiver_channels": 5,
    "random_seed": 42,
    "max_timesteps": 500,
    "emitters": [
        {"id": "E1", "type": "static", "band": "F10",
         "signal_strength": -60.0, "snr": 15.0, "active_prob": 1.0},
        {"id": "E2", "type": "periodic", "band": "F20",
         "period": 10, "duty_cycle": 0.1,
         "signal_strength": -55.0, "snr": 18.0},
        {"id": "E3", "type": "frequency_agile",
         "hop_interval": 2, "pattern_length": 20,
         "signal_strength": -65.0, "snr": 12.0},
        {"id": "E4", "type": "adaptive_evasive",
         "hop_interval": 2, "pattern_length": 20,
         "signal_strength": -63.0, "snr": 10.0},
    ],
}


def make_env(seed=42):
    cfg = dict(BASE_CONFIG)
    cfg["random_seed"] = seed
    return RFEnvironment(cfg)


def test_environment_initializes():
    env = make_env()
    assert env.timestep == -1
    assert len(env.emitters) == 4


def test_fifty_bands():
    env = make_env()
    assert len(env.bands) == 50
    assert env.bands[0] == "F01"
    assert env.bands[-1] == "F50"


def test_receiver_capacity_is_five():
    env = make_env()
    receiver = Receiver(env, k=5)
    assert receiver.k == 5


def test_receiver_rejects_over_capacity():
    env = make_env()
    receiver = Receiver(env, k=5)
    env.step()
    with pytest.raises(ReceiverCapacityError):
        receiver.observe(["F01", "F02", "F03", "F04", "F05", "F06"])


def test_receiver_returns_only_selected_bands():
    env = make_env()
    receiver = Receiver(env, k=5)
    env.step()
    selected = ["F10", "F20", "F30", "F40", "F50"]
    obs = receiver.observe(selected)
    assert set(obs.keys()) == set(selected)


def test_static_emitter_stays_on_band():
    env = make_env()
    for _ in range(15):
        env.step()
        static = next(e for e in env.emitters if e.emitter_id == "E1")
        assert static.current_band == "F10"


def test_periodic_emitter_follows_period():
    env = make_env()
    for t in range(30):
        env.step()
        periodic = next(e for e in env.emitters if e.emitter_id == "E2")
        expected_active = (t % 10) < 1  # period=10, duty_cycle=0.1
        assert periodic.active == expected_active


def test_frequency_agile_hops_deterministically():
    env_a = make_env(seed=7)
    env_b = make_env(seed=7)
    bands_a, bands_b = [], []
    for _ in range(12):
        env_a.step()
        env_b.step()
        agile_a = next(e for e in env_a.emitters if e.emitter_id == "E3")
        agile_b = next(e for e in env_b.emitters if e.emitter_id == "E3")
        bands_a.append(agile_a.current_band)
        bands_b.append(agile_b.current_band)
    assert bands_a == bands_b          # deterministic given the seed
    assert len(set(bands_a)) > 1       # actually hops, not stuck on one band


def test_same_seed_is_reproducible():
    env_a = make_env(seed=99)
    env_b = make_env(seed=99)
    for _ in range(20):
        env_a.step()
        env_b.step()
    assert env_a.full_ground_truth_snapshot() == env_b.full_ground_truth_snapshot()


def test_ground_truth_and_observation_are_separate():
    env = make_env()
    receiver = Receiver(env, k=5)
    env.step()
    obs = receiver.observe(["F10"])
    # Observation must not expose emitter identity/type at all.
    assert not hasattr(obs["F10"], "emitter_id")
    assert not hasattr(obs["F10"], "emitter_type")
    # Ground-truth log, kept separately, does carry that detail.
    assert any(r["emitter_id"] == "E1" for r in env.logger.records)
