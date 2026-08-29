"""Stage 5 demo: exploration / exploitation / prediction / balanced scores.

Run: python demo_stage5.py

Manually round-robins through all 50 bands (K=5 per step, no scheduler --
that's a later stage) for enough steps that every band gets some history,
so the four strategies show real, differentiated behaviour instead of 45
untouched bands drowning out the 5 that were ever looked at.
"""

from rf_env import (RFEnvironment, Receiver, DetectionModel, BeliefEngine,
                     TemporalEngine, BandScoringEngine)
from rf_env.config import load_config


def main():
    config = load_config("config.yaml")
    env = RFEnvironment(config)
    det_cfg = config["detection"]
    detection_model = DetectionModel(
        threshold_db=det_cfg["threshold_db"], snr_scale=det_cfg["snr_scale"],
        false_alarm_probability=det_cfg["false_alarm_probability"],
        seed=det_cfg["seed"],
    )
    receiver = Receiver(env, k=config["receiver_channels"],
                         detection_model=detection_model)
    belief = BeliefEngine(config["num_bands"], config.get("belief"))
    temporal = TemporalEngine(config["num_bands"], config.get("temporal"))
    scoring = BandScoringEngine(config["num_bands"], config.get("scoring"))

    all_bands = env.bands
    k = config["receiver_channels"]
    num_blocks = len(all_bands) // k
    t = -1
    for t in range(300):
        block = (t % num_blocks) * k
        selected_bands = all_bands[block:block + k]
        env.step()
        observations = receiver.observe(selected_bands)
        belief.update(observations)
        temporal.update(observations, t)

    scoring.update(belief.get_state(), temporal.get_state(), t)
    scores = scoring.get_scores()
    scores.sort(key=lambda s: s.balanced_score, reverse=True)

    print(f"{'Band':<6}{'Explore':>9}{'Exploit':>9}{'Predict':>9}{'Balanced':>10}")
    for s in scores[:10]:
        print(f"{s.band_id:<6}{s.exploration_score:>9.2f}"
              f"{s.exploitation_score:>9.2f}{s.prediction_score:>9.2f}"
              f"{s.balanced_score:>10.2f}")

    for strategy in ("exploration", "exploitation", "prediction", "balanced"):
        print(f"\nTop 5 {strategy}: {scoring.top_k(strategy, 5)}")


if __name__ == "__main__":
    main()
