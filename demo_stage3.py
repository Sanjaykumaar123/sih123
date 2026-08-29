"""Stage 3 demo: Bayesian belief engine fed only by receiver observations.

Run: python demo_stage3.py

Manually selects 5 bands each step (no scheduler yet -- that's a later
stage), feeds the receiver's observations into BeliefEngine, and prints a
compact probability/uncertainty/staleness table for the scanned bands.
"""

from rf_env import RFEnvironment, Receiver, DetectionModel, BeliefEngine
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

    selected_bands = ["F10", "F20", "F05", "F12", "F25"]

    for t in range(25):
        env.step()
        observations = receiver.observe(selected_bands)
        belief.update(observations)

        print(f"\n--- timestep {t} ---")
        print(f"{'Band':<6}{'P(active)':>10}{'Uncertainty':>13}{'Staleness':>11}")
        for band_id in selected_bands:
            b = belief.get_belief(band_id)
            print(f"{b.band_id:<6}{b.activity_probability:>10.3f}"
                  f"{b.uncertainty:>13.4f}{b.staleness:>11}")


if __name__ == "__main__":
    main()
