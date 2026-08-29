"""Stage 2 demo: probabilistic detection layer on top of the Stage 1 env.

Run: python demo_stage2.py

For each selected band, prints SNR, the detection probability the model
used, and the resulting hit/miss -- plus, for debugging only, whether that
hit was a true detection, a miss, or a false alarm (determined here by
comparing against ground truth, which the receiver/scheduler never sees).
"""

from rf_env import RFEnvironment, Receiver, DetectionModel
from rf_env.config import load_config


def main():
    config = load_config("config.yaml")
    env = RFEnvironment(config)
    det_cfg = config["detection"]
    detection_model = DetectionModel(
        threshold_db=det_cfg["threshold_db"],
        snr_scale=det_cfg["snr_scale"],
        false_alarm_probability=det_cfg["false_alarm_probability"],
        seed=det_cfg["seed"],
    )
    receiver = Receiver(env, k=config["receiver_channels"],
                         detection_model=detection_model)

    selected_bands = ["F10", "F20", "F05", "F12", "F25"]

    for t in range(30):
        env.step()
        observations = receiver.observe(selected_bands)

        print(f"\n--- timestep {t} ---")
        for band_id, obs in observations.items():
            truth_active = env.band_truth(band_id).active  # debug only
            if obs.hit and truth_active:
                label = "HIT (true detection)"
            elif obs.hit and not truth_active:
                label = "HIT (false alarm)"
            elif not obs.hit and truth_active:
                label = "MISS (missed detection)"
            else:
                label = "no signal, correctly quiet"
            print(f"  {band_id}: snr={obs.snr:5.1f} "
                  f"P_d={obs.detection_probability:.2f} "
                  f"hit={obs.hit} -> {label}")


if __name__ == "__main__":
    main()
