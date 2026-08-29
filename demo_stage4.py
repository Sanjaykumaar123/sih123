"""Stage 4 demo: temporal behaviour / prediction engine.

Run: python demo_stage4.py

Scans F10 (static), F20 (periodic, period=10), F05/F12/F25 (mostly empty,
occasionally hit by hopping agile emitters) for enough steps to let the
periodic band's pattern emerge, then prints a compact temporal summary.
"""

from rf_env import (RFEnvironment, Receiver, DetectionModel, TemporalEngine)
from rf_env.config import load_config


def main():
    config = load_config("config.yaml")
    env = RFEnvironment(config)
    det_cfg = config["detection"]
    # NOTE: false_alarm_probability is set to 0 here (not config.yaml's
    # real value) purely so this demo's periodicity table is easy to read
    # -- Stage 2's actual false-alarm behaviour is unchanged and still
    # exercised by demo_stage2.py / test_stage2.py. With false alarms on,
    # they interleave with a periodic emitter's true hits and visibly
    # degrade the naive interval-based periodicity score, which is an
    # honest limitation of this deterministic baseline worth knowing
    # about, not something Stage 4 tries to filter out.
    detection_model = DetectionModel(
        threshold_db=det_cfg["threshold_db"], snr_scale=det_cfg["snr_scale"],
        false_alarm_probability=0.0, seed=det_cfg["seed"],
    )
    receiver = Receiver(env, k=config["receiver_channels"],
                         detection_model=detection_model)
    temporal = TemporalEngine(config["num_bands"], config.get("temporal"))

    selected_bands = ["F10", "F20", "F05", "F12", "F25"]

    for t in range(80):
        env.step()
        observations = receiver.observe(selected_bands)
        temporal.update(observations, t)

    print(f"{'Band':<6}{'Hits':>6}{'Period':>9}{'Periodicity':>13}"
          f"{'NextActive':>12}{'Confidence':>12}{'Behaviour':>16}")
    for band_id in selected_bands:
        p = temporal.get_prediction(band_id)
        period = f"{p.estimated_period:.1f}" if p.estimated_period else "-"
        next_active = (f"t={p.predicted_next_active_time:.0f}"
                        if p.predicted_next_active_time is not None else "-")
        print(f"{p.band_id:<6}{p.number_of_hits:>6}{period:>9}"
              f"{p.periodicity_score:>13.2f}{next_active:>12}"
              f"{p.prediction_confidence:>12.2f}{p.behaviour_type:>16}")


if __name__ == "__main__":
    main()
