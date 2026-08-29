"""Stage 1 demo: hidden RF environment + receiver observation boundary.

Run: python demo_stage1.py

Shows, side by side, the full hidden GROUND TRUTH (evaluation-only) versus
what the RECEIVER OBSERVATION actually exposes for 5 manually chosen bands
out of 50 — the receiver must never leak the other 45 bands' state.
"""

from rf_env import RFEnvironment, Receiver
from rf_env.config import load_config


def main():
    config = load_config("config.yaml")
    env = RFEnvironment(config)
    receiver = Receiver(env, k=config["receiver_channels"])

    selected_bands = ["F10", "F20", "F05", "F12", "F25"]

    for t in range(20):
        env.step()
        observations = receiver.observe(selected_bands)

        print(f"\n--- timestep {t} ---")
        print("GROUND TRUTH (hidden; evaluation only):")
        active_rows = [r for r in env.logger.records[-len(env.emitters):]
                       if r["active"]]
        if not active_rows:
            print("  (no emitters active)")
        for row in active_rows:
            print(f"  {row['emitter_id']} ({row['emitter_type']}) "
                  f"-> {row['band']} strength={row['signal_strength']} "
                  f"snr={row['snr']}")

        print("RECEIVER OBSERVATION (only the 5 selected bands):")
        for band_id, obs in observations.items():
            print(f"  {band_id}: hit={obs.hit} "
                  f"strength={obs.signal_strength} snr={obs.snr}")


if __name__ == "__main__":
    main()
