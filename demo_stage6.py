"""Stage 6 demo: Q-learning strategy arbitrator, trained live.

Run: python demo_stage6.py

Closed loop each step: arbitrator observes state -> picks a strategy ->
Stage 5 ranks bands under that strategy -> receiver scans top-K -> reward
computed from those observations -> Q-table updated. "Episodes" below are
just fixed-length reporting windows over one continuous run (the
environment/belief/temporal engines are never reset mid-run) -- there is
no natural episode boundary in this simulation, so we don't pretend one.
"""

from collections import deque

from rf_env import (RFEnvironment, Receiver, DetectionModel, BeliefEngine,
                     TemporalEngine, BandScoringEngine, QLearningArbitrator, Strategy)
from rf_env.config import load_config

TOTAL_STEPS = 2000
EPISODE_LENGTH = 50
STRATEGY_NAMES = ["exploration", "exploitation", "prediction", "balanced"]


def print_q_table(arb, label):
    print(f"\n{label} (state = perf/uncertainty/detection, each 0=LOW/1=MED/2=HIGH):")
    print(f"{'state':<10}{'EXPLORE':>9}{'EXPLOIT':>9}{'PREDICT':>9}{'BALANCED':>10}")
    for p in range(3):
        for u in range(3):
            for d in range(3):
                q = arb.get_q_values((p, u, d))
                print(f"({p},{u},{d})  {q[0]:>8.3f}{q[1]:>9.3f}{q[2]:>9.3f}{q[3]:>10.3f}")


def main():
    config = load_config("config.yaml")
    env = RFEnvironment(config)
    det_cfg = config["detection"]
    detection_model = DetectionModel(
        threshold_db=det_cfg["threshold_db"], snr_scale=det_cfg["snr_scale"],
        false_alarm_probability=det_cfg["false_alarm_probability"],
        seed=det_cfg["seed"],
    )
    receiver = Receiver(env, k=config["receiver_channels"], detection_model=detection_model)
    belief = BeliefEngine(config["num_bands"], config.get("belief"))
    temporal = TemporalEngine(config["num_bands"], config.get("temporal"))
    scoring = BandScoringEngine(config["num_bands"], config.get("scoring"))
    arbitrator = QLearningArbitrator(config.get("ml_arbitrator"))
    k = config["receiver_channels"]

    print_q_table(arbitrator, "A. INITIAL Q-TABLE")

    episode_reward = 0.0
    reward_ma = deque(maxlen=200)
    episode_rewards = []
    early_counts = {s: 0 for s in STRATEGY_NAMES}
    late_counts = {s: 0 for s in STRATEGY_NAMES}
    early_cutoff = TOTAL_STEPS * 0.2
    late_cutoff = TOTAL_STEPS * 0.8
    last_decisions = deque(maxlen=5)

    print("\nB. TRAINING PROGRESS")
    for t in range(TOTAL_STEPS):
        env.step()
        state = arbitrator.get_state(belief.get_state())
        action, strategy_name = arbitrator.select_strategy(state)

        scoring.update(belief.get_state(), temporal.get_state(), t)
        selected_bands = scoring.top_k(strategy_name, k)
        observations = receiver.observe(selected_bands)
        belief.update(observations)
        temporal.update(observations, t)

        reward = arbitrator.calculate_reward(observations, t)
        next_state = arbitrator.get_state(belief.get_state())
        arbitrator.update(state, action, reward, next_state)

        episode_reward += reward
        reward_ma.append(reward)
        if t < early_cutoff:
            early_counts[strategy_name] += 1
        if t >= late_cutoff:
            late_counts[strategy_name] += 1
        last_decisions.append((state, arbitrator.get_q_values(state), strategy_name))

        if (t + 1) % EPISODE_LENGTH == 0:
            episode_rewards.append(episode_reward)
            ep_idx = (t + 1) // EPISODE_LENGTH
            if ep_idx % 8 == 0 or ep_idx == 1:
                avg_ma = sum(reward_ma) / len(reward_ma)
                print(f"  episode {ep_idx:>3} | reward={episode_reward:>6.2f} "
                      f"| moving_avg={avg_ma:>6.3f} | epsilon={arbitrator.epsilon:.3f} "
                      f"| last_strategy={strategy_name}")
            episode_reward = 0.0

    print_q_table(arbitrator, "C. FINAL Q-TABLE")

    def pct(counts):
        total = sum(counts.values()) or 1
        return {s: 100.0 * c / total for s, c in counts.items()}

    print("\nD. STRATEGY SELECTION FREQUENCY")
    print("  Early training (first 20%):",
          {s: f"{p:.0f}%" for s, p in pct(early_counts).items()})
    print("  Late training  (last 20%): ",
          {s: f"{p:.0f}%" for s, p in pct(late_counts).items()})

    print(f"\nE. FINAL REWARD MOVING AVERAGE (last {len(reward_ma)} steps): "
          f"{sum(reward_ma) / len(reward_ma):.3f}")

    print("\nF. EXAMPLE FINAL DECISIONS (last 5 steps of training)")
    for state, q_values, strategy_name in last_decisions:
        labels = ["LOW", "MEDIUM", "HIGH"]
        print(f"  STATE: strategy_performance={labels[state[0]]}, "
              f"uncertainty={labels[state[1]]}, detection_rate={labels[state[2]]}")
        print(f"  Q-values: EXPLORE={q_values[0]:.2f} EXPLOIT={q_values[1]:.2f} "
              f"PREDICT={q_values[2]:.2f} BALANCED={q_values[3]:.2f}")
        print(f"  SELECTED: {strategy_name.upper()}\n")

    print("STRATEGY STATISTICS:", arbitrator.get_strategy_statistics())


if __name__ == "__main__":
    main()
