"""Stage 7 demo: adaptive/evasive emitter vs. the full closed-loop scheduler.

Run: python demo_stage7.py

Full real closed loop, no scripted band selection: Q-learning arbitrator
picks a strategy -> Stage 5 ranks all 50 bands under it -> receiver scans
top-K -> RFEnvironment.notify_scan_results() feeds E4 (the
adaptive/evasive emitter) ONLY its own hit/miss -> belief/temporal/reward
update -> Q-table update. E4's internal state (is_evasive/evasion_count/
current_band) is read here purely for reporting -- it is never fed back
into belief/temporal/scoring/arbitrator, matching every earlier demo's use
of ground truth for debug printing only.

This does NOT model an intelligent adversary; E4 only reacts to its own
past detection outcomes (see rf_env/emitters.py). Whether the scheduler
ever discovers/re-acquires E4 organically is not guaranteed -- this demo
reports whatever actually happens.
"""

from rf_env import (RFEnvironment, Receiver, DetectionModel, BeliefEngine,
                     TemporalEngine, BandScoringEngine, QLearningArbitrator)
from rf_env.config import load_config

TOTAL_STEPS = 4000


def phase_metrics(rows):
    if not rows:
        return {"detection_rate": 0.0, "avg_reward": 0.0, "strategy_pct": {}}
    detections = sum(1 for r in rows if r["e4_hit"])
    avg_reward = sum(r["reward"] for r in rows) / len(rows)
    counts = {}
    for r in rows:
        counts[r["strategy"]] = counts.get(r["strategy"], 0) + 1
    strategy_pct = {s: round(100 * c / len(rows)) for s, c in counts.items()}
    return {"detection_rate": detections / len(rows), "avg_reward": avg_reward,
            "strategy_pct": strategy_pct}


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
    e4 = next(e for e in env.emitters if e.emitter_id == "E4")

    print(f"E4 (adaptive/evasive) starts on {e4.current_band}; "
          f"hit_threshold={e4.hit_threshold}, window={e4.observation_window}, "
          f"evasive_duration={e4.evasive_duration}\n")

    history = []
    print(f"{'t':>5}  {'state':<8}{'band':<6}{'strategy':<12}{'hit':>4}{'reward':>8}{'evasions':>9}")
    for t in range(TOTAL_STEPS):
        env.step()
        state = arbitrator.get_state(belief.get_state())
        action, strategy_name = arbitrator.select_strategy(state)

        scoring.update(belief.get_state(), temporal.get_state(), t)
        selected_bands = scoring.top_k(strategy_name, k)
        observations = receiver.observe(selected_bands)
        env.notify_scan_results(observations)
        belief.update(observations)
        temporal.update(observations, t)

        reward = arbitrator.calculate_reward(observations, t)
        next_state = arbitrator.get_state(belief.get_state())
        arbitrator.update(state, action, reward, next_state)

        e4_obs = observations.get(e4.current_band)
        row = {"t": t, "evasive": e4.is_evasive, "evasion_count": e4.evasion_count,
               "band": e4.current_band, "strategy": strategy_name, "reward": reward,
               "e4_hit": bool(e4_obs and e4_obs.hit)}
        history.append(row)
        just_changed = t > 0 and history[t - 1]["evasion_count"] != row["evasion_count"]
        detailed = row["evasion_count"] <= 3  # full hit-by-hit detail for the first 3 cycles only
        if t < 40 or just_changed or (detailed and row["e4_hit"]):
            print(f"{t:>5}  {'EVASIVE' if row['evasive'] else 'NORMAL':<8}{row['band']:<6}"
                  f"{strategy_name:<12}{int(row['e4_hit']):>4}{reward:>8.2f}{row['evasion_count']:>9}")

    # ---- segment phases around the FIRST evasion event ----
    evasion_events = e4.evasion_count
    first_evasion_row = next((r for r in history if r["evasion_count"] >= 1), None)

    print(f"\nTotal evasion events triggered: {evasion_events}")

    if first_evasion_row is None:
        print("No evasion event was ever triggered in this run -- E4 was never "
              "scanned densely enough (>= hit_threshold hits within the "
              "observation_window) to cross the evasion trigger. Reporting "
              "overall metrics only.")
        m = phase_metrics(history)
        print(f"Overall: detection_rate={m['detection_rate']:.3f} "
              f"avg_reward={m['avg_reward']:.3f} strategy%={m['strategy_pct']}")
        return

    evasion_start_t = first_evasion_row["t"]
    before = [r for r in history if r["t"] < evasion_start_t]
    during = [r for r in history if r["evasion_count"] == 1 and r["evasive"]]
    evasion_end_t = during[-1]["t"] + 1 if during else evasion_start_t
    second_evasion_row = next((r for r in history if r["evasion_count"] >= 2), None)
    after_cutoff = second_evasion_row["t"] if second_evasion_row else TOTAL_STEPS
    after = [r for r in history if evasion_end_t <= r["t"] < after_cutoff]

    reacquired_row = next((r for r in after if r["e4_hit"]), None)

    def fmt(m):
        return (f"detection_rate={m['detection_rate']:.2f} "
                f"avg_reward={m['avg_reward']:.2f} strategy%={m['strategy_pct']}")

    print("\nBEFORE evasion:                            ", fmt(phase_metrics(before)))
    print("DURING evasion:                              ", fmt(phase_metrics(during)))
    print("AFTER evasion (until next event or run end): ", fmt(phase_metrics(after)))

    if reacquired_row:
        print(f"\nRe-acquisition occurred at t={reacquired_row['t']} "
              f"({reacquired_row['t'] - evasion_end_t} steps after evasion ended).")
    else:
        print("\nRe-acquisition did NOT occur within this run's observed window "
              "(honest result -- not adjusted).")


if __name__ == "__main__":
    main()
