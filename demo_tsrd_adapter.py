"""Demonstration script for TSRD Data Adapter with Cognitive Smart Scan.

Runs one representative TSRD scenario (D:\\sih\\dataset\\scan\\test_scan\\config_1.h5)
through the full Cognitive EW Closed-Loop Scheduler (Belief + Temporal + Scoring + Q-Learning)
and prints a comprehensive operational report.
"""

import os
import sys

from data_adapter import TSRDEnvironment, FrequencyMapper, TruthManager
from rf_env import (
    Receiver,
    IntelligentSchedulerAdapter,
    EvaluationMetrics,
)

SAMPLE_FILE = r"D:\sih\dataset\scan\test_scan\config_1.h5"


def main():
    print("=" * 80)
    print("TURING SYNTHETIC RADAR DATASET (TSRD) — DATA ADAPTER DEMONSTRATION")
    print("=" * 80)

    if not os.path.exists(SAMPLE_FILE):
        print(f"Error: Sample file not found at {SAMPLE_FILE}")
        sys.exit(1)

    print(f"\n1. Loading Scenario File: {SAMPLE_FILE}")
    env = TSRDEnvironment(SAMPLE_FILE, step_duration_s=0.05, num_bands=50)
    raw = env.raw_data
    rec_meta = raw.receiver_metadata

    print("\n2. Scenario Metadata:")
    print(f"   - Total Pulses:             {raw.num_pulses:,}")
    print(f"   - Feature Columns:          {raw.features}")
    print(f"   - Active Emitter Classes:   {raw.num_emitter_classes}")
    print(f"   - Configured Transmitters:  {len(raw.transmitter_metadata)}")
    print(f"   - Collection Time:          {rec_meta.collection_time_s:.1f} seconds")
    print(f"   - Time Step Duration (dt):  {env.step_duration_s:.3f} seconds ({env.total_steps} discrete steps)")
    print(f"   - Frequency Range:          {rec_meta.freq_range_mhz[0]:.1f} - {rec_meta.freq_range_mhz[1]:.1f} MHz")
    print(f"   - Receiver Sensitivity:     {rec_meta.sensitivity_dbm:.1f} dBm")
    print(f"   - Scan Mode:                {rec_meta.scan_mode}")

    print("\n3. Discrete Frequency Band Mapping (50 Bands, 350.0 MHz each):")
    fm: FrequencyMapper = env.freq_mapper
    print(f"   - Band F01:                 {fm.get_band_info('F01').freq_min_mhz:.1f} - {fm.get_band_info('F01').freq_max_mhz:.1f} MHz (Center: {fm.get_band_info('F01').freq_center_mhz:.1f} MHz)")
    print(f"   - Band F25:                 {fm.get_band_info('F25').freq_min_mhz:.1f} - {fm.get_band_info('F25').freq_max_mhz:.1f} MHz (Center: {fm.get_band_info('F25').freq_center_mhz:.1f} MHz)")
    print(f"   - Band F50:                 {fm.get_band_info('F50').freq_min_mhz:.1f} - {fm.get_band_info('F50').freq_max_mhz:.1f} MHz (Center: {fm.get_band_info('F50').freq_center_mhz:.1f} MHz)")

    print("\n4. Initializing Receiver (K=5 Channels) & Cognitive Smart Scan Adapter:")
    receiver = Receiver(env, k=5)
    scheduler = IntelligentSchedulerAdapter(num_bands=50, k=5)
    metrics = EvaluationMetrics(redundancy_window=3)
    truth_mgr: TruthManager = env.truth_manager

    print("\n5. Running 50-Step Closed-Loop Simulation:")
    print("-" * 80)
    print(f"{'Step':<6}{'Strategy':<14}{'Selected Bands (K=5)':<25}{'Hits':<8}{'Reward':<10}{'Active Bands'}")
    print("-" * 80)

    for t in range(50):
        env.step()
        selected_bands = scheduler.select_bands(t)
        observations = receiver.observe(selected_bands)
        env.notify_scan_results(observations)
        metrics.observe_step(env, selected_bands, observations)
        reward = scheduler.learn(observations, t)

        hit_bands = [b for b, o in observations.items() if o.hit]
        active_in_env = len(env.processor.get_active_bands(t))

        if t < 10 or t % 10 == 0 or t == 49:
            bands_str = " ".join(selected_bands)
            hits_str = f"{len(hit_bands)}/{len(selected_bands)}"
            print(f"{t:<6}{scheduler.last_strategy:<14}{bands_str:<25}{hits_str:<8}{reward:<10.2f}{active_in_env} bands")

    print("-" * 80)
    print("\n6. Zero Ground-Truth Leakage Verification:")
    sample_obs = receiver.observe(["F01", "F02", "F03", "F04", "F05"])
    first_obs = list(sample_obs.values())[0]
    print(f"   - Observation Class:        {type(first_obs).__name__}")
    print(f"   - Public Attributes:        {[k for k in dir(first_obs) if not k.startswith('_')]}")
    print(f"   - Has Emitter ID?           {hasattr(first_obs, 'emitter_id')} (MUST BE False)")
    print(f"   - Has Emitter Type?         {hasattr(first_obs, 'emitter_type')} (MUST BE False)")

    print("\n7. Performance & Evaluation Summary (Step 0 to 49):")
    summary = metrics.summary()
    print("-" * 50)
    print(f"   Total Receiver Scans:       {summary['total_scans']}")
    print(f"   Total Hits Intercepted:     {summary['total_hits']}")
    print(f"   Detection Rate (Pd):        {summary['pd'] * 100:.2f}%")
    print(f"   Interception Rate:          {summary['interception_rate'] * 100:.2f}%")
    print(f"   False Alarm Rate (Pfa):     {summary['pfa'] * 100:.2f}%")
    print(f"   Average Reward / Step:      {summary['avg_reward']:.3f}")
    print(f"   Redundant Scan Rate:        {summary['redundant_scan_rate'] * 100:.2f}%")
    print("-" * 50)

    print("\n[SUCCESS] TSRD Data Adapter verification and cognitive execution completed successfully!")


if __name__ == "__main__":
    main()
