# TSRD Data Adapter for Cognitive Electronic Warfare Smart Scan

## Overview

The **TSRD Data Adapter** integrates the **Turing Synthetic Radar Dataset (TSRD)** into the Cognitive RF Smart Scan Electronic Warfare simulation architecture. It converts raw, continuous asynchronous radar Pulse Descriptor Words (PDWs) from HDF5 scenario recordings into synchronized 50-band discrete RF observation frames.

---

## Key Principles & Architectural Invariants

1. **Strict Interface Parity**:
   `TSRDEnvironment` adheres to the exact same contract as `RFEnvironment`:
   - Number of bands: $N = 50$ (`F01` to `F50`)
   - Channel observation capacity: $K = 5$ channels via standard `Receiver`
   - Observable output: `Observation` dataclass (`timestep`, `band_id`, `hit`, `signal_strength`, `snr`, `detection_probability`)

2. **Zero Ground-Truth Leakage**:
   - Emitter labels, transmitter IDs, and future pulse parameters are sequestered in `TruthManager`.
   - `Receiver.observe(selected_bands)` returns detection information **only** for the $K=5$ selected bands.
   - Ground-truth labels are completely hidden from cognitive agents (Belief, Temporal, Band Scoring, Q-Learning, Predictive ML) and accessed **only** by post-hoc `EvaluationMetrics`.

3. **On-Demand Loading & High Efficiency**:
   - Reads one `.h5` file at a time using read-only HDF5 (`h5py`).
   - Uses vectorized NumPy array indexing and group-by binning for sub-second processing.

---

## Package Architecture

```
data_adapter/
├── __init__.py           # Unified exports
├── tsrd_loader.py        # Reads raw /data, /labels, /metadata from HDF5
├── frequency_mapper.py   # Maps 500-18000 MHz continuous range to F01-F50
├── pdw_processor.py      # Bins microsecond ToA pulses into discrete time steps
├── truth_manager.py      # Tracks emitter truth, pulse counts, and re-acquisition
└── scenario_builder.py   # TSRDEnvironment and unified create_environment() factory
```

---

## Frequency and Time Mapping

### Frequency Mapping ($500 - 18000\text{ MHz} \to 50\text{ Bands}$)
- Frequency range: $f_{\min} = 500.0\text{ MHz}$, $f_{\max} = 18000.0\text{ MHz}$
- Number of bands: $N = 50$
- Bandwidth per band: $\Delta f = \frac{18000.0 - 500.0}{50} = 350.0\text{ MHz}$
- Band index formula:
  $$\text{band\_index} = \min\left(49, \max\left(0, \left\lfloor \frac{f - 500.0}{350.0} \right\rfloor\right)\right)$$
- Band ID: `F01` ($500 - 850\text{ MHz}$) to `F50` ($17650 - 18000\text{ MHz}$)

### Time Mapping
- TSRD Time of Arrival ($\text{ToA}$) is recorded in microseconds ($\mu s$).
- Standard conversion: $t_{\text{seconds}} = \text{ToA}_{\mu s} \times 10^{-6}$.
- Time step assignment ($\Delta t = 0.05\text{ s}$ by default):
  $$\text{step\_idx} = \min\left(\text{total\_steps} - 1, \max\left(0, \left\lfloor \frac{t_{\text{seconds}}}{\Delta t} \right\rfloor\right)\right)$$

### Detectability & Sensitivity Threshold
- Default receiver sensitivity: $-110.0\text{ dBm}$
- If $\max(\text{Amplitude}_{\text{band}}) \ge -110.0\text{ dBm}$, the signal is detectable.
- Derived SNR:
  $$\text{SNR}_{\text{dB}} = \max(0.0, \text{Amplitude}_{\text{max}} - (-110.0))$$

---

## Quickstart & Usage Example

```python
from data_adapter import TSRDEnvironment, EnvironmentSource, create_environment
from rf_env import Receiver, IntelligentSchedulerAdapter, EvaluationMetrics

# 1. Instantiate TSRD Environment from an HDF5 scenario
env = TSRDEnvironment(
    file_path=r"D:\sih\dataset\scan\test_scan\config_1.h5",
    step_duration_s=0.05,
    num_bands=50
)

# 2. Attach Standard 5-Channel Receiver & Intelligent Scheduler
receiver = Receiver(env, k=5)
scheduler = IntelligentSchedulerAdapter(num_bands=50, k=5)
metrics = EvaluationMetrics(redundancy_window=3)

# 3. Run Cognitive Simulation Loop
for t in range(100):
    env.step()
    selected_bands = scheduler.select_bands(t)       # Chooses 5 of 50 bands
    observations = receiver.observe(selected_bands)  # Receives ONLY 5 observations
    env.notify_scan_results(observations)
    metrics.observe_step(env, selected_bands, observations)
    scheduler.learn(observations, t)                 # Bayesian + Temporal + Q-Learning

print(metrics.summary())
```
