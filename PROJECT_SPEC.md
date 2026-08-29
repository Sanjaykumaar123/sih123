# PROJECT_SPEC.md — Cognitive RF Smart Scan (SIH 26055)

Status: **Stages 0–2 approved and implemented** (spec; RF environment +
limited receiver; probabilistic detection physics). Stage 3 not started.

---

## 1. Problem Statement

A simulated RF spectrum has **N = 50** frequency bands. The receiver can
observe only **K = 5** bands per scan cycle. An intelligent scheduler must
choose, every cycle, which 5 bands to scan so that active emitters are
detected quickly, scans are not wasted on bands unlikely to be active, and
the scheduler adapts when emitter behaviour changes (e.g. an emitter moves
to evade detection).

The scheduler is a **partial-observability sequential decision problem**,
not a classification problem: it must decide *where to look next* using
only what it has observed so far.

---

## 2. Constants

| Symbol | Meaning | Value |
|---|---|---|
| `N` | total frequency bands | 50 |
| `K` | bands observable per scan | 5 |
| `t` | discrete time step (one scan cycle = one step) | 0,1,2,... |
| `W` | sliding window (observations) used for RF feature extraction | 20 steps (tunable) |
| `M` | min. observations on a band before RF classification is attempted | 5 hits (tunable) |

Time is **discrete**. At each step `t`, the scheduler selects a set
`S_t ⊂ {1..N}, |S_t| = K`, and receives one observation per band in `S_t`.

---

## 3. Observation Contract (what the scheduler may / may not use)

**May use (per scanned band, per step):**
- `hit` (bool: signal present above detection threshold)
- `signal_strength` (float, only if hit)
- `SNR` (float, only if hit)
- `timestamp` (int, = `t`)
- Its own history of the above, for any band it has previously scanned

**May NOT use:**
- Any value from a band not in `S_t` at step `t` (no peeking)
- Ground-truth emitter type, position, or on/off schedule
- Any internal simulator state

The simulator/environment **may** keep ground truth internally, but only
for logging and post-hoc evaluation (metrics, plots) — it must never leak
into the scheduler's inputs. This separation is enforced structurally: the
environment exposes `observe(S_t) -> {band: (hit, strength, snr, t)}` and
nothing else to the scheduler.

---

## 4. Emitter Models (ground truth, simulator-internal only)

Kept deliberately simple and parametric — no behaviour is added that the
scheduler doesn't need to react to.

1. **Static** — fixed band, active with constant probability `p` per step.
2. **Periodic** — fixed band, active in a repeating window of period `T`
   and duty cycle `d` (on for `d·T` steps, off for `(1-d)·T`).
3. **Frequency-agile** — hops across a fixed subset of bands on a
   deterministic pseudo-random schedule (seeded), one band active per hop.
4. **Adaptive/evasive** — behaves like frequency-agile, but **reacts to
   being detected**: if it has been hit ≥ `h` times within the last `W`
   steps, it changes its hopping subset/seed. This is the one deliberately
   "closed-loop" emitter and is what makes the adaptation story real
   without requiring game-theoretic modelling.

Each emitter is assigned to one band-set at scenario init; multiple
emitters may be active concurrently across different bands.

---

## 5. Core Intelligence Pipeline (concrete, per stage)

```
RF Environment (ground truth, hidden)                          [Stage 1]
   -> Limited Receiver: observe(S_t) for chosen K bands          [Stage 1]
   -> Detection physics: probabilistic hit/miss (logistic P_d
      by SNR + configurable false-alarm rate)                    [Stage 2]
   -> hit/miss + strength + SNR + timestamp
   -> Bayesian belief state (per band, Beta-Bernoulli, with decay) [Stage 3]
   -> Temporal behaviour features (per band, from history window W) [Stage 4]
   -> Random Forest emitter-type classifier (only if obs. count >= M) [Stage 4]
   -> Band scoring (formula depends on active strategy)           [Stage 5]
   -> Q-learning strategy arbitrator (picks ONE strategy, not bands) [Stage 6]
   -> Top-K = argmax(band scores) under chosen strategy            [Stage 5]
   -> Scan -> reward computed -> Q-table update                    [Stage 6]
```

### 5.1 Bayesian belief state
Each band `i` has belief `Beta(alpha_i, beta_i)`, initialized `(1,1)`
(uniform). On observation: `alpha_i += 1` if hit else `beta_i += 1`.
To support non-stationary emitters (esp. adaptive/evasive), apply a decay
each step to unobserved bands' belief toward the prior:
`alpha_i, beta_i <- 1 + gamma*(alpha_i-1, beta_i-1)`, `gamma ≈ 0.98`
(tunable). This is the *entire* justification for Bayesian inference here:
cheap, closed-form, and naturally supports "forgetting" for adaptation —
no particle filter or HMM needed at prototype scale.

`P(active_i) = alpha_i / (alpha_i + beta_i)`.
`Uncertainty_i = Var[Beta_i] = alpha_i*beta_i / ((alpha_i+beta_i)^2 * (alpha_i+beta_i+1))`.

### 5.2 Temporal features (per band, computed from last W observations)
- hit rate, mean inter-hit interval, variance of inter-hit interval,
  mean SNR/strength, steps-since-last-hit.
These feed both the RF classifier and the "prediction" strategy.

### 5.3 Random Forest — emitter behaviour classifier
- **Input**: the feature vector in 5.2, computed only for bands with
  ≥ `M` recorded hits.
- **Output**: one of {static, periodic, agile, adaptive} (label available
  in simulation for training/validation only).
- **Purpose**: once a band is classified, the scheduler can predict *when
  it will next be worth revisiting* (e.g. periodic → revisit at phase;
  agile/adaptive → down-weight predictability, rely on exploration).
- Trained offline/incrementally on simulated (feature, label) pairs;
  not used before `M` observations exist for a band (falls back to
  Bayesian score alone).

### 5.4 Band scoring (per strategy — this is what turns "strategy" into "K bands")
- **Exploration**: score = `Uncertainty_i` (highest variance / least observed first).
- **Exploitation**: score = `P(active_i)` (highest belief of activity first).
- **Prediction**: score = RF-informed — e.g. for bands classified periodic,
  boost score if predicted phase says "due to be active"; for agile/adaptive,
  score = recent hit-rate decayed by time-since-last-hit.
- **Balanced**: score = `w1*P(active_i) + w2*Uncertainty_i + w3*Prediction_i`
  (fixed weights, e.g. 0.5/0.3/0.2 — tunable, not learned, to keep scope small).
- In every strategy: `Top-K = argsort(scores, descending)[:K]`.

### 5.5 Q-learning — strategy arbitrator (NOT band selection)
- **State** (discretized, small): e.g.
  `(mean_P_active_bucket, mean_uncertainty_bucket, steps_since_new_hit_bucket)`
  — a handful of bins per dimension keeps the table small and tractable.
- **Actions**: `{Explore, Exploit, Predict, Balanced}` (4 actions).
- **Reward** (per step):
  `R_t = (num_new_hits_this_step) - c * (num_redundant_scans)`
  where "new hit" = a hit on a band not already believed active
  (`P_active` was low before this scan), and "redundant scan" = a scan of
  a band with high existing confidence and no new information. `c` is a
  small tunable penalty (e.g. 0.2) that discourages wasted rescans without
  discouraging confirmation scans entirely.
- Standard tabular Q-learning update:
  `Q(s,a) <- Q(s,a) + lr*(R_t + gamma_q*max_a' Q(s',a') - Q(s,a))`.
- **Why Q-learning is scoped this way**: 4 actions x small discretized
  state space is tractable to learn in a short simulation; learning to
  arbitrate among 4 strategies is a well-posed problem, whereas learning
  to pick 5-of-50 bands directly is not, at this project's scale/timeline.

---

## 6. Approved Stage Sequence

- **Stage 0 — Problem specification and architecture** (this document). Done.
- **Stage 1 — RF environment and limited receiver**: N=50 band environment,
  4 emitter models (§4), `observe(S_t)` interface (§3), ground-truth
  logger. Done.
- **Stage 2 — Detection physics**: probabilistic hit/miss on the Receiver
  — logistic P_d(SNR) plus a configurable false-alarm rate, replacing the
  Stage 1 deterministic placeholder. §3's observation boundary is
  unchanged: still only the selected K bands, still no leakage. Done.
- **Stage 3 — Bayesian cognitive belief engine**: per-band Beta-Bernoulli
  belief with decay (§5.1). Not started.
- **Stage 4 — Temporal behaviour / prediction**: temporal features (§5.2)
  and the Random Forest emitter-type classifier (§5.3). Not started.
- **Stage 5 — Intelligent band-scoring strategies**: the four scoring
  formulas (§5.4 — Exploration/Exploitation/Prediction/Balanced) and
  Top-K selection. Not started.
- **Stage 6 — ML strategy arbitrator**: Q-learning over the 4 strategies
  (§5.5) — state/action/reward and Q-table updates. Not started.
- **Stage 7 — Adaptive/evasive emitter**: activates the reactive
  behaviour already scaffolded in Stage 1 (§4 item 4 —
  `detection_count`/`recent_detection_times`/`evasive_mode`), now driven
  by real detections once Stages 3–6 exist. Not started.
- **Stage 8 — Baseline comparison and evaluation**: round-robin and
  random-K baselines, metrics from §7. Not started.
- **Stage 9 — Predictive performance module**: reporting/validation of the
  Stage 4–6 prediction and arbitration pathway against the §7 metrics
  (RF classification accuracy, adaptation recovery time, learned-vs-
  baseline reward) — evaluation only, no new algorithm. Not started.
- **Stage 10 — Dashboard/demo**: visualization layer (Streamlit or
  similar), out of scope until Stages 1–9 are validated.

Each stage is implemented and reviewed independently; the next stage
starts only after explicit approval.

---

## 7. Success Metrics

| Metric | Definition | Goal |
|---|---|---|
| Detection latency | avg. steps from emitter activation to first hit | minimize |
| Scan efficiency | total hits / total scans (`K*T`) | maximize |
| Coverage | % of emitters detected at least once within horizon | maximize, target ~100% for static/periodic |
| Adaptation recovery time | steps to re-detect an adaptive emitter after it changes pattern | minimize; must show learned scheduler beats fixed/random baseline |
| Redundant-scan rate | % of scans that revisit high-confidence bands with no new info | minimize |
| RF classification accuracy | accuracy vs. ground-truth emitter type, post-`M`-observations | report only, not optimized directly |
| Q-learning vs baseline | cumulative reward, learned arbitrator vs. round-robin and vs. pure-random-K baselines | learned > both baselines |

Baselines to compare against (cheap to implement, required for any claim
of "intelligence"): **Round-robin scan** and **Uniform-random-K scan**.

---

## 8. Explicit Non-Goals (per project constraints)

- No real RF hardware, SDR, or spectrum data.
- No deep learning (unless Stages 3–6 empirically show the above is
  insufficient — not assumed).
- No claims of military-grade or real-world deployment performance.
- No band-level access outside the selected K per step, anywhere in the
  scheduler code path (structural constraint, not just a convention).
- No Streamlit/dashboard work until Stage 10.

---

## 9. Open Parameters (tunable later, not blocking approval)

`gamma` (belief decay), `c` (redundancy penalty), `w1/w2/w3` (balanced
weights), `lr`/`gamma_q` (Q-learning), `W`, `M` — all listed above with
starting defaults; revisit empirically once Stage 3+ produces data.
