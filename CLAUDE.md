# CLAUDE.md — Cognitive RF Smart Scan (SIH 26055)

Read this file first, every session. It exists so you don't have to
re-read/re-derive the spec or re-explore the repo to know what to do next.
This is a token-constrained academic prototype (SIH hackathon) — be terse,
don't re-open files you've already read this session, don't explore the
tree "just in case."

## Project state (source of truth: check this section before anything else)

- **Spec is FINAL and approved**: [PROJECT_SPEC.md](PROJECT_SPEC.md).
  Do not re-derive constants, formulas, or architecture from scratch —
  they're all in there (N=50, K=5, Bayesian/RF/Q-learning design, reward
  function, metrics). Read PROJECT_SPEC.md instead of asking the user to
  re-explain the design.
- **Current stage: Stage 11 (final integration, stress testing, prototype
  readiness) implemented, tests passing, awaiting user review/approval.
  This is the FINAL stage — the project's full sequence (0–11) is
  complete pending this approval. Prototype readiness: READY WITH MINOR
  FIXES (see chat for the full Stage 11 report).** Do NOT start any
  further stage/feature without being asked. (Stage 9's Random Forest is
  the FIRST place Random Forest actually appears in this project — Stage
  4's originally-planned RF emitter classifier was never built,
  deliberately deferred per that stage's own scope; Stage 9 covers a
  different job: predicting interception time/rate, not classifying
  emitter type.)
- Update the "Current stage" line above yourself whenever a stage is
  completed and approved, so the next session doesn't need to grep the
  repo or ask the user to re-explain progress.

## Stage gate rule (hard constraint from the user)

Implement **one stage at a time** (per PROJECT_SPEC.md §6), then **stop
and wait for explicit approval** before starting the next stage. Do not
pre-build later stages "while you're in there." Do not add algorithms or
files not called for by the current stage.

## Hard constraints (apply to all stages, don't re-litigate these)

- No deep learning unless Stages 1–4 empirically prove insufficient.
- No real RF hardware/SDR.
- No claims of real/military-grade performance — this is a simulation.
- Scheduler code must never read any band outside the current step's
  selected K bands — no exceptions, no debug backdoors.
- No dashboard/Streamlit work before Stage 5.
- Don't install packages without asking; keep dependencies minimal.
  Installed so far: numpy, pyyaml, pytest (Stage 1), scikit-learn (Stage 9,
  already present), streamlit + plotly (Stage 10, plotly installed via pip
  since Stage 10 explicitly required it) — nothing else assumed.
- Don't create files beyond what the current stage needs.

## Repo layout (update this as files are actually created)

- `PROJECT_SPEC.md` — full technical spec, approved.
- `CLAUDE.md` — this file.
- `config.yaml` — Stage 1 config (bands/receiver/seed/emitters).
- `rf_env/` — package: `environment.py` (RFEnvironment, hidden ground
  truth, GroundTruthLogger), `emitters.py` (Emitter base +
  Static/Periodic/FrequencyAgile/AdaptiveEvasive), `detection.py` (Stage 2:
  `DetectionModel` — logistic P_d(snr) when present, fixed
  `false_alarm_probability` when absent; returns `DetectionResult`),
  `receiver.py` (Receiver.observe — the only scheduler-facing API,
  enforces K=5, runs detection_model internally, never leaks unselected
  bands/emitter identity/ground truth), `config.py` (YAML loader).
- `demo_stage1.py` — ground truth vs. receiver observation side by side.
- `demo_stage2.py` — per-band SNR/P_d/hit + true-detection vs. miss vs.
  false-alarm labelling (labelling is debug-only, cross-checked against
  ground truth; scheduler never gets that label).
- `rf_env/belief.py` — Stage 3: `BeliefEngine` (Beta-Bernoulli per band,
  prior (1,1), decay_gamma pulls alpha/beta toward prior every `update()`
  call, uncertainty = Beta variance). Public API takes only
  `observations` dicts / band_ids — no environment/ground-truth access.
  `get_state()`/`get_belief()` return `BandBelief` (band_id,
  activity_probability, uncertainty, last_observed, staleness, hit/miss
  counts). `config.yaml`'s `belief:` section holds prior_alpha/prior_beta/
  decay_gamma.
- `demo_stage3.py` — feeds receiver observations into BeliefEngine, prints
  P(active)/uncertainty/staleness per scanned band each step.
- `rf_env/temporal.py` — Stage 4: `TemporalEngine` (bounded per-band
  history deque, `history_length` config), inter-hit interval stats ->
  periodicity_score = 1/(1+coefficient_of_variation), next-active
  prediction = last_hit + mean_interval, confidence = periodicity_score ×
  evidence_factor. behaviour_type in
  {insufficient_data, stable, periodic, intermittent}. Deterministic, no
  ML. Public API: `update(observations, current_timestep)`,
  `get_prediction(band_id)`, `get_state()`, `reset()` — observations-only,
  no environment/ground-truth access. `config.yaml`'s `temporal:` section
  holds history_length/min_hits_for_prediction/periodicity_threshold/
  stable_interval_max.
- `demo_stage4.py` — runs 80 steps, prints Hits/Period/Periodicity/
  NextActive/Confidence/Behaviour per band; uses false_alarm_probability=0
  locally (not config.yaml's real value) purely so the periodicity table
  reads cleanly — Stage 2's real false-alarm behaviour is untouched and
  still covered by demo_stage2.py/test_stage2.py.
- `rf_env/scoring.py` — Stage 5: `BandScoringEngine`, consumes only
  `BandBelief`/`TemporalPrediction` snapshots (Stage 3/4 outputs), never
  RFEnvironment/Receiver. Four bounded [0,1] scores per band:
  exploration = normalized_uncertainty × normalized_staleness (batch-max
  normalized uncertainty; inf-safe staleness via `staleness/(staleness+
  staleness_scale)`), exploitation = activity_probability directly,
  prediction = proximity(predicted_next_active_time) × confidence ×
  periodicity_score (0 if no prediction), balanced = configurable
  weighted sum (must sum to 1, enforced at construction). API:
  `update(belief_state, temporal_state, current_timestep)`,
  `score_band()`, `get_scores()`, `rank()`, `top_k()`, `reset()`.
  `config.yaml`'s `scoring:` section holds `staleness_scale` and
  `balanced_weights`.
- `demo_stage5.py` — round-robins all 50 bands (K=5/step, still no
  scheduler) for 300 steps so every band has some history, then prints
  top-10 balanced + top-5 per strategy.
- `rf_env/arbitrator.py` — Stage 6: `QLearningArbitrator` (tabular
  Q-learning, `Strategy` IntEnum EXPLORE/EXPLOIT/PREDICT/BALANCED). State
  = (perf_level, uncertainty_level, detection_level), each 0/1/2 (27
  states x 4 actions = 108 Q-values). Consumes only `BandBelief` snapshots
  (Stage 3) + its own tracked reward/hit history — never
  RFEnvironment/Receiver. Reward = new_hits − redundant_scan_penalty ×
  redundant_misses (redundant = scanned within `redundancy_window` steps
  AND still a miss). Standard Q(s,a) update, epsilon-greedy with decay,
  own seeded RNG (independent of detection's). API: `get_state()`,
  `choose_action()`/`select_strategy()`, `calculate_reward()`,
  `update()`, `get_q_values()`, `get_strategy_statistics()`, `reset()`.
  `config.yaml`'s `ml_arbitrator:` section holds all hyperparameters.
- `demo_stage6.py` — full closed-loop training harness (2000 steps,
  Stage6 arbitrator → Stage5 ranking → Stage1/2 receiver → reward → Q
  update); prints initial/final Q-tables, early-vs-late strategy
  selection frequency, reward moving average, last 5 live decisions.
- `rf_env/emitters.py` — `AdaptiveEvasiveEmitter` REPLACED (was: identical
  to FrequencyAgileEmitter). Now inherits `Emitter` directly: sits
  static-like on `current_band` until `register_detection(detected,
  timestep)` (called externally, never by itself) sees >= hit_threshold
  hits within observation_window, then hops a fresh evasive_duration-step
  seeded burst and settles on the burst's last band as its new normal.
  Own private RNG (seed from `adaptive_evasion.seed`, independent of
  detection/Q-learning). Static/Periodic/FrequencyAgileEmitter untouched.
- `rf_env/environment.py` — added `RFEnvironment.notify_scan_results(
  observations)` (new method only; existing methods unchanged): routes
  each emitter's OWN hit/miss back to it via `register_detection()` if it
  has one, using ground truth only to match emitter↔band, never leaking
  across emitters/bands/strategies. `_build_emitters()` now has a
  dedicated `adaptive_evasive` branch (was sharing FrequencyAgileEmitter's
  constructor shape) reading the new global `adaptive_evasion:` config.
  `config.yaml`'s E4 entry now has `band: F30` instead of hop params.
- `demo_stage7.py` — full closed loop (arbitrator→scoring→receiver→
  `notify_scan_results`→belief/temporal→reward→Q-update), 4000 steps;
  reports before/during/after-evasion detection rate, reward, strategy
  mix, and re-acquisition timing from whatever actually happens (no
  scripted band selection).
- `rf_env/evaluation.py` — Stage 8: `RoundRobinScheduler`/`RandomKScheduler`
  (no env/ground-truth access, `select_bands(t)` only), `IntelligentSchedulerAdapter`
  (thin wrapper reusing Stage 3-6 unmodified: `select_bands(t)`/`learn(obs,t)`),
  `RewardTracker` (standalone reimplementation of Stage 6's exact reward
  formula so baselines can be scored identically), `EvaluationMetrics`
  (ground-truth-based Pd/Pfa/sensitivity/interception_rate/avg_intercept_time/
  avg_reward/redundant_scan_rate/prediction_accuracy — read ONLY post-hoc,
  after `receiver.observe()`, never by a scheduler), `EvasionReacquisitionTracker`
  (Stage-7-specific reacquisition timing), `run_single_experiment()` +
  `aggregate_results()` (mean/std over seeds; any "insufficient_data"
  input makes the aggregate "insufficient_data" too, never silently
  dropped). `config.yaml`'s `evaluation:` section holds `num_steps`(2000)
  and `seeds`([100,200,300,400,500]).
- `demo_stage8.py` — runs all 3 schedulers × 5 seeds, same env/detection/
  emitter config per seed (only band-selection differs); prints Pd/Pfa/
  interception/reward/redundancy comparison + adaptive-evasion table;
  saves `results/stage8_results.json` + `results/stage8_summary.csv`.
  Actual last run: Intelligent Pd=0.726 vs RR=0.683 vs RandomK=0.681;
  interception_rate 0.150 vs 0.072 vs 0.068; reward 0.518 vs 0.450 vs
  0.191 — but Round Robin had the LOWEST avg_intercept_time (7.6, from
  its guaranteed full sweep) and NEVER triggered evasion at all (its
  fixed 10-step cycle structurally can't hit the same band 3x within a
  10-step window); Random-K reacquired faster (14.65) than Intelligent
  (34.5) after evasion. Reported as-is, not spun — see chat for full
  discussion of why each of these honestly cuts against "AI wins on
  everything."
- `rf_env/predictor.py` — Stage 9: `FeatureExtractor` (19-feature vector,
  `FEATURE_NAMES`, from Stage 3/4/5 public outputs + its own small
  recent-SNR/observation-count tracker; zero ground-truth access, checked
  structurally), `generate_training_samples()` (runs the REAL
  `IntelligentSchedulerAdapter` closed loop per seed, labels built by
  looking forward in a COMPLETED run's log — ground truth used only
  there, never in features), `PredictiveModelTrainer` (two
  `RandomForestRegressor`s — intercept time, interception rate — plus
  mean-baseline predictors for honest comparison), `Predictor` (runtime
  `.predict(features)`, per-tree spread as a real-but-informal
  uncertainty proxy, `prediction_quality="cold_start"` below 3
  observations). Does NOT touch Stage 6's Q-learning policy.
  `config.yaml`'s `predictive:` section holds n_estimators/max_depth/
  min_samples_leaf/prediction_horizon(100)/seed(999).
- `demo_stage9.py` — train seeds 100-109, val 400-404, test 450-454 (all
  disjoint, `run_length=600`); actual last run: intercept-time R²
  0.62 (RF) vs -0.005 (mean baseline) on held-out test; interception-rate
  R² 0.92 vs 0.00. Generation+training ≈ 40s total. Reports predicted-vs-
  actual for F10/F20/F30 using 3 spread-out test examples per band (not
  cherry-picked single rows) and feature importance (recent_average_snr /
  recent_hit_rate dominate both models).
- `dashboard/simulation_runner.py` — Stage 10: `SimulationRunner`, a thin
  wrapper reusing `IntelligentSchedulerAdapter`/`EvaluationMetrics`/
  `EvasionReacquisitionTracker`/`FeatureExtractor`/`env.notify_scan_results`
  verbatim (no second scheduler, no new ML). Ground truth (`band_truth`)
  is touched only by `EvaluationMetrics`/`EvasionReacquisitionTracker`
  (Stage 8, unchanged) and `self.last_ground_truth_debug`, strictly after
  `receiver.observe()` — never passed into `adapter.select_bands()`/
  `learn()` (checked structurally in tests).
- `dashboard/visualizations.py` — Plotly figure builders only (waterfall,
  belief line chart, Q-value bars, reward history, baseline comparison
  bars) — pure presentation over already-computed data, no new formulas.
- `app.py` — Streamlit dashboard (`streamlit run app.py`): KPI cards,
  11 tabs (Spectrum/Band Priority/Belief/Temporal/Q-Learning/Adaptive
  Evasion/Predictive ML/Baseline Comparison/Why This Band/Architecture/
  Live Metrics), sidebar Step/Run N/Reset/seed/auto-run controls. Loads
  `results/stage8_results.json` and `results/stage9_results.json` +
  `results/stage9_predictor.pkl` via `@st.cache_data`/`@st.cache_resource`
  — never retrains on refresh.
- `demo_stage10.py` — terminal walkthrough of the same real closed loop
  (no Streamlit); reports evasion/re-acquisition honestly (whatever
  actually happens that run) + Stage 8/9 loaded results.
- `demo_stage9.py` — minor ADDITIVE change only (not its algorithm): now
  also saves `results/stage9_results.json` + `results/stage9_predictor.pkl`
  at the end, mirroring `demo_stage8.py`'s existing save pattern, so
  Stage 10 can load rather than retrain. Re-run once with identical
  seeds/config — numbers unchanged from the original Stage 9 report.
- `demo_stage11.py` — final integration demo: real 1000-step run (seed 42)
  via `SimulationRunner`, validating Q-learning (11/108 Q-values non-zero,
  largest 10.11), Bayesian adaptation (F10 P(active) 0.5→0.763 over 914
  obs), temporal prediction (real inter-hit intervals + a genuine
  prediction-error example, 0.7 steps), band-scoring (top-5 per strategy,
  all 50 bands finite), adaptive evasion (11 events this run; first
  trigger t=63→end t=72→reacquired t=88, matching demo_stage10.py's
  independent run at the same seed exactly — confirms reproducibility),
  baseline comparison (loads `results/stage8_results.json`, does not
  re-run), predictor validation (loads `results/stage9_predictor.pkl`,
  confirms cold-start flag), and 5 stress scenarios (A normal, B harder
  detection, C faster evasion cycling, D single-step cold start, E sparse
  detections) — all PASS, config-only variants, no algorithm changes.
- `tests/test_stage11.py` (10) — new integration/edge-case tests: K=1,
  K=50, no-detections config, all-hits config, empty observations across
  every engine, never-observed band through the full pipeline,
  insufficient temporal history, multiple evasion events through the real
  `RFEnvironment`+`Receiver`+`notify_scan_results` loop (not just the
  emitter object in isolation, unlike Stage 7's own test), finite scores
  cold and hot, K-constraint held across a real run.
- Stage 11 structural re-verification: grepped `band_truth`/
  `GroundTruthLogger`/`.emitter_id`/`.emitter_type`/`RFEnvironment`
  against `IntelligentSchedulerAdapter`, `FeatureExtractor`, `Predictor`,
  `Receiver`, `BeliefEngine`, `TemporalEngine`, `BandScoringEngine`,
  `QLearningArbitrator` class bodies specifically — all clean except
  `Receiver` itself (which legitimately reads `band_truth` internally to
  build the stripped `Observation` it returns — by design since Stage 1).
  **No new leak found.**
- `tests/test_stage1.py` (10) + `tests/test_stage2.py` (6) +
  `tests/test_stage3.py` (10) + `tests/test_stage4.py` (13) +
  `tests/test_stage5.py` (17) + `tests/test_stage6.py` (19) +
  `tests/test_stage7.py` (19) + `tests/test_stage8.py` (12) +
  `tests/test_stage9.py` (13) + `tests/test_stage10.py` (12) +
  `tests/test_stage11.py` (10) — all pass: `python -m pytest tests/ -q`
  (141 total).
- Launch dashboard: `streamlit run app.py`. Terminal demos:
  `python demo_stage10.py` / `python demo_stage11.py`.
- `Observation` now carries `detection_probability` in addition to
  timestep/band_id/hit/signal_strength/snr.
- `Receiver(environment, k, detection_model=None)` — `detection_model`
  defaults to `DetectionModel()` (threshold_db=10, snr_scale=3,
  false_alarm_probability=0.05, seed=42) if omitted, so Stage 1 call
  sites/tests still work unchanged. `config.yaml`'s `detection:` section
  holds the real tuned values (seed=123, independent of `random_seed` so
  detection randomness never perturbs emitter reproducibility).
- AdaptiveEvasiveEmitter currently behaves identically to
  FrequencyAgileEmitter by design (Stage 1 scope) — its
  `detection_count`/`recent_detection_times`/`evasive_mode` fields exist
  but are unused until a scheduler exists (later stage wires reactions).

## Working conventions

- This is a **non-git** local folder as of Stage 0. If a stage's work
  should be versioned, ask before running `git init`.
- Keep responses and diffs scoped to the current stage only.
- When resuming a session: read this file + PROJECT_SPEC.md's §6 stage
  list, check "Current stage" above, and confirm with the user what to
  build next rather than re-scanning the whole repo.
