# NBA Draft AI — Project State & Handoff

> **Purpose of this file.** A complete, self-contained snapshot of the project so another
> assistant (or developer) can understand its full state **without reading the source**. It maps
> every requirement from the original spec (`instructions.md`) to how it was solved, documents
> each module, reports results, and records the operational details and gotchas needed to work on
> it. Written 2026-06-25.

---

## 0. Quick orientation

- **What it is:** a human-in-the-loop decision-support system for NBA draft decisions. For each
  prospect it projects NBA impact, estimates contribution to winning, evaluates fit with a
  specific roster, and quantifies uncertainty.
- **Philosophy:** *validity over spectacle.* The architecture is built defense-first against the
  inferential traps that ruin draft models (leakage, survivorship, etc.).
- **Two modes:**
  1. **Fixture-first** — the whole system runs on deterministic *synthetic* data, so the
     architecture is verifiable with zero data access. (All 162 tests use this.)
  2. **Real data** — a working pipeline pulls `nba_api` + CollegeBasketballData.com, builds real
     labels, and trains/evaluates real models.
- **Headline real result:** on 2011–2020 drafts (389 trainable players), a tuned gradient-boosting
  **production model** (draft pick + college production + age + Combine) **beats the
  draft-position baseline** at ranking prospects by realized NBA impact: **Spearman 0.336 vs
  0.233**. Progression: consensus 0.233 → public data alone 0.302 → consensus + data 0.336.
- **Status:** all 12 spec phases complete + real-data extension. 162 tests pass; `ruff` and
  `mypy --strict` clean; GitHub Actions CI green. Private repo
  `github.com/fernandogg2004/NBA-Draft-AI`, tagged `v0.1.0` with a published release.

### Environment & conventions (important for a fresh session)

- **OS:** Windows 11. Project root: `C:\Users\ferna\Desktop\NBA Draft AI`.
- **Python:** venv at `.venv` (local Python 3.14; CI uses 3.12; package requires ≥3.11). Invoke as
  `./.venv/Scripts/python.exe ...` from the Bash tool, or `.venv\Scripts\python.exe` in PowerShell.
- **Data stack:** **Polars** is the primary dataframe library (not Pandas). `scikit-learn`,
  XGBoost/LightGBM, lifelines, PyMC (installed but Bayesian uses sklearn `BayesianRidge`), SHAP,
  FastAPI, Streamlit, MLflow, Optuna.
- **Lint/types:** `ruff` (line length 100) + `mypy --strict` (target py3.12). Both must stay clean.
- **Tests:** `pytest` — 162 tests, **all offline** (external services are injected/faked), so the
  suite is deterministic and CI-safe.
- **Secrets:** the CollegeBasketballData API key is read from env var `CBD_API_KEY`. It must never
  be written to a tracked file. (`.claude/settings.local.json` is gitignored because the harness
  logged commands there.)
- **gitignored:** `.venv/`, `data/{raw,interim,processed,external}/` (API caches), `artifacts/`,
  `experiments/*.db`, `mlartifacts/`, `.claude/settings.local.json`.

### One-command entrypoints (`scripts/`)

| command | what it does | needs |
|---|---|---|
| `python scripts/reproduce.py` | smoke: temporal split → baseline → metrics (synthetic) | core |
| `python scripts/run_eda.py` | EDA report (+ plots) on dev set → `artifacts/eda/` | `[eda]` for plots |
| `python scripts/run_modeling.py` | baseline vs ridge vs gbm (synthetic) → `artifacts/modeling/` | `[models]` |
| `python scripts/run_evaluation.py` | calibration + error analysis + comparison | `[models]` |
| `python scripts/run_pipeline.py` | full MLOps pipeline (synthetic): integrate→eval→train→register→drift | `[models,mlops]` |
| `python scripts/run_ingest.py` | **live** raw NBA pull (local only) | `[ingest]` |
| `python scripts/verify_cbd.py` | verify CBD key + endpoints, print schema | `[ingest]`, `CBD_API_KEY` |
| `python scripts/run_real_pipeline.py` | **live** real end-to-end (nba_api + CBD) | `[ingest,models]`, `CBD_API_KEY` |

---

## 1. Mapping to the original spec (`instructions.md`)

### Role & objective
Delivered exactly: a decision-support tool (not a scout-replacement) that projects production,
estimates winning contribution, evaluates roster fit, and quantifies uncertainty.

### Critical domain considerations — how each of the 8 risks is handled

1. **Temporal leakage** → All validation is temporal: `holdout_split` (untouchable recent
   classes) + `walk_forward_folds` (train years strictly < val years; raises `LeakageError`
   otherwise). All learned preprocessing is fit on the train fold only via `FoldPreprocessor`.
   A guard (`assert_pre_draft_safe`) rejects post-draft columns from the feature matrix.
2. **Survivorship bias** → Hurdle/two-part target: `EV = P(reach)·E(impact|reach) +
   (1−P(reach))·replacement`. `P(reach)` is fit over *all* prospects; conditional impact only on
   players with stable minutes, labeled as conditional.
3. **Opportunity confound** → Rate-based features (per-100 / per-40, true shooting, usage); the
   draft pick is kept as a *baseline* and only added to the explicit "production" model, not
   conflated with talent.
4. **Era effects** → eBPM is league-centered within season; dynamic strength-of-schedule is
   standardized within `(league, season)`; inter-league translation re-expresses production on a
   reference scale.
5. **Small, heterogeneous samples** → Simple/regularized models first; strong regularization;
   Optuna tuning inside CV; honesty that rare tiers are uncertain. Age handled as a first-class
   feature.
6. **Fat tails / high variance** → Uncertainty is mandatory: prediction intervals (quantile,
   conformal, Bayesian, ensemble) + per-prospect outcome-tier *distributions* (bust…superstar).
7. **Dataset size** → Modest model complexity justified; the tuned GBM converged to shallow
   (depth 2), slow-learning, many-tree config = strong regularization.
8. **Transfer Portal / NIL volatility** → Dynamic (per-season) strength-of-schedule, plus
   sequence features: `sos_jump` (conference jump), `usage_change` (role change), and
   `efficiency_held_up` (did TS survive a rise in competition). NIL/CBA rules treated as
   year-varying config to verify, not hard-coded.
9. **Data disparity across leagues** → "Not measured" is kept as `null` (never zero); the
   `LeakageSafeImputer` fills from comparable leagues with flags + an uncertainty `sd`. The system
   works for an international prospect with only basic stats + Combine.

### Methodology — Phases 0–12 (all complete)

- **Phase 0 — Targets** (`config/targets.yaml`, `src/nba_draft/targets/definitions.py`). Discussed
  and confirmed: impact spine = box-score (BPM/VORP family); hybrid outcome tiers (honors for top,
  bands below); horizon debut-anchored, capped at 4 years. Targets: reach, peak impact, cumulative
  value, outcome-tier distribution, longevity, Real Surplus Value. Functions: `reached_role`,
  `peak_impact`, `cumulative_value`, `outcome_tier`, `unconditional_value`, `is_label_resolved`
  (explicit right-censoring of unfinished classes).
- **Phase 1 — Acquisition** (`config/sources.yaml`, `src/nba_draft/ingestion/`). Source ToS/robots
  verified via web. **Excluded** (robots/ToS): Basketball-Reference, Bart Torvik, KenPom.
  **Primary:** stats.nba.com via `nba_api` (local-only). **College:** CollegeBasketballData.com
  (Bearer key). Built source-agnostic core: registry (gate on enabled/scraping_allowed), caching
  (content-addressed + provenance sidecar with license + SHA-256), rate limiting, retries, robots
  respect (fails closed).
- **Phase 2 — Cleaning/integration** (`src/nba_draft/cleaning/`). Name/league normalization,
  union-find entity resolution (strong blocking on name+draft-year + guarded fuzzy), explicit
  imputation strategy (leakage-safe, comparable-league, flags + uncertainty), versioned master
  dataset with manifest.
- **Phase 3 — EDA** (`src/nba_draft/eda/`, `scripts/run_eda.py`). Distributions + missingness,
  rank correlations, age-vs-success, league-vs-success, base rates, bias-by-league. Runs on the
  **development set only** (never the holdout).
- **Phase 4 — Feature engineering** (`src/nba_draft/features/`). Stateless transforms + learned
  context model (translation + dynamic SoS), with a leakage guard. Transfer-Portal/NIL sequence
  features included.
- **Phase 5 — Validation** (`src/nba_draft/validation/`, `docs/validation.md`). Formalized the
  temporal protocol: untouchable holdout, walk-forward folds, the `FoldPreprocessor` firewall, the
  `walk_forward_evaluate` runner, and out-of-fold predictions.
- **Phase 6 — Modeling** (`src/nba_draft/models/`). Baselines → GBM → survival; Optuna tuning
  inside the temporal CV; per-target families.
- **Phase 7 — Evaluation** (`src/nba_draft/evaluation/`). Ranking + calibration metrics, error
  analysis by segment, honest baseline comparison, out-of-fold predictions.
- **Phase 8 — Fit** (`src/nba_draft/fit/`, `config/cba_rules.yaml`). Archetypes, need, synergy,
  lineup Net-Rating simulation, Real Surplus Value under the CBA/aprons.
- **Phase 9 — Uncertainty** (`src/nba_draft/uncertainty/`). Quantile / conformal / Bayesian /
  ensemble intervals + scenario-tier distributions.
- **Phase 10 — Interpretability** (`src/nba_draft/interpretability/`). SHAP, permutation
  importance, partial dependence, counterfactuals.
- **Phase 11 — Deployment** (`src/nba_draft/service/`, `api/`, `dashboard/`). FastAPI service +
  Streamlit GM dashboard over a shared service layer.
- **Phase 12 — MLOps** (`src/nba_draft/mlops/`, `dvc.yaml`, `params.yaml`). MLflow tracking, model
  registry, PSI drift, retraining policy, one-command reproducible pipeline.

### Fit modeling, uncertainty, interpretability
All three dedicated sections of the spec are implemented — see modules below. Fit outputs are
explicitly flagged `exploratory=True`; CBA projections and the $-per-win conversion are surfaced
as assumptions.

### Technical requirements
Python, Polars-first, the suggested stack, typed + docstring'd + logged code, tests for critical
logic, centralized config, fixed seeds, declared environment (`pyproject.toml` extras), clean
modular structure. ✅

### Use of skills
`find-skills` was consulted at the start of every phase. The only skill adopted was
`developing-with-streamlit` (used for the dashboard). Everything else was assessed and built by
hand (the available skills were low-install/generic or off-domain).

### Deliverables (all present)
Reproducible pipeline; trained/evaluated models with temporal validation + baseline comparison;
fit module; uncertainty + interpretability; API + dashboard; MLOps; documentation
(`README.md`, `docs/`).

---

## 2. Module-by-module reference

> Paths are under `src/nba_draft/` unless noted. Everything is typed and tested.

### `config.py` + `config/`
Typed pydantic loaders. YAML config files: `config.yaml` (seed, horizon, validation scheme, eval
k's), `targets.yaml` (target/tier definitions), `sources.yaml` (data-source registry + ToS/robots
review + enable gates), `cba_rules.yaml` (cap/aprons/rookie scale per season), `leagues.yaml`,
`normalization.yaml` (name suffixes + league aliases).

### `ingestion/`
- `registry.py` — `Source` model + `load_sources`/`get_source`. A source can't be fetched unless
  `enabled` and (for scrapes) `scraping_allowed`.
- `http.py` — `RateLimiter`, `RobotsChecker` (fails closed), `PoliteClient` (cache → policy check →
  rate-limit → retries w/ backoff), `make_requests_fetcher(headers)` (adds e.g. Bearer auth). The
  network fetcher is injectable, so tests are offline.
- `cache.py` — `FileCache`: content-addressed payloads + `.prov.json` sidecars.
- `provenance.py` — `Provenance` (source, url, UTC time, license, SHA-256, n_bytes).
- `nba_stats.py` — `NbaStatsIngester`: cached/rate-limited wrappers over `nba_api` endpoints —
  `draft_history`, `draft_combine_stats`, `player_season_stats` (Base/Advanced), `player_info`
  (CommonPlayerInfo → birthdate).
- `college_bb_data.py` — `CollegeBasketballDataIngester`: Bearer-auth (env `CBD_API_KEY`),
  endpoints `player_season_stats`, `teams`, `team_roster`. Gate = API-key presence.
- `parse.py` — pure JSON→Polars parsers: `parse_draft_history`, `parse_combine`,
  `parse_player_season` (joins Base+Advanced → per-100), `parse_player_info` (birthdate),
  `parse_cbd_player_season` (flattens nested fields; per-40; scales percents→fractions).

### `cleaning/`
- `normalize.py` — `normalize_name`, `name_match_key` (accent/suffix/order folding),
  `normalize_league`.
- `entity_resolution.py` — `resolve_entities`: union-find, strong blocking on
  `(name_key, draft_year)` + guarded fuzzy (difflib ≥ 0.88, birthdate-compatible); deterministic
  `player_id`.
- `schema.py` — column groups (`BASIC_COLUMNS`, `ADVANCED_COLUMNS`, `COMBINE_COLUMNS`,
  `IMPUTABLE_COLUMNS`); `add_missing_flags`.
- `imputation.py` — `LeakageSafeImputer` (fit/transform): comparable-league group means with
  global fallback; sets `<col>_imputed` flags and `<col>_impute_sd`. **Fit on train fold only.**
- `master.py` — `build_master`: entity-resolve → normalize → concat → flags → 3 versioned tables
  (`identity`, `prospect_season`, `combine`) + `manifest.json` with a deterministic content hash.

### `targets/`
- `definitions.py` — `TargetConfig` + `PlayerOutcome`/`SeasonStat` + the label functions listed in
  Phase 0. `unconditional_value` combines the hurdle parts.
- `impact.py` — **eBPM** (`estimated_bpm`): a transparent, league-centered box plus-minus *proxy*
  (documented heuristic weights; NOT BBRef BPM). `vorp = (eBPM+2)·minutes/(240·82)` (exact).
  `add_impact_metrics`; `pie_rank_agreement` (validation vs official PIE).
- `outcomes.py` — `build_player_outcomes` (NBA seasons + draft history → `PlayerOutcome`),
  `build_labels_frame`, `season_str_to_year`.

### `features/`
- `transforms.py` — `add_stateless_features` (`versatility_index`, `playmaking_share`,
  `scoring_load`), `add_combine_features` (`agility_score`, `leanness_score`, `explosiveness`),
  `sequence_features` (`sos_jump`, `usage_change`, `ts_change`, `efficiency_held_up`,
  `pts_yoy_delta`, `n_pre_draft_seasons`).
- `learned.py` — `LeagueSeasonContextModel` (fit/transform): inter-league translation
  (`<stat>_translated`) + dynamic SoS (`sos_z`). Fit on train only.
- `assembler.py` — `assemble_prospect_features` (stateless, safe once), `build_feature_matrix`
  (adds fitted context), `primary_pre_draft_season`, `assert_pre_draft_safe`,
  `FORBIDDEN_POST_DRAFT_COLUMNS`.

### `validation/`
- `temporal.py` — `holdout_split`, `walk_forward_folds`, `TemporalFold`, `LeakageError`.
- `pipeline.py` — `FoldPreprocessor`: bundles context model + imputer + train-median backfill;
  guarantees a null-free matrix; fit on train only.
- `runner.py` — `walk_forward_evaluate` (the modeling protocol), `walk_forward_predictions`
  (out-of-fold), `make_data_split`/`DataSplit`, `EvaluationReport`, `default_metrics`, `Estimator`
  protocol.

### `models/`
- `base.py` — `Estimator` protocol (shared, avoids import cycles).
- `zoo.py` — `mean/ridge/lasso/elasticnet_regressor`, `gbm_regressor` (XGBoost→LightGBM→sklearn
  fallback; coerces float hyperparams to int for Optuna), `logistic_classifier`/`gbm_classifier`
  via `ProbaAdapter` (so probabilistic targets score with calibration metrics).
- `baseline.py` — `DraftPositionBaseline` and `DraftPositionEstimator` (runs the baseline through
  the same runner for honest comparison).
- `tuning.py` — `tune_estimator`: Optuna TPE optimizing the mean walk-forward metric (leakage-safe).
- `survival.py` — `CoxSurvivalModel` (lifelines) + `concordance`. (Builds the pandas frame manually
  because `pyarrow` is not installed — see gotchas.)

### `evaluation/`
- `metrics.py` — `spearman_corr`, `kendall_tau`, `top_k_hit_rate`, `rmse`, `brier_score`,
  `expected_calibration_error` (NumPy-only).
- `calibration.py` — `calibration_table` (reliability curve).
- `error_analysis.py` — `residual_segments` (bias/MAE/RMSE per segment), `largest_errors`.
- `comparison.py` — `compare_models` + `make_spec` (uplift vs a named baseline).

### `fit/`
- `types.py` — `Player`, `SKILL_DIMS` (6 functional skills), `TeamContext`.
- `archetypes.py` — `ArchetypeModel` (KMeans on standardized skills).
- `need.py` — `functional_need` (supply-based gaps).
- `synergy.py` — `synergy_score` (complementarity − redundancy).
- `lineup.py` — `simulate_net_rating`, `lineup_upgrade` (replace weakest link → before/after/delta
  + GM narrative). Transparent proxy (Σ impact + spacing/rim synergy), not a calibrated RAPM model.
- `financial.py` — `CBAConfig`/`load_cba`, `rookie_cost_total`, `real_surplus_value`,
  `apron_pressure_multiplier` (over second apron = 1.5×).
- `score.py` — `player_team_fit` → `FitResult` (component-wise, narrative, `exploratory=True`,
  assumptions list).

### `uncertainty/`
- `quantile.py` — `QuantileGBM` (pinball loss per quantile; sorted to remove crossing).
- `conformal.py` — `SplitConformalRegressor` (finite-sample marginal coverage).
- `bayesian.py` — `BayesianLinearModel` (BayesianRidge predictive mean + std).
- `ensemble.py` — `BootstrapEnsemble` (samples → intervals + scenario probs + floor/ceiling).
- `scenarios.py` — `scenario_probabilities_from_normal/_from_samples`, `ceiling_floor`,
  `interval_coverage`.

### `interpretability/`
- `attribution.py` — `ShapExplainer` (global + additive local), `permutation_importance_table`
  (model-agnostic).
- `pdp.py` — `partial_dependence`.
- `counterfactual.py` — `counterfactual_single_feature`, `greedy_counterfactual`.

### `service/`, `api/`, `dashboard/`
- `service/board.py` — `DraftBoardService` (`rank`, `explain`, `fit_for_team`),
  `build_demo_service` (trains on synthetic dev), `prospect_to_player` (features→skills bridge,
  exploratory), `TIER_EDGES`/`TIER_LABELS`.
- `api/main.py` — FastAPI: `GET /health`, `GET /prospects`, `GET /explain/{player_id}`,
  `POST /fit`. Thin over the service; tested with `TestClient`.
- `dashboard/streamlit_app.py` — GM dashboard (sidebar: roster style / cap situation / pick;
  ranked board; per-prospect metrics + tier distribution + SHAP; fit narrative).

### `mlops/`
- `tracking.py` — `ExperimentTracker` (MLflow; graceful no-op if disabled/missing; default backend
  is **SQLite** `experiments/mlflow.db` because MLflow 3.x deprecated the file store).
- `registry.py` — file-based model registry: `register_model`, `load_model` (by version/stage/
  latest), `list_models`, `promote_model`; each version stores `model.joblib` + `meta.json`
  (metrics, feature columns, data version, stage) + a `registry.json` index.
- `drift.py` — `population_stability_index` (PSI), `feature_drift_report`.
- `retraining.py` — `RetrainingPolicy` + `should_retrain` (annual cadence + off-cycle on drift).
- `pipeline.py` — `run_pipeline`: integrate → evaluate → train → register → monitor (synthetic).

### `realdata/` (the real-data extension, beyond the original spec)
- `build.py` — `pull_real_frames` (pull+parse+impact), `build_real_modeling_table` (one row per
  drafted player: Combine + pick + real labels + `resolved`/censoring flag, + college + age when
  available), `run_real_pipeline` (pull → labels → temporal CV → Optuna-tune GBM → register best).
- `college.py` — `link_college_features` (match drafted player → CBD college season by name +
  draft year, school tiebreak) + `COLLEGE_FEATURE_COLUMNS` (per-40 production + TS/usage/efg/3P% +
  off/def/net rating + PORPAG + win_shares_per40).
- `age.py` — `age_at_draft` (years to late-June draft day) + `pull_player_ages` (resilient per
  player; failures → null age → imputed).

---

## 3. Data, in detail

- **stats.nba.com via `nba_api`** (primary NBA source; **run locally** — cloud/datacenter IPs are
  banned). Endpoints used: `DraftHistory`, `DraftCombineStats`, `LeagueDashPlayerStats`
  (Base+Advanced), `CommonPlayerInfo` (birthdate). License: library MIT; data under NBA.com ToU.
- **CollegeBasketballData.com** (NCAA pre-draft production). Bearer-token API, free key at
  `collegebasketballdata.com/key`. Endpoints: `/stats/player/season` (one call returns the whole
  season, ~9,800 players), `/teams`, `/teams/roster`. Fields include per-game/total counting
  stats, `trueShootingPct` (fraction), `usage`/`effectiveFieldGoalPct` (percent), `offensiveRating`/
  `defensiveRating`/`netRating`, `PORPAG`, nested `winShares`/`rebounds`/shot splits.
- **Excluded** (verified via web 2026-06-25): Basketball-Reference and Bart Torvik (`robots.txt`
  disallows their data paths), KenPom (paid). Because of this, **BPM/VORP are computed in-house**
  as the eBPM proxy + exact VORP.
- **eBPM validation:** Spearman vs the official NBA **PIE** metric ≈ **0.92** on 2023-24 (572
  player-seasons) — rankings are trustworthy; magnitudes are approximate.
- **CBA figures** (`config/cba_rules.yaml`): 2025-26 official (cap $154.647M, tax $187.895M, apron
  1 $195.945M, apron 2 $207.824M); 2026-27 projected (flagged). Rookie scale picks 1–14
  (1–10 verified); valuation knobs (2.7 wins/VORP, $3.5M/win) flagged as assumptions.
- **Synthetic fixtures** (`src/nba_draft/data/fixtures.py`): `make_synthetic_prospects`
  (single-source w/ target, fat-tailed), `make_multisource_fixture` (entity-resolution test),
  `make_feature_fixture` (multi-season/transfer cases). Used by every test and the demo service.

---

## 4. Results

**Real pipeline** — 2011–2020 drafts; 600 drafted, **389 trainable** (resolved & reached); 21
features (Combine + college + age) or 22 (+ draft pick); target = peak eBPM over years 1–4,
conditional on reaching; strictly temporal walk-forward CV.

| model | features | Spearman ↑ | RMSE ↓ |
|---|---|---|---|
| production (tuned GBM) ⭐ registered | pick + college + age + combine | **0.336** | 0.697 |
| production (ridge) | pick + college + age + combine | 0.314 | 0.704 |
| data-only (tuned GBM) | college + age + combine | 0.302 | 0.704 |
| data-only (ridge) | college + age + combine | 0.302 | 0.709 |
| gbm (untuned) | college + age + combine | 0.257 | 0.769 |
| draft-position baseline | pick only | 0.233 | 0.700 |

- **Progression:** consensus 0.233 → public data alone 0.302 → consensus + data 0.336.
- **Tuned GBM config:** `n_estimators≈549, learning_rate≈0.0101, max_depth=2` (shallow, slow,
  many trees = strong regularization — appropriate for ~389 rows).
- **Feature value ladder (data-only):** Combine alone ≈ 0.00 → + college production ≈ 0.19 → + age
  ≈ 0.30. Age was the decisive lever; GBM exploited its nonlinearity (ridge benefited less).
- **Synthetic demos** (tooling verification only, no basketball meaning) live under `artifacts/`.

Caveat: one target, one era window, ~389 players — the **direction/ordering is robust**; treat the
exact magnitudes as indicative.

---

## 5. Operational details & gotchas (read before touching real data)

- **`nba_api` is local-only.** stats.nba.com bans cloud IPs. Runs on a residential machine.
- **~9% of `CommonPlayerInfo` birthdate calls fail** with `KeyError 'resultSet'` (nba_api raises
  on a malformed/empty payload). This is **structural and reproducible per player-id** — those
  players **never reached the NBA** (no profile). Verified: **0 of 389 trainable players** have a
  null age, so the impact model's age data is complete. The pull is resilient (per-player
  try/except → null age → imputed); failures aren't cached so they retry but keep failing.
- **`pyarrow` is NOT installed.** `polars.to_pandas()` therefore fails; the survival model builds
  its pandas frame manually from numpy. If you need pandas conversion elsewhere, install pyarrow or
  build frames manually.
- **MLflow 3.x deprecated the file store** → the tracker defaults to SQLite
  (`sqlite:///experiments/mlflow.db`). View runs with
  `mlflow ui --backend-store-uri sqlite:///experiments/mlflow.db`.
- **Caching:** every external response is cached under `data/raw/<source>/`. Re-runs are cheap and
  resume from cache. The first real run pulls ~25 endpoint calls + ~600 birthdate calls
  (~6–10 min at polite spacing); subsequent runs are fast.
- **Windows line endings:** git warns LF→CRLF on commit — harmless.
- **Production model uses `draft_pick` as a feature** — valid (it's known at draft time and encodes
  scouting the box score can't). At inference you'd score a prospect at a candidate slot. The
  data-only models exist precisely to show public data alone already matches the consensus.

---

## 6. Repository, CI, release

- **Repo:** `github.com/fernandogg2004/NBA-Draft-AI` (private). Authenticated via `gh` CLI
  (installed at `C:\Program Files\GitHub CLI\gh.exe`; account `fernandogg2004`).
- **CI:** `.github/workflows/ci.yml` runs `ruff` + `mypy --strict` + `pytest` on every push/PR to
  `main` (Python 3.12, all extras). Currently **green**.
- **Release:** tag `v0.1.0` + a published GitHub Release with notes.
- **Key commits:** `be0345a` initial; `f6cd277` CI; `807cdef` comprehensive README.
- **Test count:** 162. **Lint/type:** clean.

---

## 7. What's done beyond the spec

- A complete **real-data pipeline** (the spec only required the architecture; this adds working
  ingestion → labels → models on real NBA + college data).
- **In-house eBPM/VORP** computation (since BBRef can't be scraped), PIE-validated.
- **CollegeBasketballData** integration + **age-at-draft** source + entity resolution linking
  college players to drafted players.
- A **production model** that fuses consensus + data and beats the baseline.
- **Optuna tuning** wired into the real pipeline; **best-model auto-registration**.
- **GitHub repo + CI + tagged release**; comprehensive `README.md` and `docs/`.

---

## 8. Suggested next steps (not yet done)

- **Backfill the ~9% missing ages** for non-reach players via an alternate source — only needed if
  building a **reach-classification** model (the impact model is unaffected).
- **Honors source** (All-Star/All-NBA) so the top outcome tiers aren't BPM-band-only.
- **Wider/more windows** and separate tuning of the production feature set.
- **Dynamic SoS from CBD** team ratings; **international** pre-draft features (EuroLeague feeds).
- **Deploy a real-data service** (point `service.build_demo_service` at the rebuilt master
  dataset) and add the AllStar/honors + a reach classifier to complete the hurdle in production.
- **Branch protection** requiring CI to pass; Dependabot.

---

## 9. How to verify the current state quickly

```bash
# from project root, with the venv active or via ./.venv/Scripts/python.exe
pytest                                          # expect: 162 passed
ruff check src tests scripts api dashboard      # expect: All checks passed!
mypy                                            # expect: Success: no issues found
python scripts/run_pipeline.py                  # synthetic end-to-end (needs [models,mlops])
```

For the real result, set `CBD_API_KEY`, install `[ingest,models,explain]`, and run
`python scripts/run_real_pipeline.py` on a local machine (writes
`artifacts/real_pipeline/real_run_summary.json` with the leaderboard above).
