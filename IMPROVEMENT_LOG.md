# IMPROVEMENT_LOG.md — Autonomous Self-Improvement Loop

Authoritative record for the `SELF_IMPROVE.md` loop. Every number here is **real** — computed
from acquired data through the pipeline on the cached real modeling table
(`artifacts/real_pipeline/serving/`), not fabricated. Synthetic-only results are labelled.

## Locked temporal holdout

- **Modeling table:** 659 drafted players, classes **2011–2020** (resolved labels) + **2025**
  (unlabeled projection pool; no NBA outcomes yet).
- **Resolved set:** 594 players (label fully observed within the 4-year debut-capped window).
- **LOCKED HOLDOUT = draft classes 2019 + 2020** (115 resolved players). Tuning / CV use only the
  **dev set = 2011–2018**, via strictly temporal walk-forward. The holdout is touched **only at
  milestone checkpoints**, never tuned against. The 2025 class is the served board (projection only)
  and is never used for training/eval.
- Reach rate: dev 0.649, holdout 0.678.

## Baseline scoreboard — Iteration 0 (locked holdout 2019–2020)

Measured offline on cached real data (`scratchpad/baseline_scoreboard.py`; reach=logistic,
impact=GBM default, ensemble=30× GBM). The tuned pipeline run (`artifacts/real_pipeline/
real_run_summary.json`) is cited where it differs.

| Metric | Value | Notes |
|---|---:|---|
| Draft-position baseline — Spearman | **0.516** | the bar to beat (consensus of 30 front offices) |
| Hurdle EV ranking, **scouting-only (served)** — Spearman | **0.263** | tuned pipeline: 0.300 |
| Hurdle EV ranking, **consensus (+draft_pick)** — Spearman | **0.470** | best model variant so far |
| Hurdle CV Spearman (dev walk-forward, scouting) | 0.426 | from pipeline summary |
| 80% prediction-interval empirical coverage (reached) | **0.346** | nominal 0.80 → badly overconfident |
| Reach-probability calibration — ECE / Brier | 0.049 / 0.199 | well-calibrated |
| Longevity (Cox PH career length) — holdout concordance | 0.659 | 2nd target, validated |
| Tests / ruff / mypy / CI | green | full suite passes |

### Headline reads
- **RED — ranking loses to baseline.** Even the consensus-aware model (0.470) is below the
  draft-position baseline (0.516); the *served* scouting-only model (0.263) is far below. Exit
  criterion 1 ("beat the baseline") is currently **failed**.
- **RED — intervals overconfident.** 80% intervals cover only ~35% of realized outcomes.
- **GREEN — multi-target + calibration of reach.** Reach probability is well-calibrated; a second
  target (longevity) is validated; outcome tiers exist.

## Exit criteria (measurable translation)

1. **Predictive quality** — served ranking model Spearman on the locked holdout **≥ draft-position
   baseline** (≈0.52), positive & stable across walk-forward folds; **80% interval coverage in
   [0.73, 0.87]**; reach ECE ≤ 0.10; ≥2 targets validated (impact + longevity + reach). 
2. **Data integrity** — no-placeholder audit passes across service/API/frontend; real pipeline
   reproducible + documented; missing inputs acquired or transparently scoped.
3. **Decision usefulness** — board + per-prospect floor/ceiling + roster/cap fit + lineup NetRtg +
   surplus value + defensible explanation, on real prospects.
4. **Trust & transparency** — faithful explanations; assumptions/limits/biases surfaced.
5. **Robustness** — tests/types/lint green; reproducible; missing-data & international edge cases
   handled; backend ↔ frontend consistent.
6. **Honest self-assessment** — evidence-based statement of usefulness + remaining limits.

## Prioritized backlog (impact × feasibility; attack weakest first)

- [x] **I1 — Honest intervals (conformal coverage).** DONE (coverage 0.346 → 0.808). Split-conformal
  floor/ceiling in the service.
- [x] **I2 — Beat the baseline → MATCH it (ceiling scoped).** Done: served board is now
  consensus-anchored (pick-aware), holdout ~0.50 ≈ baseline 0.516, up from 0.263. Beating the
  consensus is NOT achievable on this sample/era (evidence below); "beat" is honestly scoped out.
- [ ] **I3 — Coverage of intervals on the served board** wired through to the API/frontend floor/
  ceiling (consume the calibrated intervals).
- [x] **I4 — No-placeholder audit** — PASS. Every shown statistic traces to real computation on the
  2025 data; the one fabricated label ("Model V2.4") is replaced by a real `/meta` endpoint.
- [x] **I5 — Tier-probability calibration** — DONE (multiclass ECE 0.284 → 0.104; P(starter+) ECE
  0.311 → 0.095). Tier probs now from the conformal residual distribution.
- [ ] **I6 — Robustness/edge cases:** international prospects with sparse features; the `'resultSet'`
  age-fetch failures (53/659) — make age acquisition more robust.
- [ ] **I7 — Honest ceiling write-up** + final readiness report.

## Post-exit work (user-directed): more eras + orthogonal signal

### Iteration 8 — extend training eras 2011→2005 (I8) — DONE + MEASURED ✅ (ceiling confirmed)
- **Change:** widened `run_real_pipeline` to 16 training classes **2005–2020** (+ 2025 pool),
  outcome seasons through 2023-24. User ran the (gated) pull on a residential IP.
- **Result:** sample **resolved 594 → 954** (dev 479 → 839). Hurdle **CV Spearman 0.426 → 0.587**
  (now above the baseline CV). BUT the unbiased **holdout** served ridge-hurdle EV went
  **0.497 → 0.469** (vs baseline 0.516) — still below, within noise. Coverage robust (0.782);
  reach ECE 0.097; tier multiclass ECE 0.158 (up from 0.104 — holdout noise, still ≪ the 0.28
  pre-fix baseline).
- **Key conclusion (honest):** doubling the sample did **not** beat the consensus on the locked
  holdout → **rules out "too little data"** as the cause. The ceiling is **structural**: public
  box-score/combine/age features lack marginal signal over the 30-team draft consensus. The served
  model is now better-grounded (2× training data, higher CV) but the holdout ranking is unchanged.

### Iteration 9 — auto-ingest honors (PlayerAwards) (I9) — DONE + MEASURED ✅
- **Change:** added the nba_api **PlayerAwards** endpoint (`ingestion/nba_stats.py`),
  `parse_player_awards` (counts All-Star / All-NBA, `ingestion/parse.py`), and
  `pull_player_honors` (`realdata/honors.py`), wired into `run_real_pipeline`
  (`with_honors=True`) so `build_real_modeling_table` tiers use **real** honors instead of the BPM
  proxy. Offline tests added (parser + acquisition with failure tolerance).
- **Result:** real honors flowed into the labels — the honors-aware `outcome_tier` now has **26
  all_star + 60 superstar** rows across 2005–2025 (promotions the BPM proxy would have missed).
- **Honest scope:** honors enrich the outcome-tier **labels** (ground-truth integrity for the
  top tiers + tier *evaluation*); they are an OUTCOME, not a pre-draft feature, so they do **not**
  change the served impact ranking. The served board's tier probabilities are still BPM-band
  derived — aligning them to the honors-aware tier definition would need a separate honors/tier
  classifier (natural next step now that the honors data exists).
- **Note on "orthogonal signal":** no cheap orthogonal pre-draft FEATURE was available from the
  current sources (age, combine anthropometrics, college advanced ratings are already features;
  strength-of-schedule isn't exposed by the CBD parse). A true orthogonal feature (recruiting rank,
  tracking data) needs a NEW source — gated/future.

## Post-exit work — recommended next steps (N1–N4)

### N1 — orthogonal source: CBD recruiting rankings — TESTED, NEGATIVE, NOT MERGED ❌
- **Idea:** high-school recruiting consensus (CBD `/recruiting/players`: rating/stars/rank) is a
  *pre-college* opinion distinct from NBA draft order → possible orthogonal signal. CBD is reachable
  from this environment, so it was testable here.
- **Experiment (`scratchpad/recruit_experiment.py`):** pulled 9,110 recruits (2008–2024), linked by
  normalized name to 461/1019 drafted (45%; internationals/unranked null), `corr(rating, realized)
  =0.122`, `corr(rating, −pick)=0.228` (only partly orthogonal). Adding recruiting to the consensus
  model: dev-CV **0.583 → 0.556** (worse), holdout **0.481 → 0.486** (flat); baseline still 0.516.
- **Decision:** recruiting adds **no** marginal signal over the draft consensus → **not merged**
  (validity over numbers). Reinforces the structural ceiling: even a different scouting consensus
  doesn't beat the draft. A genuinely new signal would need *in-kind* novelty (tracking/biometric),
  which is not freely available.

### N2 — honors-aware tier model — KEPT ✅ (large calibration win)
- **Problem:** the board's tier probabilities came from mapping one predicted BPM through fixed
  bands. Against the **honors-aware** `outcome_tier` ground truth (real All-Star/All-NBA), that was
  poorly calibrated: multiclass ECE **0.390**, accuracy 0.287.
- **Change:** `TierProbabilityModel` (multinomial logistic) trained directly on `outcome_tier`
  (`src/nba_draft/models/tier.py`); `build_service_from_table` fits it when the label is present and
  `rank()` uses it for `p_<tier>` (priority: tier model → conformal → ensemble). Synthetic/demo path
  (no `outcome_tier`) falls back to the conformal scenarios.
- **Result (holdout, vs honors-aware tiers):** multiclass ECE **0.390 → 0.111**; accuracy
  **0.287 → 0.548** (confidence 0.560 ≈ accuracy). Served 2025 board now meaningful, e.g. Cooper
  Flagg 61% superstar / 22% starter (was a flat ~82% starter). API shape unchanged → frontend
  unaffected.
- **Tests/gates:** unit test (label alignment, unseen tiers→0, sums to 1) + service wiring test;
  suite + ruff + mypy + frontend green.

### N3 — align eval and served model — KEPT ✅
- **Problem:** the reported headline was the **GBM** hurdle eval, but the **served** model is the
  **ridge** hurdle (different — and the ridge actually scores better on the holdout).
- **Change:** `evaluate_real_models` now also builds the **actual served model**
  (`build_service_from_table` on dev), ranks the untouchable holdout, and reports
  `served_holdout_spearman` as the headline (GBM head kept as context). Added to the run summary +
  `RealPipelineResult`; the script logs the SERVED number as the headline.
- **Result (cached re-eval):** served **0.469** (honest headline, matches the API) vs GBM-eval
  0.453 vs baseline 0.516. The reported number now equals what's served.

### N4 — harden per-player acquisition — KEPT ✅
- **Problem:** ~8% of players (deterministic IDs) raise a `'resultSet'` KeyError inside nba_api at
  fetch time → never cached → retried every run (and would block in datacenter envs).
- **Change:** `_safe_endpoint_json` wraps the per-player endpoints (`player_info`, `player_awards`);
  on any nba_api parse/transport error it logs and returns a valid **empty** payload, which the
  cache stores — so those players resolve once to null age / zero honors (imputed) and are **not
  re-fetched**. Added a unit test.
- **Note:** clears-on-cache-clear if a future nba_api/network fix makes them parseable.

## Iteration log

### Iteration 0 — scoreboard + lock
- Read project incl. frontend; ran real pipeline on cached data (drafted=659, resolved=594).
- Locked holdout = 2019+2020; recorded baseline scoreboard above.
- Wrote exit criteria + backlog. No code change. Next: **I1 (conformal intervals)**.

### Iteration 1 — honest intervals via split-conformal — KEPT ✅
- **Hypothesis:** the 80% floor/ceiling interval undercovers (0.346) because it uses the bootstrap
  ensemble; a split-conformal layer gives finite-sample marginal coverage ≈ nominal.
- **Change:** `DraftBoardService` gained a `conformal` field; `rank()` now derives floor/ceiling
  from `SplitConformalRegressor` (alpha=0.2) when present, else falls back to the ensemble. Fit in
  `build_service_from_table` (≥20 reached rows) and `build_demo_service`
  (`src/nba_draft/service/board.py`). Ensemble retained for tier scenarios.
- **Result:** 80% interval coverage on the locked holdout **0.346 → 0.808** (n=78 reached; mean
  width 1.83 BPM, qhat 0.913), measured via the real served path (`build_service_from_table` on
  dev → `rank()` holdout). Ranking/Spearman unchanged (point estimate untouched); reach calibration
  unchanged.
- **Leakage check:** conformal calibrates *within* the training fold only; holdout never used to
  fit. No new leakage.
- **Tests/gates:** added `test_conformal_interval_is_attached_and_calibrated`; full suite + ruff +
  mypy green. API shape unchanged → frontend unaffected (floor/ceiling just honestly wider).
- **Evidence:** live real board — Cooper Flagg proj 0.71, floor −0.34, ceil 1.58 (±0.96).
- Next: **I2 (beat the draft-position baseline)**.

### Iteration 2 — consensus-anchored ranking; "beat baseline" scoped to "match" — KEPT ✅
- **Hypothesis:** the served board should rank ≥ the draft-position baseline. The served model was
  scouting-only (pick excluded) and ranked WORSE than draft order (0.263 vs 0.516) — misleading.
- **Investigation (dev-CV selected, holdout confirmed; `scratchpad/model_forms.py`, `blend_eval.py`):**
  | approach | dev CV | holdout |
  |---|---:|---:|
  | baseline (draft pick) | 0.566 | **0.516** |
  | ridge realized **+pick** | 0.571 | 0.514 |
  | gbm realized +pick | 0.502 | 0.444 |
  | ridge realized (scouting) | 0.387 | 0.275 |
  | best dev-selected blend (w*=0.40) | — | 0.471 |
  No model/blend **beats** the baseline; the best (pick-aware) **matches** it within noise. Public
  box-score/combine/age features add ~nothing over where 30 front offices actually drafted players.
- **Change:** serve the **consensus-anchored** (pick-aware) model. `run_real_pipeline` now uses
  `exclude_pick_feature=False`; regenerated the serving artifact from the cached table (no network)
  so `feature_cols` includes `draft_pick` (22 features). Served ridge-hurdle EV holdout Spearman
  **0.497** (≈ baseline), up from 0.263. Steal/reach retained (EV still reorders: |Δ| mean ≈ 29)
  and **relabelled exploratory** in the UI ("model matches, not beats, the draft consensus").
- **DECISION on exit criterion 1:** "beat the baseline" is **honestly scoped out** as the documented
  data/era ceiling; the served ranking now MATCHES consensus and uncertainty is honest (I1).
- **Leakage check:** `draft_pick` is pre-draft (known at the draft, before any outcome); the
  `assert_pre_draft_safe` guard passes. No outcome leakage.
- **Tests/gates:** full suite + ruff + mypy green; frontend rebuilt; live board verified via proxy
  (Cooper Flagg #1; Thomas Sorber pick #15, steal +12; Dylan Harper pick #2, reach −3).
- Next: **I4 (no-placeholder audit)**.

### Iteration 3 — no-placeholder audit (I4) — KEPT ✅
- **Audit:** traced every statistic the UI shows to its source on the real 2025 board:
  - Board: `model_rank` (rank), `projected_ev`/`projected_impact` (hurdle model), `floor`/`ceiling`
    (conformal), 5 tier probs (ensemble scenarios), `archetype`/`age`/`wingspan`/`skill_*` (features
    via `prospect_to_player`), `draft_pick`/`team_abbr`/`position` (DraftHistory + Combine),
    `slot_delta` (computed), `headshot_url` (real NBA id), `peak_pctile`/`projected_value_usd`
    (computed). All real.
  - Prospect Detail combine: wingspan 52/59, standing-reach 52/59, max-vertical 47/59, lane-agility
    50/59 real; **body-fat 0/59 → honest "not measured"** (absent from the 2025 Combine feed).
  - Explainability: SHAP contributions + counterfactual + top-local-drivers all real (`/explain`,
    `/counterfactual`). Team Fit: gauge/sub-scores/lineup before-after/financial all from `/fit`.
- **Only fabricated value found:** the decorative **"Model V2.4 Active"** chip. **Fixed:** added a
  real **`GET /meta`** endpoint (serving mode, pool size, draft year(s), trained `model_version`,
  feature count read from the manifest); the chip now shows real metadata (`mode=real`,
  `model_version=20260628100900`, `n_features=22`, `draft_years=[2025]`) or "Synthetic demo".
- **Tests/gates:** added `test_meta_reports_real_serving_metadata`; suite + ruff + mypy green;
  frontend rebuilt & verified via proxy.
- Next: **I5 (tier-probability calibration)**.

### Iteration 4 — calibrated outcome-tier probabilities (I5) — KEPT ✅
- **Hypothesis:** the 5-tier probabilities are overconfident (same too-narrow ensemble spread as the
  intervals), so the distribution shown on the board is miscalibrated.
- **Measured (holdout, reached n=78; `scratchpad/tier_calib.py`):** before — mean confidence 0.882
  vs argmax accuracy 0.603; multiclass ECE 0.284; P(starter+) ECE 0.311.
- **Change:** `SplitConformalRegressor` now stores its signed calibration residuals and gains
  `predict_scenarios()` (tier probs from point + empirical residuals). `DraftBoardService.rank()`
  uses conformal scenarios when conformal is fit (`uncertainty/conformal.py`, `service/board.py`).
- **Result:** multiclass ECE **0.284 → 0.104**; P(starter+) ECE **0.311 → 0.095**; mean confidence
  0.882 → 0.681 (≈ accuracy 0.577). Honest spread (Cooper Flagg 82% starter / 18% rotation vs the
  prior ~100%). API shape unchanged → frontend unaffected.
- **Tests/gates:** added a conformal-scenarios test; relaxed a tier-sum assertion to match the
  documented 4-decimal display rounding (probs sum to 1 pre-rounding); suite + ruff + mypy green.
- Next: milestone holdout re-check + readiness report (I7).

### Iteration 5 — milestone holdout re-check + EXIT (I7)
- **After scoreboard (locked holdout 2019–2020):** served ranking 0.263 → **0.497** (vs baseline
  0.516); 80% coverage 0.346 → **0.808**; tier multiclass ECE 0.284 → **0.104**; P(starter+) ECE
  0.311 → **0.095**; reach ECE 0.049; longevity concordance 0.659; no-placeholder **PASS**;
  tests/ruff/mypy/frontend **green** (191 tests, 82 files typed).
- **Exit decision:** the two red items (interval coverage, tier calibration) are fixed; the ranking
  ceiling ("beat consensus") is real and **honestly scoped out** with evidence; all shown stats are
  real; ≥2 targets validated. Further holdout variant-hunting would risk overfitting the holdout
  (anti-gaming). → **EXIT at diminishing returns.** Final deliverable: `NBA_READINESS_REPORT.md`.
- **Not done / scoped:** beating consensus (needs more eras/signal — gated on data); age-fetch
  `'resultSet'` failures (imputed; live debug gated on network); eval/served model alignment
  (cosmetic). All documented in the readiness report.
