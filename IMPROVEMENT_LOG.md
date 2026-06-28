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
- [ ] **I2 — Beat the baseline.** The served board must rank ≥ the draft-position baseline. Try:
  serve the consensus-aware EV (include `draft_pick`) as the primary ranking; blend model + consensus
  (rank averaging); regularize/tune to stop the GBM adding noise over pick. Keep a separate pick-free
  model only for the "independent scout" steal/reach signal. *High impact, medium feasibility.*
- [ ] **I3 — Coverage of intervals on the served board** wired through to the API/frontend floor/
  ceiling (consume the calibrated intervals).
- [ ] **I4 — No-placeholder audit** end to end (service → API → frontend), re-run after each output
  change; confirm every shown statistic is real on the 2025 board.
- [ ] **I5 — Tier-probability calibration** measured + improved (reliability of the 5-tier
  distribution shown on the board).
- [ ] **I6 — Robustness/edge cases:** international prospects with sparse features; the `'resultSet'`
  age-fetch failures (53/659) — make age acquisition more robust.
- [ ] **I7 — Honest ceiling write-up** + final readiness report.

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
