# Design decisions

Why the system is built the way it is. Validity over spectacle throughout.

## Problem framing (Phase 0)
- **Hurdle / two-part target.** `EV = P(reach) · E(impact | reach) + (1−P(reach)) · replacement`.
  Modeling impact only on players who logged NBA minutes conditions on the outcome and over-rates
  busts who never got minutes. The reach gate (fit over *all* prospects) defeats survivorship.
- **Impact spine = box-score BPM/VORP.** Free, deep history, fully reproducible and license-clean.
  Premium metrics (EPM/LEBRON) are optional enrichment, never the maintainable backbone.
- **Outcome tiers = hybrid** (honors for the top tiers, BPM bands below) — cleaner top labels.
- **Horizon = debut-anchored, capped at 4 years** so draft-and-stash internationals aren't marked
  busts during stash years; non-debut within the cap is a definitive non-reach.

## Anti-leakage architecture (Phases 1, 4, 5)
- **Temporal validation only.** `holdout_split` locks the most-recent classes; `walk_forward_folds`
  guarantees every training year strictly precedes every validation year and raises on any breach.
- **Two feature classes.** *Stateless* transforms (a player's own pre-draft stats) are leakage-safe
  by construction. *Learned* steps (inter-league translation, dynamic SoS, imputation, scaling) are
  bundled in `FoldPreprocessor`, **fit on the train fold only**, and applied to validation/test.
- **Leakage guard.** `assert_pre_draft_safe` rejects any post-draft column reaching the matrix.

## Data handling (Phase 2)
- **"Not measured" ≠ zero ≠ bad.** Absent advanced/Combine metrics stay `null`; the imputer fills
  from comparable leagues, flags each filled cell, and propagates an imputation `sd`.
- **Versioned master dataset** with deterministic content-hash version + provenance per source.

## Strength of schedule & era (Phases 2, 4)
- **Dynamic SoS** standardized within `(league, season)` — not a static player attribute — capturing
  Transfer-Portal/NIL jumps in competition as *signal*, plus conference-jump / role-change features.
- **Inter-league translation** re-expresses production on a reference scale so leagues are comparable.

## Modeling (Phase 6)
- **Simple first.** Regularized regression + the draft-position baseline; gradient boosting and
  survival added only where they earn it under temporal CV. Tiny, heterogeneous samples ⇒ strong
  regularization, modest model complexity, Optuna tuning *inside* the CV.

## Uncertainty (Phase 9)
- **Distributions, not points.** Quantile / conformal / Bayesian / ensemble intervals plus a
  per-prospect outcome-tier distribution — a high-ceiling/low-floor prospect is a different decision.

## Fit (Phase 8)
- **Lineup Net-Rating simulation** (concrete "+Y → +Z") + **Real Surplus Value** under the CBA/aprons,
  modulated by cap pressure. Flagged exploratory; CBA projections + $-per-win are explicit assumptions.

## Interpretability (Phase 10)
- **SHAP + permutation importance + PDP + counterfactuals** so an analyst can challenge every call.

## MLOps (Phase 12)
- **One-command pipeline** (`scripts/run_pipeline.py`), MLflow tracking (SQLite backend), a file-based
  **model registry** with stages, **PSI drift** monitoring, and an **annual retraining** policy.
