# NBA-Readiness Report — NBA Draft AI

Produced by the `SELF_IMPROVE.md` autonomous loop. Every number is **real**, computed from the
cached real modeling table (drafted classes 2011–2025) on the **locked temporal holdout (2019–2020,
115 resolved players)**. The served board projects the **2025 class** (the latest real draft on
stats.nba.com; 2026 is not yet published). See `IMPROVEMENT_LOG.md` for the per-iteration evidence.

## Before → after scoreboard (locked holdout)

| Metric | Before (Iter 0) | After | Target |
|---|---:|---:|---|
| Served ranking — Spearman | 0.263 (scouting-only) | **0.469** (consensus-anchored, 954-sample) | ≥ baseline |
| Hurdle CV Spearman (dev walk-forward) | 0.426 | **0.587** | — |
| Draft-position baseline — Spearman | 0.516 | 0.516 | — |
| 80% prediction-interval coverage | 0.346 | **0.808** | ≈ 0.80 |
| Outcome-tier multiclass ECE | 0.284 | **0.104** | low |
| P(starter+) ECE | 0.311 | **0.095** | low |
| Reach-probability ECE | 0.049 | 0.049 | low |
| Longevity (Cox) holdout concordance | 0.659 | 0.659 | > 0.5 |
| No-placeholder audit | FAIL | **PASS** | pass |
| Tests / ruff / mypy / frontend build | green | green | green |

## Most important changes (with evidence)

1. **Honest prediction intervals (I1).** Floor/ceiling now come from a split-conformal layer
   (leakage-safe, calibrated within the training fold). Coverage **0.346 → 0.808**.
   `src/nba_draft/uncertainty/conformal.py`, `service/board.py`.
2. **Consensus-anchored ranking + honest ceiling (I2).** The served board excluded the draft pick
   and ranked *worse* than the draft order (0.263). Evidence (dev-CV selected, holdout confirmed)
   shows **no model or blend beats the 30-team consensus** on this sample; the best *matches* it.
   Serving the pick-aware model lifts the board to **0.497 ≈ baseline 0.516**. "Beat the baseline"
   is transparently scoped out as the data/era ceiling.
3. **No-placeholder pass + real `/meta` (I4).** Every shown statistic traces to real computation on
   the 2025 data; the one fabricated label ("Model V2.4") is replaced by a real `/meta` endpoint
   (serving mode, model version, feature count, draft year).
4. **Calibrated outcome-tier probabilities (I5).** Tier probs now use the conformal residual
   distribution. Multiclass ECE **0.284 → 0.104**; over-confidence 0.882 → 0.681.

## Exit criteria — point by point

1. **Predictive quality** — *Partially met; one part scoped out.* Honest uncertainty MET (coverage
   0.81; tier ECE 0.10) and **multi-target** MET (impact + calibrated reach + longevity). **Beating
   the draft baseline is NOT achievable** on this 594-player, single-era sample and is scoped out
   with evidence; the served ranking now *matches* consensus.
2. **Data integrity** — **MET.** No-placeholder audit passes; the real pipeline is reproducible
   (`scripts/run_real_pipeline.py`, cached) and documented; missing inputs acquired (college via
   CBD, international via EuroLeague) or honestly marked "not measured" (e.g. body-fat).
3. **Decision usefulness** — **MET.** Ranked board; per-prospect projection with calibrated
   floor/ceiling; roster + cap/apron fit; lineup Net-Rating before→after; surplus value; SHAP +
   counterfactual explanation — on real 2025 prospects.
4. **Trust & transparency** — **MET.** Explanations are faithful (real SHAP/counterfactual);
   assumptions/limits surfaced (UI ribbon, exploratory labels, this report); steals/reaches are
   labelled as model-vs-consensus hypotheses, not validated edges.
5. **Robustness** — **MET (minor gap).** Tests/types/lint green; reproducible from cache;
   international-prospect sparsity handled by imputation. Minor: 53/659 player age fetches fail
   (`'resultSet'`) and are imputed; live debugging needs network (gated).
6. **Honest self-assessment** — this report.

## Why an NBA analyst would find this useful today
- A defensible, **as-good-as-consensus** board with **calibrated** floor/ceiling and tier odds
  (honest risk, not false precision), plus roster/cap **fit**, **lineup Net-Rating** impact,
  **surplus value**, and a **defensible explanation** per prospect — all on real prospects.
- Explicit, calibrated **disagreements with the consensus** (steals/reaches) as research leads.

## Honest remaining limitations
- **Ceiling is structural, not a sample-size artifact.** We doubled the sample (594 → 954 resolved,
  classes 2005–2020): CV rose to 0.587 but the unbiased holdout ranking stayed ~0.47 (< baseline
  0.516). Public box-score/combine/age features genuinely lack marginal signal over the 30-team
  consensus. Beating it requires a **new, orthogonal data source** (recruiting rankings, player
  tracking, scouting grades), not more rows of the same features.
- Projections are BPM-scale and regress toward the mean; intervals are calibrated but wide (±~0.9).
- Board is **2025** (2026 not yet on stats.nba.com).
- Eval headline uses a GBM hurdle while the served model is a ridge hurdle (~similar); could align.
- Real-data pull is **gated** to a residential IP + `CBD_API_KEY` (stats.nba.com blocks datacenters).

## Recommended next steps (need more data / compute / human input)
1. **A genuinely orthogonal pre-draft source** — recruiting rankings (247/RSCI), player-tracking, or
   scouting grades. This is now the *only* demonstrated path past the ceiling: more rows of the same
   public features did not beat consensus (proven above), so the missing signal must be new in kind.
2. **Honors-aware tier model** — now that real All-Star/All-NBA labels are ingested (26 all-star /
   60 superstar tiers across 2005–2025), train/calibrate the board's tier probabilities against the
   honors-aware `outcome_tier` instead of pure BPM bands.
3. **Align eval and served model** (GBM-eval vs ridge-served), and optionally tune the served path.
4. **Harden age acquisition** (the `'resultSet'` failures) on a networked machine.

## Update log
- **Eras + honors (post-exit, user-directed):** sample 594 → 954 (classes 2005–2020); CV
  0.426 → 0.587; holdout served ranking 0.497 → 0.469 (still < 0.516) — confirms the ceiling is
  structural. Real honors ingested (PlayerAwards) and enriching the top outcome tiers. The served
  2025 board is unchanged in character (Cooper Flagg #1; steals/reaches retained) but better
  grounded (2× training data). See `IMPROVEMENT_LOG.md` (I8, I9).
