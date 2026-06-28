"""End-to-end pipeline on REAL nba_api data (run locally).

Pulls draft history + Combine + player seasons (cached), builds real outcome labels, runs a
temporal-CV comparison of Combine-feature models vs the draft-position baseline, and trains +
registers a model. Honest scope: with only nba_api the pre-draft features are Combine + pick.

    python scripts/run_real_pipeline.py
"""

from __future__ import annotations

import os

from nba_draft.ingestion.college_bb_data import CollegeBasketballDataIngester
from nba_draft.ingestion.euroleague import EuroLeagueIngester
from nba_draft.ingestion.http import RateLimiter
from nba_draft.ingestion.nba_stats import NbaStatsIngester
from nba_draft.realdata import run_real_pipeline
from nba_draft.utils.logging import get_logger

log = get_logger("run_real_pipeline")

# Training window: 10 draft classes (2011-2020), whose 4-year debut windows all resolve from the
# outcome seasons below (through 2023-24). PLUS the most-recent class to PROJECT: it has no resolved
# NBA outcomes yet, so it is auto-excluded from training/eval and becomes the served board's pool
# (build_service_from_master holds out the latest draft_year). Adding it only pulls that class's
# endpoints; 2011-2020 stay cache hits. 2026 isn't on stats.nba.com yet, so the latest real draft is
# 2025 — set PROJECT_DRAFT_YEAR to whatever the newest available class is.
TRAIN_DRAFT_YEARS = list(range(2011, 2021))
PROJECT_DRAFT_YEAR = 2025
DRAFT_YEARS = [*TRAIN_DRAFT_YEARS, PROJECT_DRAFT_YEAR]
OUTCOME_SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(2011, 2024)]


def main() -> None:
    # Slightly faster spacing for the many lightweight per-player bio (age) calls; cached
    # season pulls are unaffected (cache hits skip rate limiting).
    ing = NbaStatsIngester("data/raw", rate_limiter=RateLimiter(1.2))
    # College features if a CBD key is set (CBD_API_KEY); otherwise Combine + pick only.
    cbd = CollegeBasketballDataIngester("data/raw") if os.environ.get("CBD_API_KEY") else None
    if cbd is None:
        log.warning("CBD_API_KEY not set -> running with Combine+pick only (no college features).")
    # International (EuroLeague) features for non-NCAA prospects; enable with NBA_DRAFT_AI_INTL=1.
    intl = EuroLeagueIngester("data/raw") if os.environ.get("NBA_DRAFT_AI_INTL") else None
    result = run_real_pipeline(
        ing, draft_years=DRAFT_YEARS, outcome_seasons=OUTCOME_SEASONS,
        cbd_ingester=cbd, intl_ingester=intl, min_train_years=4, n_holdout_years=2,
        tune=True, n_trials=30,
        # Consensus-anchored board: the draft pick IS legitimate pre-draft information (the 30-team
        # consensus, known before any outcome). A pick-free model ranks WORSE than the draft order
        # (holdout Spearman ~0.26 vs ~0.52) — misleading as a board. Including the pick makes the
        # served ranking MATCH the baseline (~0.50) while the EV still reorders enough to surface
        # model-vs-consensus disagreements (steals/reaches), which are exploratory, not validated to
        # beat the room. See IMPROVEMENT_LOG.md (I2) for the measured ceiling.
        exclude_pick_feature=False,
    )
    log.info(
        "drafted=%d  resolved=%d  model=%s  holdout=%s",
        result.n_drafted, result.n_resolved, result.model_version, result.holdout_years,
    )
    log.info("Survivorship-robust HURDLE ranking (target = realized value over ALL prospects):")
    log.info("  hurdle    CV spearman      = %.3f", result.hurdle_cv_spearman)
    log.info("  hurdle    HOLDOUT spearman = %.3f  (headline)", result.hurdle_holdout_spearman)
    log.info("  baseline  HOLDOUT spearman = %.3f", result.baseline_holdout_spearman)
    log.info("summary -> %s", result.summary_path)


if __name__ == "__main__":
    main()
