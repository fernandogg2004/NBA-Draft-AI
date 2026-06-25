"""End-to-end pipeline on REAL nba_api data (run locally).

Pulls draft history + Combine + player seasons (cached), builds real outcome labels, runs a
temporal-CV comparison of Combine-feature models vs the draft-position baseline, and trains +
registers a model. Honest scope: with only nba_api the pre-draft features are Combine + pick.

    python scripts/run_real_pipeline.py
"""

from __future__ import annotations

import os

from nba_draft.ingestion.college_bb_data import CollegeBasketballDataIngester
from nba_draft.ingestion.http import RateLimiter
from nba_draft.ingestion.nba_stats import NbaStatsIngester
from nba_draft.realdata import run_real_pipeline
from nba_draft.utils.logging import get_logger

log = get_logger("run_real_pipeline")

# Widened window: 10 draft classes (2011-2020), whose 4-year debut windows all resolve from
# the outcome seasons below (through 2023-24). More classes -> more walk-forward folds.
DRAFT_YEARS = list(range(2011, 2021))
OUTCOME_SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(2011, 2024)]


def main() -> None:
    # Slightly faster spacing for the many lightweight per-player bio (age) calls; cached
    # season pulls are unaffected (cache hits skip rate limiting).
    ing = NbaStatsIngester("data/raw", rate_limiter=RateLimiter(1.2))
    # College features if a CBD key is set (CBD_API_KEY); otherwise Combine + pick only.
    cbd = CollegeBasketballDataIngester("data/raw") if os.environ.get("CBD_API_KEY") else None
    if cbd is None:
        log.warning("CBD_API_KEY not set -> running with Combine+pick only (no college features).")
    result = run_real_pipeline(
        ing, draft_years=DRAFT_YEARS, outcome_seasons=OUTCOME_SEASONS,
        cbd_ingester=cbd, min_train_years=4, tune=True, n_trials=30,
    )
    log.info(
        "drafted=%d  trainable=%d  model=%s",
        result.n_drafted, result.n_trainable, result.model_version,
    )
    log.info("Models vs draft-position baseline (target = peak eBPM, conditional on reaching):")
    for row in sorted(result.comparison, key=lambda r: -r["spearman_mean"]):
        log.info(
            "  %-18s spearman=%.3f rmse=%.3f",
            row["model"], row["spearman_mean"], row["rmse_mean"],
        )
    log.info("summary -> %s", result.summary_path)


if __name__ == "__main__":
    main()
