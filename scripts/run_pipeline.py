"""One command to reproduce the whole project end-to-end (Phase 12).

    python scripts/run_pipeline.py

Integrates the master dataset, runs temporal-CV model comparison, trains + registers the final
model, and produces a drift report — all logged to MLflow (local file store) with a JSON summary.
"""

from __future__ import annotations

from nba_draft.mlops.pipeline import run_pipeline
from nba_draft.utils.logging import get_logger

log = get_logger("run_pipeline")


def main() -> None:
    result = run_pipeline(tracking_enabled=True)
    log.info("master_version = %s", result.master_version)
    log.info("model_version  = %s", result.model_version)
    log.info("summary        = %s", result.summary_path)
    log.info("Top models by Spearman:")
    for row in result.comparison[:4]:
        log.info("  %-18s spearman=%.3f", row["model"], row["spearman_mean"])
    drifted = [d["feature"] for d in result.drift if d.get("drifted")]
    log.info("Drifted features: %s", drifted or "none")


if __name__ == "__main__":
    main()
