"""Phase 6 modeling comparison: real models (pre-draft features) vs the draft-position baseline.

All models are evaluated through the SAME temporal-CV protocol (walk_forward_evaluate) on the
development set; the holdout is untouched. The baseline uses the draft pick (the league's
consensus); the real models use only pre-draft features.

    python scripts/run_modeling.py
"""

from __future__ import annotations

from nba_draft.config import load_config
from nba_draft.data.fixtures import (
    FEATURE_COLUMNS,
    PICK_COLUMN,
    TARGET_COLUMN,
    YEAR_COLUMN,
    make_synthetic_prospects,
)
from nba_draft.models import DraftPositionEstimator, gbm_regressor, mean_regressor, ridge_regressor
from nba_draft.utils.logging import get_logger
from nba_draft.validation import FoldPreprocessor, make_data_split, walk_forward_evaluate

log = get_logger("run_modeling")


def main() -> None:
    cfg = load_config()
    df = make_synthetic_prospects(seed=cfg.seed)
    split = make_data_split(
        df, year_col=YEAR_COLUMN, n_holdout_years=cfg.validation.n_holdout_years
    )
    dev = split.dev
    log.info("Dev set %d rows; holdout years %s locked.", dev.height, split.holdout_years)

    feats = list(FEATURE_COLUMNS)
    common = dict(
        target_col=TARGET_COLUMN,
        year_col=YEAR_COLUMN,
        min_train_years=cfg.validation.min_train_years,
    )

    contenders = {
        "baseline_draftpos": dict(
            feature_cols=[PICK_COLUMN],
            model_factory=lambda: DraftPositionEstimator(0),
            preprocessor_factory=lambda: FoldPreprocessor([PICK_COLUMN]),
        ),
        "mean": dict(
            feature_cols=feats,
            model_factory=mean_regressor,
            preprocessor_factory=lambda: FoldPreprocessor(feats),
        ),
        "ridge": dict(
            feature_cols=feats,
            model_factory=lambda: ridge_regressor(alpha=1.0),
            preprocessor_factory=lambda: FoldPreprocessor(feats),
        ),
        "gbm": dict(
            feature_cols=feats,
            model_factory=lambda: gbm_regressor(),
            preprocessor_factory=lambda: FoldPreprocessor(feats),
        ),
    }

    lines = ["# Phase 6 modeling comparison (SYNTHETIC dev set)\n",
             "| model | spearman | top10 | rmse |", "| --- | --- | --- | --- |"]
    for name, kw in contenders.items():
        report = walk_forward_evaluate(dev, **common, **kw)  # type: ignore[arg-type]
        a = report.aggregate
        lines.append(
            f"| {name} | {a['spearman_mean']:.3f}±{a['spearman_std']:.3f} "
            f"| {a['top10_mean']:.3f} | {a['rmse_mean']:.3f} |"
        )
        log.info("%s: %s", name, {k: round(v, 3) for k, v in a.items()})

    out_dir = cfg.path("artifacts") / "modeling"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Wrote %s", out_dir / "comparison.md")


if __name__ == "__main__":
    main()
