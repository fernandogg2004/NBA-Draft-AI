"""Phase 7 evaluation report: baseline comparison, calibration, and error analysis.

All temporal-CV on the development set; the holdout stays locked. On the synthetic fixture this
exercises the evaluation tooling end-to-end; point it at the real master dataset for real results.

    python scripts/run_evaluation.py
"""

from __future__ import annotations

import polars as pl

from nba_draft.config import load_config
from nba_draft.data.fixtures import (
    FEATURE_COLUMNS,
    PICK_COLUMN,
    TARGET_COLUMN,
    YEAR_COLUMN,
    make_synthetic_prospects,
)
from nba_draft.eda.report import df_to_markdown
from nba_draft.evaluation import (
    calibration_table,
    compare_models,
    largest_errors,
    make_spec,
    residual_segments,
)
from nba_draft.evaluation.metrics import brier_score, expected_calibration_error
from nba_draft.models import (
    DraftPositionEstimator,
    gbm_regressor,
    logistic_classifier,
    mean_regressor,
    ridge_regressor,
)
from nba_draft.utils.logging import get_logger
from nba_draft.validation import (
    FoldPreprocessor,
    make_data_split,
    walk_forward_predictions,
)

log = get_logger("run_evaluation")


def main() -> None:
    cfg = load_config()
    df = make_synthetic_prospects(seed=cfg.seed)
    dev = make_data_split(
        df, year_col=YEAR_COLUMN, n_holdout_years=cfg.validation.n_holdout_years
    ).dev
    feats = list(FEATURE_COLUMNS)
    mty = cfg.validation.min_train_years

    parts: list[str] = [
        "# Evaluation Report (SYNTHETIC dev set)\n",
        "> Temporal CV on the development set; holdout locked. Tooling demo only.\n",
    ]

    # 1) Model comparison vs the draft-position baseline.
    specs = {
        "baseline_draftpos": make_spec(
            [PICK_COLUMN], lambda: DraftPositionEstimator(0),
            lambda: FoldPreprocessor([PICK_COLUMN])
        ),
        "mean": make_spec(feats, mean_regressor, lambda: FoldPreprocessor(feats)),
        "ridge": make_spec(feats, lambda: ridge_regressor(1.0), lambda: FoldPreprocessor(feats)),
        "gbm": make_spec(feats, gbm_regressor, lambda: FoldPreprocessor(feats)),
    }
    table = compare_models(
        dev, specs, target_col=TARGET_COLUMN, min_train_years=mty, baseline_name="baseline_draftpos"
    )
    parts.append("## 1. Models vs draft-position baseline (ranking-first)\n")
    parts.append(df_to_markdown(table.select(
        "model", "spearman_mean", "top10_mean", "rmse_mean", "uplift_spearman_mean"
    )))

    # 2) Error analysis (out-of-fold) by league tier.
    oof = walk_forward_predictions(
        dev, feature_cols=feats, target_col=TARGET_COLUMN, year_col=YEAR_COLUMN,
        model_factory=lambda: ridge_regressor(1.0),
        preprocessor_factory=lambda: FoldPreprocessor(feats), min_train_years=mty,
    )
    parts.append("\n\n## 2. Error analysis by league tier (ridge, out-of-fold)\n")
    parts.append("_bias = mean signed error (model − actual); MAE/RMSE = error magnitude._\n")
    seg = residual_segments(oof, segment_col="league_tier", target_col=TARGET_COLUMN)
    parts.append(df_to_markdown(seg))

    parts.append("\n\n## 3. Largest individual misses\n")
    parts.append(df_to_markdown(
        largest_errors(oof, target_col=TARGET_COLUMN, id_cols=["player_id", "draft_year"], k=10)
    ))

    # 4) Calibration of a reach-probability classifier.
    reach_df = df.with_columns((pl.col(TARGET_COLUMN) > 0).cast(pl.Float64).alias("reached"))
    reach_dev = make_data_split(reach_df, year_col=YEAR_COLUMN, n_holdout_years=2).dev
    reach_oof = walk_forward_predictions(
        reach_dev, feature_cols=feats, target_col="reached", year_col=YEAR_COLUMN,
        model_factory=lambda: logistic_classifier(1.0),
        preprocessor_factory=lambda: FoldPreprocessor(feats), min_train_years=mty,
    )
    y_true = reach_oof["reached"].to_numpy()
    p = reach_oof["y_pred"].to_numpy()
    parts.append("\n\n## 4. Reach-probability calibration (logistic, out-of-fold)\n")
    parts.append(
        f"Brier = {brier_score(y_true, p):.3f}; ECE = {expected_calibration_error(y_true, p):.3f}\n"
    )
    parts.append(df_to_markdown(calibration_table(y_true, p, n_bins=10)))

    out_dir = cfg.path("artifacts") / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evaluation_report.md").write_text("\n".join(parts) + "\n", encoding="utf-8")
    log.info("Wrote %s", out_dir / "evaluation_report.md")


if __name__ == "__main__":
    main()
