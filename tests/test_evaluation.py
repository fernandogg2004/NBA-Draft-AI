"""Tests for Phase 7 evaluation: calibration, error analysis, model comparison, OOF preds."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nba_draft.data.fixtures import (
    FEATURE_COLUMNS,
    PICK_COLUMN,
    TARGET_COLUMN,
    YEAR_COLUMN,
    make_synthetic_prospects,
)
from nba_draft.evaluation import (
    calibration_table,
    compare_models,
    largest_errors,
    make_spec,
    residual_segments,
)
from nba_draft.models import DraftPositionEstimator, mean_regressor, ridge_regressor
from nba_draft.validation import (
    FoldPreprocessor,
    make_data_split,
    walk_forward_predictions,
)


# --------------------------------------------------------------- calibration
def test_calibration_table_perfect_model():
    # predictions equal outcomes -> each bin's mean_pred matches frac_pos
    y = np.array([0, 0, 1, 1, 1, 0, 1, 0], dtype=float)
    p = y.copy()
    tbl = calibration_table(y, p, n_bins=5)
    populated = tbl.filter(pl.col("n") > 0)
    for row in populated.iter_rows(named=True):
        assert row["mean_pred"] == pytest.approx(row["frac_pos"])


def test_calibration_rejects_out_of_range():
    with pytest.raises(ValueError):
        calibration_table(np.array([0.0, 1.0]), np.array([0.5, 1.5]))


# --------------------------------------------------------------- OOF predictions + error analysis
def _oof() -> pl.DataFrame:
    df = make_synthetic_prospects(seed=2)
    dev = make_data_split(df, year_col=YEAR_COLUMN, n_holdout_years=2).dev
    feats = list(FEATURE_COLUMNS)
    return walk_forward_predictions(
        dev,
        feature_cols=feats,
        target_col=TARGET_COLUMN,
        year_col=YEAR_COLUMN,
        model_factory=lambda: ridge_regressor(1.0),
        preprocessor_factory=lambda: FoldPreprocessor(feats),
        min_train_years=4,
    )


def test_oof_predictions_cover_validated_rows_once():
    oof = _oof()
    assert "y_pred" in oof.columns
    assert "fold" in oof.columns
    # one prediction per validated prospect (no duplicates across folds)
    assert oof.height == oof["player_id"].n_unique()


def test_residual_segments_report_bias_and_mae():
    oof = _oof()
    seg = residual_segments(oof, segment_col="league_tier", target_col=TARGET_COLUMN)
    assert set(["league_tier", "n", "bias", "mae", "rmse"]).issubset(seg.columns)
    assert seg.height == oof["league_tier"].n_unique()
    assert (seg["mae"] >= 0).all()


def test_largest_errors_surface_biggest_misses():
    oof = _oof()
    top = largest_errors(oof, target_col=TARGET_COLUMN, id_cols=["player_id"], k=5)
    assert top.height == 5
    # sorted by absolute residual descending
    ar = top["abs_residual"].to_list()
    assert ar == sorted(ar, reverse=True)


# --------------------------------------------------------------- model comparison vs baseline
def test_compare_models_ranks_and_computes_uplift():
    df = make_synthetic_prospects(seed=2)
    dev = make_data_split(df, year_col=YEAR_COLUMN, n_holdout_years=2).dev
    feats = list(FEATURE_COLUMNS)
    specs = {
        "baseline": make_spec(
            [PICK_COLUMN],
            lambda: DraftPositionEstimator(0),
            lambda: FoldPreprocessor([PICK_COLUMN]),
        ),
        "mean": make_spec(feats, mean_regressor, lambda: FoldPreprocessor(feats)),
        "ridge": make_spec(feats, lambda: ridge_regressor(1.0), lambda: FoldPreprocessor(feats)),
    }
    table = compare_models(
        dev, specs, target_col=TARGET_COLUMN, min_train_years=4, baseline_name="baseline"
    )
    assert "uplift_spearman_mean" in table.columns
    # table sorted by spearman desc; ridge should top mean
    models = table["model"].to_list()
    assert models.index("ridge") < models.index("mean")
    # baseline's own uplift is zero
    base_uplift = table.filter(pl.col("model") == "baseline")["uplift_spearman_mean"][0]
    assert base_uplift == pytest.approx(0.0)
