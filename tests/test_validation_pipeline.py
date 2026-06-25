"""Tests for the Phase 5 validation protocol: fold preprocessor + walk-forward runner."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import Ridge

from nba_draft.cleaning.imputation import LeakageSafeImputer
from nba_draft.data.fixtures import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    YEAR_COLUMN,
    make_synthetic_prospects,
)
from nba_draft.features.learned import LeagueSeasonContextModel
from nba_draft.validation import (
    FoldPreprocessor,
    make_data_split,
    walk_forward_evaluate,
)


# --------------------------------------------------------------- FoldPreprocessor
def _context_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "draft_year": [2015, 2015, 2016, 2016],
            "league_id": ["ncaa", "ncaa", "euroleague", "euroleague"],
            "season": [2014, 2014, 2015, 2015],
            "pts_per100": [20.0, 10.0, 15.0, None],
            "strength_of_schedule": [8.0, 6.0, None, None],
        }
    )


def test_preprocessor_fills_all_nulls_and_is_fit_on_train_only():
    df = _context_frame()
    feature_cols = ["pts_per100", "pts_per100_translated", "sos_z"]
    pp = FoldPreprocessor(
        feature_cols,
        context_model=LeagueSeasonContextModel(stat_cols=("pts_per100",)),
        imputer=LeakageSafeImputer(columns=("pts_per100",), group_col="league_id"),
    )
    train = df.filter(pl.col("draft_year") == 2015)
    val = df.filter(pl.col("draft_year") == 2016)
    pp.fit(train)
    out = pp.transform_matrix(val)
    # no nulls survive preprocessing
    assert not np.isnan(out.to_numpy().astype(float)).any()
    assert out.columns == feature_cols
    assert out.height == val.height


def test_preprocessor_requires_fit():
    with pytest.raises(RuntimeError):
        FoldPreprocessor(["pts_per100"]).transform(pl.DataFrame({"pts_per100": [1.0]}))


def test_preprocessor_median_backfill_uses_train_median():
    train = pl.DataFrame({"draft_year": [2015, 2015], "x": [2.0, 4.0]})  # median 3.0
    val = pl.DataFrame({"draft_year": [2016], "x": [None]})
    pp = FoldPreprocessor(["x"]).fit(train)
    out = pp.transform_matrix(val)
    assert out["x"][0] == pytest.approx(3.0)


# --------------------------------------------------------------- walk-forward runner
def test_make_data_split_locks_recent_years():
    df = make_synthetic_prospects(seed=1)
    split = make_data_split(df, year_col=YEAR_COLUMN, n_holdout_years=2)
    assert max(split.dev[YEAR_COLUMN].to_list()) < min(split.holdout[YEAR_COLUMN].to_list())
    assert len(split.holdout_years) == 2


def test_walk_forward_evaluate_is_temporal_and_scored():
    df = make_synthetic_prospects(seed=1)
    split = make_data_split(df, year_col=YEAR_COLUMN, n_holdout_years=2)

    report = walk_forward_evaluate(
        split.dev,
        feature_cols=list(FEATURE_COLUMNS),
        target_col=TARGET_COLUMN,
        year_col=YEAR_COLUMN,
        model_factory=lambda: Ridge(alpha=1.0),
        preprocessor_factory=lambda: FoldPreprocessor(list(FEATURE_COLUMNS)),
        min_train_years=4,
        val_horizon_years=1,
    )
    assert len(report.per_fold) >= 1
    # every fold strictly temporal
    for f in report.per_fold:
        assert max(f["train_years"]) < min(f["val_years"])
    # aggregate metrics present and finite
    assert np.isfinite(report.aggregate["spearman_mean"])
    assert np.isfinite(report.aggregate["rmse_mean"])
    # a real model should achieve positive rank correlation on this signal-bearing fixture
    assert report.aggregate["spearman_mean"] > 0.2


def test_runner_rejects_holdout_leakage_by_construction():
    # The runner only ever sees the dev set we pass; confirm holdout rows are absent.
    df = make_synthetic_prospects(seed=1)
    split = make_data_split(df, year_col=YEAR_COLUMN, n_holdout_years=2)
    dev_years = set(split.dev[YEAR_COLUMN].to_list())
    holdout_years = set(split.holdout[YEAR_COLUMN].to_list())
    assert dev_years.isdisjoint(holdout_years)
