"""Tests for the hurdle model and its temporal-CV evaluation (A1)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nba_draft.data.fixtures import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    YEAR_COLUMN,
    make_synthetic_prospects,
)
from nba_draft.models.hurdle import REPLACEMENT_BPM, HurdleModel, realized_value
from nba_draft.validation import (
    FoldPreprocessor,
    make_data_split,
    walk_forward_hurdle_evaluate,
)


def test_realized_value_uses_replacement_for_non_reachers():
    reached = np.array([1.0, 0.0, 1.0])
    impact = np.array([5.0, np.nan, 1.0])
    rv = realized_value(reached, impact, replacement=-2.0)
    assert rv[0] == 5.0
    assert rv[1] == -2.0   # non-reacher -> replacement, not NaN
    assert rv[2] == 1.0


def test_hurdle_fit_predict_ev_and_parts():
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=(n, 3))
    skill = x[:, 0]
    reached = (skill + rng.normal(scale=0.5, size=n) > 0).astype(float)
    impact = np.where(reached > 0.5, 3.0 * skill + rng.normal(scale=0.3, size=n), np.nan)
    model = HurdleModel().fit(x, reached, impact)
    p, imp = model.predict_parts(x)
    assert p.min() >= 0.0 and p.max() <= 1.0
    ev = model.predict(x)
    assert ev.shape == (n,)
    # higher skill -> higher EV (both reach prob and conditional impact rise with skill)
    assert np.corrcoef(skill, ev)[0, 1] > 0.5


def test_hurdle_requires_reached_to_fit():
    x = np.zeros((5, 2))
    with pytest.raises(ValueError):
        HurdleModel().fit(x, np.zeros(5), np.full(5, np.nan))  # nobody reached


def test_hurdle_requires_fit_before_predict():
    with pytest.raises(RuntimeError):
        HurdleModel().predict(np.zeros((2, 2)))


def test_walk_forward_hurdle_evaluate_runs_and_ranks():
    df = make_synthetic_prospects(seed=2)
    dev = make_data_split(df, year_col=YEAR_COLUMN, n_holdout_years=2).dev
    reached = (dev[TARGET_COLUMN].to_numpy() > REPLACEMENT_BPM).astype(float)
    dev = dev.with_columns(
        pl.Series("reached", reached),
        pl.col(TARGET_COLUMN).alias("impact"),
        pl.Series("realized", realized_value(reached, dev[TARGET_COLUMN].to_numpy())),
    )
    feats = list(FEATURE_COLUMNS)
    report = walk_forward_hurdle_evaluate(
        dev, feature_cols=feats, reached_col="reached", impact_col="impact",
        realized_col="realized", preprocessor_factory=lambda: FoldPreprocessor(feats),
        min_train_years=4,
    )
    assert len(report.per_fold) >= 1
    for f in report.per_fold:
        assert max(f["train_years"]) < min(f["val_years"])   # temporal
    # the unconditional ranking should be clearly better than chance on signal-bearing data
    assert report.aggregate["spearman_mean"] > 0.2
