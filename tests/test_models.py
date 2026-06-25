"""Tests for Phase 6 modeling: zoo factories, baseline estimator, tuning, survival."""

from __future__ import annotations

import numpy as np
import polars as pl

from nba_draft.data.fixtures import (
    FEATURE_COLUMNS,
    PICK_COLUMN,
    TARGET_COLUMN,
    YEAR_COLUMN,
    make_synthetic_prospects,
)
from nba_draft.evaluation.metrics import brier_score
from nba_draft.models import (
    DraftPositionEstimator,
    gbm_regressor,
    logistic_classifier,
    mean_regressor,
    ridge_regressor,
)
from nba_draft.validation import FoldPreprocessor, make_data_split, walk_forward_evaluate


# ----------------------------------------------------------------- zoo factories
def test_regressors_fit_and_predict():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 3))
    y = x[:, 0] * 2.0 + rng.normal(scale=0.1, size=40)
    for build in (mean_regressor, lambda: ridge_regressor(0.5), gbm_regressor):
        model = build()
        model.fit(x, y)
        pred = model.predict(x)
        assert pred.shape == (40,)


def test_gbm_regressor_uses_a_real_backend():
    # xgboost is installed in this env; the factory should produce a working booster.
    model = gbm_regressor(n_estimators=20)
    backends = {"XGBRegressor", "LGBMRegressor", "HistGradientBoostingRegressor"}
    assert type(model).__name__ in backends


def test_proba_adapter_returns_probabilities():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(60, 3))
    y = (x[:, 0] > 0).astype(float)
    clf = logistic_classifier(C=1.0)
    clf.fit(x, y)
    p = clf.predict(x)
    assert p.shape == (60,)
    assert float(p.min()) >= 0.0 and float(p.max()) <= 1.0


# ----------------------------------------------------------------- baseline through the protocol
def test_draft_position_estimator_beats_mean_under_temporal_cv():
    df = make_synthetic_prospects(seed=2)
    dev = make_data_split(df, year_col=YEAR_COLUMN, n_holdout_years=2).dev
    common = dict(target_col=TARGET_COLUMN, year_col=YEAR_COLUMN, min_train_years=4)

    base = walk_forward_evaluate(
        dev, feature_cols=[PICK_COLUMN],
        model_factory=lambda: DraftPositionEstimator(0),
        preprocessor_factory=lambda: FoldPreprocessor([PICK_COLUMN]),
        **common,
    )
    mean = walk_forward_evaluate(
        dev, feature_cols=list(FEATURE_COLUMNS),
        model_factory=mean_regressor,
        preprocessor_factory=lambda: FoldPreprocessor(list(FEATURE_COLUMNS)),
        **common,
    )
    # draft position is a real signal; mean predictor has ~0 rank correlation
    assert base.aggregate["spearman_mean"] > 0.3
    assert base.aggregate["spearman_mean"] > mean.aggregate["spearman_mean"]


def test_classification_target_scored_with_brier():
    df = make_synthetic_prospects(seed=2).with_columns(
        (pl.col(TARGET_COLUMN) > 0).cast(pl.Float64).alias("reached")
    )
    dev = make_data_split(df, year_col=YEAR_COLUMN, n_holdout_years=2).dev
    report = walk_forward_evaluate(
        dev, feature_cols=list(FEATURE_COLUMNS), target_col="reached", year_col=YEAR_COLUMN,
        model_factory=lambda: logistic_classifier(1.0),
        preprocessor_factory=lambda: FoldPreprocessor(list(FEATURE_COLUMNS)),
        min_train_years=4,
        metrics={"brier": brier_score},
    )
    # a real classifier should beat the trivial 0.25 Brier of always-predicting-0.5
    assert report.aggregate["brier_mean"] < 0.25


# ----------------------------------------------------------------- tuning (optuna)
def test_tune_estimator_returns_best_in_range():
    from nba_draft.models.tuning import tune_estimator

    df = make_synthetic_prospects(seed=2)
    dev = make_data_split(df, year_col=YEAR_COLUMN, n_holdout_years=2).dev
    feats = list(FEATURE_COLUMNS)
    result = tune_estimator(
        dev,
        feature_cols=feats,
        target_col=TARGET_COLUMN,
        build_fn=ridge_regressor,
        param_space={"alpha": ("float", 0.01, 10.0, True)},
        preprocessor_factory=lambda: FoldPreprocessor(feats),
        min_train_years=4,
        n_trials=8,
        seed=0,
    )
    assert 0.01 <= result.best_params["alpha"] <= 10.0
    assert np.isfinite(result.best_value)


# ----------------------------------------------------------------- survival (lifelines)
def test_cox_survival_concordance_above_chance():
    from nba_draft.models.survival import CoxSurvivalModel, concordance

    rng = np.random.default_rng(3)
    n = 200
    skill = rng.normal(size=n)
    # higher skill -> longer career; censor ~30%
    true_dur = np.clip(6 + 3 * skill + rng.normal(scale=1.0, size=n), 1, None)
    censor_t = rng.uniform(2, 12, size=n)
    duration = np.minimum(true_dur, censor_t)
    event = (true_dur <= censor_t).astype(float)
    df = pl.DataFrame({"skill": skill, "duration": duration, "event": event})

    model = CoxSurvivalModel(penalizer=0.1).fit(
        df, feature_cols=["skill"], duration_col="duration", event_col="event"
    )
    risk = model.predict_risk(df)
    c = concordance(duration, event, risk)
    assert c > 0.6  # clearly better than chance
