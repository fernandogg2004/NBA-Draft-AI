"""Tests for Phase 9 uncertainty: quantile, conformal, bayesian, ensemble, scenarios."""

from __future__ import annotations

import numpy as np
import pytest

from nba_draft.models.zoo import ridge_regressor
from nba_draft.uncertainty import (
    BayesianLinearModel,
    BootstrapEnsemble,
    QuantileGBM,
    SplitConformalRegressor,
    ceiling_floor,
    interval_coverage,
    scenario_probabilities_from_normal,
    scenario_probabilities_from_samples,
)

TIER_EDGES = [-1e9, -2.0, 0.0, 3.0, 6.0, 1e9]
TIER_LABELS = ["bust", "rotation", "starter", "all_star", "superstar"]


def _linear_data(n: int = 400, noise: float = 1.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    y = 2.0 * x[:, 0] - 1.0 * x[:, 1] + rng.normal(scale=noise, size=n)
    return x, y


# ----------------------------------------------------------------- scenarios
def test_scenario_from_normal_sums_to_one_and_peaks_correctly():
    probs = scenario_probabilities_from_normal(7.0, 0.5, TIER_EDGES, TIER_LABELS)
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)
    assert max(probs, key=probs.get) == "superstar"  # high mean, low sd


def test_scenario_from_normal_spreads_with_high_sd():
    tight = scenario_probabilities_from_normal(0.0, 0.3, TIER_EDGES, TIER_LABELS)
    wide = scenario_probabilities_from_normal(0.0, 5.0, TIER_EDGES, TIER_LABELS)
    # wide uncertainty puts meaningful mass on the tails (bust & superstar)
    assert wide["superstar"] > tight["superstar"]
    assert wide["bust"] > tight["bust"]


def test_scenario_from_samples_matches_fractions():
    samples = np.array([-3.0, -1.0, 1.0, 4.0, 7.0])  # one in each tier
    probs = scenario_probabilities_from_samples(samples, TIER_EDGES, TIER_LABELS)
    for label in TIER_LABELS:
        assert probs[label] == pytest.approx(0.2)


def test_ceiling_floor_and_coverage():
    s = np.linspace(0, 10, 101)
    floor, ceil_ = ceiling_floor(s, floor_q=0.1, ceiling_q=0.9)
    assert floor < ceil_
    # first two points inside their intervals, third (9.0) outside [0,2] -> 2/3 coverage
    cov = interval_coverage([1.0, 5.0, 9.0], [0.0, 4.0, 0.0], [2.0, 6.0, 2.0])
    assert cov == pytest.approx(2 / 3)


# ----------------------------------------------------------------- quantile
def test_quantile_intervals_are_ordered_and_cover():
    x, y = _linear_data(noise=1.0)
    model = QuantileGBM(quantiles=(0.1, 0.5, 0.9)).fit(x, y)
    lo, hi = model.predict_interval(x)
    assert np.all(lo <= hi)
    # in-sample 80% interval should cover most points
    assert interval_coverage(y, lo, hi) > 0.7


# ----------------------------------------------------------------- conformal
def test_conformal_interval_achieves_nominal_coverage():
    x, y = _linear_data(n=600, noise=1.0, seed=1)
    n_tr = 400
    model = SplitConformalRegressor(lambda: ridge_regressor(1.0), alpha=0.1, seed=1)
    model.fit(x[:n_tr], y[:n_tr])
    lo, hi = model.predict_interval(x[n_tr:])
    cov = interval_coverage(y[n_tr:], lo, hi)
    assert cov >= 0.83  # ~0.90 nominal, allow sampling slack


# ----------------------------------------------------------------- bayesian
def test_bayesian_predicts_mean_and_positive_std():
    x, y = _linear_data()
    model = BayesianLinearModel().fit(x, y)
    mean, std = model.predict_mean_std(x)
    assert mean.shape == (x.shape[0],)
    assert np.all(std > 0)


# ----------------------------------------------------------------- ensemble
def test_ensemble_samples_intervals_and_scenarios():
    x, y = _linear_data()
    ens = BootstrapEnsemble(lambda: ridge_regressor(1.0), n_estimators=20, seed=3).fit(x, y)
    samples = ens.predict_samples(x[:5])
    assert samples.shape == (5, 20)
    lo, hi = ens.predict_interval(x[:5], alpha=0.2)
    assert np.all(lo <= hi)
    scen = ens.predict_scenarios(x[:5], TIER_EDGES, TIER_LABELS)
    assert len(scen) == 5
    for row in scen:
        assert sum(row.values()) == pytest.approx(1.0, abs=1e-6)
