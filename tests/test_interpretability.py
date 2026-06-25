"""Tests for Phase 10 interpretability: SHAP, permutation importance, PDP, counterfactuals."""

from __future__ import annotations

import numpy as np
import pytest

from nba_draft.interpretability import (
    ShapExplainer,
    counterfactual_single_feature,
    greedy_counterfactual,
    partial_dependence,
    permutation_importance_table,
)
from nba_draft.models.zoo import ridge_regressor

FEATURES = ["f0", "f1", "f2"]


def _linear_model_and_data(seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(150, 3))
    # f0 dominates, f1 weak, f2 irrelevant
    y = 3.0 * x[:, 0] + 0.5 * x[:, 1] + rng.normal(scale=0.2, size=150)
    model = ridge_regressor(0.1)
    model.fit(x, y)
    return model, x, y


# ----------------------------------------------------------------- SHAP
def test_shap_global_importance_ranks_dominant_feature_first():
    model, x, _ = _linear_model_and_data()
    expl = ShapExplainer(model, x, FEATURES, max_background=50)
    table = expl.global_importance(x[:30])
    assert table["feature"][0] == "f0"  # strongest driver


def test_shap_local_explanation_is_additive():
    model, x, _ = _linear_model_and_data()
    expl = ShapExplainer(model, x, FEATURES, max_background=50)
    row = x[0]
    table, base = expl.local_explanation(row)
    recon = base + float(table["shap_value"].sum())
    pred = float(model.predict(row.reshape(1, -1))[0])
    assert recon == pytest.approx(pred, abs=0.1)  # SHAP additivity


# ----------------------------------------------------------------- permutation importance
def test_permutation_importance_ranks_dominant_feature_first():
    model, x, y = _linear_model_and_data()
    table = permutation_importance_table(model, x, y, FEATURES, n_repeats=5)
    assert table["feature"][0] == "f0"
    # the irrelevant feature should have ~zero importance
    f2 = table.filter(table["feature"] == "f2")["importance"][0]
    assert abs(f2) < 0.1


# ----------------------------------------------------------------- PDP
def test_partial_dependence_increasing_in_positive_feature():
    model, x, _ = _linear_model_and_data()
    grid, means = partial_dependence(model, x, feature_index=0, n_points=10)
    # positive coefficient -> PDP increases with the feature
    assert means[-1] > means[0]
    assert grid.shape == means.shape


# ----------------------------------------------------------------- counterfactual
def test_counterfactual_single_feature_finds_needed_increase():
    model, x, y = _linear_model_and_data()
    row = x[0].copy()
    current_pred = float(model.predict(row.reshape(1, -1))[0])
    target = current_pred + 3.0
    new_val = counterfactual_single_feature(
        model, row, feature_index=0, target=target, bounds=(-5.0, 5.0)
    )
    assert new_val is not None
    row2 = row.copy()
    row2[0] = new_val
    assert float(model.predict(row2.reshape(1, -1))[0]) >= target - 1e-6


def test_greedy_counterfactual_reaches_target():
    model, x, _ = _linear_model_and_data()
    row = x[0].copy()
    target = float(model.predict(row.reshape(1, -1))[0]) + 4.0
    changes = greedy_counterfactual(
        model, row, target, feature_bounds={0: (-5, 5), 1: (-5, 5)}, max_features=2
    )
    assert changes  # at least one change proposed
    row2 = row.copy()
    for j, v in changes.items():
        row2[j] = v
    assert float(model.predict(row2.reshape(1, -1))[0]) >= target - 0.5
