"""Counterfactuals: what would have to change to raise a prospect's projection.

Answers the scout's natural follow-up — "what is this prospect missing?" — by finding the
smallest feature change(s) that lift the prediction to a target. Model-agnostic (uses predict).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def counterfactual_single_feature(
    model: Any,
    x_row: NDArray[np.float64],
    feature_index: int,
    target: float,
    *,
    bounds: tuple[float, float],
    n_points: int = 101,
) -> float | None:
    """Smallest-change value of one feature that brings the prediction to >= target.

    Returns the new feature value (closest to the current one that reaches the target), or
    ``None`` if no value within ``bounds`` achieves it.
    """
    x = np.asarray(x_row, dtype=np.float64).reshape(1, -1).copy()
    current = x[0, feature_index]
    grid = np.linspace(bounds[0], bounds[1], n_points)
    achieved: list[float] = []
    for v in grid:
        x[0, feature_index] = v
        if float(model.predict(x)[0]) >= target:
            achieved.append(float(v))
    if not achieved:
        return None
    return min(achieved, key=lambda v: abs(v - current))


def greedy_counterfactual(
    model: Any,
    x_row: NDArray[np.float64],
    target: float,
    feature_bounds: dict[int, tuple[float, float]],
    *,
    max_features: int = 3,
    n_points: int = 51,
) -> dict[int, float]:
    """Greedily change a few features to reach the target prediction.

    At each step, for each still-unused candidate feature, find its best single-feature move
    (largest prediction gain within bounds) and apply the best one. Stops when the target is met
    or ``max_features`` changes are made. Returns {feature_index: new_value} for changed features.
    """
    x = np.asarray(x_row, dtype=np.float64).reshape(1, -1).copy()
    changes: dict[int, float] = {}
    remaining = set(feature_bounds)
    for _ in range(max_features):
        if float(model.predict(x)[0]) >= target:
            break
        best_feat: int | None = None
        best_val = 0.0
        best_pred = float(model.predict(x)[0])
        for j in remaining:
            lo, hi = feature_bounds[j]
            for v in np.linspace(lo, hi, n_points):
                trial = x.copy()
                trial[0, j] = v
                pred = float(model.predict(trial)[0])
                if pred > best_pred:
                    best_pred, best_feat, best_val = pred, j, float(v)
        if best_feat is None:
            break  # no improving move
        x[0, best_feat] = best_val
        changes[best_feat] = best_val
        remaining.discard(best_feat)
    return changes
