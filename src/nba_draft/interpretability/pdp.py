"""Partial dependence: how a projection moves as one feature varies, averaging over the rest.

Model-agnostic (only needs ``predict``), so it works for every estimator in the project.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def partial_dependence(
    model: Any,
    X: NDArray[np.float64],
    feature_index: int,
    *,
    grid: NDArray[np.float64] | None = None,
    n_points: int = 20,
    use_quantile_grid: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return (grid_values, mean_prediction) for the partial-dependence curve of a feature.

    Args:
        model: anything with ``predict``.
        X: feature matrix.
        feature_index: which column to vary.
        grid: explicit grid; if None it is built from the feature's distribution.
        use_quantile_grid: build the grid from quantiles (robust to outliers) vs linspace.
    """
    X = np.asarray(X, dtype=np.float64)
    col = X[:, feature_index]
    if grid is None:
        if use_quantile_grid:
            grid = np.unique(np.quantile(col, np.linspace(0.0, 1.0, n_points)))
        else:
            grid = np.linspace(float(col.min()), float(col.max()), n_points)
    means = np.empty(grid.shape[0], dtype=np.float64)
    for i, v in enumerate(grid):
        xc = X.copy()
        xc[:, feature_index] = v
        means[i] = float(np.mean(model.predict(xc)))
    return np.asarray(grid, dtype=np.float64), means
