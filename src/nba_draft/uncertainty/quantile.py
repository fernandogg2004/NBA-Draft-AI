"""Quantile-regression prediction intervals via gradient boosting.

Fits one quantile model per requested quantile (pinball loss). Quantile crossing is fixed by
sorting predictions across quantiles, so the interval is always well-formed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


class QuantileGBM:
    """Gradient-boosted quantile regressors for a set of quantiles."""

    def __init__(
        self,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
        *,
        max_iter: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 3,
        seed: int = 42,
    ) -> None:
        if not all(0.0 < q < 1.0 for q in quantiles):
            raise ValueError("quantiles must be in (0, 1).")
        self.quantiles = tuple(sorted(quantiles))
        self._kw = dict(max_iter=max_iter, learning_rate=learning_rate, max_depth=max_depth)
        self.seed = seed
        self._models: dict[float, Any] = {}

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> QuantileGBM:
        from sklearn.ensemble import HistGradientBoostingRegressor

        for q in self.quantiles:
            model = HistGradientBoostingRegressor(
                loss="quantile", quantile=q, random_state=self.seed, **self._kw
            )
            model.fit(X, y)
            self._models[q] = model
        return self

    def predict_quantiles(self, X: NDArray[np.float64]) -> dict[float, NDArray[np.float64]]:
        if not self._models:
            raise RuntimeError("QuantileGBM must be fit before predict.")
        # Stack and sort across quantiles to remove crossing.
        preds = np.vstack([self._models[q].predict(X) for q in self.quantiles])
        preds = np.sort(preds, axis=0)
        return {q: preds[i] for i, q in enumerate(self.quantiles)}

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Point prediction = the median quantile if present, else the central one."""
        qs = self.predict_quantiles(X)
        median_q = min(self.quantiles, key=lambda q: abs(q - 0.5))
        return qs[median_q]

    def predict_interval(
        self, X: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """(lower, upper) using the extreme requested quantiles."""
        qs = self.predict_quantiles(X)
        return qs[self.quantiles[0]], qs[self.quantiles[-1]]
