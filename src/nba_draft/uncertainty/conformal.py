"""Split-conformal prediction intervals.

Wraps any base estimator to produce intervals with a finite-sample marginal coverage guarantee
(~1-alpha) under exchangeability, with no distributional assumptions. Within the temporal CV the
base model and calibration set both come from the training fold, so it stays leakage-safe.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray


class SplitConformalRegressor:
    """Split-conformal regressor using absolute-residual nonconformity scores."""

    def __init__(
        self,
        base_factory: Callable[[], Any],
        *,
        alpha: float = 0.1,
        calib_fraction: float = 0.3,
        seed: int = 42,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1).")
        self.base_factory = base_factory
        self.alpha = alpha
        self.calib_fraction = calib_fraction
        self.seed = seed
        self._base: Any | None = None
        self._qhat: float | None = None
        self._residuals: NDArray[np.float64] | None = None  # signed calibration residuals

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> SplitConformalRegressor:
        rng = np.random.default_rng(self.seed)
        n = X.shape[0]
        idx = rng.permutation(n)
        n_calib = max(1, int(round(self.calib_fraction * n)))
        calib_idx, train_idx = idx[:n_calib], idx[n_calib:]
        if train_idx.size == 0:
            raise ValueError("Not enough data to split into proper-train and calibration.")

        self._base = self.base_factory()
        self._base.fit(X[train_idx], y[train_idx])
        signed = np.asarray(y[calib_idx] - self._base.predict(X[calib_idx]), dtype=np.float64)
        self._residuals = signed  # signed residuals -> empirical predictive distribution
        residuals = np.abs(signed)
        # Conformal quantile level with finite-sample correction.
        level = min(1.0, np.ceil((n_calib + 1) * (1 - self.alpha)) / n_calib)
        self._qhat = float(np.quantile(residuals, level, method="higher"))
        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._base is None:
            raise RuntimeError("SplitConformalRegressor must be fit before predict.")
        return np.asarray(self._base.predict(X), dtype=np.float64)

    def predict_interval(
        self, X: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if self._qhat is None:
            raise RuntimeError("SplitConformalRegressor must be fit before predict_interval.")
        point = self.predict(X)
        return point - self._qhat, point + self._qhat

    def predict_scenarios(
        self, X: NDArray[np.float64], edges: list[float], labels: list[str]
    ) -> list[dict[str, float]]:
        """Per-row outcome-tier probabilities from the empirical predictive distribution.

        Each prediction is spread by the *calibration residual* distribution (point + signed
        residuals), so the tier probabilities inherit the conformal model's honest predictive
        variance instead of an overconfident parametric/ensemble spread.
        """
        from nba_draft.uncertainty.scenarios import scenario_probabilities_from_samples

        if self._residuals is None:
            raise RuntimeError("SplitConformalRegressor must be fit before predict_scenarios.")
        point = self.predict(X)
        return [
            scenario_probabilities_from_samples(point[i] + self._residuals, edges, labels)
            for i in range(point.shape[0])
        ]
