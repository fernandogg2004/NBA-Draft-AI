"""Bayesian linear predictive uncertainty (fast; apt for scarce data).

BayesianRidge yields an analytic predictive mean AND standard deviation, giving a Gaussian
predictive distribution per prospect with no sampling cost — a pragmatic Bayesian route for the
small-sample draft problem. (A full hierarchical PyMC model is a future option; PyMC is
installed but heavier to run.)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


class BayesianLinearModel:
    """Standardized Bayesian ridge regression exposing predictive mean and std."""

    def __init__(self) -> None:
        self._scaler: Any | None = None
        self._model: Any | None = None

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> BayesianLinearModel:
        from sklearn.linear_model import BayesianRidge
        from sklearn.preprocessing import StandardScaler

        self._scaler = StandardScaler().fit(X)
        self._model = BayesianRidge()
        self._model.fit(self._scaler.transform(X), y)
        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._model is None or self._scaler is None:
            raise RuntimeError("BayesianLinearModel must be fit before predict.")
        return np.asarray(self._model.predict(self._scaler.transform(X)), dtype=np.float64)

    def predict_mean_std(
        self, X: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Predictive mean and standard deviation (Gaussian predictive distribution)."""
        if self._model is None or self._scaler is None:
            raise RuntimeError("BayesianLinearModel must be fit before predict_mean_std.")
        mean, std = self._model.predict(self._scaler.transform(X), return_std=True)
        return np.asarray(mean, dtype=np.float64), np.asarray(std, dtype=np.float64)
