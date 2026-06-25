"""Bootstrap ensemble: predictive samples -> intervals + scenario probabilities.

Each member is trained on a bootstrap resample; the spread of member predictions is the
predictive distribution. This is a model-agnostic way to get the outcome-tier distribution
(P(bust/.../superstar)) the decision needs, for any base estimator.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from nba_draft.uncertainty.scenarios import (
    ceiling_floor,
    scenario_probabilities_from_samples,
)


class BootstrapEnsemble:
    """Bag of base estimators over bootstrap resamples."""

    def __init__(
        self,
        base_factory: Callable[[], Any],
        *,
        n_estimators: int = 30,
        seed: int = 42,
    ) -> None:
        self.base_factory = base_factory
        self.n_estimators = n_estimators
        self.seed = seed
        self._models: list[Any] = []

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> BootstrapEnsemble:
        rng = np.random.default_rng(self.seed)
        n = X.shape[0]
        self._models = []
        for _ in range(self.n_estimators):
            idx = rng.integers(0, n, size=n)  # bootstrap resample with replacement
            model = self.base_factory()
            model.fit(X[idx], y[idx])
            self._models.append(model)
        return self

    def predict_samples(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """(n_rows, n_estimators) matrix of member predictions."""
        if not self._models:
            raise RuntimeError("BootstrapEnsemble must be fit before predicting.")
        return np.column_stack([m.predict(X) for m in self._models])

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(self.predict_samples(X).mean(axis=1), dtype=np.float64)

    def predict_interval(
        self, X: NDArray[np.float64], *, alpha: float = 0.2
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        samples = self.predict_samples(X)
        lo = np.quantile(samples, alpha / 2, axis=1)
        hi = np.quantile(samples, 1 - alpha / 2, axis=1)
        return np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)

    def predict_scenarios(
        self, X: NDArray[np.float64], edges: list[float], labels: list[str]
    ) -> list[dict[str, float]]:
        """Per-row outcome-tier probabilities from the ensemble's predictive samples."""
        samples = self.predict_samples(X)
        return [
            scenario_probabilities_from_samples(samples[i], edges, labels)
            for i in range(samples.shape[0])
        ]

    def predict_floor_ceiling(
        self, X: NDArray[np.float64], *, floor_q: float = 0.1, ceiling_q: float = 0.9
    ) -> list[tuple[float, float]]:
        samples = self.predict_samples(X)
        return [
            ceiling_floor(samples[i], floor_q=floor_q, ceiling_q=ceiling_q)
            for i in range(samples.shape[0])
        ]
