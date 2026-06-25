"""The baseline to beat: draft position.

instructions.md (Phase 6 & 7) requires an honest comparison against the draft-position
baseline. Draft order already encodes the aggregated judgement of every NBA front office,
so it is a strong, hard-to-beat predictor of NBA impact. Any model we build must demonstrate
it improves on simply trusting the draft order — under temporal validation.

This baseline learns a monotone-ish mapping from pick number to expected impact on the
TRAIN set only (no leakage), then predicts expected impact for validation picks. Because the
ranking it implies is just "earlier pick = better", its ranking metrics equal those of raw
draft order; the fitted magnitude only matters for error-based metrics.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


class DraftPositionBaseline:
    """Predict NBA impact from draft pick via a log-pick linear fit.

    impact_hat = a + b * log(pick). Fitted on training data; ``b`` is expected to be
    negative (earlier pick -> higher impact).
    """

    def __init__(self) -> None:
        self._a: float | None = None
        self._b: float | None = None

    def fit(self, picks: ArrayLike, targets: ArrayLike) -> DraftPositionBaseline:
        p = np.asarray(picks, dtype=np.float64)
        t = np.asarray(targets, dtype=np.float64)
        if p.ndim != 1 or t.ndim != 1 or p.shape != t.shape:
            raise ValueError("picks and targets must be 1-D arrays of equal length.")
        if np.any(p < 1):
            raise ValueError("Draft picks must be >= 1.")
        x = np.log(p)
        # Ordinary least squares via polyfit (degree 1); robust enough for a baseline.
        b, a = np.polyfit(x, t, deg=1)
        self._b, self._a = float(b), float(a)
        return self

    def predict(self, picks: ArrayLike) -> NDArray[np.float64]:
        if self._a is None or self._b is None:
            raise RuntimeError("Baseline must be fit before predict().")
        p = np.asarray(picks, dtype=np.float64)
        if np.any(p < 1):
            raise ValueError("Draft picks must be >= 1.")
        return np.asarray(self._a + self._b * np.log(p), dtype=np.float64)


class DraftPositionEstimator:
    """Runner-compatible adapter: the draft-position baseline as a fit/predict estimator.

    Reads the draft pick from a fixed column of the feature matrix, so the baseline competes
    against real models through the exact same temporal-CV protocol (the comparison Phase 7 needs).
    """

    def __init__(self, pick_index: int) -> None:
        self.pick_index = pick_index
        self._model = DraftPositionBaseline()

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> DraftPositionEstimator:
        self._model.fit(X[:, self.pick_index], y)
        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._model.predict(X[:, self.pick_index])
