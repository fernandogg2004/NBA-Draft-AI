"""Shared estimator protocol.

Lives in a dependency-free module so both the modeling zoo and the validation runner can refer
to it without an import cycle. Any sklearn-style estimator (fit/predict on numpy) satisfies it.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class Estimator(Protocol):
    """Minimal sklearn-style estimator interface used throughout modeling/evaluation."""

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> object: ...
    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]: ...
