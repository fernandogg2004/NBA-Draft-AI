"""Hurdle model: the survivorship-robust ranking the project is built around.

Combines the two heads defined in Phase 0 into one unconditional expected value:

    EV = P(reach) * E(impact | reached) + (1 - P(reach)) * replacement

- P(reach) is trained over ALL prospects (it learns who washes out).
- E(impact | reached) is trained only on prospects who reached a role.
- The product down-weights likely non-reachers, so the final ranking is NOT conditioned on
  having reached — which is exactly what defeats survivorship bias (instructions.md risk #2).

This is the production ranking head. It is fit on the training fold only (the reach classifier
and the impact regressor each see train data only), so it is leakage-safe inside the temporal CV.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from nba_draft.models.zoo import logistic_classifier, ridge_regressor

REPLACEMENT_BPM = -2.0


def realized_value(
    reached: NDArray[np.float64],
    impact: NDArray[np.float64],
    *,
    replacement: float = REPLACEMENT_BPM,
) -> NDArray[np.float64]:
    """The honest unconditional realized value per prospect: impact if reached, else replacement.

    This is the target the hurdle EV is evaluated against across ALL prospects (reached or not),
    so non-reachers are scored at replacement rather than dropped.
    """
    reached_arr = np.asarray(reached, dtype=float)
    impact_arr = np.asarray(impact, dtype=float)
    out = np.where(reached_arr > 0.5, impact_arr, replacement)
    # NaN impact for non-reachers is irrelevant (already replaced); guard reached-but-NaN too.
    return np.where(np.isnan(out), replacement, out).astype(np.float64)


class HurdleModel:
    """Two-part hurdle: reach classifier × conditional impact regressor → unconditional EV."""

    def __init__(
        self,
        *,
        reach_factory: Callable[[], Any] = logistic_classifier,
        impact_factory: Callable[[], Any] = lambda: ridge_regressor(1.0),
        replacement: float = REPLACEMENT_BPM,
    ) -> None:
        # Factories are needed only at fit time; they are excluded from the pickled state
        # (see __getstate__) so a fitted HurdleModel serializes cleanly into the model registry.
        self.reach_factory: Callable[[], Any] | None = reach_factory
        self.impact_factory: Callable[[], Any] | None = impact_factory
        self.replacement = replacement
        self._reach: Any | None = None
        self._impact: Any | None = None

    def __getstate__(self) -> dict[str, Any]:
        # Drop the (possibly lambda/partial) factories; keep the fitted models + replacement.
        return {"_reach": self._reach, "_impact": self._impact, "replacement": self.replacement}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._reach = state["_reach"]
        self._impact = state["_impact"]
        self.replacement = state["replacement"]
        self.reach_factory = None
        self.impact_factory = None

    def fit(
        self,
        X: NDArray[np.float64],
        reached: NDArray[np.float64],
        impact: NDArray[np.float64],
    ) -> HurdleModel:
        """Fit the reach head on all rows and the impact head on reached rows only.

        Args:
            X: feature matrix for all prospects.
            reached: 0/1 reach label per row.
            impact: conditional impact per row (value for reached; NaN/ignored for non-reached).
        """
        if self.reach_factory is None or self.impact_factory is None:
            raise RuntimeError("HurdleModel was deserialized without factories; cannot refit.")
        reached_arr = np.asarray(reached, dtype=float)
        mask = reached_arr > 0.5
        if mask.sum() < 2:
            raise ValueError("Need at least 2 reached prospects to fit the impact head.")
        self._reach = self.reach_factory()
        self._reach.fit(X, reached_arr)
        self._impact = self.impact_factory()
        self._impact.fit(X[mask], np.asarray(impact, dtype=float)[mask])
        return self

    def predict_parts(
        self, X: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return (P(reach), conditional impact) for transparency."""
        if self._reach is None or self._impact is None:
            raise RuntimeError("HurdleModel must be fit before predict.")
        p_reach = np.clip(np.asarray(self._reach.predict(X), dtype=float), 0.0, 1.0)
        impact = np.asarray(self._impact.predict(X), dtype=float)
        return p_reach, impact

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Unconditional expected value used to rank prospects."""
        p_reach, impact = self.predict_parts(X)
        return p_reach * impact + (1.0 - p_reach) * self.replacement
