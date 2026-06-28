"""Honors-aware outcome-tier classifier.

Predicts the probability of each outcome tier directly from pre-draft features, trained on the
honors-aware ``outcome_tier`` label (real All-Star/All-NBA selections, not just BPM bands). This
replaces mapping a single predicted BPM through fixed bands, which is poorly calibrated against the
true tier definition. Multinomial logistic regression: well-calibrated and stable on small samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class TierProbabilityModel:
    """A fitted multiclass tier classifier mapped onto a fixed tier-label ordering."""

    estimator: Any
    labels: list[str]            # full ordered tier vocabulary (e.g. TIER_LABELS)
    _classes: list[int]          # tier indices the estimator actually learned

    @classmethod
    def fit(
        cls,
        x: NDArray[np.float64],
        tier_index: NDArray[np.int64],
        labels: list[str],
        *,
        seed: int = 42,
    ) -> TierProbabilityModel:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        est = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed)
        )
        est.fit(x, tier_index)
        return cls(estimator=est, labels=labels, _classes=[int(c) for c in est.classes_])

    def predict_proba_matrix(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """(n, n_labels) tier probabilities aligned to ``labels`` (unseen tiers -> 0)."""
        proba = np.asarray(self.estimator.predict_proba(x), dtype=np.float64)
        out = np.zeros((x.shape[0], len(self.labels)), dtype=np.float64)
        for j, tier_idx in enumerate(self._classes):
            out[:, tier_idx] = proba[:, j]
        return out

    def predict_scenarios(self, x: NDArray[np.float64]) -> list[dict[str, float]]:
        """Per-row {tier_label: probability} (rounded), matching the ensemble/conformal API."""
        mat = self.predict_proba_matrix(x)
        return [
            {label: round(float(mat[i, k]), 4) for k, label in enumerate(self.labels)}
            for i in range(mat.shape[0])
        ]
