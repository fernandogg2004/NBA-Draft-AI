"""Feature attribution: SHAP (global + local) and model-agnostic permutation importance.

SHAP gives additive, per-prediction credit ("this prospect ranks high *because* of youth +
efficiency"). Permutation importance is a robust, library-light cross-check that works for any
object exposing ``predict`` (including our adapters), scored on a ranking metric by default.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from nba_draft.evaluation.metrics import spearman_corr


class ShapExplainer:
    """SHAP explainer built on a model's ``predict`` and a background sample.

    Function-based so it works uniformly across pipelines, boosters, and our adapters.
    """

    def __init__(
        self,
        model: Any,
        background: NDArray[np.float64],
        feature_names: list[str],
        *,
        max_background: int = 100,
        seed: int = 42,
    ) -> None:
        import shap

        self.feature_names = list(feature_names)
        bg = np.asarray(background, dtype=np.float64)
        if bg.shape[0] > max_background:
            rng = np.random.default_rng(seed)
            bg = bg[rng.choice(bg.shape[0], max_background, replace=False)]
        self._explainer = shap.Explainer(model.predict, bg)

    def _explain(self, X: NDArray[np.float64]) -> Any:
        return self._explainer(np.asarray(X, dtype=np.float64))

    def global_importance(self, X: NDArray[np.float64]) -> pl.DataFrame:
        """Mean absolute SHAP value per feature (global importance), sorted descending."""
        values = self._explain(X).values
        mean_abs = np.abs(values).mean(axis=0)
        return pl.DataFrame(
            {"feature": self.feature_names, "mean_abs_shap": [float(v) for v in mean_abs]}
        ).sort("mean_abs_shap", descending=True)

    def local_explanation(self, x_row: NDArray[np.float64]) -> tuple[pl.DataFrame, float]:
        """Per-feature SHAP contributions for ONE prospect, plus the base value.

        Returns (table sorted by absolute contribution, base_value). By SHAP additivity,
        base_value + sum(shap) ≈ model.predict(x_row).
        """
        x = np.asarray(x_row, dtype=np.float64).reshape(1, -1)
        exp = self._explain(x)
        values = np.asarray(exp.values[0], dtype=float)
        base = exp.base_values[0]
        base_val = float(np.asarray(base).ravel()[0])
        table = pl.DataFrame(
            {"feature": self.feature_names, "shap_value": [float(v) for v in values]}
        ).with_columns(pl.col("shap_value").abs().alias("abs"))
        return table.sort("abs", descending=True).drop("abs"), base_val


def permutation_importance_table(
    model: Any,
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    feature_names: list[str],
    *,
    metric_fn: Callable[[NDArray[np.float64], NDArray[np.float64]], float] = spearman_corr,
    n_repeats: int = 5,
    seed: int = 42,
) -> pl.DataFrame:
    """Model-agnostic permutation importance: drop in a (higher-is-better) metric when a feature
    is shuffled. Larger drop => more important. Works for any model with ``predict``.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rng = np.random.default_rng(seed)
    base = metric_fn(y, np.asarray(model.predict(X), dtype=float))
    rows: list[dict[str, object]] = []
    for j, name in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            xp = X.copy()
            xp[:, j] = rng.permutation(xp[:, j])
            score = metric_fn(y, np.asarray(model.predict(xp), dtype=float))
            drops.append(base - score)
        rows.append({"feature": name, "importance": float(np.mean(drops))})
    return pl.DataFrame(rows).sort("importance", descending=True)
