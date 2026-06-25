"""Survival modeling for career longevity (target T5), via lifelines Cox PH.

Longevity is right-censored: recent draft classes have not finished their careers, so we must
not treat "still playing / not yet observed" as a short career. A Cox proportional-hazards model
handles censoring directly. lifelines is an optional dependency (``models`` extra); imports lazy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray


def _to_pandas(df: pl.DataFrame, cols: list[str]) -> Any:
    """Build a pandas DataFrame from numpy columns (avoids the pyarrow dependency)."""
    import pandas as pd

    return pd.DataFrame({c: df[c].to_numpy() for c in cols})


class CoxSurvivalModel:
    """Cox proportional-hazards model over pre-draft features.

    Higher predicted risk == shorter expected career. ``concordance`` (below) is the natural,
    censoring-aware ranking metric.
    """

    def __init__(self, *, penalizer: float = 0.1, l1_ratio: float = 0.0) -> None:
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self._fitter: Any | None = None
        self._feature_cols: list[str] = []

    def fit(
        self,
        df: pl.DataFrame,
        *,
        feature_cols: list[str],
        duration_col: str,
        event_col: str,
    ) -> CoxSurvivalModel:
        from lifelines import CoxPHFitter

        self._feature_cols = list(feature_cols)
        pdf = _to_pandas(df, [*feature_cols, duration_col, event_col])
        fitter = CoxPHFitter(penalizer=self.penalizer, l1_ratio=self.l1_ratio)
        fitter.fit(pdf, duration_col=duration_col, event_col=event_col)
        self._fitter = fitter
        return self

    def predict_risk(self, df: pl.DataFrame) -> NDArray[np.float64]:
        """Partial-hazard risk score (higher = higher hazard = shorter career)."""
        if self._fitter is None:
            raise RuntimeError("CoxSurvivalModel must be fit before predict_risk().")
        pdf = _to_pandas(df, self._feature_cols)
        risk = self._fitter.predict_partial_hazard(pdf).to_numpy()
        return np.asarray(risk, dtype=np.float64)


def concordance(
    durations: NDArray[np.float64],
    events: NDArray[np.float64],
    risk: NDArray[np.float64],
) -> float:
    """Censoring-aware concordance index (0.5 = chance, 1.0 = perfect ordering).

    ``risk`` is higher-is-shorter-career, so we pass its negative as the survival-ordering score.
    """
    from lifelines.utils import concordance_index

    return float(concordance_index(durations, -np.asarray(risk, dtype=float), events))
