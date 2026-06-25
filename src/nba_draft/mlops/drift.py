"""Drift monitoring via the Population Stability Index (PSI).

Each year a new draft class arrives and the college/international landscape shifts (Transfer
Portal/NIL, rule changes). PSI compares a feature's distribution in new data against a reference
(the training data), flagging features whose distribution has moved enough to threaten the model.

Rule of thumb: PSI < 0.1 stable; 0.1-0.25 moderate shift; > 0.25 significant shift.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from numpy.typing import ArrayLike

_EPS = 1e-6


def population_stability_index(
    expected: ArrayLike, actual: ArrayLike, *, n_bins: int = 10
) -> float:
    """PSI between a reference (`expected`) and new (`actual`) sample of one feature.

    Bins are quantile edges of the reference, so each reference bin holds ~equal mass.
    """
    exp = np.asarray(expected, dtype=float)
    act = np.asarray(actual, dtype=float)
    exp = exp[~np.isnan(exp)]
    act = act[~np.isnan(act)]
    if exp.size == 0 or act.size == 0:
        return float("nan")
    edges = np.unique(np.quantile(exp, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    exp_prop = np.histogram(exp, bins=edges)[0] / exp.size
    act_prop = np.histogram(act, bins=edges)[0] / act.size
    exp_prop = np.clip(exp_prop, _EPS, None)
    act_prop = np.clip(act_prop, _EPS, None)
    return float(np.sum((act_prop - exp_prop) * np.log(act_prop / exp_prop)))


def feature_drift_report(
    reference: pl.DataFrame,
    current: pl.DataFrame,
    feature_cols: list[str],
    *,
    n_bins: int = 10,
    psi_threshold: float = 0.25,
) -> pl.DataFrame:
    """PSI per feature with a drift flag, sorted by PSI descending."""
    rows: list[dict[str, object]] = []
    for c in feature_cols:
        if c not in reference.columns or c not in current.columns:
            continue
        psi = population_stability_index(
            reference[c].to_numpy(), current[c].to_numpy(), n_bins=n_bins
        )
        rows.append(
            {"feature": c, "psi": round(psi, 4), "drifted": bool(psi > psi_threshold)}
        )
    return pl.DataFrame(rows).sort("psi", descending=True, nulls_last=True)
