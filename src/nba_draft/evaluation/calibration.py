"""Calibration tooling for probabilistic targets (e.g. T1 reach-probability).

For classification the question is not only "are the rankings right" but "do the probabilities
mean what they say" — if the model says 30%, do ~30% of such prospects actually reach? The
reliability table answers that; Brier/ECE (in metrics.py) summarize it.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from numpy.typing import ArrayLike


def calibration_table(
    y_true: ArrayLike, p_pred: ArrayLike, n_bins: int = 10
) -> pl.DataFrame:
    """Reliability table: per probability bin, the mean predicted prob vs observed frequency.

    A well-calibrated model has ``mean_pred`` ≈ ``frac_pos`` in every populated bin.
    """
    yt = np.asarray(y_true, dtype=float)
    pp = np.asarray(p_pred, dtype=float)
    if yt.shape != pp.shape:
        raise ValueError("y_true and p_pred must have equal shape.")
    if np.any((pp < 0) | (pp > 1)):
        raise ValueError("p_pred must be probabilities in [0, 1].")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, object]] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (pp > lo) & (pp <= hi) if lo > 0 else (pp >= lo) & (pp <= hi)
        count = int(in_bin.sum())
        rows.append(
            {
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "n": count,
                "mean_pred": float(pp[in_bin].mean()) if count else None,
                "frac_pos": float(yt[in_bin].mean()) if count else None,
            }
        )
    return pl.DataFrame(rows)
