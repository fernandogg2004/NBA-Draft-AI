"""Turn a predictive distribution into decision-facing outputs.

The headline deliverable of Phase 9: outcome-tier probabilities (P(bust / rotation / starter /
star / superstar)) and floor/ceiling, from either Monte-Carlo samples (ensemble) or a Gaussian
summary (mean, sd). Tier edges/labels come from config/targets.yaml.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _normal_cdf(x: float, mean: float, sd: float) -> float:
    if sd <= 0:
        return 1.0 if x >= mean else 0.0
    return 0.5 * (1.0 + math.erf((x - mean) / (sd * math.sqrt(2.0))))


def scenario_probabilities_from_normal(
    mean: float, sd: float, edges: list[float], labels: list[str]
) -> dict[str, float]:
    """Tier probabilities from a Gaussian predictive distribution.

    ``edges`` are tier boundaries (use ±1e9 for open ends); ``labels`` has len(edges)-1 entries.
    """
    if len(labels) != len(edges) - 1:
        raise ValueError("labels must have len(edges)-1 entries.")
    probs: dict[str, float] = {}
    for lo, hi, name in zip(edges[:-1], edges[1:], labels, strict=True):
        probs[name] = max(0.0, _normal_cdf(hi, mean, sd) - _normal_cdf(lo, mean, sd))
    total = sum(probs.values()) or 1.0
    return {k: round(v / total, 4) for k, v in probs.items()}


def scenario_probabilities_from_samples(
    samples: ArrayLike, edges: list[float], labels: list[str]
) -> dict[str, float]:
    """Tier probabilities from Monte-Carlo predictive samples (fraction of mass per tier)."""
    if len(labels) != len(edges) - 1:
        raise ValueError("labels must have len(edges)-1 entries.")
    s = np.asarray(samples, dtype=float)
    n = s.size
    if n == 0:
        raise ValueError("samples is empty.")
    # interior edges -> bin index in [0, len(labels)-1]
    idx = np.clip(np.digitize(s, edges[1:-1], right=False), 0, len(labels) - 1)
    counts = np.bincount(idx, minlength=len(labels))
    return {name: round(float(c) / n, 4) for name, c in zip(labels, counts, strict=True)}


def ceiling_floor(
    samples: ArrayLike, *, floor_q: float = 0.1, ceiling_q: float = 0.9
) -> tuple[float, float]:
    """Floor and ceiling as low/high quantiles of the predictive samples."""
    s = np.asarray(samples, dtype=float)
    if s.size == 0:
        raise ValueError("samples is empty.")
    return float(np.quantile(s, floor_q)), float(np.quantile(s, ceiling_q))


def interval_coverage(
    y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike
) -> float:
    """Empirical fraction of true values within [lower, upper] — interval calibration check."""
    yt = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if not (yt.shape == lo.shape == hi.shape):
        raise ValueError("y_true, lower, upper must share shape.")
    inside: NDArray[np.bool_] = (yt >= lo) & (yt <= hi)
    return float(inside.mean())
