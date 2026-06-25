"""Metrics for the draft problem.

Ranking metrics (Spearman, Kendall tau, top-k hit rate) are primary because the tool's job
is to order prospects. Calibration metrics (Brier, ECE) are the classification-target
counterpart for Phase 9. RMSE is kept only as a secondary error gauge.

Implemented with NumPy/SciPy-free code to keep the Milestone-0 dependency surface tiny.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _rankdata(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Average-rank of values (ties share the mean rank). Mirrors scipy.stats.rankdata."""
    a = np.asarray(values, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.shape[0], dtype=np.float64)
    ranks[order] = np.arange(1, a.shape[0] + 1, dtype=np.float64)
    # Resolve ties to average rank.
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.shape[0], dtype=np.float64)
    np.add.at(sums, inv, ranks)
    avg = sums / counts
    return np.asarray(avg[inv], dtype=np.float64)


def spearman_corr(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Spearman rank correlation between truth and prediction."""
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    if yt.shape != yp.shape or yt.ndim != 1:
        raise ValueError("y_true and y_pred must be 1-D arrays of equal length.")
    if yt.shape[0] < 2:
        raise ValueError("Need at least 2 observations.")
    rt, rp = _rankdata(yt), _rankdata(yp)
    rt -= rt.mean()
    rp -= rp.mean()
    denom = np.sqrt((rt**2).sum() * (rp**2).sum())
    if denom == 0:
        return 0.0
    return float((rt * rp).sum() / denom)


def kendall_tau(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Kendall tau-a: (concordant - discordant) / total pairs. O(n^2), fine for draft sizes."""
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    if yt.shape != yp.shape or yt.ndim != 1:
        raise ValueError("y_true and y_pred must be 1-D arrays of equal length.")
    n = yt.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 observations.")
    concordant = discordant = 0
    for i in range(n - 1):
        dt = yt[i + 1 :] - yt[i]
        dp = yp[i + 1 :] - yp[i]
        s = np.sign(dt) * np.sign(dp)
        concordant += int((s > 0).sum())
        discordant += int((s < 0).sum())
    total = n * (n - 1) // 2
    return float((concordant - discordant) / total)


def top_k_hit_rate(y_true: ArrayLike, y_pred: ArrayLike, k: int) -> float:
    """Fraction of the true top-k that appear in the predicted top-k.

    Directly answers "how many of the genuinely best prospects did we rank in our top k".
    """
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    if yt.shape != yp.shape or yt.ndim != 1:
        raise ValueError("y_true and y_pred must be 1-D arrays of equal length.")
    n = yt.shape[0]
    if k < 1:
        raise ValueError("k must be >= 1.")
    k = min(k, n)
    true_top = set(np.argsort(-yt, kind="mergesort")[:k].tolist())
    pred_top = set(np.argsort(-yp, kind="mergesort")[:k].tolist())
    return len(true_top & pred_top) / k


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root mean squared error (secondary, error-magnitude metric)."""
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have equal shape.")
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def brier_score(y_true: ArrayLike, p_pred: ArrayLike) -> float:
    """Brier score for binary outcomes (calibration of probabilistic milestone targets)."""
    yt, pp = np.asarray(y_true, float), np.asarray(p_pred, float)
    if yt.shape != pp.shape:
        raise ValueError("y_true and p_pred must have equal shape.")
    if np.any((pp < 0) | (pp > 1)):
        raise ValueError("p_pred must be probabilities in [0, 1].")
    return float(np.mean((pp - yt) ** 2))


def expected_calibration_error(
    y_true: ArrayLike, p_pred: ArrayLike, n_bins: int = 10
) -> float:
    """Expected Calibration Error with equal-width probability bins."""
    yt, pp = np.asarray(y_true, float), np.asarray(p_pred, float)
    if yt.shape != pp.shape:
        raise ValueError("y_true and p_pred must have equal shape.")
    if np.any((pp < 0) | (pp > 1)):
        raise ValueError("p_pred must be probabilities in [0, 1].")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = yt.shape[0]
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (pp > lo) & (pp <= hi) if lo > 0 else (pp >= lo) & (pp <= hi)
        count = int(in_bin.sum())
        if count == 0:
            continue
        conf = float(pp[in_bin].mean())
        acc = float(yt[in_bin].mean())
        ece += (count / n) * abs(acc - conf)
    return ece
