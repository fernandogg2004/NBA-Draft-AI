"""Strictly temporal splitting by draft year.

Why this exists (instructions.md, domain risk #1): the real task is to predict the future
of players who have not debuted. Random k-fold splits leak the future into training and
inflate every metric. So ALL validation here is temporal:

  * ``holdout_split`` carves the most-recent N draft classes off as an untouchable test set.
  * ``walk_forward_folds`` produces train/validation folds where EVERY training row is from
    a strictly earlier draft year than EVERY validation row.

Both are guarded: if any fold would put a training year on or after a validation year, a
:class:`LeakageError` is raised rather than silently returning a leaky split.

We split on the *group* (draft year), never on rows, so all prospects from a class stay
together — splitting within a class would also leak (same-class players share era/context).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


class LeakageError(RuntimeError):
    """Raised when a requested split would leak future information into training."""


@dataclass(frozen=True)
class TemporalFold:
    """One walk-forward fold.

    Attributes:
        train_idx: Row indices (into the original array) for training.
        val_idx: Row indices for validation.
        train_years: Sorted distinct draft years in the training set.
        val_years: Sorted distinct draft years in the validation set.
    """

    train_idx: NDArray[np.intp]
    val_idx: NDArray[np.intp]
    train_years: tuple[int, ...]
    val_years: tuple[int, ...]

    def assert_no_leakage(self) -> None:
        """Fail loudly if max(train year) >= min(val year)."""
        if not self.train_years or not self.val_years:
            raise LeakageError("A fold has an empty train or validation set.")
        if max(self.train_years) >= min(self.val_years):
            raise LeakageError(
                f"Temporal leakage: train years {self.train_years} overlap/exceed "
                f"validation years {self.val_years}."
            )


def _as_year_array(years: ArrayLike) -> NDArray[np.int64]:
    arr = np.asarray(years)
    if arr.ndim != 1:
        raise ValueError("`years` must be 1-dimensional (one draft year per row).")
    if arr.size == 0:
        raise ValueError("`years` is empty.")
    return arr.astype(np.int64)


def holdout_split(
    years: ArrayLike,
    n_holdout_years: int,
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Split rows into (development, holdout) by draft year.

    The holdout is the most-recent ``n_holdout_years`` distinct draft classes. It is the
    untouchable test set: no tuning, no peeking, until final evaluation.

    Args:
        years: Draft year per row.
        n_holdout_years: Number of most-recent distinct classes to reserve.

    Returns:
        ``(dev_idx, holdout_idx)`` row-index arrays.

    Raises:
        LeakageError: if the resulting split is not strictly temporal, or there are not
            enough distinct years to leave any development data.
    """
    yr = _as_year_array(years)
    distinct = np.unique(yr)
    if n_holdout_years < 1:
        raise ValueError("n_holdout_years must be >= 1.")
    if n_holdout_years >= distinct.size:
        raise LeakageError(
            f"n_holdout_years={n_holdout_years} leaves no development years "
            f"(only {distinct.size} distinct draft years available)."
        )
    cutoff = distinct[-n_holdout_years]  # first year that belongs to the holdout
    dev_idx = np.nonzero(yr < cutoff)[0].astype(np.intp)
    holdout_idx = np.nonzero(yr >= cutoff)[0].astype(np.intp)
    if dev_idx.size == 0 or holdout_idx.size == 0:
        raise LeakageError("Holdout split produced an empty side.")
    if yr[dev_idx].max() >= yr[holdout_idx].min():
        raise LeakageError("Holdout split is not strictly temporal.")
    return dev_idx, holdout_idx


def walk_forward_folds(
    years: ArrayLike,
    *,
    min_train_years: int,
    val_horizon_years: int = 1,
    step_years: int = 1,
    expanding: bool = True,
) -> Iterator[TemporalFold]:
    """Yield strictly-temporal walk-forward folds over draft years.

    Starting after the first ``min_train_years`` distinct classes, each fold trains on past
    classes and validates on the next ``val_horizon_years`` classes, then advances by
    ``step_years``. With ``expanding=True`` the training window grows to include all prior
    years; with ``expanding=False`` it slides, keeping a fixed ``min_train_years`` span.

    Args:
        years: Draft year per row (typically the development set, holdout already removed).
        min_train_years: Distinct classes required before the first validation fold.
        val_horizon_years: Distinct classes each validation fold spans.
        step_years: Stride (in distinct classes) between successive folds.
        expanding: Expanding (True) vs sliding (False) training window.

    Yields:
        :class:`TemporalFold` objects, each self-checked for leakage before being yielded.

    Raises:
        LeakageError: if no valid fold can be produced (e.g. too few distinct years).
    """
    yr = _as_year_array(years)
    distinct = np.unique(yr)  # ascending
    n = distinct.size

    if min_train_years < 1 or val_horizon_years < 1 or step_years < 1:
        raise ValueError("min_train_years, val_horizon_years, step_years must be >= 1.")
    if min_train_years + val_horizon_years > n:
        raise LeakageError(
            f"Not enough distinct draft years ({n}) for min_train_years="
            f"{min_train_years} + val_horizon_years={val_horizon_years}."
        )

    produced = 0
    start = min_train_years  # index into `distinct` where the first validation block begins
    while start + val_horizon_years <= n:
        val_years = distinct[start : start + val_horizon_years]
        if expanding:
            train_years = distinct[:start]
        else:
            train_lo = max(0, start - min_train_years)
            train_years = distinct[train_lo:start]

        train_mask = np.isin(yr, train_years)
        val_mask = np.isin(yr, val_years)
        fold = TemporalFold(
            train_idx=np.nonzero(train_mask)[0].astype(np.intp),
            val_idx=np.nonzero(val_mask)[0].astype(np.intp),
            train_years=tuple(int(y) for y in train_years),
            val_years=tuple(int(y) for y in val_years),
        )
        fold.assert_no_leakage()  # belt-and-suspenders: never yield a leaky fold
        produced += 1
        yield fold
        start += step_years

    if produced == 0:
        raise LeakageError("No walk-forward folds could be produced from the given years.")
