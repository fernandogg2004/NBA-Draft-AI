"""Tests for the temporal validation primitives — the project's anti-leakage core.

These are the most important tests in Milestone 0: they prove no fold can train on the future.
"""

from __future__ import annotations

import numpy as np
import pytest

from nba_draft.validation.temporal import (
    LeakageError,
    holdout_split,
    walk_forward_folds,
)


def _years(spec: dict[int, int]) -> np.ndarray:
    """Build a year-per-row array from {year: count}."""
    out: list[int] = []
    for yr, n in spec.items():
        out.extend([yr] * n)
    return np.array(out)


def test_holdout_reserves_most_recent_years():
    years = _years({2018: 3, 2019: 2, 2020: 2, 2021: 4})
    dev, hold = holdout_split(years, n_holdout_years=2)
    assert set(years[dev].tolist()) == {2018, 2019}
    assert set(years[hold].tolist()) == {2020, 2021}
    # strictly temporal
    assert years[dev].max() < years[hold].min()


def test_holdout_rejects_when_no_dev_left():
    years = _years({2020: 5, 2021: 5})
    with pytest.raises(LeakageError):
        holdout_split(years, n_holdout_years=2)


def test_walk_forward_is_strictly_temporal():
    years = _years({y: 5 for y in range(2010, 2020)})
    folds = list(
        walk_forward_folds(years, min_train_years=4, val_horizon_years=1, step_years=1)
    )
    assert len(folds) == 6  # years 2014..2019 each become a validation fold
    for f in folds:
        # core invariant: every train year strictly precedes every val year
        assert max(f.train_years) < min(f.val_years)
        # indices are disjoint
        assert set(f.train_idx.tolist()).isdisjoint(f.val_idx.tolist())
        f.assert_no_leakage()


def test_walk_forward_expanding_vs_sliding():
    years = _years({y: 2 for y in range(2010, 2018)})
    expanding = list(walk_forward_folds(years, min_train_years=3, expanding=True))
    sliding = list(walk_forward_folds(years, min_train_years=3, expanding=False))
    # Expanding window grows; sliding stays fixed-width.
    assert len(expanding[-1].train_years) > len(sliding[-1].train_years)
    assert len(sliding[-1].train_years) == 3
    # First fold identical for both.
    assert expanding[0].train_years == sliding[0].train_years


def test_walk_forward_multiyear_horizon_and_step():
    years = _years({y: 1 for y in range(2010, 2020)})
    folds = list(
        walk_forward_folds(years, min_train_years=4, val_horizon_years=2, step_years=2)
    )
    for f in folds:
        assert len(f.val_years) == 2
        assert max(f.train_years) < min(f.val_years)


def test_walk_forward_raises_when_insufficient_years():
    years = _years({2010: 5, 2011: 5})
    with pytest.raises(LeakageError):
        list(walk_forward_folds(years, min_train_years=4, val_horizon_years=1))


def test_empty_input_rejected():
    with pytest.raises(ValueError):
        holdout_split(np.array([], dtype=int), n_holdout_years=1)
