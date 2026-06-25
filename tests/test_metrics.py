"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from nba_draft.evaluation.metrics import (
    brier_score,
    expected_calibration_error,
    kendall_tau,
    rmse,
    spearman_corr,
    top_k_hit_rate,
)


def test_spearman_perfect_monotone():
    yt = np.array([1.0, 2, 3, 4, 5])
    yp = np.array([10.0, 20, 33, 41, 55])  # same order
    assert spearman_corr(yt, yp) == pytest.approx(1.0)
    assert spearman_corr(yt, yp[::-1]) == pytest.approx(-1.0)


def test_spearman_handles_ties():
    yt = np.array([1.0, 1, 2, 3])
    yp = np.array([5.0, 5, 6, 7])
    assert spearman_corr(yt, yp) == pytest.approx(1.0)


def test_kendall_tau_bounds():
    yt = np.array([1.0, 2, 3, 4])
    assert kendall_tau(yt, yt) == pytest.approx(1.0)
    assert kendall_tau(yt, yt[::-1]) == pytest.approx(-1.0)


def test_top_k_hit_rate():
    yt = np.array([10.0, 9, 8, 1, 2, 3])
    yp = np.array([100.0, 90, 1, 2, 3, 4])  # top-3 truth = {0,1,2}; pred top-3 = {0,1,5}
    assert top_k_hit_rate(yt, yp, k=3) == pytest.approx(2 / 3)
    # k larger than n clamps
    assert top_k_hit_rate(yt, yp, k=99) == pytest.approx(1.0)


def test_rmse():
    assert rmse(np.array([1.0, 2, 3]), np.array([1.0, 2, 3])) == 0.0
    assert rmse(np.array([0.0, 0]), np.array([3.0, 4])) == pytest.approx(np.sqrt(12.5))


def test_brier_and_ece_perfect():
    y = np.array([0.0, 1, 0, 1])
    p = np.array([0.0, 1, 0, 1])
    assert brier_score(y, p) == 0.0
    assert expected_calibration_error(y, p, n_bins=5) == pytest.approx(0.0)


def test_brier_rejects_out_of_range():
    with pytest.raises(ValueError):
        brier_score(np.array([0.0, 1]), np.array([0.5, 1.5]))
