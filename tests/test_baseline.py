"""Tests for the draft-position baseline and the synthetic fixture."""

from __future__ import annotations

import numpy as np
import pytest

from nba_draft.data.fixtures import (
    FEATURE_COLUMNS,
    PICK_COLUMN,
    TARGET_COLUMN,
    YEAR_COLUMN,
    make_synthetic_prospects,
)
from nba_draft.evaluation.metrics import spearman_corr
from nba_draft.models.baseline import DraftPositionBaseline


def test_baseline_learns_negative_slope():
    # earlier pick -> higher impact, so slope on log(pick) should be negative
    picks = np.arange(1, 31)
    targets = 10 - 2 * np.log(picks) + 0.0
    model = DraftPositionBaseline().fit(picks, targets)
    pred = model.predict(picks)
    assert spearman_corr(targets, pred) == pytest.approx(1.0, abs=1e-6)
    assert model.predict(np.array([1]))[0] > model.predict(np.array([30]))[0]


def test_baseline_requires_fit_first():
    with pytest.raises(RuntimeError):
        DraftPositionBaseline().predict(np.array([1, 2, 3]))


def test_baseline_rejects_invalid_picks():
    with pytest.raises(ValueError):
        DraftPositionBaseline().fit(np.array([0, 1]), np.array([1.0, 2.0]))


def test_fixture_is_deterministic_and_well_formed():
    a = make_synthetic_prospects(seed=7)
    b = make_synthetic_prospects(seed=7)
    assert a.equals(b)  # deterministic
    for col in [YEAR_COLUMN, PICK_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS]:
        assert col in a.columns
    # picks within a class are 1..N unique
    first_year = a[YEAR_COLUMN].min()
    cls = a.filter(a[YEAR_COLUMN] == first_year)
    picks = sorted(cls[PICK_COLUMN].to_list())
    assert picks == list(range(1, len(picks) + 1))


def test_fixture_baseline_has_predictive_signal():
    # On the fixture, draft position should carry real (imperfect) signal about impact.
    df = make_synthetic_prospects(seed=1)
    picks = df[PICK_COLUMN].to_numpy()
    target = df[TARGET_COLUMN].to_numpy()
    model = DraftPositionBaseline().fit(picks, target)
    sp = spearman_corr(target, model.predict(picks))
    assert sp > 0.2  # informative but far from perfect — beatable, as intended
