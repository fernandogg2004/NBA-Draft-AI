"""Tests for Phase 12 MLOps: registry, drift, retraining, tracking, end-to-end pipeline."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from nba_draft.mlops import (
    ExperimentTracker,
    RetrainingPolicy,
    feature_drift_report,
    list_models,
    load_model,
    population_stability_index,
    promote_model,
    register_model,
    run_pipeline,
    should_retrain,
)
from nba_draft.models.zoo import ridge_regressor


# ----------------------------------------------------------------- registry
def test_registry_roundtrip_and_promote(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 2))
    model = ridge_regressor(1.0)
    model.fit(x, x[:, 0])

    register_model(
        model, name="m", version="v1", metrics={"cv_spearman": 0.6},
        feature_cols=["a", "b"], data_version="d1", root=tmp_path,
    )
    register_model(model, name="m", version="v2", metrics={"cv_spearman": 0.7}, root=tmp_path)

    # latest resolves to v2; list is newest-first
    loaded = load_model("m", "latest", root=tmp_path)
    assert hasattr(loaded, "predict")
    versions = [m["version"] for m in list_models("m", root=tmp_path)]
    assert set(versions) == {"v1", "v2"}

    promote_model("m", "v1", "production", root=tmp_path)
    prod = load_model("m", "production", root=tmp_path)
    assert hasattr(prod, "predict")


def test_load_unknown_model_raises(tmp_path):
    with pytest.raises(KeyError):
        load_model("nope", root=tmp_path)


# ----------------------------------------------------------------- drift
def test_psi_zero_for_identical_and_large_for_shifted():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=2000)
    same = rng.normal(size=2000)
    shifted = rng.normal(loc=3.0, size=2000)
    assert population_stability_index(ref, same) < 0.1
    assert population_stability_index(ref, shifted) > 0.25


def test_feature_drift_report_flags_shifted_feature():
    rng = np.random.default_rng(2)
    ref = pl.DataFrame({"a": rng.normal(size=1000), "b": rng.normal(size=1000)})
    cur = pl.DataFrame({"a": rng.normal(size=1000), "b": rng.normal(loc=4.0, size=1000)})
    rep = feature_drift_report(ref, cur, ["a", "b"])
    flagged = dict(zip(rep["feature"], rep["drifted"], strict=True))
    assert flagged["b"] is True
    assert flagged["a"] is False


# ----------------------------------------------------------------- retraining
def test_should_retrain_on_new_class_or_drift():
    policy = RetrainingPolicy()
    # new draft class -> retrain
    decide, reasons = should_retrain(last_trained_year=2024, current_year=2025, policy=policy)
    assert decide and reasons
    # same year, no drift -> no retrain
    decide, _ = should_retrain(last_trained_year=2025, current_year=2025, max_feature_psi=0.05)
    assert decide is False
    # same year but big drift -> retrain
    decide, reasons = should_retrain(last_trained_year=2025, current_year=2025, max_feature_psi=0.4)
    assert decide and any("drift" in r for r in reasons)


# ----------------------------------------------------------------- tracking (graceful)
def test_tracker_disabled_is_noop():
    with ExperimentTracker(enabled=False) as t:
        t.log_params({"x": 1})
        t.log_metrics({"y": 2.0})
        t.set_tags({"z": "a"})  # must not raise


# ----------------------------------------------------------------- end-to-end pipeline
def test_run_pipeline_end_to_end(tmp_path):
    result = run_pipeline(
        output_root=tmp_path / "pipeline",
        model_root=tmp_path / "models",
        tracking_enabled=False,
    )
    # master built + model registered + summary written
    assert result.master_version
    assert result.model_version
    summary = json.loads((tmp_path / "pipeline" / "run_summary.json").read_text())
    assert summary["model_version"] == result.model_version

    # the registered model is loadable
    model = load_model("impact_regressor", "latest", root=tmp_path / "models")
    assert hasattr(model, "predict")

    # ridge should beat the draft-position baseline on the synthetic dev set
    comp = {row["model"]: row["spearman_mean"] for row in result.comparison}
    assert comp["ridge"] > comp["baseline_draftpos"]
