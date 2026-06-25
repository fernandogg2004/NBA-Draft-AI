"""The one-command, end-to-end reproducible pipeline.

Stages (all on synthetic fixtures for the demo; swap in real ingestion in production):
  1. integrate  — build the versioned master dataset (entity resolution + provenance)
  2. evaluate   — temporal-CV comparison of baseline vs real models (the bar to beat)
  3. train      — fit the final impact model on the development set
  4. register   — version the model with its metrics + data version
  5. monitor    — feature-drift report (dev = reference, holdout = "new class")
Everything is logged to MLflow (if enabled) and a JSON run summary is written.

Run:  python scripts/run_pipeline.py     (or `dvc repro` once git+DVC are initialized)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from nba_draft.cleaning.master import build_master
from nba_draft.config import load_config
from nba_draft.data.fixtures import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    YEAR_COLUMN,
    make_multisource_fixture,
    make_synthetic_prospects,
)
from nba_draft.evaluation.comparison import compare_models, make_spec
from nba_draft.mlops.drift import feature_drift_report
from nba_draft.mlops.registry import register_model
from nba_draft.mlops.tracking import ExperimentTracker
from nba_draft.models import DraftPositionEstimator, gbm_regressor, mean_regressor, ridge_regressor
from nba_draft.utils.logging import get_logger
from nba_draft.utils.seeds import set_global_seed
from nba_draft.validation import FoldPreprocessor, make_data_split

log = get_logger("mlops.pipeline")


@dataclass
class PipelineResult:
    master_version: str
    comparison: list[dict[str, Any]] = field(default_factory=list)
    model_name: str = "impact_regressor"
    model_version: str = ""
    drift: list[dict[str, Any]] = field(default_factory=list)
    summary_path: str = ""


def run_pipeline(
    *,
    seed: int | None = None,
    output_root: str | Path = "artifacts/pipeline",
    model_root: str | Path = "artifacts/models",
    tracking_enabled: bool = True,
    tracking_uri: str | None = None,
) -> PipelineResult:
    """Run the full pipeline and return a :class:`PipelineResult`."""
    cfg = load_config()
    seed = cfg.seed if seed is None else seed
    set_global_seed(seed)
    out = Path(output_root)
    out.mkdir(parents=True, exist_ok=True)
    feats = list(FEATURE_COLUMNS)
    mty = cfg.validation.min_train_years

    with ExperimentTracker(
        run_name=f"pipeline-{datetime.now(UTC):%Y%m%d-%H%M%S}",
        tracking_uri=tracking_uri,
        enabled=tracking_enabled,
    ) as tracker:
        tracker.log_params({"seed": seed, "min_train_years": mty, "n_features": len(feats)})

        # 1. integrate -> versioned master dataset (entity resolution across sources)
        fx = make_multisource_fixture()
        master = build_master(
            {"college_stats": fx["college_stats"], "intl_stats": fx["intl_stats"]},
            {"combine": fx["combine"]},
            output_root=out / "master",
        )
        tracker.set_tags({"master_version": master.version})

        # 2. evaluate (temporal CV) — models vs draft-position baseline
        df = make_synthetic_prospects(seed=seed)
        split = make_data_split(
            df, year_col=YEAR_COLUMN, n_holdout_years=cfg.validation.n_holdout_years
        )
        dev = split.dev
        specs = {
            "baseline_draftpos": make_spec(
                ["draft_pick"], lambda: DraftPositionEstimator(0),
                lambda: FoldPreprocessor(["draft_pick"]),
            ),
            "mean": make_spec(feats, mean_regressor, lambda: FoldPreprocessor(feats)),
            "ridge": make_spec(
                feats, lambda: ridge_regressor(1.0), lambda: FoldPreprocessor(feats)
            ),
            "gbm": make_spec(feats, gbm_regressor, lambda: FoldPreprocessor(feats)),
        }
        comparison = compare_models(
            dev, specs, target_col=TARGET_COLUMN, min_train_years=mty,
            baseline_name="baseline_draftpos",
        )
        best = comparison.row(0, named=True)
        tracker.log_metrics(
            {
                "best_spearman": float(best["spearman_mean"]),
                "baseline_spearman": float(
                    comparison.filter(pl.col("model") == "baseline_draftpos")["spearman_mean"][0]
                ),
            }
        )

        # 3. train final impact model on the dev set
        pp = FoldPreprocessor(feats).fit(dev)
        x_tr = pp.transform_matrix(dev).to_numpy()
        y_tr = dev[TARGET_COLUMN].to_numpy().astype(float)
        model = ridge_regressor(1.0)
        model.fit(x_tr, y_tr)

        # 4. register the model with metrics + data lineage
        version = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        ridge_spearman = float(
            comparison.filter(pl.col("model") == "ridge")["spearman_mean"][0]
        )
        register_model(
            model,
            name="impact_regressor",
            version=version,
            metrics={"cv_spearman": ridge_spearman},
            feature_cols=feats,
            data_version=master.version,
            root=model_root,
        )

        # 5. monitor — drift between dev (reference) and the held-out "new class"
        drift = feature_drift_report(dev, split.holdout, feats)

        summary = {
            "created_at": datetime.now(UTC).isoformat(),
            "seed": seed,
            "master_version": master.version,
            "model_name": "impact_regressor",
            "model_version": version,
            "comparison": comparison.to_dicts(),
            "drift": drift.to_dicts(),
        }
        summary_path = out / "run_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        tracker.log_artifact(summary_path)
        log.info("Pipeline OK. master=%s model=%s -> %s", master.version, version, summary_path)

    return PipelineResult(
        master_version=master.version,
        comparison=comparison.to_dicts(),
        model_version=version,
        drift=drift.to_dicts(),
        summary_path=str(summary_path),
    )
