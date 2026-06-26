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
import yaml

from nba_draft.cleaning.master import build_master
from nba_draft.config import REPO_ROOT, load_config
from nba_draft.data.fixtures import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    YEAR_COLUMN,
    make_multisource_fixture,
    make_synthetic_prospects,
)
from nba_draft.evaluation.comparison import compare_models, make_spec
from nba_draft.evaluation.metrics import spearman_corr
from nba_draft.mlops.drift import feature_drift_report
from nba_draft.mlops.registry import register_model
from nba_draft.mlops.tracking import ExperimentTracker
from nba_draft.models import DraftPositionEstimator, gbm_regressor, mean_regressor, ridge_regressor
from nba_draft.models.hurdle import REPLACEMENT_BPM, realized_value
from nba_draft.utils.logging import get_logger
from nba_draft.utils.seeds import set_global_seed
from nba_draft.validation import (
    FoldPreprocessor,
    make_data_split,
    walk_forward_hurdle_evaluate,
)

log = get_logger("mlops.pipeline")


@dataclass
class PipelineResult:
    master_version: str
    comparison: list[dict[str, Any]] = field(default_factory=list)
    model_name: str = "impact_regressor"
    model_version: str = ""
    drift: list[dict[str, Any]] = field(default_factory=list)
    summary_path: str = ""


def load_params(path: str | Path | None = None) -> dict[str, Any]:
    """Load DVC-tracked pipeline parameters from params.yaml (the source DVC `repro` versions).

    Editing params.yaml changes the pipeline (so `dvc repro` is meaningful); config/config.yaml
    supplies defaults for anything params.yaml omits.
    """
    p = Path(path) if path is not None else REPO_ROOT / "params.yaml"
    if not p.exists():
        return {}
    data: dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return data


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
    params = load_params()
    validation_params = params.get("validation", {})
    seed = params.get("seed", cfg.seed) if seed is None else seed
    set_global_seed(seed)
    out = Path(output_root)
    out.mkdir(parents=True, exist_ok=True)
    feats = list(FEATURE_COLUMNS)
    mty = int(validation_params.get("min_train_years", cfg.validation.min_train_years))
    n_holdout = int(validation_params.get("n_holdout_years", cfg.validation.n_holdout_years))
    alpha = float(params.get("model", {}).get("alpha", 1.0))
    psi_threshold = float(params.get("drift", {}).get("psi_threshold", 0.25))

    with ExperimentTracker(
        run_name=f"pipeline-{datetime.now(UTC):%Y%m%d-%H%M%S}",
        tracking_uri=tracking_uri,
        enabled=tracking_enabled,
    ) as tracker:
        tracker.log_params(
            {"seed": seed, "min_train_years": mty, "n_features": len(feats),
             "alpha": alpha, "psi_threshold": psi_threshold}
        )

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
        split = make_data_split(df, year_col=YEAR_COLUMN, n_holdout_years=n_holdout)
        dev = split.dev
        specs = {
            "baseline_draftpos": make_spec(
                ["draft_pick"], lambda: DraftPositionEstimator(0),
                lambda: FoldPreprocessor(["draft_pick"]),
            ),
            "mean": make_spec(feats, mean_regressor, lambda: FoldPreprocessor(feats)),
            "ridge": make_spec(
                feats, lambda: ridge_regressor(alpha), lambda: FoldPreprocessor(feats)
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

        # 2b. HURDLE ranking (survivorship-robust): rank ALL prospects by unconditional EV.
        # reached := impact above replacement; realized := impact if reached else replacement.
        def _with_hurdle_cols(frame: pl.DataFrame) -> pl.DataFrame:
            reached = (pl.col(TARGET_COLUMN) > REPLACEMENT_BPM).cast(pl.Float64)
            return frame.with_columns(
                reached.alias("reached"),
                pl.col(TARGET_COLUMN).alias("impact"),
                pl.Series(
                    "realized",
                    realized_value(
                        (frame[TARGET_COLUMN].to_numpy() > REPLACEMENT_BPM).astype(float),
                        frame[TARGET_COLUMN].to_numpy(),
                    ),
                ),
            )

        dev_h = _with_hurdle_cols(dev)
        hurdle = walk_forward_hurdle_evaluate(
            dev_h, feature_cols=feats, reached_col="reached", impact_col="impact",
            realized_col="realized", preprocessor_factory=lambda: FoldPreprocessor(feats),
            min_train_years=mty,
        )
        tracker.log_metrics({"hurdle_spearman": float(hurdle.aggregate["spearman_mean"])})
        log.info(
            "Hurdle (unconditional EV) spearman=%.3f vs realized value",
            hurdle.aggregate["spearman_mean"],
        )

        # 3. train final impact model on the dev set
        pp = FoldPreprocessor(feats).fit(dev)
        x_tr = pp.transform_matrix(dev).to_numpy()
        y_tr = dev[TARGET_COLUMN].to_numpy().astype(float)
        model = ridge_regressor(alpha)
        model.fit(x_tr, y_tr)

        # 3b. FINAL evaluation on the untouchable holdout (fit on dev, scored on locked classes).
        x_ho = pp.transform_matrix(split.holdout).to_numpy()
        holdout_pred = model.predict(x_ho)
        holdout_spearman = spearman_corr(
            split.holdout[TARGET_COLUMN].to_numpy(), holdout_pred
        )
        tracker.log_metrics({"holdout_spearman": float(holdout_spearman)})
        log.info("FINAL holdout spearman=%.3f (years %s)", holdout_spearman, split.holdout_years)

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
        drift = feature_drift_report(dev, split.holdout, feats, psi_threshold=psi_threshold)

        summary = {
            "created_at": datetime.now(UTC).isoformat(),
            "seed": seed,
            "master_version": master.version,
            "model_name": "impact_regressor",
            "model_version": version,
            "comparison": comparison.to_dicts(),
            "hurdle_spearman": float(hurdle.aggregate["spearman_mean"]),
            "holdout_spearman": float(holdout_spearman),
            "holdout_years": list(split.holdout_years),
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
