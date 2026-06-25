"""Phase 12 — MLOps and maintenance.

Makes the project reproducible and maintainable:
  * tracking   — MLflow experiment tracking (graceful no-op if MLflow is absent/disabled)
  * registry   — versioned model registry on disk (save/load/list/promote)
  * drift      — feature-distribution drift monitoring (PSI) for the annual new class
  * retraining — the retraining policy + a should_retrain decision
  * pipeline   — the one-command, end-to-end reproducible pipeline

DVC wiring (``dvc.yaml`` + ``params.yaml``) defines the data/stage versioning; the whole
project reproduces from scratch via ``python scripts/run_pipeline.py`` (or ``dvc repro``).
"""

from nba_draft.mlops.drift import feature_drift_report, population_stability_index
from nba_draft.mlops.pipeline import PipelineResult, run_pipeline
from nba_draft.mlops.registry import (
    list_models,
    load_model,
    promote_model,
    register_model,
)
from nba_draft.mlops.retraining import RetrainingPolicy, should_retrain
from nba_draft.mlops.tracking import ExperimentTracker

__all__ = [
    "ExperimentTracker",
    "PipelineResult",
    "RetrainingPolicy",
    "feature_drift_report",
    "list_models",
    "load_model",
    "population_stability_index",
    "promote_model",
    "register_model",
    "run_pipeline",
    "should_retrain",
]
