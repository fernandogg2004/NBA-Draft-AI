# MLOps & maintenance (Phase 12)

## One-command reproducibility
`python scripts/run_pipeline.py` runs the whole project end-to-end:
1. **integrate** — build the versioned master dataset (entity resolution + provenance)
2. **evaluate** — temporal-CV comparison of real models vs the draft-position baseline
3. **train** — fit the final impact model on the development set
4. **register** — version the model with its metrics + data version
5. **monitor** — feature-drift (PSI) report

Outputs: `artifacts/pipeline/run_summary.json`, a registered model under `artifacts/models/`, and
an MLflow run. `dvc.yaml` + `params.yaml` wrap this as a DVC stage (`dvc repro`).

## Experiment tracking — MLflow
`nba_draft.mlops.ExperimentTracker` wraps MLflow and **degrades gracefully** (no-op if MLflow is
absent or disabled). MLflow 3.x deprecated the file store, so the default backend is **SQLite**
(`experiments/mlflow.db`). View runs: `mlflow ui --backend-store-uri sqlite:///experiments/mlflow.db`.

## Model registry
`nba_draft.mlops.registry` is a file-based registry: each version is `artifacts/models/<name>/<version>/`
with `model.joblib` + `meta.json` (metrics, feature columns, **data version**, stage). A `registry.json`
indexes versions, `latest`, and stages. API: `register_model`, `load_model` (by version/stage/latest),
`list_models`, `promote_model` (none → staging → production → archived).

## Drift monitoring
`feature_drift_report(reference, current, feature_cols)` computes **PSI** per feature (< 0.1 stable,
0.1–0.25 moderate, > 0.25 significant) and flags drifted features. Run dev (reference) vs the new
draft class each year.

## Retraining policy
`RetrainingPolicy` + `should_retrain(...)`: retrain **annually** with each new draft class (it adds a
year of resolved labels and a new class to project), and **off-cycle** if max-feature PSI exceeds the
threshold (default 0.25). The decision returns human-readable reasons for the audit trail.

## Data versioning — DVC
`dvc.yaml` defines the pipeline stage and its `deps`/`params`/`outs`; `params.yaml` holds tunables that
trigger re-runs. Initialize once with `git init && dvc init`, then `dvc repro`. Raw/processed data
under `data/` is gitignored and DVC-tracked.
