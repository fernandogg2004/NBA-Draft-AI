"""MLflow experiment tracking with graceful degradation.

Wraps MLflow so the rest of the codebase never imports it directly. If MLflow is unavailable or
tracking is disabled, every method is a no-op — the pipeline still runs and tests stay fast.
Defaults to a local file store under ``experiments/mlruns`` (no server needed).
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Any

from nba_draft.utils.logging import get_logger

log = get_logger("mlops.tracking")


class ExperimentTracker:
    """Context manager around an MLflow run; no-op when disabled or MLflow missing."""

    def __init__(
        self,
        experiment: str = "nba-draft",
        *,
        run_name: str | None = None,
        tracking_uri: str | None = None,
        enabled: bool = True,
    ) -> None:
        self.experiment = experiment
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self.enabled = enabled
        self._mlflow: Any | None = None
        self._active = False

    def __enter__(self) -> ExperimentTracker:
        if not self.enabled:
            return self
        try:
            import mlflow
        except ImportError:
            log.warning("MLflow not installed; tracking disabled for this run.")
            return self
        self._mlflow = mlflow
        # MLflow 3.x deprecated the file store; default to a local SQLite backend.
        if self.tracking_uri:
            uri = self.tracking_uri
        else:
            db = Path("experiments")
            db.mkdir(parents=True, exist_ok=True)
            uri = f"sqlite:///{(db / 'mlflow.db').resolve().as_posix()}"
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(self.experiment)
        mlflow.start_run(run_name=self.run_name)
        self._active = True
        return self

    def log_params(self, params: dict[str, Any]) -> None:
        if self._active and self._mlflow is not None:
            self._mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        if self._active and self._mlflow is not None:
            self._mlflow.log_metrics({k: float(v) for k, v in metrics.items()})

    def set_tags(self, tags: dict[str, Any]) -> None:
        if self._active and self._mlflow is not None:
            self._mlflow.set_tags(tags)

    def log_artifact(self, path: str | Path) -> None:
        if self._active and self._mlflow is not None:
            self._mlflow.log_artifact(str(path))

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._active and self._mlflow is not None:
            self._mlflow.end_run(status="FAILED" if exc_type else "FINISHED")
            self._active = False
