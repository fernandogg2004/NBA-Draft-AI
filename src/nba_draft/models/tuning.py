"""Hyperparameter tuning with Optuna, inside the temporal-CV protocol.

Each trial is scored by :func:`walk_forward_evaluate` (train-only preprocessing per fold), so
tuning never leaks: we optimize the mean validation metric across walk-forward folds. The
untouchable holdout is still never seen here — it is reserved for the single final evaluation.

Optuna is an optional dependency (``models`` extra). Import is lazy so the package loads without it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import polars as pl

from nba_draft.models.base import Estimator
from nba_draft.validation.pipeline import FoldPreprocessor

# spec: name -> ("float"|"int", low, high, log?)
ParamSpace = dict[str, tuple[str, float, float, bool]]


@dataclass(frozen=True)
class TuningResult:
    best_params: dict[str, Any]
    best_value: float
    n_trials: int
    direction: str


def _suggest(trial: Any, name: str, spec: tuple[str, float, float, bool]) -> float:
    kind, low, high, log = spec
    if kind == "int":
        return float(trial.suggest_int(name, int(low), int(high), log=log))
    return float(trial.suggest_float(name, low, high, log=log))


def tune_estimator(
    df: pl.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    build_fn: Callable[..., Estimator],
    param_space: ParamSpace,
    preprocessor_factory: Callable[[], FoldPreprocessor],
    min_train_years: int,
    year_col: str = "draft_year",
    metric: str = "spearman_mean",
    direction: str = "maximize",
    n_trials: int = 25,
    seed: int = 42,
) -> TuningResult:
    """Search ``build_fn``'s hyperparameters by maximizing a temporal-CV metric.

    Args:
        build_fn: Called with sampled params to build a fresh estimator (e.g. ``ridge_regressor``).
        param_space: Hyperparameter ranges to search.
        metric: Aggregate metric key from the evaluation report (e.g. ``"spearman_mean"``).
        direction: "maximize" (ranking metrics) or "minimize" (error metrics like rmse_mean).

    Returns:
        A :class:`TuningResult` with the best params and score.
    """
    import optuna

    from nba_draft.validation.runner import walk_forward_evaluate

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: Any) -> float:
        params = {name: _suggest(trial, name, spec) for name, spec in param_space.items()}
        report = walk_forward_evaluate(
            df,
            feature_cols=feature_cols,
            target_col=target_col,
            year_col=year_col,
            model_factory=lambda: build_fn(**params),
            preprocessor_factory=preprocessor_factory,
            min_train_years=min_train_years,
        )
        return report.aggregate[metric]

    study = optuna.create_study(
        direction=direction, sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials)
    return TuningResult(
        best_params=dict(study.best_params),
        best_value=float(study.best_value),
        n_trials=n_trials,
        direction=direction,
    )
