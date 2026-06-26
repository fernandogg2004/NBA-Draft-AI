"""Walk-forward evaluation runner — the project's modeling protocol.

For each strictly-temporal fold it: fits the fold preprocessor on TRAIN, transforms train+val,
fits the estimator on TRAIN, predicts VAL, and scores. Every leakage-sensitive step is fit on
train only, so no model evaluated through this runner can see the future. Phase 6 plugs models
in; Phase 7 uses it to compare against the draft-position baseline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from numpy.typing import NDArray

from nba_draft.evaluation.metrics import rmse, spearman_corr, top_k_hit_rate
from nba_draft.models.base import Estimator
from nba_draft.utils.logging import get_logger
from nba_draft.validation.pipeline import FoldPreprocessor
from nba_draft.validation.temporal import holdout_split, walk_forward_folds

log = get_logger("validation.runner")

MetricFn = Callable[[NDArray[np.float64], NDArray[np.float64]], float]


def default_metrics() -> dict[str, MetricFn]:
    """Ranking-first defaults: Spearman, top-10 hit rate, and RMSE."""
    return {
        "spearman": spearman_corr,
        "top10": lambda yt, yp: top_k_hit_rate(yt, yp, 10),
        "rmse": rmse,
    }


@dataclass(frozen=True)
class DataSplit:
    """Development vs untouchable-holdout split (the holdout is never tuned on)."""

    dev: pl.DataFrame
    holdout: pl.DataFrame
    holdout_years: tuple[int, ...]


def make_data_split(
    df: pl.DataFrame, *, year_col: str = "draft_year", n_holdout_years: int = 2
) -> DataSplit:
    """Carve the most-recent draft classes off as the untouchable test set."""
    years = df[year_col].to_numpy()
    dev_idx, hold_idx = holdout_split(years, n_holdout_years)
    holdout = df[hold_idx.tolist()]
    return DataSplit(
        dev=df[dev_idx.tolist()],
        holdout=holdout,
        holdout_years=tuple(sorted({int(y) for y in holdout[year_col].to_list()})),
    )


@dataclass
class EvaluationReport:
    """Per-fold and aggregate metrics from a walk-forward evaluation."""

    per_fold: list[dict[str, object]] = field(default_factory=list)
    aggregate: dict[str, float] = field(default_factory=dict)

    def format(self) -> str:
        lines = ["walk-forward evaluation:"]
        for f in self.per_fold:
            metric_str = " ".join(
                f"{k}={v:.3f}" for k, v in f.items() if isinstance(v, float)
            )
            lines.append(
                f"  fold {f['fold']}: train{f['train_years']} -> "
                f"val{f['val_years']} | {metric_str}"
            )
        agg = " ".join(f"{k}={v:.3f}" for k, v in self.aggregate.items())
        lines.append(f"  AGG | {agg}")
        return "\n".join(lines)


def walk_forward_evaluate(
    df: pl.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    model_factory: Callable[[], Estimator],
    preprocessor_factory: Callable[[], FoldPreprocessor],
    year_col: str = "draft_year",
    min_train_years: int,
    val_horizon_years: int = 1,
    step_years: int = 1,
    expanding: bool = True,
    metrics: dict[str, MetricFn] | None = None,
) -> EvaluationReport:
    """Evaluate an estimator across strictly-temporal walk-forward folds.

    Args:
        df: Per-prospect frame with `year_col`, `target_col`, and raw feature inputs.
        feature_cols: Columns the preprocessor materializes and the model consumes.
        target_col: Regression target.
        model_factory: Returns a fresh estimator per fold.
        preprocessor_factory: Returns a fresh :class:`FoldPreprocessor` per fold.
        Remaining args mirror :func:`walk_forward_folds`.

    Returns:
        An :class:`EvaluationReport` with per-fold and mean/std aggregate metrics.
    """
    metrics = metrics or default_metrics()
    years = df[year_col].to_numpy()
    report = EvaluationReport()
    scores: dict[str, list[float]] = {name: [] for name in metrics}

    for i, fold in enumerate(
        walk_forward_folds(
            years,
            min_train_years=min_train_years,
            val_horizon_years=val_horizon_years,
            step_years=step_years,
            expanding=expanding,
        )
    ):
        train_df = df[fold.train_idx.tolist()]
        val_df = df[fold.val_idx.tolist()]

        pp = preprocessor_factory().fit(train_df)
        x_tr = pp.transform_matrix(train_df).to_numpy()
        x_va = pp.transform_matrix(val_df).to_numpy()
        if np.isnan(x_tr).any() or np.isnan(x_va).any():
            raise ValueError(f"Fold {i}: feature matrix has nulls after preprocessing.")
        y_tr = train_df[target_col].to_numpy().astype(float)
        y_va = val_df[target_col].to_numpy().astype(float)

        model = model_factory()
        model.fit(x_tr, y_tr)
        pred = np.asarray(model.predict(x_va), dtype=float)

        fold_rec: dict[str, object] = {
            "fold": i,
            "train_years": fold.train_years,
            "val_years": fold.val_years,
            "n_val": int(val_df.height),
        }
        for name, fn in metrics.items():
            val = float(fn(y_va, pred))
            fold_rec[name] = val
            scores[name].append(val)
        report.per_fold.append(fold_rec)

    for name, vals in scores.items():
        report.aggregate[f"{name}_mean"] = float(np.mean(vals)) if vals else float("nan")
        report.aggregate[f"{name}_std"] = float(np.std(vals)) if vals else float("nan")
    return report


def walk_forward_predictions(
    df: pl.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    model_factory: Callable[[], Estimator],
    preprocessor_factory: Callable[[], FoldPreprocessor],
    year_col: str = "draft_year",
    min_train_years: int,
    val_horizon_years: int = 1,
    step_years: int = 1,
    expanding: bool = True,
    pred_col: str = "y_pred",
) -> pl.DataFrame:
    """Collect out-of-fold predictions for every validated prospect (for error analysis).

    Each prospect is predicted exactly once — in the fold where its draft class is the
    validation set — by a model trained only on earlier classes. The returned frame is the
    original rows (so any segment column is available) plus ``pred_col`` and ``fold``.
    """
    years = df[year_col].to_numpy()
    parts: list[pl.DataFrame] = []
    for i, fold in enumerate(
        walk_forward_folds(
            years,
            min_train_years=min_train_years,
            val_horizon_years=val_horizon_years,
            step_years=step_years,
            expanding=expanding,
        )
    ):
        train_df = df[fold.train_idx.tolist()]
        val_df = df[fold.val_idx.tolist()]
        pp = preprocessor_factory().fit(train_df)
        x_tr = pp.transform_matrix(train_df).to_numpy()
        x_va = pp.transform_matrix(val_df).to_numpy()
        if np.isnan(x_tr).any() or np.isnan(x_va).any():
            raise ValueError(f"Fold {i}: feature matrix has nulls after preprocessing.")
        model = model_factory()
        model.fit(x_tr, train_df[target_col].to_numpy().astype(float))
        pred = np.asarray(model.predict(x_va), dtype=float)
        parts.append(
            val_df.with_columns(
                pl.Series(pred_col, pred, dtype=pl.Float64),
                pl.lit(i, dtype=pl.Int64).alias("fold"),
            )
        )
    return pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()


def walk_forward_hurdle_evaluate(
    df: pl.DataFrame,
    *,
    feature_cols: list[str],
    reached_col: str,
    impact_col: str,
    realized_col: str,
    preprocessor_factory: Callable[[], FoldPreprocessor],
    reach_factory: Callable[[], object] | None = None,
    impact_factory: Callable[[], object] | None = None,
    replacement: float = -2.0,
    year_col: str = "draft_year",
    min_train_years: int,
    val_horizon_years: int = 1,
    step_years: int = 1,
    expanding: bool = True,
) -> EvaluationReport:
    """Temporal-CV evaluation of the HurdleModel over ALL prospects (reached or not).

    Each fold fits the reach + impact heads on the train fold only and ranks the validation
    prospects by unconditional EV, scored against the realized value (impact if reached, else
    replacement). This measures the survivorship-robust ranking the system is meant to produce.
    """
    from nba_draft.models.hurdle import HurdleModel
    from nba_draft.models.zoo import logistic_classifier, ridge_regressor

    reach_f = reach_factory or logistic_classifier
    impact_f = impact_factory or (lambda: ridge_regressor(1.0))
    metrics = default_metrics()
    years = df[year_col].to_numpy()
    report = EvaluationReport()
    scores: dict[str, list[float]] = {name: [] for name in metrics}

    for i, fold in enumerate(
        walk_forward_folds(
            years,
            min_train_years=min_train_years,
            val_horizon_years=val_horizon_years,
            step_years=step_years,
            expanding=expanding,
        )
    ):
        train_df = df[fold.train_idx.tolist()]
        val_df = df[fold.val_idx.tolist()]

        pp = preprocessor_factory().fit(train_df)
        x_tr = pp.transform_matrix(train_df).to_numpy()
        x_va = pp.transform_matrix(val_df).to_numpy()
        if np.isnan(x_tr).any() or np.isnan(x_va).any():
            raise ValueError(f"Fold {i}: feature matrix has nulls after preprocessing.")

        model = HurdleModel(
            reach_factory=reach_f, impact_factory=impact_f, replacement=replacement,
        )
        model.fit(
            x_tr,
            train_df[reached_col].to_numpy().astype(float),
            train_df[impact_col].to_numpy().astype(float),
        )
        ev = np.asarray(model.predict(x_va), dtype=float)
        realized = val_df[realized_col].to_numpy().astype(float)

        fold_rec: dict[str, object] = {
            "fold": i,
            "train_years": fold.train_years,
            "val_years": fold.val_years,
            "n_val": int(val_df.height),
        }
        for name, fn in metrics.items():
            val = float(fn(realized, ev))
            fold_rec[name] = val
            scores[name].append(val)
        report.per_fold.append(fold_rec)

    for name, vals in scores.items():
        report.aggregate[f"{name}_mean"] = float(np.mean(vals)) if vals else float("nan")
        report.aggregate[f"{name}_std"] = float(np.std(vals)) if vals else float("nan")
    return report
