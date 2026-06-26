"""Honest model comparison against the draft-position baseline (instructions.md Phase 7).

Runs each contender through the SAME temporal-CV protocol and reports aggregate metrics plus
uplift over the baseline. The baseline is the bar: a model that does not beat draft order under
temporal validation has not earned its complexity.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import polars as pl

# NOTE: import walk_forward_evaluate lazily inside compare_models. evaluation.runner-style code
# and validation.runner both reference evaluation.metrics, so a top-level import here creates a
# circular import (evaluation.__init__ -> comparison -> validation.runner -> evaluation.metrics).


def compare_models(
    df: pl.DataFrame,
    specs: dict[str, dict[str, Any]],
    *,
    target_col: str,
    year_col: str = "draft_year",
    min_train_years: int,
    baseline_name: str | None = None,
    primary_metric: str = "spearman_mean",
) -> pl.DataFrame:
    """Evaluate several model specs and tabulate metrics (+ uplift vs baseline).

    Args:
        specs: name -> dict with keys ``feature_cols``, ``model_factory``, ``preprocessor_factory``.
        baseline_name: if given, adds an uplift column relative to that row's primary metric.

    Returns:
        One row per model with the standard aggregate metrics, sorted by `primary_metric` desc.
    """
    from nba_draft.validation.runner import walk_forward_evaluate

    rows: list[dict[str, object]] = []
    for name, spec in specs.items():
        report = walk_forward_evaluate(
            df,
            feature_cols=spec["feature_cols"],
            target_col=target_col,
            year_col=year_col,
            model_factory=spec["model_factory"],
            preprocessor_factory=spec["preprocessor_factory"],
            min_train_years=min_train_years,
        )
        rows.append({"model": name, **report.aggregate})
    table = pl.DataFrame(rows).sort(primary_metric, descending=True, nulls_last=True)

    if baseline_name is not None and baseline_name in table["model"].to_list():
        base_val = table.filter(pl.col("model") == baseline_name)[primary_metric][0]
        table = table.with_columns(
            (pl.col(primary_metric) - base_val).alias(f"uplift_{primary_metric}")
        )
    return table


def make_spec(
    feature_cols: list[str],
    model_factory: Callable[[], Any],
    preprocessor_factory: Callable[[], Any],
) -> dict[str, Any]:
    """Small helper to build a model spec dict for :func:`compare_models`."""
    return {
        "feature_cols": feature_cols,
        "model_factory": model_factory,
        "preprocessor_factory": preprocessor_factory,
    }
