"""Error analysis: where does the model fail, and on which kinds of player?

The spec is explicit that statistical metrics aren't enough — we must know the model's blind
spots (e.g. does it systematically under-rate young bigs, or international guards?). These
functions operate on out-of-fold predictions (`validation.walk_forward_predictions`) so the
errors reflect genuine generalization, not training fit.
"""

from __future__ import annotations

import polars as pl


def residual_segments(
    pred_df: pl.DataFrame,
    *,
    segment_col: str,
    target_col: str,
    pred_col: str = "y_pred",
) -> pl.DataFrame:
    """Per-segment error summary: count, bias (mean signed error), MAE, RMSE.

    `bias` reveals systematic over/under-rating for a group; MAE/RMSE its magnitude. Sorted by
    MAE descending so the worst-served segments surface first.
    """
    resid = pl.col(pred_col) - pl.col(target_col)
    return (
        pred_df.with_columns(resid.alias("_resid"))
        .group_by(segment_col)
        .agg(
            pl.len().alias("n"),
            pl.col("_resid").mean().alias("bias"),
            pl.col("_resid").abs().mean().alias("mae"),
            (pl.col("_resid") ** 2).mean().sqrt().alias("rmse"),
        )
        .sort("mae", descending=True, nulls_last=True)
    )


def largest_errors(
    pred_df: pl.DataFrame,
    *,
    target_col: str,
    pred_col: str = "y_pred",
    id_cols: list[str] | None = None,
    k: int = 10,
) -> pl.DataFrame:
    """The k biggest misses, for human inspection (the most informative individual cases)."""
    id_cols = id_cols or []
    keep = [*id_cols, target_col, pred_col]
    keep = [c for c in keep if c in pred_df.columns]
    return (
        pred_df.with_columns((pl.col(pred_col) - pl.col(target_col)).alias("residual"))
        .with_columns(pl.col("residual").abs().alias("abs_residual"))
        .sort("abs_residual", descending=True)
        .select([*keep, "residual", "abs_residual"])
        .head(k)
    )
