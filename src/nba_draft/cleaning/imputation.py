"""Leakage-safe imputation for sparse prospect data.

Two domain risks meet here:
  * #1 leakage: the imputer is FIT ON TRAINING ROWS ONLY and then applied to validation/test.
    It is never fit on the full dataset, so validation distributions cannot leak into training.
  * #9 data disparity: an international prospect missing Torvik-style advanced metrics must NOT
    be scored as if those metrics were bad. We fill from COMPARABLE-LEAGUE group statistics,
    flag every filled cell (`<col>_imputed = True`), and emit a per-cell imputation standard
    deviation (`<col>_impute_sd`) so the added uncertainty propagates downstream.

This is a fit/transform transformer (sklearn-style) intended to live INSIDE the temporal CV
folds (Phase 5/6), not to be baked into the master dataset.
"""

from __future__ import annotations

from typing import cast

import polars as pl

from nba_draft.cleaning.schema import (
    IMPUTABLE_COLUMNS,
    impute_sd_name,
    missing_flag_name,
)


class LeakageSafeImputer:
    """Comparable-league mean imputer with flags and per-cell uncertainty.

    For each imputable column it learns, on TRAIN only: the per-group mean and within-group
    standard deviation (fallback to global mean/sd). At transform time, null cells are filled
    with the group mean (or global), flagged, and assigned the group sd as their uncertainty.
    """

    def __init__(
        self,
        columns: tuple[str, ...] = IMPUTABLE_COLUMNS,
        *,
        group_col: str | None = "league_id",
    ) -> None:
        self.columns = tuple(columns)
        self.group_col = group_col
        self._global_mean: dict[str, float | None] = {}
        self._global_sd: dict[str, float | None] = {}
        self._group_mean: dict[str, dict[object, float]] = {}
        self._group_sd: dict[str, dict[object, float]] = {}
        self._fitted = False

    def fit(self, train: pl.DataFrame) -> LeakageSafeImputer:
        use_groups = self.group_col is not None and self.group_col in train.columns
        for c in self.columns:
            if c not in train.columns:
                continue
            obs = train.select(pl.col(c)).drop_nulls()
            self._global_mean[c] = float(cast(float, obs[c].mean())) if obs.height > 0 else None
            self._global_sd[c] = float(cast(float, obs[c].std())) if obs.height > 1 else None
            if use_groups:
                assert self.group_col is not None
                stats = train.group_by(self.group_col).agg(
                    pl.col(c).mean().alias("m"), pl.col(c).std().alias("s")
                )
                self._group_mean[c] = {
                    row[self.group_col]: float(row["m"])
                    for row in stats.iter_rows(named=True)
                    if row["m"] is not None
                }
                self._group_sd[c] = {
                    row[self.group_col]: float(row["s"])
                    for row in stats.iter_rows(named=True)
                    if row["s"] is not None
                }
            else:
                self._group_mean[c] = {}
                self._group_sd[c] = {}
        self._fitted = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self._fitted:
            raise RuntimeError("LeakageSafeImputer must be fit before transform().")
        use_groups = self.group_col is not None and self.group_col in df.columns
        out = df
        for c in self.columns:
            if c not in df.columns:
                continue
            was_null = pl.col(c).is_null()
            if use_groups and self._group_mean.get(c):
                assert self.group_col is not None
                mean_expr = pl.col(self.group_col).replace_strict(
                    self._group_mean[c], default=self._global_mean[c], return_dtype=pl.Float64
                )
                sd_expr = pl.col(self.group_col).replace_strict(
                    self._group_sd[c], default=self._global_sd[c], return_dtype=pl.Float64
                )
            else:
                mean_expr = pl.lit(self._global_mean[c], dtype=pl.Float64)
                sd_expr = pl.lit(self._global_sd[c], dtype=pl.Float64)

            filled = pl.when(was_null).then(mean_expr).otherwise(pl.col(c)).alias(c)
            flag = (was_null & mean_expr.is_not_null()).alias(missing_flag_name(c))
            sd = (
                pl.when(was_null)
                .then(sd_expr.fill_null(0.0))
                .otherwise(0.0)
                .alias(impute_sd_name(c))
            )
            out = out.with_columns(filled, flag, sd)
        return out

    def fit_transform(self, train: pl.DataFrame) -> pl.DataFrame:
        """Convenience for the TRAIN fold only. Never call on combined train+val data."""
        return self.fit(train).transform(train)
