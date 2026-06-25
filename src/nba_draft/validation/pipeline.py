"""Fold-local preprocessing pipeline — the leakage firewall for modeling.

A :class:`FoldPreprocessor` bundles every step that must learn parameters from data
(inter-league translation + dynamic SoS, comparable-league imputation, and a final
median backfill) into one fit/transform unit. It is FIT ON THE TRAIN FOLD ONLY and then
applied to validation/test. Stateless features are assumed already assembled upstream
(`features.assemble_prospect_features`), so they need no fitting here.

Order: context model -> imputer -> median backfill. The backfill guarantees the resulting
feature matrix has no nulls (so estimators get clean input), using TRAIN medians only.
"""

from __future__ import annotations

from typing import cast

import polars as pl

from nba_draft.cleaning.imputation import LeakageSafeImputer
from nba_draft.features.learned import LeagueSeasonContextModel


class FoldPreprocessor:
    """Leakage-safe, fit-on-train preprocessing for one fold (or for final-fit inference)."""

    def __init__(
        self,
        feature_cols: list[str],
        *,
        context_model: LeagueSeasonContextModel | None = None,
        imputer: LeakageSafeImputer | None = None,
    ) -> None:
        self.feature_cols = list(feature_cols)
        self.context_model = context_model
        self.imputer = imputer
        self._medians: dict[str, float | None] = {}
        self._fitted = False

    def _apply_learned(self, df: pl.DataFrame) -> pl.DataFrame:
        out = df
        if self.context_model is not None:
            out = self.context_model.transform(out)
        if self.imputer is not None:
            out = self.imputer.transform(out)
        return out

    def fit(self, train: pl.DataFrame) -> FoldPreprocessor:
        if self.context_model is not None:
            self.context_model.fit(train)
        learned = self.context_model.transform(train) if self.context_model is not None else train
        if self.imputer is not None:
            self.imputer.fit(learned)
            learned = self.imputer.transform(learned)
        # Final backfill medians for every requested feature, from TRAIN only.
        learned = learned.with_columns(
            [pl.col(c).cast(pl.Float64) for c in self.feature_cols if c in learned.columns]
        )
        self._medians = {
            c: (cast(float, learned[c].median()) if c in learned.columns else None)
            for c in self.feature_cols
        }
        self._fitted = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self._fitted:
            raise RuntimeError("FoldPreprocessor must be fit before transform().")
        out = self._apply_learned(df)
        out = out.with_columns(
            [pl.col(c).cast(pl.Float64) for c in self.feature_cols if c in out.columns]
        )
        # Any missing feature column is created as null, then median-filled (train medians).
        for c in self.feature_cols:
            if c not in out.columns:
                out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias(c))
            med = self._medians.get(c)
            if med is not None:
                out = out.with_columns(pl.col(c).fill_null(med))
        return out

    def transform_matrix(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform and return ONLY the feature columns, in declared order."""
        return self.transform(df).select(self.feature_cols)
