"""Learned, leakage-safe context features: inter-league translation + dynamic SoS.

Both require cross-player baselines, so — exactly like the imputer — this is a fit/transform
model FIT ON TRAINING CLASSES ONLY and applied to validation/test inside the temporal folds.

  * Inter-league translation (risk #5/#9): production is z-scored within its own league and
    re-expressed on a reference league's scale, so a EuroLeague line and an NCAA line become
    comparable. Factors are learned from TRAIN league distributions.
  * Dynamic strength-of-schedule (risk #8): SoS is standardized within (league, season), NOT
    treated as a static player attribute — a given raw SoS means different things in different
    seasons/leagues. Falls back league-level then global when a (league, season) is unseen.
"""

from __future__ import annotations

import polars as pl

DEFAULT_STAT_COLS = ("pts_per100", "ast_per100", "reb_per100", "true_shooting", "usage")


def _mean_std(df: pl.DataFrame, col: str) -> tuple[float | None, float | None]:
    obs = df.select(pl.col(col)).drop_nulls()
    if obs.height == 0:
        return None, None
    mean = float(obs[col].mean())  # type: ignore[arg-type]
    std = float(obs[col].std()) if obs.height > 1 else None  # type: ignore[arg-type]
    return mean, std


class LeagueSeasonContextModel:
    """Fit league/season baselines on train; emit translated stats + dynamic SoS z-scores."""

    def __init__(
        self,
        stat_cols: tuple[str, ...] = DEFAULT_STAT_COLS,
        *,
        league_col: str = "league_id",
        season_col: str = "season",
        sos_col: str = "strength_of_schedule",
        reference_league: str = "ncaa",
    ) -> None:
        self.stat_cols = tuple(stat_cols)
        self.league_col = league_col
        self.season_col = season_col
        self.sos_col = sos_col
        self.reference_league = reference_league
        self._league_mean: dict[str, dict[object, float]] = {}
        self._league_std: dict[str, dict[object, float]] = {}
        self._ref_mean: dict[str, float | None] = {}
        self._ref_std: dict[str, float | None] = {}
        self._global_std: dict[str, float | None] = {}
        self._sos_ls_mean: dict[str, float] = {}
        self._sos_ls_std: dict[str, float] = {}
        self._sos_global_mean: float | None = None
        self._sos_global_std: float | None = None
        self._fitted = False

    def _ls_key(self, league: object, season: object) -> str:
        return f"{league}|{season}"

    def fit(self, train: pl.DataFrame) -> LeagueSeasonContextModel:
        # Per-league mean/std for each stat (translation factors) + reference + global std.
        for c in self.stat_cols:
            if c not in train.columns:
                continue
            self._league_mean[c] = {}
            self._league_std[c] = {}
            for lg, g in train.group_by(self.league_col):
                lg_val = lg[0] if isinstance(lg, tuple) else lg
                m, s = _mean_std(g, c)
                if m is not None:
                    self._league_mean[c][lg_val] = m
                if s is not None and s > 0:
                    self._league_std[c][lg_val] = s
            ref = train.filter(pl.col(self.league_col) == self.reference_league)
            rm, rs = _mean_std(ref if ref.height else train, c)
            self._ref_mean[c] = rm
            self._ref_std[c] = rs
            _, gstd = _mean_std(train, c)
            self._global_std[c] = gstd

        # Dynamic SoS baselines per (league, season), with global fallback.
        if self.sos_col in train.columns:
            self._sos_global_mean, self._sos_global_std = _mean_std(train, self.sos_col)
            for key, g in train.group_by(self.league_col, self.season_col):
                m, s = _mean_std(g, self.sos_col)
                k = self._ls_key(key[0], key[1])
                if m is not None:
                    self._sos_ls_mean[k] = m
                if s is not None and s > 0:
                    self._sos_ls_std[k] = s
        self._fitted = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self._fitted:
            raise RuntimeError("LeagueSeasonContextModel must be fit before transform().")
        out = df

        # Inter-league translation: (x - mean_league)/std_league * ref_std + ref_mean.
        for c in self.stat_cols:
            if c not in df.columns:
                continue
            gstd = self._global_std.get(c) or 1.0
            mean_l = pl.col(self.league_col).replace_strict(
                self._league_mean.get(c, {}), default=self._ref_mean.get(c), return_dtype=pl.Float64
            )
            std_l = pl.col(self.league_col).replace_strict(
                self._league_std.get(c, {}), default=gstd, return_dtype=pl.Float64
            )
            std_safe = pl.when(std_l.is_null() | (std_l <= 0)).then(gstd).otherwise(std_l)
            ref_std = self._ref_std.get(c) or gstd
            ref_mean = self._ref_mean.get(c)
            translated = (pl.col(c) - mean_l) / std_safe * ref_std + ref_mean
            out = out.with_columns(translated.alias(f"{c}_translated"))

        # Dynamic SoS z-score within (league, season).
        if self.sos_col in df.columns:
            gmean = self._sos_global_mean
            gstd = self._sos_global_std or 1.0
            key = (
                pl.col(self.league_col).cast(pl.Utf8)
                + pl.lit("|")
                + pl.col(self.season_col).cast(pl.Utf8)
            )
            mean_ls = key.replace_strict(self._sos_ls_mean, default=gmean, return_dtype=pl.Float64)
            std_ls = key.replace_strict(self._sos_ls_std, default=gstd, return_dtype=pl.Float64)
            std_safe = pl.when(std_ls.is_null() | (std_ls <= 0)).then(gstd).otherwise(std_ls)
            out = out.with_columns(((pl.col(self.sos_col) - mean_ls) / std_safe).alias("sos_z"))
        return out
