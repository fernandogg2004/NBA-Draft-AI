"""Estimated box impact (BPM proxy) and VORP from per-100 production.

IMPORTANT — honesty: this is a TRANSPARENT PROXY, not Basketball-Reference's BPM 2.0. BBRef's BPM
is a RAPM-calibrated regression we cannot reproduce from public data alone, and BBRef's pages are
not scrapeable (see config/sources.yaml). So we compute an estimated box plus-minus ("eBPM") from
per-100 box production with documented heuristic weights, then league-center within season so it
averages ~0 on the BPM scale. Validate it (see `pie_rank_agreement`) and recalibrate before
trusting magnitudes; rankings are more reliable than absolute values.

VORP uses the standard definition and is exact given eBPM and minutes:
    VORP = (eBPM - (-2.0)) * (player minutes / team-season minutes) * (team games / 82)
With a full 82-game team season the team-games factors cancel, leaving
    VORP = (eBPM + 2.0) * minutes / (240 * 82)      # 240 = 5 players * 48 min per team-game
"""

from __future__ import annotations

import numpy as np
import polars as pl

REPLACEMENT_BPM = -2.0
TEAM_MINUTES_PER_GAME = 240.0
FULL_SEASON_GAMES = 82.0

# Heuristic per-100 weights (documented proxy, NOT fitted to RAPM). Defense is under-captured by
# box stats, so steals/blocks proxy it imperfectly — a known limitation.
_WEIGHTS = {
    "pts_per100": 0.10,
    "ast_per100": 0.10,
    "oreb_per100": 0.07,
    "dreb_per100": 0.04,
    "stl_per100": 0.20,
    "blk_per100": 0.12,
    "tov_per100": -0.18,
}
_TS_BASELINE = 0.55       # league-ish true-shooting anchor for the efficiency adjustment
_TS_COEF = 6.0            # reward/penalize scoring efficiency vs the anchor


def estimated_bpm(
    df: pl.DataFrame, *, season_col: str = "season", minutes_col: str = "minutes"
) -> pl.Series:
    """Compute the league-centered eBPM proxy (minutes-weighted mean ~ 0 within each season)."""
    raw = pl.lit(0.0)
    for col, w in _WEIGHTS.items():
        if col in df.columns:
            raw = raw + w * pl.col(col).fill_null(0.0)
    # efficiency adjustment: scale by how far TS% sits from the anchor
    if "true_shooting" in df.columns:
        raw = raw + _TS_COEF * (pl.col("true_shooting").fill_null(_TS_BASELINE) - _TS_BASELINE)

    work = df.with_columns(raw.alias("_raw"))
    # minutes-weighted league mean per season, then center so eBPM ~ 0 on average.
    centered = work.with_columns(
        (
            (pl.col("_raw") * pl.col(minutes_col)).sum().over(season_col)
            / pl.col(minutes_col).sum().over(season_col)
        ).alias("_season_mean")
    ).with_columns((pl.col("_raw") - pl.col("_season_mean")).alias("ebpm"))
    return centered["ebpm"].round(3)


def vorp(ebpm: pl.Series, minutes: pl.Series) -> pl.Series:
    """Exact VORP from eBPM and total minutes (full-season normalization)."""
    e = ebpm.to_numpy().astype(float)
    m = minutes.to_numpy().astype(float)
    v = (e - REPLACEMENT_BPM) * m / (TEAM_MINUTES_PER_GAME * FULL_SEASON_GAMES)
    return pl.Series("vorp", np.round(v, 3))


def add_impact_metrics(df: pl.DataFrame) -> pl.DataFrame:
    """Attach eBPM and VORP columns to a per-season production frame."""
    ebpm = estimated_bpm(df)
    out = df.with_columns(ebpm.alias("ebpm"))
    return out.with_columns(vorp(out["ebpm"], out["minutes"]).alias("vorp"))


def pie_rank_agreement(df: pl.DataFrame) -> float:
    """Sanity check: Spearman correlation between eBPM and the official PIE metric.

    PIE (Player Impact Estimate) is an independent NBA box metric carried through from the
    Advanced endpoint. Strong positive agreement indicates the proxy ranks players sensibly.
    Returns NaN if PIE is unavailable.
    """
    if "pie" not in df.columns or "ebpm" not in df.columns:
        return float("nan")
    from nba_draft.evaluation.metrics import spearman_corr

    sub = df.select("ebpm", "pie").drop_nulls()
    if sub.height < 2:
        return float("nan")
    return spearman_corr(sub["ebpm"].to_numpy(), sub["pie"].to_numpy())
