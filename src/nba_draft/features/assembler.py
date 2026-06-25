"""Assemble the per-prospect feature matrix and guard against post-draft leakage.

Grain: one row per prospect. The prospect's most recent PRE-DRAFT season is the primary row;
year-over-year sequence features summarize their trajectory; Combine measurements are joined on.

The learned context model (translation + dynamic SoS) must already be FIT ON TRAIN before being
passed in, so transform here cannot leak. `assert_pre_draft_safe` is a belt-and-suspenders check
that no known post-draft outcome column ever reaches the feature matrix.
"""

from __future__ import annotations

import polars as pl

from nba_draft.features.learned import LeagueSeasonContextModel
from nba_draft.features.transforms import (
    add_combine_features,
    add_stateless_features,
    sequence_features,
)

# Columns that describe what happened AFTER the draft — must never appear as features.
FORBIDDEN_POST_DRAFT_COLUMNS: frozenset[str] = frozenset(
    {
        "nba_impact",
        "draft_pick",
        "reached",
        "peak_impact",
        "cumulative_value",
        "outcome_tier",
        "all_star_count",
        "all_nba_count",
        "debut_year",
    }
)


def primary_pre_draft_season(
    prospect_season: pl.DataFrame,
    *,
    player_col: str = "player_id",
    season_col: str = "season",
) -> pl.DataFrame:
    """Take each prospect's most recent pre-draft season as their primary feature row."""
    return (
        prospect_season.sort([player_col, season_col])
        .group_by(player_col, maintain_order=True)
        .last()
    )


def assert_pre_draft_safe(
    df: pl.DataFrame, forbidden: frozenset[str] = FORBIDDEN_POST_DRAFT_COLUMNS
) -> None:
    """Raise if any forbidden post-draft column is present in a feature frame."""
    leaked = sorted(set(df.columns) & forbidden)
    if leaked:
        raise ValueError(f"Post-draft leakage: feature matrix contains {leaked}.")


def assemble_prospect_features(
    prospect_season: pl.DataFrame,
    combine: pl.DataFrame,
    *,
    identity: pl.DataFrame | None = None,
    player_col: str = "player_id",
) -> pl.DataFrame:
    """Build the STATELESS per-prospect feature frame (no learned/context steps yet).

    Safe to compute once over the whole dataset: every feature here depends only on a
    prospect's own pre-draft stats. The learned context model + imputer are applied later,
    inside each temporal fold (see ``validation.pipeline``).
    """
    primary = primary_pre_draft_season(prospect_season, player_col=player_col)
    primary = add_stateless_features(primary)
    seq = sequence_features(prospect_season, player_col=player_col)
    combine_feat = add_combine_features(combine)

    out = primary.join(seq, on=player_col, how="left")
    out = out.join(combine_feat, on=player_col, how="left")
    if identity is not None:
        id_cols = [c for c in ("full_name",) if c in identity.columns]
        if id_cols:
            out = out.join(identity.select([player_col, *id_cols]), on=player_col, how="left")

    assert_pre_draft_safe(out)
    return out


def build_feature_matrix(
    prospect_season: pl.DataFrame,
    combine: pl.DataFrame,
    context_model: LeagueSeasonContextModel,
    *,
    identity: pl.DataFrame | None = None,
    player_col: str = "player_id",
) -> pl.DataFrame:
    """Assemble stateless features then apply an ALREADY-FIT context model.

    Convenience for non-CV use (e.g. final inference). Inside temporal CV, use
    ``assemble_prospect_features`` once and let the fold preprocessor fit/apply the context
    model per fold so nothing leaks.
    """
    raw = assemble_prospect_features(
        prospect_season, combine, identity=identity, player_col=player_col
    )
    out = context_model.transform(raw)
    assert_pre_draft_safe(out)
    return out
