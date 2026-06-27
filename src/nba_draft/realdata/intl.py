"""Link EuroLeague player-seasons to drafted international prospects.

Fills the same pre-draft production feature slots as the NCAA college source (a subset — EuroLeague
exposes fewer advanced metrics), for players who never played college ball. Matching is by
normalized name within the draft window (an international prospect's final pre-draft EuroLeague
season is typically the season ending in their draft year, i.e. season code draft_year-1 or
draft_year). Unmatched prospects keep nulls, which the imputer handles.
"""

from __future__ import annotations

import polars as pl

from nba_draft.cleaning.normalize import name_match_key

# Subset of COLLEGE_FEATURE_COLUMNS that EuroLeague can populate (shared names so the features
# unify: a prospect gets college OR international values in the same columns).
INTL_FEATURE_COLUMNS: list[str] = [
    "pts_per40",
    "ast_per40",
    "reb_per40",
    "stl_per40",
    "blk_per40",
    "tov_per40",
    "true_shooting",
]


def link_intl_features(
    draft_history: pl.DataFrame,
    intl_seasons: pl.DataFrame,
) -> pl.DataFrame:
    """Return player_id + INTL_FEATURE_COLUMNS for drafted players matched to a EuroLeague season.

    Matches by name within {draft_year-1, draft_year}, preferring the latest available season.
    """
    # Index EuroLeague rows by name_key -> list of (season, row).
    index: dict[str, list[tuple[int, dict[str, object]]]] = {}
    for row in intl_seasons.iter_rows(named=True):
        if row["season"] is None or row["full_name"] is None:
            continue
        key = name_match_key(str(row["full_name"]))
        index.setdefault(key, []).append((int(row["season"]), row))

    out_rows: list[dict[str, object]] = []
    for d in draft_history.iter_rows(named=True):
        pid = int(d["player_id"])
        key = name_match_key(str(d["full_name"]))
        draft_year = int(d["draft_year"])
        candidates = [
            (season, row)
            for season, row in index.get(key, [])
            if season in (draft_year - 1, draft_year)
        ]
        chosen = max(candidates, key=lambda sr: sr[0])[1] if candidates else None
        rec: dict[str, object] = {"player_id": pid}
        for col in INTL_FEATURE_COLUMNS:
            rec[col] = chosen[col] if chosen is not None else None
        out_rows.append(rec)

    schema = {"player_id": pl.Int64, **{c: pl.Float64 for c in INTL_FEATURE_COLUMNS}}
    return pl.DataFrame(out_rows, schema_overrides=schema)
