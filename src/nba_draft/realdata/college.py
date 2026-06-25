"""Link CollegeBasketballData player-seasons to drafted players, yielding pre-draft features.

A drafted player's final college season is the spring of their draft year (CBD `season` ==
`draft_year`). We match on normalized name within that year, using the school vs the draft
`organization` as a tiebreak when a name is ambiguous. Unmatched (international / non-NCAA)
prospects get null college features, which the imputer handles downstream.
"""

from __future__ import annotations

import polars as pl

from nba_draft.cleaning.normalize import name_match_key, normalize_name

# Curated college features (kept modest given the small drafted-player sample).
COLLEGE_FEATURE_COLUMNS: list[str] = [
    "pts_per40",
    "ast_per40",
    "reb_per40",
    "stl_per40",
    "blk_per40",
    "tov_per40",
    "true_shooting",
    "usage",
    "efg",
    "three_pt_pct",
    "off_rating",
    "def_rating",
    "net_rating",
    "porpag",
    "win_shares_per40",
]


def link_college_features(
    draft_history: pl.DataFrame,
    cbd_seasons: pl.DataFrame,
    *,
    org_col: str = "organization",
) -> pl.DataFrame:
    """Return player_id + college features for drafted players matched to a college season.

    Args:
        draft_history: parsed draft history (player_id, full_name, draft_year, organization).
        cbd_seasons: parsed CBD player-seasons across the relevant years.
        org_col: draft-history column naming the pre-NBA organization (college).
    """
    # Index CBD rows by (name_key, season) -> list of row dicts (handle name collisions by school).
    index: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in cbd_seasons.iter_rows(named=True):
        key = (name_match_key(str(row["full_name"])), int(row["season"]))
        index.setdefault(key, []).append(row)

    out_rows: list[dict[str, object]] = []
    for d in draft_history.iter_rows(named=True):
        pid = int(d["player_id"])
        key = (name_match_key(str(d["full_name"])), int(d["draft_year"]))
        candidates = index.get(key, [])
        chosen: dict[str, object] | None = None
        if len(candidates) == 1:
            chosen = candidates[0]
        elif candidates:
            org = normalize_name(str(d.get(org_col) or ""))
            chosen = next(
                (c for c in candidates if normalize_name(str(c["school"])) == org),
                candidates[0],
            )
        rec: dict[str, object] = {"player_id": pid}
        for col in COLLEGE_FEATURE_COLUMNS:
            rec[col] = chosen[col] if chosen is not None else None
        out_rows.append(rec)

    schema = {"player_id": pl.Int64, **{c: pl.Float64 for c in COLLEGE_FEATURE_COLUMNS}}
    return pl.DataFrame(out_rows, schema_overrides=schema)
