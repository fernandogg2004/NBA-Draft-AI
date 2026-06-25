"""Parse raw nba_api JSON into canonical Polars frames.

Maps the exact nba_api column names (verified against live responses) onto the schema the rest of
the project uses. Pure functions over JSON strings, so they are unit-tested with small fixtures
and need no network.

Endpoints handled:
  * DraftHistory          -> draft slot per player (pick, draft year, pre-NBA organization)
  * DraftCombineStats     -> physical measurements (canonical Combine columns)
  * LeagueDashPlayerStats -> per-season production; Base + Advanced joined into per-100 + rates
"""

from __future__ import annotations

import json
from typing import Any

import polars as pl


def _pct(value: Any) -> float | None:
    """CollegeBasketballData percentages are 0-100; convert to a 0-1 fraction."""
    return None if value is None else float(value) / 100.0


def parse_cbd_player_season(raw_json: str) -> pl.DataFrame:
    """Parse CollegeBasketballData /stats/player/season (a JSON array) into canonical college rows.

    College data has no possessions, so production is expressed per-40-minutes. Rates are
    normalized to fractions (usage/efg/3P% are percents in the source; TS% is already a fraction).
    Nested objects (rebounds, winShares, shot splits) are flattened. Keeps name + school + season
    for linking to drafted players.
    """
    rows = json.loads(raw_json)
    out: list[dict[str, object]] = []
    for r in rows:
        minutes = r.get("minutes") or 0.0

        def per40(v: Any, _min: float = minutes) -> float | None:
            return float(v) / _min * 40.0 if _min and v is not None else None

        reb = r.get("rebounds") or {}
        ws = r.get("winShares") or {}
        tp = r.get("threePointFieldGoals") or {}
        usage = r.get("usage")
        out.append(
            {
                "cbd_athlete_id": r.get("athleteId"),
                "full_name": r.get("name"),
                "school": r.get("team"),
                "conference": r.get("conference"),
                "season": r.get("season"),
                "position": r.get("position"),
                "games": r.get("games"),
                "minutes": float(minutes),
                "pts_per40": per40(r.get("points")),
                "ast_per40": per40(r.get("assists")),
                "reb_per40": per40(reb.get("total")),
                "stl_per40": per40(r.get("steals")),
                "blk_per40": per40(r.get("blocks")),
                "tov_per40": per40(r.get("turnovers")),
                "true_shooting": r.get("trueShootingPct"),         # already a fraction
                "usage": None if usage is None else float(usage) / 100.0,
                "efg": _pct(r.get("effectiveFieldGoalPct")),
                "three_pt_pct": _pct(tp.get("pct")),
                "oreb_pct": _pct(r.get("offensiveReboundPct")),
                "off_rating": r.get("offensiveRating"),
                "def_rating": r.get("defensiveRating"),
                "net_rating": r.get("netRating"),
                "porpag": r.get("PORPAG"),
                "win_shares_per40": ws.get("totalPer40"),
                "ast_to": r.get("assistsTurnoverRatio"),
            }
        )
    schema = {
        "cbd_athlete_id": pl.Int64, "full_name": pl.Utf8, "school": pl.Utf8,
        "conference": pl.Utf8, "season": pl.Int64, "position": pl.Utf8,
        "games": pl.Int64, "minutes": pl.Float64,
        "pts_per40": pl.Float64, "ast_per40": pl.Float64, "reb_per40": pl.Float64,
        "stl_per40": pl.Float64, "blk_per40": pl.Float64, "tov_per40": pl.Float64,
        "true_shooting": pl.Float64, "usage": pl.Float64, "efg": pl.Float64,
        "three_pt_pct": pl.Float64, "oreb_pct": pl.Float64,
        "off_rating": pl.Float64, "def_rating": pl.Float64, "net_rating": pl.Float64,
        "porpag": pl.Float64, "win_shares_per40": pl.Float64, "ast_to": pl.Float64,
    }
    if not out:
        return pl.DataFrame({k: [] for k in schema}, schema=schema)
    return pl.DataFrame(out, schema_overrides=schema)


def _result_frame(raw_json: str, result_index: int = 0) -> pl.DataFrame:
    """Build a Polars frame from an nba_api resultSets payload (headers + rowSet)."""
    payload = json.loads(raw_json)
    sets = payload.get("resultSets") or payload.get("resultSet")
    rs = sets[result_index] if isinstance(sets, list) else sets
    rows = rs["rowSet"]
    headers = rs["headers"]
    if not rows:
        return pl.DataFrame({h: [] for h in headers})
    return pl.DataFrame(rows, schema=headers, orient="row")


def parse_draft_history(raw_json: str) -> pl.DataFrame:
    """DraftHistory -> [player_id, full_name, draft_year, draft_pick, draft_round, organization]."""
    df = _result_frame(raw_json)
    return df.select(
        pl.col("PERSON_ID").cast(pl.Int64).alias("player_id"),
        pl.col("PLAYER_NAME").cast(pl.Utf8).alias("full_name"),
        pl.col("SEASON").cast(pl.Int64).alias("draft_year"),
        pl.col("OVERALL_PICK").cast(pl.Int64).alias("draft_pick"),
        pl.col("ROUND_NUMBER").cast(pl.Int64).alias("draft_round"),
        pl.col("ORGANIZATION").cast(pl.Utf8).alias("organization"),
        pl.col("ORGANIZATION_TYPE").cast(pl.Utf8).alias("organization_type"),
    )


def parse_combine(raw_json: str) -> pl.DataFrame:
    """DraftCombineStats -> canonical Combine measurement columns (inches / seconds / pct)."""
    df = _result_frame(raw_json)

    def num(col: str) -> pl.Expr:
        return pl.col(col).cast(pl.Float64, strict=False)

    return df.select(
        pl.col("PLAYER_ID").cast(pl.Int64).alias("player_id"),
        pl.col("PLAYER_NAME").cast(pl.Utf8).alias("full_name"),
        pl.col("SEASON").cast(pl.Int64, strict=False).alias("draft_year"),
        pl.col("POSITION").cast(pl.Utf8).alias("position"),
        num("WINGSPAN").alias("wingspan_in"),
        num("STANDING_REACH").alias("standing_reach_in"),
        num("MAX_VERTICAL_LEAP").alias("max_vertical_in"),
        num("LANE_AGILITY_TIME").alias("lane_agility_s"),
        num("BODY_FAT_PCT").alias("body_fat_pct"),
    )


def parse_player_info(raw_json: str) -> dict[str, object]:
    """CommonPlayerInfo -> {player_id, birthdate (ISO date str | None), draft_year (int | None)}."""
    df = _result_frame(raw_json)
    if df.height == 0:
        return {"player_id": None, "birthdate": None, "draft_year": None}
    row = df.row(0, named=True)
    birth = row.get("BIRTHDATE")
    person_id = row.get("PERSON_ID")
    dy = row.get("DRAFT_YEAR")
    draft_year_int: int | None = None
    if dy is not None:
        try:
            draft_year_int = int(dy)
        except (TypeError, ValueError):
            draft_year_int = None
    return {
        "player_id": None if person_id is None else int(person_id),
        "birthdate": str(birth)[:10] if birth else None,   # 'YYYY-MM-DD'
        "draft_year": draft_year_int,
    }


def parse_player_season(base_json: str, advanced_json: str, season: str) -> pl.DataFrame:
    """Join LeagueDashPlayerStats Base + Advanced into per-100 production + rate stats.

    Per-100-possession stats are derived from counting totals and POSS (from Advanced). TS% and
    USG% are taken directly; PIE and NET_RATING are carried for validation/diagnostics.
    """
    base = _result_frame(base_json)
    adv = _result_frame(advanced_json)

    adv_sel = adv.select(
        pl.col("PLAYER_ID").cast(pl.Int64).alias("player_id"),
        pl.col("POSS").cast(pl.Float64, strict=False).alias("poss"),
        pl.col("TS_PCT").cast(pl.Float64, strict=False).alias("true_shooting"),
        pl.col("USG_PCT").cast(pl.Float64, strict=False).alias("usage"),
        pl.col("PIE").cast(pl.Float64, strict=False).alias("pie"),
        pl.col("NET_RATING").cast(pl.Float64, strict=False).alias("net_rating"),
    )
    base_sel = base.select(
        pl.col("PLAYER_ID").cast(pl.Int64).alias("player_id"),
        pl.col("PLAYER_NAME").cast(pl.Utf8).alias("full_name"),
        pl.col("AGE").cast(pl.Float64, strict=False).alias("age"),
        pl.col("GP").cast(pl.Int64, strict=False).alias("gp"),
        pl.col("MIN").cast(pl.Float64, strict=False).alias("minutes"),
        pl.col("PTS").cast(pl.Float64, strict=False).alias("_pts"),
        pl.col("AST").cast(pl.Float64, strict=False).alias("_ast"),
        pl.col("OREB").cast(pl.Float64, strict=False).alias("_oreb"),
        pl.col("DREB").cast(pl.Float64, strict=False).alias("_dreb"),
        pl.col("REB").cast(pl.Float64, strict=False).alias("_reb"),
        pl.col("STL").cast(pl.Float64, strict=False).alias("_stl"),
        pl.col("BLK").cast(pl.Float64, strict=False).alias("_blk"),
        pl.col("TOV").cast(pl.Float64, strict=False).alias("_tov"),
    )
    joined = base_sel.join(adv_sel, on="player_id", how="inner")

    def per100(total: str) -> pl.Expr:
        rate = pl.col(total) / pl.col("poss") * 100.0
        return pl.when(pl.col("poss") > 0).then(rate).otherwise(None)

    return joined.with_columns(
        pl.lit(season).alias("season"),
        per100("_pts").alias("pts_per100"),
        per100("_ast").alias("ast_per100"),
        per100("_oreb").alias("oreb_per100"),
        per100("_dreb").alias("dreb_per100"),
        per100("_reb").alias("reb_per100"),
        per100("_stl").alias("stl_per100"),
        per100("_blk").alias("blk_per100"),
        per100("_tov").alias("tov_per100"),
    ).drop("_pts", "_ast", "_oreb", "_dreb", "_reb", "_stl", "_blk", "_tov")
