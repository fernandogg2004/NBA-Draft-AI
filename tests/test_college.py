"""Tests for the CBD player-season parser and the college->draft entity linker."""

from __future__ import annotations

import json

import polars as pl
import pytest

from nba_draft.ingestion.parse import parse_cbd_player_season
from nba_draft.realdata.college import COLLEGE_FEATURE_COLUMNS, link_college_features


def _cbd_payload() -> str:
    # mirrors the real CBD schema (nested rebounds/winShares, percent-scaled usage/efg)
    return json.dumps(
        [
            {
                "athleteId": 160, "name": "Jeremy Roach", "team": "Duke", "conference": "ACC",
                "season": 2024, "position": "G", "games": 35, "minutes": 1144,
                "points": 489, "assists": 114, "steals": 39, "blocks": 5, "turnovers": 49,
                "offensiveRating": 124.8, "defensiveRating": 102.9, "netRating": 21.9,
                "PORPAG": 4.1, "usage": 20.7, "assistsTurnoverRatio": 2.33,
                "offensiveReboundPct": 23.9, "effectiveFieldGoalPct": 54.4,
                "trueShootingPct": 0.598,
                "threePointFieldGoals": {"made": 54, "attempted": 126, "pct": 42.9},
                "rebounds": {"offensive": 21, "defensive": 67, "total": 88},
                "winShares": {"total": 5.1, "totalPer40": 0.178},
            }
        ]
    )


def test_parse_cbd_scales_and_per40():
    df = parse_cbd_player_season(_cbd_payload())
    row = df.row(0, named=True)
    assert row["full_name"] == "Jeremy Roach" and row["school"] == "Duke"
    assert row["true_shooting"] == pytest.approx(0.598)        # already a fraction
    assert row["usage"] == pytest.approx(0.207)                # percent -> fraction
    assert row["efg"] == pytest.approx(0.544)
    assert row["three_pt_pct"] == pytest.approx(0.429)
    assert row["pts_per40"] == pytest.approx(489 / 1144 * 40)  # per-40 from totals
    assert row["reb_per40"] == pytest.approx(88 / 1144 * 40)   # nested rebounds.total
    assert row["win_shares_per40"] == pytest.approx(0.178)


def test_parse_cbd_empty_returns_typed_empty_frame():
    df = parse_cbd_player_season("[]")
    assert df.height == 0
    assert "pts_per40" in df.columns


def test_link_college_features_matches_by_name_and_year():
    draft_history = pl.DataFrame(
        {
            "player_id": [101, 102, 103],
            "full_name": ["Jeremy Roach", "Some Intl Guy", "Paolo Banchero"],
            "draft_year": [2024, 2024, 2022],
            "organization": ["Duke", "Real Madrid", "Duke"],
        }
    )
    cbd = parse_cbd_player_season(_cbd_payload()).with_columns(
        pl.col("season").cast(pl.Int64)
    )
    linked = link_college_features(draft_history, cbd)
    assert linked.height == 3
    assert set(COLLEGE_FEATURE_COLUMNS).issubset(linked.columns)

    roach = linked.filter(pl.col("player_id") == 101).row(0, named=True)
    assert roach["pts_per40"] == pytest.approx(489 / 1144 * 40)
    # international player + wrong-year player -> no college match -> null (imputed later)
    intl = linked.filter(pl.col("player_id") == 102).row(0, named=True)
    assert intl["pts_per40"] is None
    banchero = linked.filter(pl.col("player_id") == 103).row(0, named=True)
    assert banchero["pts_per40"] is None  # 2022 season not in the fixture


def test_link_disambiguates_by_school():
    # two players with the same name in the same season -> pick the one matching the draft school
    payload = json.dumps(
        [
            {"athleteId": 1, "name": "John Smith", "team": "Duke", "season": 2023,
             "minutes": 1000, "points": 500, "rebounds": {"total": 100}, "trueShootingPct": 0.6},
            {"athleteId": 2, "name": "John Smith", "team": "Kansas", "season": 2023,
             "minutes": 1000, "points": 200, "rebounds": {"total": 100}, "trueShootingPct": 0.5},
        ]
    )
    cbd = parse_cbd_player_season(payload)
    dh = pl.DataFrame(
        {"player_id": [9], "full_name": ["John Smith"], "draft_year": [2023],
         "organization": ["Kansas"]}
    )
    linked = link_college_features(dh, cbd)
    # Kansas John Smith scored 200 pts -> per40 = 200/1000*40 = 8.0
    assert linked.row(0, named=True)["pts_per40"] == pytest.approx(8.0)
