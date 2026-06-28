"""Tests for the EuroLeague (international) parser + entity linker (offline)."""

from __future__ import annotations

import json

import polars as pl
import pytest

from nba_draft.ingestion.parse import parse_euroleague_player_season
from nba_draft.realdata.build import build_real_modeling_table
from nba_draft.realdata.intl import INTL_FEATURE_COLUMNS, link_intl_features


def _el_payload() -> str:
    # euroleague_api "traditional / Accumulated" style records (totals).
    return json.dumps(
        [
            {
                "playerName": "Luka Doncic", "season": "E2017", "gamesPlayed": 33,
                "minutesPlayed": 900.0, "pointsScored": 500, "assistances": 150,
                "totalRebounds": 160, "steals": 40, "blocksFavour": 10, "turnovers": 80,
                "fieldGoalsAttempted2": 250, "fieldGoalsAttempted3": 150,
                "freeThrowsAttempted": 120,
            },
            # different casing / dotted key -> case-insensitive resolution must still work
            {
                "PLAYERNAME": "Varied Casing", "Season": 2016, "GamesPlayed": 20,
                "minutes": 400.0, "points": 200, "assists": 50, "rebounds": 80,
            },
        ]
    )


def test_parse_euroleague_per40_and_true_shooting():
    df = parse_euroleague_player_season(_el_payload())
    luka = df.filter(pl.col("full_name") == "Luka Doncic").row(0, named=True)
    assert luka["season"] == 2017                                  # parsed from "E2017"
    assert luka["pts_per40"] == pytest.approx(500 / 900 * 40)
    assert luka["reb_per40"] == pytest.approx(160 / 900 * 40)
    ts = 500 / (2 * ((250 + 150) + 0.44 * 120))
    assert luka["true_shooting"] == pytest.approx(ts, abs=1e-4)


def test_parse_euroleague_case_insensitive_and_missing():
    df = parse_euroleague_player_season(_el_payload())
    varied = df.filter(pl.col("full_name") == "Varied Casing").row(0, named=True)
    assert varied["season"] == 2016
    assert varied["pts_per40"] == pytest.approx(200 / 400 * 40)
    # no shot attempts in that record -> true_shooting stays null (not an error)
    assert varied["true_shooting"] is None


def test_parse_euroleague_empty():
    df = parse_euroleague_player_season("[]")
    assert df.height == 0
    assert set(INTL_FEATURE_COLUMNS).issubset(df.columns)


def test_euroleague_ingester_tolerates_unavailable_season(tmp_path, monkeypatch):
    """A not-yet-played season (the API 404s) must yield empty WITHOUT caching, so the pull
    doesn't crash and a later run can still fetch it once the season exists."""
    from nba_draft.ingestion.euroleague import EuroLeagueIngester
    from nba_draft.ingestion.http import RateLimiter

    ing = EuroLeagueIngester(tmp_path, rate_limiter=RateLimiter(0.0))
    # Simulate the 404 path (e.g. SeasonCode E2026 before the 2026-27 season starts).
    monkeypatch.setattr(ing, "_fetch_records", lambda *a, **k: None)
    assert ing.player_season_stats(2026) == "[]"
    assert parse_euroleague_player_season(ing.player_season_stats(2026)).height == 0
    # Not cached -> a later successful fetch is still picked up (no stale empty cache).
    monkeypatch.setattr(ing, "_fetch_records", lambda *a, **k: '[{"playerName": "X"}]')
    assert "playerName" in ing.player_season_stats(2026)


def test_link_intl_matches_within_draft_window():
    draft = pl.DataFrame(
        {
            "player_id": [77, 78],
            "full_name": ["Luka Doncic", "Never Match"],
            "draft_year": [2018, 2018],
            "organization": ["Real Madrid", "Some Team"],
        }
    )
    intl = parse_euroleague_player_season(_el_payload())
    linked = link_intl_features(draft, intl)
    luka = linked.filter(pl.col("player_id") == 77).row(0, named=True)
    assert luka["pts_per40"] == pytest.approx(500 / 900 * 40)   # matched season 2017 == 2018-1
    nomatch = linked.filter(pl.col("player_id") == 78).row(0, named=True)
    assert nomatch["pts_per40"] is None


def test_build_table_coalesces_intl_features():
    # one drafted intl player, no college source -> intl fills the production feature columns
    draft = pl.DataFrame(
        {
            "player_id": [77], "full_name": ["Luka Doncic"], "draft_year": [2018],
            "draft_pick": [3], "organization": ["Real Madrid"],
        }
    )
    combine = pl.DataFrame({"player_id": [77]})
    player_seasons = pl.DataFrame(
        {
            "player_id": [77, 77], "season": ["2018-19", "2019-20"],
            "minutes": [2000.0, 2200.0], "ebpm": [3.0, 5.0], "vorp": [2.0, 3.0],
        }
    )
    intl = parse_euroleague_player_season(_el_payload())
    table = build_real_modeling_table(
        draft, combine, player_seasons, data_through_year=2023, intl_seasons=intl
    )
    row = table.filter(pl.col("player_id") == 77).row(0, named=True)
    assert row["pts_per40"] == pytest.approx(500 / 900 * 40)   # intl value populated the column
