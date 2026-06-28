"""Tests for the real-data adapters: nba_api JSON parsing, eBPM/VORP, outcome labels."""

from __future__ import annotations

import json

import polars as pl
import pytest

from nba_draft.ingestion.parse import (
    parse_combine,
    parse_draft_history,
    parse_player_season,
)
from nba_draft.targets import (
    add_impact_metrics,
    build_labels_frame,
    build_player_outcomes,
    estimated_bpm,
    pie_rank_agreement,
)
from nba_draft.targets.impact import vorp


def _payload(headers: list[str], rows: list[list]) -> str:
    return json.dumps({"resultSets": [{"headers": headers, "rowSet": rows}]})


# ----------------------------------------------------------------- parsing
def test_parse_draft_history():
    raw = _payload(
        ["PERSON_ID", "PLAYER_NAME", "SEASON", "ROUND_NUMBER", "ROUND_PICK", "OVERALL_PICK",
         "DRAFT_TYPE", "TEAM_ID", "TEAM_CITY", "TEAM_NAME", "TEAM_ABBREVIATION",
         "ORGANIZATION", "ORGANIZATION_TYPE", "PLAYER_PROFILE_FLAG"],
        [[1, "Jane Doe", 2023, 1, 3, 3, "Draft", 10, "X", "Y", "XYZ", "Duke", "College", 1]],
    )
    df = parse_draft_history(raw)
    row = df.row(0, named=True)
    assert row["player_id"] == 1 and row["draft_year"] == 2023
    assert row["draft_pick"] == 3 and row["organization"] == "Duke"
    # NBA team that made the pick is captured for the post-draft view.
    assert row["team_abbr"] == "XYZ" and row["team_name"] == "Y"


def test_parse_draft_history_tolerates_missing_team_columns():
    # Older payloads without team columns must yield nulls, not raise.
    raw = _payload(
        ["PERSON_ID", "PLAYER_NAME", "SEASON", "ROUND_NUMBER", "OVERALL_PICK",
         "ORGANIZATION", "ORGANIZATION_TYPE"],
        [[1, "Jane Doe", 2014, 1, 3, "Duke", "College"]],
    )
    row = parse_draft_history(raw).row(0, named=True)
    assert row["team_abbr"] is None and row["team_name"] is None


def test_parse_combine_casts_measurements():
    raw = _payload(
        ["SEASON", "PLAYER_ID", "FIRST_NAME", "LAST_NAME", "PLAYER_NAME", "POSITION",
         "WINGSPAN", "STANDING_REACH", "MAX_VERTICAL_LEAP", "LANE_AGILITY_TIME", "BODY_FAT_PCT"],
        [[2023, 7, "A", "B", "A B", "SG", 80.5, 104.0, 38.0, 11.1, 6.2],
         [2023, 8, "C", "D", "C D", "C", None, None, None, None, None]],
    )
    df = parse_combine(raw)
    assert df["wingspan_in"][0] == pytest.approx(80.5)
    assert df["wingspan_in"][1] is None  # not measured -> null, not zero
    assert df["draft_year"][0] == 2023


def test_parse_player_season_computes_per100():
    base = _payload(
        ["PLAYER_ID", "PLAYER_NAME", "NICKNAME", "TEAM_ID", "TEAM_ABBREVIATION", "AGE", "GP",
         "W", "L", "W_PCT", "MIN", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
         "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB", "AST", "TOV", "STL", "BLK", "BLKA",
         "PF", "PFD", "PTS"],
        [[1, "Jane Doe", "", 10, "XYZ", 22.0, 70, 40, 30, 0.57, 2100, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 50, 150, 200, 300, 100, 80, 40, 0, 0, 0, 1000]],
    )
    adv = _payload(
        ["PLAYER_ID", "POSS", "TS_PCT", "USG_PCT", "PIE", "NET_RATING"],
        [[1, 2000.0, 0.60, 0.25, 0.12, 3.5]],
    )
    df = parse_player_season(base, adv, "2023-24")
    row = df.row(0, named=True)
    assert row["pts_per100"] == pytest.approx(1000 / 2000 * 100)   # 50
    assert row["ast_per100"] == pytest.approx(15.0)
    assert row["true_shooting"] == pytest.approx(0.60)
    assert row["season"] == "2023-24"


# ----------------------------------------------------------------- impact metrics
def _season_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_id": [1, 2, 3],
            "season": ["2023-24"] * 3,
            "minutes": [2000.0, 1500.0, 500.0],
            "pts_per100": [30.0, 22.0, 12.0],
            "ast_per100": [8.0, 5.0, 2.0],
            "oreb_per100": [2.0, 3.0, 1.0],
            "dreb_per100": [10.0, 8.0, 5.0],
            "stl_per100": [2.0, 1.5, 0.5],
            "blk_per100": [1.0, 2.0, 0.3],
            "tov_per100": [3.0, 2.5, 1.5],
            "true_shooting": [0.62, 0.57, 0.50],
            "pie": [0.18, 0.12, 0.05],
        }
    )


def test_ebpm_is_league_centered_near_zero():
    df = _season_frame()
    ebpm = estimated_bpm(df)
    # minutes-weighted mean within the season should be ~0
    w = df["minutes"].to_numpy()
    wm = float((ebpm.to_numpy() * w).sum() / w.sum())
    assert abs(wm) < 1e-2  # ~0 up to 3-decimal rounding of eBPM


def test_vorp_formula_and_replacement():
    # a replacement-level player (eBPM = -2) has VORP 0 regardless of minutes
    z = vorp(pl.Series([-2.0]), pl.Series([2000.0]))
    assert z[0] == pytest.approx(0.0)
    # positive eBPM + minutes -> positive VORP
    v = vorp(pl.Series([4.0]), pl.Series([1968.0]))  # (4+2)*1968/(240*82)=0.6
    assert v[0] == pytest.approx(0.6, abs=1e-3)


def test_add_impact_metrics_and_pie_agreement():
    df = add_impact_metrics(_season_frame())
    assert "ebpm" in df.columns and "vorp" in df.columns
    # eBPM should rank players the same way the official PIE does, on this clean example
    assert pie_rank_agreement(df) == pytest.approx(1.0)


# ----------------------------------------------------------------- outcome labels
def test_build_outcomes_and_labels_from_nba_frames():
    # player 1 plays 4 strong seasons from 2018; player 2 never appears (non-reach)
    player_seasons = pl.DataFrame(
        {
            "player_id": [1, 1, 1, 1],
            "season": ["2018-19", "2019-20", "2020-21", "2021-22"],
            "minutes": [2000.0, 2100.0, 2200.0, 2300.0],
            "ebpm": [1.0, 3.0, 5.0, 4.0],
            "vorp": [1.0, 2.0, 3.0, 2.5],
        }
    )
    draft_history = pl.DataFrame({"player_id": [1, 2], "draft_year": [2018, 2018]})
    outcomes = build_player_outcomes(player_seasons, draft_history)
    assert outcomes[1].debut_year == 2018
    assert outcomes[2].debut_year is None  # never played

    labels = build_labels_frame(outcomes).sort("player_id")
    r1 = labels.filter(pl.col("player_id") == 1).row(0, named=True)
    r2 = labels.filter(pl.col("player_id") == 2).row(0, named=True)
    assert r1["reached"] is True
    assert r1["peak_impact"] == pytest.approx((5.0 + 4.0) / 2)  # mean of top-2 BPM seasons
    assert r2["reached"] is False and r2["outcome_tier"] == "bust"
