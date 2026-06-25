"""Tests for the real-data join logic (offline; the pull step is exercised separately)."""

from __future__ import annotations

import polars as pl

from nba_draft.realdata.build import (
    FEATURE_COLUMNS,
    PICK_COLUMN,
    TARGET_COLUMN,
    build_real_modeling_table,
)


def _frames():
    draft_history = pl.DataFrame(
        {
            "player_id": [1, 2, 3],
            "full_name": ["Reached Star", "Bust", "Censored Recent"],
            "draft_year": [2018, 2018, 2022],
            "draft_pick": [3, 25, 5],
        }
    )
    combine = pl.DataFrame(
        {
            "player_id": [1, 3],   # player 2 never attended the Combine
            "wingspan_in": [84.0, 86.0],
            "standing_reach_in": [110.0, 112.0],
            "max_vertical_in": [35.0, 33.0],
            "lane_agility_s": [11.0, 11.5],
            "body_fat_pct": [6.0, 7.0],
        }
    )
    # player 1 plays 4 strong seasons (resolved, reached); player 3 has one recent season (censored)
    player_seasons = pl.DataFrame(
        {
            "player_id": [1, 1, 1, 1, 3],
            "season": ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23"],
            "minutes": [2000.0, 2100.0, 2200.0, 2300.0, 1500.0],
            "ebpm": [2.0, 4.0, 5.0, 3.0, 1.0],
            "vorp": [1.0, 2.0, 3.0, 2.0, 0.8],
        }
    )
    return draft_history, combine, player_seasons


def test_real_table_has_features_labels_and_resolved_flag():
    dh, combine, ps = _frames()
    table = build_real_modeling_table(dh, combine, ps, data_through_year=2023)

    assert table.height == 3
    for col in (PICK_COLUMN, "draft_year", "reached", TARGET_COLUMN, "resolved", *FEATURE_COLUMNS):
        assert col in table.columns

    p1 = table.filter(pl.col("player_id") == 1).row(0, named=True)
    assert p1["reached"] is True and p1["resolved"] is True
    assert p1["peak_impact"] is not None
    assert p1["wingspan_in"] == 84.0

    # player 2: drafted, never played -> bust, no Combine features (null, not zero)
    p2 = table.filter(pl.col("player_id") == 2).row(0, named=True)
    assert p2["reached"] is False
    assert p2["wingspan_in"] is None

    # player 3: 2022 class, window not complete by 2023 -> censored (resolved False)
    p3 = table.filter(pl.col("player_id") == 3).row(0, named=True)
    assert p3["resolved"] is False


def test_trainable_subset_excludes_bust_and_censored():
    dh, combine, ps = _frames()
    table = build_real_modeling_table(dh, combine, ps, data_through_year=2023)
    trainable = table.filter(
        pl.col("resolved") & pl.col("reached") & pl.col(TARGET_COLUMN).is_not_null()
    )
    assert trainable["player_id"].to_list() == [1]  # only the resolved, reached star
