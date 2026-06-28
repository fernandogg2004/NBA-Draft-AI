"""Tests for the All-Star/All-NBA honors source feeding the top outcome tiers."""

from __future__ import annotations

import polars as pl

from nba_draft.targets import (
    build_labels_frame,
    build_player_outcomes,
    honors_from_frame,
    load_honors,
)


def _seasons() -> pl.DataFrame:
    # a solid-but-not-elite scorer by box metrics (peak eBPM ~ +2.5, below the +3 All-Star band)
    return pl.DataFrame(
        {
            "player_id": [1, 1, 1, 1],
            "season": ["2018-19", "2019-20", "2020-21", "2021-22"],
            "minutes": [2000.0, 2000.0, 2000.0, 2000.0],
            "ebpm": [2.0, 2.5, 2.5, 2.0],
            "vorp": [2.0, 2.5, 2.5, 2.0],
        }
    )


def test_pull_player_honors_builds_counts_and_survives_failures():
    """The acquisition layer maps player_id -> (all_star, all_nba), tolerating per-player errors."""
    import json as _json

    from nba_draft.realdata.honors import pull_player_honors

    def _awards(rows: list[list[object]]) -> str:
        return _json.dumps(
            {"resultSets": [{"name": "PlayerAwards",
                             "headers": ["PERSON_ID", "DESCRIPTION"], "rowSet": rows}]}
        )

    class FakeIngester:
        def player_awards(self, pid: int) -> str:
            if pid == 1:
                return _awards([[1, "All-Star"], [1, "All-NBA"], [1, "All-Star"]])
            if pid == 2:
                return _awards([])               # no honors
            raise RuntimeError("transient API error")  # pid 3 fails -> (0, 0)

    draft = pl.DataFrame({"player_id": [1, 2, 3], "draft_year": [2018, 2018, 2018]})
    honors = pull_player_honors(FakeIngester(), draft)
    assert honors[1] == (2, 1)
    assert honors[2] == (0, 0)
    assert honors[3] == (0, 0)  # failure -> no honors, batch not aborted


def test_honors_promote_top_tier_over_bpm_band():
    draft = pl.DataFrame({"player_id": [1], "draft_year": [2018]})

    # Without honors -> BPM band puts this player in "starter" (peak ~2.5 < 3.0).
    no_honors = build_labels_frame(build_player_outcomes(_seasons(), draft))
    assert no_honors.row(0, named=True)["outcome_tier"] == "starter"

    # With an All-Star selection -> promoted to "all_star".
    honors = {1: (2, 0)}  # 2 All-Star selections, 0 All-NBA
    with_honors = build_labels_frame(build_player_outcomes(_seasons(), draft, honors=honors))
    assert with_honors.row(0, named=True)["outcome_tier"] == "all_star"


def test_honors_from_frame_and_load(tmp_path):
    df = pl.DataFrame(
        {"player_id": [10, 11], "all_star_count": [3, 0], "all_nba_count": [1, 0]}
    )
    h = honors_from_frame(df)
    assert h[10] == (3, 1)
    assert h[11] == (0, 0)

    csv = tmp_path / "honors.csv"
    df.write_csv(csv)
    assert load_honors(csv) == h
