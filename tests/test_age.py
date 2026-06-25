"""Tests for the age-at-draft source (computation + pull + parsing), offline."""

from __future__ import annotations

import json

import polars as pl
import pytest

from nba_draft.ingestion.parse import parse_player_info
from nba_draft.realdata.age import age_at_draft, pull_player_ages


def test_age_at_draft_computation():
    # born 2004-01-04, drafted 2023 (draft day ~ June 26) -> ~19.5 years
    assert age_at_draft("2004-01-04", 2023) == pytest.approx(19.48, abs=0.05)
    # older prospect
    assert age_at_draft("1998-06-26", 2021) == pytest.approx(23.0, abs=0.05)
    # missing birthdate -> None (not zero)
    assert age_at_draft(None, 2023) is None
    assert age_at_draft("not-a-date", 2023) is None


def test_parse_player_info_extracts_birthdate():
    raw = json.dumps(
        {
            "resultSets": [
                {
                    "headers": ["PERSON_ID", "DISPLAY_FIRST_LAST", "BIRTHDATE", "DRAFT_YEAR"],
                    "rowSet": [[1641705, "Victor Wembanyama", "2004-01-04T00:00:00", "2023"]],
                }
            ]
        }
    )
    info = parse_player_info(raw)
    assert info["player_id"] == 1641705
    assert info["birthdate"] == "2004-01-04"
    assert info["draft_year"] == 2023


class _FakeIngester:
    """Returns canned CommonPlayerInfo JSON keyed by player_id."""

    def __init__(self, birthdays: dict[int, str | None]) -> None:
        self._b = birthdays

    def player_info(self, player_id: int) -> str:
        b = self._b[player_id]
        birth = f"{b}T00:00:00" if b else None
        return json.dumps(
            {
                "resultSets": [
                    {
                        "headers": ["PERSON_ID", "BIRTHDATE", "DRAFT_YEAR"],
                        "rowSet": [[player_id, birth, "2020"]],
                    }
                ]
            }
        )


def test_pull_player_ages_computes_and_handles_missing():
    draft_history = pl.DataFrame(
        {"player_id": [1, 2], "draft_year": [2020, 2020]}
    )
    ing = _FakeIngester({1: "2000-06-26", 2: None})
    ages = pull_player_ages(ing, draft_history)
    a1 = ages.filter(pl.col("player_id") == 1).row(0, named=True)
    a2 = ages.filter(pl.col("player_id") == 2).row(0, named=True)
    assert a1["age_at_draft"] == pytest.approx(20.0, abs=0.05)
    assert a2["age_at_draft"] is None  # birthdate missing -> null (imputed downstream)
