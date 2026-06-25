"""Offline tests for the CollegeBasketballData ingester (no network; fetcher injected)."""

from __future__ import annotations

import pytest

from nba_draft.ingestion import (
    CollegeBasketballDataIngester,
    FileCache,
    MissingApiKeyError,
    PoliteClient,
    RateLimiter,
    get_source,
)


def test_missing_key_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("CBD_API_KEY", raising=False)
    with pytest.raises(MissingApiKeyError):
        CollegeBasketballDataIngester(tmp_path)


def test_get_uses_cache_and_hits_endpoint_once(tmp_path):
    calls: list[str] = []

    def fetcher(url, params):
        calls.append(url)
        return 200, b'[{"team": "Duke", "points": 100}]'

    source = get_source("college_bb_data").model_copy(update={"enabled": True})
    client = PoliteClient(source, FileCache(tmp_path, ext="json"), fetcher=fetcher,
                          rate_limiter=RateLimiter(0.0))
    ing = CollegeBasketballDataIngester(tmp_path, api_key="fake", client=client)

    raw1 = ing.player_season_stats(2024, team="Duke")
    raw2 = ing.player_season_stats(2024, team="Duke")  # served from cache
    assert raw1 == raw2 == '[{"team": "Duke", "points": 100}]'
    assert len(calls) == 1  # second call cached
    # the request targets the configured base URL + player-season path
    assert calls[0].startswith("https://api.collegebasketballdata.com/stats/player/season")


def test_explicit_api_key_builds_authenticated_client(tmp_path):
    # with a key but no injected client, it constructs without error (no network until a call)
    ing = CollegeBasketballDataIngester(tmp_path, api_key="abc123")
    assert ing.api_key == "abc123"
    assert ing.source.enabled is True
