"""Ingester for the CollegeBasketballData.com API (pre-draft college production).

Provides the pre-draft FEATURE source the nba_api-only pipeline lacks. Auth is a Bearer token
(free key at https://collegebasketballdata.com/key). Obtaining a key constitutes accepting the
provider's terms, so the key's presence is the gate here (rather than the registry `enabled`
flag). Reuses PoliteClient for caching, rate limiting, retries, and provenance.

Endpoint paths are kept configurable and default to the documented REST conventions; verify them
against https://api.collegebasketballdata.com/docs with `scripts/verify_cbd.py` once you have a key,
and adjust the constants here if the live API differs.
"""

from __future__ import annotations

import os
from pathlib import Path

from nba_draft.ingestion.cache import FileCache
from nba_draft.ingestion.http import PoliteClient
from nba_draft.ingestion.registry import Source, get_source
from nba_draft.utils.logging import get_logger

log = get_logger("ingestion.cbd")

SOURCE_ID = "college_bb_data"
API_KEY_ENV = "CBD_API_KEY"

# Default REST paths (verify against /docs; override if the live API differs).
PATH_PLAYER_SEASON_STATS = "/stats/player/season"
PATH_TEAMS = "/teams"
PATH_TEAM_ROSTER = "/teams/roster"


class MissingApiKeyError(RuntimeError):
    """Raised when no CollegeBasketballData API key is available."""


class CollegeBasketballDataIngester:
    """Cached, rate-limited, Bearer-authenticated client for CollegeBasketballData.com."""

    def __init__(
        self,
        cache_root: str | Path,
        *,
        api_key: str | None = None,
        source: Source | None = None,
        client: PoliteClient | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get(API_KEY_ENV)
        if not self.api_key and client is None:
            raise MissingApiKeyError(
                f"No API key. Get a free key at https://collegebasketballdata.com/key and set "
                f"the {API_KEY_ENV} environment variable (or pass api_key=...)."
            )
        # Build an enabled in-memory source view (key presence == terms accepted).
        base = source or get_source(SOURCE_ID)
        self.source = base.model_copy(update={"enabled": True})
        self.cache = FileCache(cache_root, ext="json")
        self._client = client or PoliteClient(
            self.source,
            self.cache,
            headers={"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"},
        )

    def _url(self, path: str) -> str:
        return self.source.base_url.rstrip("/") + path

    def get(self, path: str, params: dict[str, str] | None = None) -> str:
        """Fetch (cached) a raw JSON response for an endpoint path + params."""
        return self._client.get(self._url(path), params=params).decode("utf-8")

    def player_season_stats(
        self, season: int, *, team: str | None = None, conference: str | None = None
    ) -> str:
        params = {"season": str(season)}
        if team:
            params["team"] = team
        if conference:
            params["conference"] = conference
        return self.get(PATH_PLAYER_SEASON_STATS, params)

    def teams(self, *, season: int | None = None, conference: str | None = None) -> str:
        params: dict[str, str] = {}
        if season:
            params["season"] = str(season)
        if conference:
            params["conference"] = conference
        return self.get(PATH_TEAMS, params)

    def team_roster(self, team: str, *, season: int | None = None) -> str:
        params = {"team": team}
        if season:
            params["season"] = str(season)
        return self.get(PATH_TEAM_ROSTER, params)
