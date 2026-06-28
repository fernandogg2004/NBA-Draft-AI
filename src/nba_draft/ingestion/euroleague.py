"""Ingester for EuroLeague player-season stats (international pre-draft features).

Fills pre-draft production for international prospects who never played NCAA. Backed by the
``euroleague_api`` package (MIT), which wraps EuroLeague's public feeds; we cache its result as
records JSON with provenance, mirroring the nba_api / CollegeBasketballData ingesters.

Runs locally / online only; gated on the optional ``ingest`` extra. The exact returned columns
vary by EuroLeague endpoint, so the parser resolves them case-insensitively and
``scripts/verify_euroleague.py`` prints the live schema to confirm the mapping.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nba_draft.ingestion.cache import FileCache, cache_key
from nba_draft.ingestion.http import RateLimiter
from nba_draft.ingestion.provenance import Provenance
from nba_draft.ingestion.registry import Source, get_source
from nba_draft.utils.logging import get_logger

log = get_logger("ingestion.euroleague")

SOURCE_ID = "euroleague"


class EuroLeagueIngester:
    """Cached, rate-limited wrapper over ``euroleague_api`` player-season stats."""

    def __init__(
        self,
        cache_root: str | Path,
        *,
        source: Source | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        # Enable in-memory (the package handles the public feeds; registry entry stays gated).
        base = source or get_source(SOURCE_ID)
        self.source = base.model_copy(update={"enabled": True})
        self.cache = FileCache(cache_root, ext="json")
        self._rate = rate_limiter or RateLimiter(self.source.min_interval_s)

    def player_season_stats(
        self, season: int, *, endpoint: str = "traditional", statistic_mode: str = "Accumulated"
    ) -> str:
        """Return (cached) EuroLeague player season stats as records JSON for a season."""
        params = {"season": str(season), "endpoint": endpoint, "mode": statistic_mode}
        key = cache_key("euroleague://player_season", params)
        cached = self.cache.get(self.source.id, key)
        if cached is not None:
            return cached.decode("utf-8")

        self._rate.wait()
        payload = self._fetch_records(season, endpoint, statistic_mode)
        if payload is None:
            # Season not available yet (e.g. the upcoming EuroLeague season around a just-happened
            # draft -> the API 404s). Return empty WITHOUT caching, so a later run can fetch it
            # once the season exists; downstream parsing/concat tolerate an empty frame.
            log.warning("EuroLeague season %s not available yet; treating as empty.", season)
            return "[]"
        data = payload.encode("utf-8")
        prov = Provenance.create(
            source_id=self.source.id, url="euroleague_api://PlayerStats", data=data,
            license=self.source.license, attribution_required=self.source.attribution_required,
            params=params,
        )
        self.cache.put(self.source.id, key, data, prov)
        return payload

    def _fetch_records(self, season: int, endpoint: str, statistic_mode: str) -> str | None:
        """Fetch one season's records as JSON, or ``None`` if the season isn't available (404)."""
        import requests
        from euroleague_api.player_stats import PlayerStats

        try:
            df = PlayerStats().get_player_stats_single_season(
                endpoint=endpoint, season=season, statistic_mode=statistic_mode
            )
        except requests.exceptions.HTTPError as exc:
            resp = exc.response
            if resp is not None and resp.status_code == 404:
                return None  # season not played yet -> caller treats as empty
            raise
        # euroleague_api returns a pandas DataFrame; serialize as a list-of-records JSON.
        records: list[dict[str, Any]] = df.to_dict(orient="records")
        return json.dumps(records)
