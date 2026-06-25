"""Concrete ingester for NBA Stats via the ``nba_api`` library (PRIMARY NBA source).

Runs locally only (cloud IPs get banned). ``nba_api`` performs the HTTP itself, so here we
wrap its endpoint calls with our own rate limiting, caching, and provenance. The JSON returned
by each endpoint is cached verbatim under ``data/raw/nba_stats/`` with a provenance sidecar.

This module is NOT exercised by the offline test suite (it needs network + the optional
``ingest`` extra: ``pip install -e ".[ingest]"``). It is the code you run to pull real data.

Endpoints we rely on (all available pre-draft for prospects, or post-draft for NBA outcomes):
  * DraftHistory          — picks by draft year (links prospects to draft slots)
  * DraftCombineStats     — wingspan, standing reach, vertical, lane agility, sprint, body fat
  * LeagueDashPlayerStats — player season box + advanced (basis to COMPUTE BPM/VORP outcomes)
  * LeagueDashTeamStats   — team season context for the BPM team adjustment
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from nba_draft.ingestion.cache import FileCache, cache_key
from nba_draft.ingestion.http import RateLimiter
from nba_draft.ingestion.provenance import Provenance
from nba_draft.ingestion.registry import Source, get_source
from nba_draft.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

log = get_logger("ingestion.nba_stats")

SOURCE_ID = "nba_stats"


class NbaStatsIngester:
    """Cached, rate-limited wrapper over selected ``nba_api`` endpoints."""

    def __init__(
        self,
        cache_root: str | Path,
        *,
        source: Source | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self.source = source or get_source(SOURCE_ID)
        if not self.source.enabled:
            raise RuntimeError(f"Source {SOURCE_ID!r} is disabled in the registry.")
        self.cache = FileCache(cache_root, ext="json")
        self._rate = rate_limiter or RateLimiter(self.source.min_interval_s)

    def _cached_json(
        self, logical_url: str, params: dict[str, str], produce: Callable[[], str]
    ) -> str:
        """Return cached JSON for a logical request, else produce (rate-limited) and cache it."""
        key = cache_key(logical_url, params)
        cached = self.cache.get(self.source.id, key)
        if cached is not None:
            log.debug("cache hit %s", logical_url)
            return cached.decode("utf-8")
        self._rate.wait()
        payload = produce()
        data = payload.encode("utf-8")
        prov = Provenance.create(
            source_id=self.source.id,
            url=logical_url,
            data=data,
            license=self.source.license,
            attribution_required=self.source.attribution_required,
            params=params,
        )
        self.cache.put(self.source.id, key, data, prov)
        return payload

    def draft_history(self, season_year: int) -> str:
        params = {"season": str(season_year)}

        def produce() -> str:
            from nba_api.stats.endpoints import drafthistory

            return _endpoint_json(drafthistory.DraftHistory(season_year_nullable=season_year))

        return self._cached_json("nba_api://DraftHistory", params, produce)

    def draft_combine_stats(self, season_all_time: str) -> str:
        params = {"season_all_time": season_all_time}

        def produce() -> str:
            from nba_api.stats.endpoints import draftcombinestats

            return _endpoint_json(
                draftcombinestats.DraftCombineStats(season_all_time=season_all_time)
            )

        return self._cached_json("nba_api://DraftCombineStats", params, produce)

    def player_info(self, player_id: int) -> str:
        """CommonPlayerInfo for one player (birthdate, draft year, bio). One call per player."""
        params = {"player_id": str(player_id)}

        def produce() -> str:
            from nba_api.stats.endpoints import commonplayerinfo

            return _endpoint_json(commonplayerinfo.CommonPlayerInfo(player_id=player_id))

        return self._cached_json("nba_api://CommonPlayerInfo", params, produce)

    def player_season_stats(self, season: str, measure_type: str = "Base") -> str:
        params = {"season": season, "measure_type": measure_type}

        def produce() -> str:
            from nba_api.stats.endpoints import leaguedashplayerstats

            return _endpoint_json(
                leaguedashplayerstats.LeagueDashPlayerStats(
                    season=season, measure_type_detailed_defense=measure_type
                )
            )

        return self._cached_json("nba_api://LeagueDashPlayerStats", params, produce)


def _endpoint_json(endpoint: Any) -> str:
    """Extract the raw JSON string from an nba_api endpoint object."""
    return str(endpoint.get_json())
