"""Phase 1 — data acquisition.

A source-agnostic ingestion core (registry, provenance, caching, rate-limiting, retries,
robots respect) on top of which thin per-source ingesters live. The core is fully unit-tested
offline: the network fetcher is injectable, so no test makes a live request.

Posture (instructions.md Phase 1): prefer APIs/open datasets; never access paths a source's
robots.txt disallows; record provenance + license for everything; rate-limit, cache, retry.
"""

from nba_draft.ingestion.cache import FileCache
from nba_draft.ingestion.college_bb_data import (
    CollegeBasketballDataIngester,
    MissingApiKeyError,
)
from nba_draft.ingestion.euroleague import EuroLeagueIngester
from nba_draft.ingestion.http import (
    FetchBlockedError,
    PoliteClient,
    RateLimiter,
    RobotsChecker,
)
from nba_draft.ingestion.parse import (
    parse_cbd_player_season,
    parse_combine,
    parse_draft_history,
    parse_euroleague_player_season,
    parse_player_season,
)
from nba_draft.ingestion.provenance import Provenance, sha256_bytes
from nba_draft.ingestion.registry import Source, get_source, load_sources

__all__ = [
    "CollegeBasketballDataIngester",
    "EuroLeagueIngester",
    "FetchBlockedError",
    "FileCache",
    "MissingApiKeyError",
    "PoliteClient",
    "Provenance",
    "RateLimiter",
    "RobotsChecker",
    "Source",
    "get_source",
    "load_sources",
    "parse_cbd_player_season",
    "parse_combine",
    "parse_draft_history",
    "parse_euroleague_player_season",
    "parse_player_season",
    "sha256_bytes",
]
