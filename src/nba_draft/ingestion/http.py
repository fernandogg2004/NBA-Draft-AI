"""Polite HTTP access: rate limiting, retries, robots respect, caching, provenance.

The network call itself is an injectable ``fetcher`` (default uses ``requests``), so the whole
policy layer is unit-tested offline with a fake fetcher — no test touches the network.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

from nba_draft.ingestion.cache import FileCache, cache_key
from nba_draft.ingestion.provenance import Provenance
from nba_draft.ingestion.registry import Source
from nba_draft.utils.logging import get_logger

log = get_logger("ingestion.http")

# (status_code, content) — keeps the requests dependency out of the type surface.
Fetcher = Callable[[str, dict[str, str] | None], tuple[int, bytes]]

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
DEFAULT_USER_AGENT = "nba-draft-ai/0.1 (research; contact: configure-me)"


class FetchBlockedError(RuntimeError):
    """Raised when policy forbids a fetch (source disabled, scraping not allowed, robots block)."""


class FetchError(RuntimeError):
    """Raised when a fetch ultimately fails after retries."""


def make_requests_fetcher(headers: dict[str, str] | None = None) -> Fetcher:
    """Build a requests-based fetcher, optionally carrying extra headers (e.g. Bearer auth)."""
    merged = {"User-Agent": DEFAULT_USER_AGENT, **(headers or {})}

    def _fetch(url: str, params: dict[str, str] | None) -> tuple[int, bytes]:
        import requests  # lazy: requests is an optional 'ingest' dependency

        resp = requests.get(url, params=params, timeout=30, headers=merged)
        return resp.status_code, resp.content

    return _fetch


def _default_fetcher(url: str, params: dict[str, str] | None) -> tuple[int, bytes]:
    return make_requests_fetcher()(url, params)


class RateLimiter:
    """Enforce a minimum interval between calls (clock/sleep injectable for tests)."""

    def __init__(
        self,
        min_interval_s: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.min_interval_s = max(0.0, min_interval_s)
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    def wait(self) -> None:
        if self.min_interval_s <= 0:
            return
        now = self._clock()
        if self._last is not None:
            elapsed = now - self._last
            remaining = self.min_interval_s - elapsed
            if remaining > 0:
                self._sleep(remaining)
                now = self._clock()
        self._last = now


class RobotsChecker:
    """robots.txt gate. Fetches and caches a parser per host; fails closed on fetch error."""

    def __init__(
        self,
        fetch_text: Callable[[str], str],
        *,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        self._fetch_text = fetch_text
        self._user_agent = user_agent
        self._parsers: dict[str, RobotFileParser] = {}

    def _parser_for(self, url: str) -> RobotFileParser:
        parsed = urlparse(url)
        host = f"{parsed.scheme}://{parsed.netloc}"
        if host not in self._parsers:
            rp = RobotFileParser()
            try:
                content = self._fetch_text(f"{host}/robots.txt")
                rp.parse(content.splitlines())
            except Exception as exc:  # noqa: BLE001 - fail closed, but record why
                log.warning("Could not read robots.txt for %s (%s); failing closed.", host, exc)
                rp.disallow_all = True  # type: ignore[attr-defined]
            self._parsers[host] = rp
        return self._parsers[host]

    def allowed(self, url: str) -> bool:
        return self._parser_for(url).can_fetch(self._user_agent, url)


class PoliteClient:
    """Fetch artifacts for one source under all the politeness/legal policies."""

    def __init__(
        self,
        source: Source,
        cache: FileCache,
        *,
        fetcher: Fetcher | None = None,
        headers: dict[str, str] | None = None,
        rate_limiter: RateLimiter | None = None,
        robots: RobotsChecker | None = None,
        max_retries: int = 3,
        backoff_base_s: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.source = source
        self.cache = cache
        self._fetch = fetcher or make_requests_fetcher(headers)
        self._rate = rate_limiter or RateLimiter(source.min_interval_s)
        self._robots = robots
        self._max_retries = max_retries
        self._backoff_base_s = backoff_base_s
        self._sleep = sleep

    def _check_policy(self, url: str) -> None:
        if not self.source.enabled:
            raise FetchBlockedError(f"Source {self.source.id!r} is disabled in the registry.")
        if self.source.kind == "scrape":
            if not self.source.scraping_allowed:
                raise FetchBlockedError(
                    f"Source {self.source.id!r}: scraping is not allowed (per ToS/robots)."
                )
            if self._robots is not None and not self._robots.allowed(url):
                raise FetchBlockedError(f"robots.txt disallows fetching {url!r}.")

    def get(self, url: str, params: dict[str, str] | None = None) -> bytes:
        """Return artifact bytes, using cache when present, else fetch under policy."""
        key = cache_key(url, params)
        cached = self.cache.get(self.source.id, key)
        if cached is not None:
            log.debug("cache hit %s %s", self.source.id, key)
            return cached

        self._check_policy(url)
        data = self._fetch_with_retries(url, params)

        provenance = Provenance.create(
            source_id=self.source.id,
            url=url,
            data=data,
            license=self.source.license,
            attribution_required=self.source.attribution_required,
            params=params,
        )
        self.cache.put(self.source.id, key, data, provenance)
        return data

    def _fetch_with_retries(self, url: str, params: dict[str, str] | None) -> bytes:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            self._rate.wait()
            try:
                status, data = self._fetch(url, params)
            except Exception as exc:  # noqa: BLE001 - network errors are retryable
                last_exc = exc
                status, data = -1, b""
            if status == 200:
                return data
            if status in _RETRYABLE_STATUS or status == -1:
                if attempt < self._max_retries:
                    backoff = self._backoff_base_s * (2**attempt)
                    log.warning(
                        "fetch %s status=%s attempt=%d; retrying in %.1fs",
                        url, status, attempt, backoff,
                    )
                    self._sleep(backoff)
                    continue
            else:
                raise FetchError(f"Non-retryable status {status} for {url!r}.")
        raise FetchError(f"Failed to fetch {url!r} after {self._max_retries} retries: {last_exc}")
