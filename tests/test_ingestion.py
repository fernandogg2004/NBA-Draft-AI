"""Offline tests for the Phase 1 ingestion core. No test makes a network call: the fetcher,
clock, and sleep are all injected.
"""

from __future__ import annotations

import json

import pytest

from nba_draft.ingestion import (
    FetchBlockedError,
    FileCache,
    PoliteClient,
    Provenance,
    RateLimiter,
    RobotsChecker,
    Source,
    load_sources,
    sha256_bytes,
)
from nba_draft.ingestion.cache import cache_key
from nba_draft.ingestion.http import FetchError


# --------------------------------------------------------------------------- registry
def test_real_registry_loads_and_is_consistent():
    sources = load_sources()
    assert "nba_stats" in sources
    # disallowed scrapers must be disabled in the shipped registry
    for sid in ("basketball_reference", "bart_torvik", "kenpom"):
        assert sources[sid].enabled is False
        assert sources[sid].scraping_allowed is False
    # enabled network source must declare a positive rate limit
    assert sources["nba_stats"].enabled is True
    assert sources["nba_stats"].rate_limit_per_min > 0


def test_source_requires_declared_license():
    with pytest.raises(ValueError):
        Source(id="x", name="x", kind="api", base_url="u", license="UNKNOWN")


def test_enabled_network_source_needs_rate_limit():
    with pytest.raises(ValueError):
        Source(id="x", name="x", kind="scrape", base_url="u", license="MIT", enabled=True)


def test_min_interval_from_rate_limit():
    s = Source(id="x", name="x", kind="api", base_url="u", license="MIT", rate_limit_per_min=20)
    assert s.min_interval_s == pytest.approx(3.0)


# --------------------------------------------------------------------------- cache
def test_cache_roundtrip_with_provenance(tmp_path):
    cache = FileCache(tmp_path)
    data = b"hello"
    prov = Provenance.create(
        source_id="s", url="http://x", data=data, license="MIT", attribution_required=True
    )
    key = cache_key("http://x", {"a": "1"})
    assert cache.get("s", key) is None
    cache.put("s", key, data, prov)
    assert cache.get("s", key) == data
    # provenance sidecar exists and records the right hash
    sidecar = tmp_path / "s" / f"{key}.prov.json"
    assert json.loads(sidecar.read_text())["sha256"] == sha256_bytes(data)


def test_cache_key_is_param_order_independent():
    assert cache_key("u", {"a": "1", "b": "2"}) == cache_key("u", {"b": "2", "a": "1"})


# --------------------------------------------------------------------------- rate limiter
def test_rate_limiter_spaces_calls():
    now = [0.0]
    slept: list[float] = []

    def fake_sleep(s: float) -> None:
        slept.append(s)
        now[0] += s

    rl = RateLimiter(2.0, clock=lambda: now[0], sleep=fake_sleep)
    rl.wait()              # first call, no wait
    rl.wait()              # immediately after -> must sleep ~2s
    assert slept == [pytest.approx(2.0)]


def test_rate_limiter_zero_interval_never_sleeps():
    slept: list[float] = []
    rl = RateLimiter(0.0, clock=lambda: 0.0, sleep=lambda s: slept.append(s))
    rl.wait()
    rl.wait()
    assert slept == []


# --------------------------------------------------------------------------- robots
_ROBOTS = "User-agent: *\nDisallow: /private/\n"


def test_robots_allows_and_blocks():
    checker = RobotsChecker(lambda url: _ROBOTS, user_agent="test-bot")
    assert checker.allowed("https://site.com/public/page") is True
    assert checker.allowed("https://site.com/private/secret") is False


def test_robots_fails_closed_on_error():
    def boom(url: str) -> str:
        raise RuntimeError("no robots")

    checker = RobotsChecker(boom)
    assert checker.allowed("https://site.com/anything") is False


# --------------------------------------------------------------------------- PoliteClient
def _scrape_source(**kw) -> Source:
    base = dict(
        id="scr", name="scr", kind="scrape", base_url="https://site.com", license="MIT",
        scraping_allowed=True, enabled=True, rate_limit_per_min=60,
    )
    base.update(kw)
    return Source(**base)


def _no_wait_rl() -> RateLimiter:
    return RateLimiter(0.0)


def test_disabled_source_is_blocked(tmp_path):
    client = PoliteClient(_scrape_source(enabled=False), FileCache(tmp_path),
                          fetcher=lambda u, p: (200, b"x"), rate_limiter=_no_wait_rl())
    with pytest.raises(FetchBlockedError):
        client.get("https://site.com/a")


def test_scrape_not_allowed_is_blocked(tmp_path):
    client = PoliteClient(_scrape_source(scraping_allowed=False), FileCache(tmp_path),
                          fetcher=lambda u, p: (200, b"x"), rate_limiter=_no_wait_rl())
    with pytest.raises(FetchBlockedError):
        client.get("https://site.com/a")


def test_robots_block_prevents_fetch(tmp_path):
    robots = RobotsChecker(lambda url: _ROBOTS)
    calls: list[str] = []

    def fetcher(u, p):
        calls.append(u)
        return 200, b"x"

    client = PoliteClient(_scrape_source(), FileCache(tmp_path), fetcher=fetcher,
                          rate_limiter=_no_wait_rl(), robots=robots)
    with pytest.raises(FetchBlockedError):
        client.get("https://site.com/private/x")
    assert calls == []  # never fetched


def test_successful_fetch_caches_and_records_provenance(tmp_path):
    fetches: list[str] = []

    def fetcher(u, p):
        fetches.append(u)
        return 200, b"payload"

    client = PoliteClient(_scrape_source(), FileCache(tmp_path), fetcher=fetcher,
                          rate_limiter=_no_wait_rl())
    assert client.get("https://site.com/a") == b"payload"
    # second call served from cache (no new fetch)
    assert client.get("https://site.com/a") == b"payload"
    assert len(fetches) == 1


def test_retries_then_succeeds(tmp_path):
    statuses = [503, 503, 200]
    slept: list[float] = []

    def fetcher(u, p):
        return statuses.pop(0), b"ok"

    client = PoliteClient(_scrape_source(), FileCache(tmp_path), fetcher=fetcher,
                          rate_limiter=_no_wait_rl(), backoff_base_s=0.01,
                          sleep=lambda s: slept.append(s))
    assert client.get("https://site.com/a") == b"ok"
    assert len(slept) == 2  # two backoffs before success


def test_non_retryable_status_raises(tmp_path):
    client = PoliteClient(_scrape_source(), FileCache(tmp_path),
                          fetcher=lambda u, p: (404, b""), rate_limiter=_no_wait_rl())
    with pytest.raises(FetchError):
        client.get("https://site.com/missing")


def test_exhausted_retries_raise(tmp_path):
    client = PoliteClient(_scrape_source(), FileCache(tmp_path),
                          fetcher=lambda u, p: (503, b""), rate_limiter=_no_wait_rl(),
                          max_retries=2, backoff_base_s=0.0, sleep=lambda s: None)
    with pytest.raises(FetchError):
        client.get("https://site.com/flaky")
