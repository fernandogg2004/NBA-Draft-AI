"""Live NBA Stats pull (Phase 1) — RUN LOCALLY ONLY.

stats.nba.com bans datacenter/cloud IPs; run this on a personal machine. Pulls draft history,
Combine measurements, and player season stats (Base + Advanced) through the rate-limited, cached
ingester, writing raw JSON + provenance to data/raw/nba_stats/. Cached requests are skipped, so
re-running is cheap and safe.

Requires the ingest extra:  pip install -e ".[ingest]"

Examples:
  python scripts/run_ingest.py                                  # sensible recent defaults
  python scripts/run_ingest.py --draft-years 2022,2023,2024 \
      --combine-seasons 2022-23,2023-24 --player-seasons 2022-23,2023-24
"""

from __future__ import annotations

import argparse

from nba_draft.ingestion.nba_stats import NbaStatsIngester
from nba_draft.utils.logging import get_logger

log = get_logger("run_ingest")


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Live NBA Stats pull (local only).")
    parser.add_argument("--cache", default="data/raw", help="cache root (default data/raw)")
    parser.add_argument("--draft-years", default="2022,2023,2024", type=_csv)
    parser.add_argument("--combine-seasons", default="2022-23,2023-24", type=_csv)
    parser.add_argument("--player-seasons", default="2022-23,2023-24", type=_csv)
    args = parser.parse_args()

    ing = NbaStatsIngester(args.cache)
    log.info(
        "Pulling NBA Stats (rate-limited %.0f req/min, cached). This runs locally only.",
        ing.source.rate_limit_per_min,
    )

    for year in args.draft_years:
        ing.draft_history(int(year))
        log.info("draft history %s ✓", year)

    for season in args.combine_seasons:
        ing.draft_combine_stats(season)
        log.info("combine %s ✓", season)

    for season in args.player_seasons:
        ing.player_season_stats(season, "Base")
        ing.player_season_stats(season, "Advanced")
        log.info("player stats %s (Base+Advanced) ✓", season)

    log.info("Done. Raw JSON + provenance under %s/nba_stats/.", args.cache)
    log.info("Next: build the master dataset, then point the service at it (see usage guide).")


if __name__ == "__main__":
    main()
