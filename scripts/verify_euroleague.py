"""Verify the EuroLeague source + reveal the live player-season schema (run locally/online).

The euroleague_api column names vary by endpoint/version, so this prints the fields of one season
and the canonical columns our parser extracts. If a stat is missing, extend the candidate lists in
``ingestion/parse.py::_EL_CANDIDATES``.

Requires the ingest extra:  pip install -e ".[ingest]"

    python scripts/verify_euroleague.py --season 2017
"""

from __future__ import annotations

import argparse
import json

from nba_draft.ingestion.euroleague import EuroLeagueIngester
from nba_draft.ingestion.parse import parse_euroleague_player_season
from nba_draft.utils.logging import get_logger

log = get_logger("verify_euroleague")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify EuroLeague player-season schema.")
    parser.add_argument("--season", type=int, default=2017)
    parser.add_argument("--cache", default="data/raw")
    args = parser.parse_args()

    ing = EuroLeagueIngester(args.cache)
    log.info("Pulling EuroLeague player stats for season %s ...", args.season)
    try:
        raw = ing.player_season_stats(args.season)
    except Exception as exc:  # noqa: BLE001 - surface install/network errors plainly
        log.error("Call failed: %s", exc)
        log.error("Ensure `pip install -e \".[ingest]\"` (euroleague-api) and network access.")
        raise SystemExit(1) from exc

    records = json.loads(raw)
    if records:
        log.info("rows=%d; raw fields = %s", len(records), sorted(records[0].keys()))
    parsed = parse_euroleague_player_season(raw)
    log.info("parsed %d rows; canonical cols = %s", parsed.height, parsed.columns)
    if parsed.height:
        log.info("sample parsed row: %s", parsed.head(1).to_dicts()[0])
    log.info("If a stat is null/missing, extend _EL_CANDIDATES in ingestion/parse.py.")


if __name__ == "__main__":
    main()
