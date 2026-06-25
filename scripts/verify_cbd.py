"""Verify the CollegeBasketballData.com API key + endpoints, and reveal the response schema.

Run AFTER getting a free key (https://collegebasketballdata.com/key) and setting it:
    setx CBD_API_KEY "your-key"        # Windows (new shell after), or set for the session
    python scripts/verify_cbd.py --season 2024

It makes one small authenticated call per endpoint and prints the JSON field names so the parser
can be written against the real schema. Adjust the PATH_* constants in
src/nba_draft/ingestion/college_bb_data.py if any endpoint path differs.
"""

from __future__ import annotations

import argparse
import json

from nba_draft.ingestion.college_bb_data import (
    CollegeBasketballDataIngester,
    MissingApiKeyError,
)
from nba_draft.utils.logging import get_logger

log = get_logger("verify_cbd")


def _show(label: str, raw: str) -> None:
    data = json.loads(raw)
    if isinstance(data, list) and data:
        log.info("%s: %d rows; fields = %s", label, len(data), sorted(data[0].keys()))
        log.info("  sample row: %s", json.dumps(data[0])[:400])
    elif isinstance(data, list):
        log.info("%s: 0 rows returned", label)
    else:
        log.info("%s: non-list response: %s", label, json.dumps(data)[:300])


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify CBD API key + endpoints.")
    parser.add_argument("--season", type=int, default=2024)
    parser.add_argument("--team", default="Duke")
    parser.add_argument("--cache", default="data/raw")
    args = parser.parse_args()

    try:
        ing = CollegeBasketballDataIngester(args.cache)
    except MissingApiKeyError as exc:
        log.error("%s", exc)
        raise SystemExit(1) from exc

    log.info("Key found. Calling endpoints for season %s ...", args.season)
    try:
        _show("teams", ing.teams(season=args.season))
        _show("player_season_stats", ing.player_season_stats(args.season, team=args.team))
        _show("team_roster", ing.team_roster(args.team, season=args.season))
    except Exception as exc:  # noqa: BLE001 - surface auth/path errors plainly
        log.error("Call failed: %s", exc)
        log.error("401/403 -> re-check the key; 404 -> the path may differ (see /docs).")
        raise SystemExit(2) from exc

    log.info("Verification OK. Share this output and I'll write the parser against these fields.")


if __name__ == "__main__":
    main()
