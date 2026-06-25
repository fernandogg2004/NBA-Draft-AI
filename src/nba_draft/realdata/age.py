"""Age-at-draft features from nba_api CommonPlayerInfo birthdates.

Age is one of the strongest pre-draft predictors (a 19-year-old producing like a 22-year-old is
far more promising). We compute age on draft day (~late June of the draft year) from the player's
birthdate. One API call per player (cached); birthdate-missing players get null age (imputed).
"""

from __future__ import annotations

from datetime import date
from typing import Any

import polars as pl

from nba_draft.ingestion.parse import parse_player_info
from nba_draft.utils.logging import get_logger

log = get_logger("realdata.age")

AGE_FEATURE_COLUMNS: list[str] = ["age_at_draft"]
_DRAFT_MONTH, _DRAFT_DAY = 6, 26   # NBA draft is held in late June


def age_at_draft(birthdate_iso: str | None, draft_year: int) -> float | None:
    """Years between birthdate and the draft day of `draft_year` (None if birthdate missing)."""
    if not birthdate_iso:
        return None
    try:
        birth = date.fromisoformat(birthdate_iso)
    except ValueError:
        return None
    draft_day = date(draft_year, _DRAFT_MONTH, _DRAFT_DAY)
    return round((draft_day - birth).days / 365.25, 2)


def pull_player_ages(ingester: Any, draft_history: pl.DataFrame) -> pl.DataFrame:
    """Fetch birthdates for drafted players and compute age-at-draft.

    Args:
        ingester: an NbaStatsIngester (its `player_info` is cached + rate-limited).
        draft_history: parsed draft history (player_id, draft_year).

    Returns:
        [player_id, age_at_draft] (age null where birthdate is unavailable).
    """
    draft_year_by_pid = {
        int(r["player_id"]): int(r["draft_year"])
        for r in draft_history.iter_rows(named=True)
    }
    rows: list[dict[str, object]] = []
    n = len(draft_year_by_pid)
    n_failed = 0
    for i, (pid, dy) in enumerate(draft_year_by_pid.items(), 1):
        # Resilient per player: a transient API error must not abort the whole batch. Successful
        # calls are cached, so re-running resumes and only retries the ones that failed.
        try:
            info = parse_player_info(ingester.player_info(pid))
            birthdate = info["birthdate"]
            age = age_at_draft(birthdate if isinstance(birthdate, str) else None, dy)
        except Exception as exc:  # noqa: BLE001 - network/parse hiccup -> null age, keep going
            log.warning("age fetch failed for player %s: %s", pid, exc)
            age, n_failed = None, n_failed + 1
        rows.append({"player_id": pid, "age_at_draft": age})
        if i % 50 == 0:
            log.info("ages: %d/%d (%d failed)", i, n, n_failed)
    if n_failed:
        log.warning("age pull: %d/%d players failed (null age; re-run to retry).", n_failed, n)
    return pl.DataFrame(rows, schema_overrides={"player_id": pl.Int64, "age_at_draft": pl.Float64})
