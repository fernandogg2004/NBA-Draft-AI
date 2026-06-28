"""Acquire All-Star / All-NBA honors from nba_api PlayerAwards (one call per player).

Honors enrich the TOP outcome tiers: without them, `outcome_tier` falls back to BPM bands for the
All-Star/Superstar tiers, under-rating players the box score misses. This builds the
``{player_id: (all_star_count, all_nba_count)}`` mapping that ``build_real_modeling_table`` consumes
(``honors=``). Pure acquisition layer; the loader/usage live in ``nba_draft.targets.honors``.

Runs locally/online only (needs a residential IP). Successful calls are cached, so re-running
resumes and only retries failures.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from nba_draft.ingestion.parse import parse_player_awards
from nba_draft.targets.honors import Honors
from nba_draft.utils.logging import get_logger

log = get_logger("realdata.honors")


def pull_player_honors(ingester: Any, draft_history: pl.DataFrame) -> Honors:
    """Fetch All-Star / All-NBA selection counts for every drafted player.

    Args:
        ingester: an NbaStatsIngester (its ``player_awards`` is cached + rate-limited).
        draft_history: parsed draft history (needs a ``player_id`` column).

    Returns:
        ``{player_id: (all_star_count, all_nba_count)}``; players whose awards call fails get
        ``(0, 0)`` (treated as no honors) so the batch never aborts.
    """
    pids = [int(p) for p in draft_history["player_id"].unique().to_list()]
    out: Honors = {}
    n, n_failed = len(pids), 0
    for i, pid in enumerate(pids, 1):
        try:
            out[pid] = parse_player_awards(ingester.player_awards(pid))
        except Exception as exc:  # noqa: BLE001 - network/parse hiccup -> no honors, keep going
            log.warning("honors fetch failed for player %s: %s", pid, exc)
            out[pid] = (0, 0)
            n_failed += 1
        if i % 50 == 0:
            log.info("honors: %d/%d (%d failed)", i, n, n_failed)
    if n_failed:
        log.warning("honors pull: %d/%d players failed (no honors; re-run to retry).", n_failed, n)
    return out
