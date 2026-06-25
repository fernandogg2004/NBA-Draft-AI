"""Build target labels from real NBA outcomes (nba_api) using the Phase 0 definitions.

Turns parsed per-season production (with eBPM/VORP) + draft history into `PlayerOutcome` records,
then applies the hurdle/tier label functions. Honesty: All-Star / All-NBA honors are not pulled
from nba_api here, so they default to 0 — tiering then falls back to the BPM bands until an honors
source is wired (the top-tier labels are therefore conservative).
"""

from __future__ import annotations

import polars as pl

from nba_draft.targets.definitions import (
    OutcomeTier,
    PlayerOutcome,
    SeasonStat,
    TargetConfig,
    cumulative_value,
    load_target_config,
    outcome_tier,
    peak_impact,
    reached_role,
)


def season_str_to_year(season: str | int) -> int:
    """'2023-24' -> 2023; passes ints through."""
    if isinstance(season, int):
        return season
    return int(str(season).split("-")[0])


def build_player_outcomes(
    player_seasons: pl.DataFrame,
    draft_history: pl.DataFrame,
    *,
    bpm_col: str = "ebpm",
) -> dict[int, PlayerOutcome]:
    """Assemble {player_id: PlayerOutcome} from per-season production + draft slots.

    Args:
        player_seasons: rows with player_id, season, minutes, `bpm_col`, vorp.
        draft_history: rows with player_id, draft_year.
        bpm_col: which column holds the (estimated) BPM.
    """
    seasons_by_player: dict[int, list[SeasonStat]] = {}
    for row in player_seasons.iter_rows(named=True):
        pid = int(row["player_id"])
        seasons_by_player.setdefault(pid, []).append(
            SeasonStat(
                season_year=season_str_to_year(row["season"]),
                minutes=float(row["minutes"] or 0.0),
                bpm=float(row[bpm_col] if row[bpm_col] is not None else 0.0),
                vorp=float(row["vorp"] if row["vorp"] is not None else 0.0),
            )
        )

    outcomes: dict[int, PlayerOutcome] = {}
    for row in draft_history.iter_rows(named=True):
        pid = int(row["player_id"])
        seasons = tuple(sorted(seasons_by_player.get(pid, []), key=lambda s: s.season_year))
        debut = seasons[0].season_year if seasons else None
        outcomes[pid] = PlayerOutcome(
            draft_year=int(row["draft_year"]),
            debut_year=debut,
            seasons=seasons,
            all_star_count=0,   # honors not sourced from nba_api here
            all_nba_count=0,
        )
    return outcomes


def build_labels_frame(
    outcomes: dict[int, PlayerOutcome], cfg: TargetConfig | None = None
) -> pl.DataFrame:
    """Compute the hurdle/tier labels for each prospect -> a modeling-ready frame."""
    cfg = cfg or load_target_config()
    rows: list[dict[str, object]] = []
    for pid, out in outcomes.items():
        peak = peak_impact(out, cfg)
        tier: OutcomeTier = outcome_tier(out, cfg)
        rows.append(
            {
                "player_id": pid,
                "draft_year": out.draft_year,
                "reached": reached_role(out, cfg),
                "peak_impact": peak,
                "cumulative_value": cumulative_value(out, cfg),
                "outcome_tier": tier.value,
            }
        )
    return pl.DataFrame(rows)
