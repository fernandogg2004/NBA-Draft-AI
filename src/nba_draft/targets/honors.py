"""All-Star / All-NBA honors source for the top outcome tiers.

Without honors, `outcome_tier` falls back to BPM bands for the All-Star/Superstar tiers, which is
conservative. This module loads a small honors table — one row per player with their selection
counts — so tiering can promote genuine All-Stars/All-NBA players the box score under-rates.

The honors data itself is provided by the user (nba_api has no clean honors endpoint): a CSV with
columns ``player_id, all_star_count, all_nba_count``. The loader is pure and offline-testable.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

Honors = dict[int, tuple[int, int]]


def honors_from_frame(df: pl.DataFrame) -> Honors:
    """Build {player_id: (all_star_count, all_nba_count)} from a frame with those columns."""
    required = {"player_id", "all_star_count", "all_nba_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Honors frame missing columns: {sorted(missing)}")
    out: Honors = {}
    for row in df.iter_rows(named=True):
        out[int(row["player_id"])] = (
            int(row["all_star_count"] or 0),
            int(row["all_nba_count"] or 0),
        )
    return out


def load_honors(path: str | Path) -> Honors:
    """Load honors from a CSV with columns player_id, all_star_count, all_nba_count."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Honors file not found: {p}")
    return honors_from_frame(pl.read_csv(p))
