"""Build the versioned, reproducible master dataset from raw per-source frames.

Output = three canonical tables written under ``data/processed/<version>/``:
  * ``identity``        — one row per canonical prospect (from entity resolution)
  * ``prospect_season`` — one row per (player, league, season); basic + advanced, nulls kept
  * ``combine``         — one row per prospect with Combine measurements (nulls kept)
plus ``manifest.json`` recording the dataset version, table fingerprints, sources, and counts.

The master keeps raw values with ``null`` preserved and seeds ``<col>_imputed`` flags from
null-ness. It does NOT impute: imputation is leakage-safe and happens inside the CV folds.
The version is a deterministic hash of table contents + schema version, so identical inputs
reproduce an identical version (DVC-friendly).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from nba_draft.cleaning.entity_resolution import resolve_entities
from nba_draft.cleaning.normalize import normalize_league
from nba_draft.cleaning.schema import (
    ADVANCED_COLUMNS,
    COMBINE_COLUMNS,
    add_missing_flags,
)
from nba_draft.utils.logging import get_logger

log = get_logger("cleaning.master")

SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class MasterDataset:
    version: str
    root: Path
    identity: pl.DataFrame
    prospect_season: pl.DataFrame
    combine: pl.DataFrame
    manifest: dict[str, object]


def _fingerprint(df: pl.DataFrame) -> str:
    """Deterministic content hash: sort columns + rows, serialize to CSV, sha256."""
    if df.is_empty():
        return "empty"
    cols = sorted(df.columns)
    ordered = df.select(cols).sort(cols)
    return hashlib.sha256(ordered.write_csv().encode("utf-8")).hexdigest()


def build_master(
    season_frames: dict[str, pl.DataFrame],
    combine_frames: dict[str, pl.DataFrame] | None = None,
    *,
    output_root: str | Path,
    league_col: str = "league",
) -> MasterDataset:
    """Integrate raw frames into the versioned master dataset and write it to disk.

    Args:
        season_frames: source_id -> prospect-season frame (needs full_name, draft_year, league).
        combine_frames: source_id -> combine frame (needs full_name, draft_year + measurements).
        output_root: directory under which ``<version>/`` is created.
        league_col: name of the raw league column on the season frames.
    """
    combine_frames = combine_frames or {}

    # 1) Entity resolution across ALL frames so combine links to season rows.
    all_frames = {**season_frames, **{f"combine::{k}": v for k, v in combine_frames.items()}}
    res = resolve_entities(all_frames)
    resolved = res.frames

    # 2) Canonical prospect_season: normalize league, align columns, concat, seed flags.
    season_parts: list[pl.DataFrame] = []
    for src in season_frames:
        df = resolved[src]
        if league_col in df.columns:
            league_ids = [normalize_league(str(x)) for x in df[league_col].to_list()]
            df = df.with_columns(pl.Series("league_id", league_ids, dtype=pl.Utf8))
        season_parts.append(df.with_columns(pl.lit(src).alias("source_id")))
    prospect_season = (
        pl.concat(season_parts, how="diagonal_relaxed") if season_parts else pl.DataFrame()
    )
    if not prospect_season.is_empty():
        # Ensure advanced columns exist (as null) so flags are consistent across sources.
        for c in ADVANCED_COLUMNS:
            if c not in prospect_season.columns:
                prospect_season = prospect_season.with_columns(
                    pl.lit(None, dtype=pl.Float64).alias(c)
                )
        prospect_season = add_missing_flags(prospect_season, ADVANCED_COLUMNS)

    # 3) Canonical combine: concat combine frames, one row per player_id, seed flags.
    combine_parts = [
        resolved[f"combine::{k}"].with_columns(pl.lit(k).alias("source_id"))
        for k in combine_frames
    ]
    combine = pl.concat(combine_parts, how="diagonal_relaxed") if combine_parts else pl.DataFrame()
    if not combine.is_empty():
        for c in COMBINE_COLUMNS:
            if c not in combine.columns:
                combine = combine.with_columns(pl.lit(None, dtype=pl.Float64).alias(c))
        combine = combine.unique(subset=["player_id"], keep="first")
        combine = add_missing_flags(combine, COMBINE_COLUMNS)

    identity = res.identity

    # 4) Deterministic version from content fingerprints + schema version.
    fps = {
        "identity": _fingerprint(identity),
        "prospect_season": _fingerprint(prospect_season),
        "combine": _fingerprint(combine),
    }
    version = hashlib.sha256(
        (SCHEMA_VERSION + "|" + "|".join(f"{k}:{v}" for k, v in sorted(fps.items()))).encode()
    ).hexdigest()[:12]

    manifest: dict[str, object] = {
        "version": version,
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "n_entities": res.n_entities,
        "sources": sorted({*season_frames, *combine_frames}),
        "tables": {
            name: {"rows": df.height, "cols": df.width, "sha256": fps[name]}
            for name, df in (
                ("identity", identity),
                ("prospect_season", prospect_season),
                ("combine", combine),
            )
        },
        "imputation": "deferred to temporal CV folds (leakage-safe); master keeps nulls + flags",
    }

    # 5) Write versioned outputs.
    root = Path(output_root) / version
    root.mkdir(parents=True, exist_ok=True)
    if not identity.is_empty():
        identity.write_parquet(root / "identity.parquet")
    if not prospect_season.is_empty():
        prospect_season.write_parquet(root / "prospect_season.parquet")
    if not combine.is_empty():
        combine.write_parquet(root / "combine.parquet")
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), "utf-8")
    log.info("Master dataset v%s written to %s (%d entities).", version, root, res.n_entities)

    return MasterDataset(
        version=version,
        root=root,
        identity=identity,
        prospect_season=prospect_season,
        combine=combine,
        manifest=manifest,
    )
