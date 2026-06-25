"""Entity resolution: assign one canonical ``player_id`` to a prospect across all sources.

Strategy (deliberately conservative; prospects are few, so precision matters more than recall):
  * Strong blocking key = (name_match_key, draft_year). Same key + same draft year => same player.
  * Fuzzy fallback within a draft year: near-identical names (difflib ratio >= threshold) are
    merged only when birth dates are compatible (equal, or at least one missing).
The canonical id is a deterministic hash of the cluster's representative key + draft year, so
re-running on the same inputs yields identical ids (reproducibility).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from difflib import SequenceMatcher

import polars as pl

from nba_draft.cleaning.normalize import name_match_key

DEFAULT_FUZZY_THRESHOLD = 0.88


@dataclass(frozen=True)
class _Rec:
    """Internal flattened record for one source row during resolution."""

    frame: str
    row: int
    key: str
    year: int
    bdate: str | None
    name: str


@dataclass(frozen=True)
class ResolutionResult:
    """Output of entity resolution."""

    frames: dict[str, pl.DataFrame]   # each input frame + a `player_id` column
    identity: pl.DataFrame            # one row per canonical prospect
    n_entities: int


class _UnionFind:
    def __init__(self, n: int) -> None:
        self._parent = list(range(n))

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[max(ra, rb)] = min(ra, rb)


def _birthdates_compatible(a: str | None, b: str | None) -> bool:
    return a is None or b is None or a == b


def _player_id(rep_key: str, year: int) -> str:
    digest = hashlib.sha1(f"{rep_key}|{year}".encode()).hexdigest()
    return f"p_{digest[:12]}"


def resolve_entities(
    frames: dict[str, pl.DataFrame],
    *,
    name_col: str = "full_name",
    draft_year_col: str = "draft_year",
    birth_date_col: str | None = "birth_date",
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
) -> ResolutionResult:
    """Resolve player identities across `frames` and stamp a canonical ``player_id`` on each.

    Args:
        frames: Mapping of source name -> DataFrame (must contain `name_col`, `draft_year_col`).
        name_col: Column holding the display name.
        draft_year_col: Column holding the draft year (the blocking dimension).
        birth_date_col: Optional birth-date column (string) used to guard fuzzy merges.
        fuzzy_threshold: Minimum difflib ratio for a fuzzy name merge.

    Returns:
        A :class:`ResolutionResult`.
    """
    # 1) Flatten all rows into typed records.
    records: list[_Rec] = []
    for fname, df in frames.items():
        if name_col not in df.columns or draft_year_col not in df.columns:
            raise ValueError(f"Frame {fname!r} missing required columns.")
        names = df[name_col].to_list()
        years = df[draft_year_col].to_list()
        bdates = (
            df[birth_date_col].to_list()
            if birth_date_col and birth_date_col in df.columns
            else [None] * df.height
        )
        for i, (nm, yr, bd) in enumerate(zip(names, years, bdates, strict=True)):
            records.append(
                _Rec(
                    frame=fname,
                    row=i,
                    key=name_match_key(str(nm)),
                    year=int(yr),
                    bdate=None if bd is None else str(bd),
                    name=str(nm),
                )
            )

    uf = _UnionFind(len(records))

    # 2) Strong blocking: identical (key, year).
    blocks: dict[tuple[str, int], list[int]] = {}
    for idx, rec in enumerate(records):
        blocks.setdefault((rec.key, rec.year), []).append(idx)
    for members in blocks.values():
        for j in members[1:]:
            uf.union(members[0], j)

    # 3) Fuzzy fallback within the same draft year.
    by_year: dict[int, list[int]] = {}
    for idx, rec in enumerate(records):
        by_year.setdefault(rec.year, []).append(idx)
    for members in by_year.values():
        for a_pos in range(len(members)):
            for b_pos in range(a_pos + 1, len(members)):
                ia, ib = members[a_pos], members[b_pos]
                ra, rb = records[ia], records[ib]
                if uf.find(ia) == uf.find(ib):
                    continue
                if not _birthdates_compatible(ra.bdate, rb.bdate):
                    continue
                if SequenceMatcher(None, ra.key, rb.key).ratio() >= fuzzy_threshold:
                    uf.union(ia, ib)

    # 4) Assign canonical ids using each cluster's representative (min) key.
    cluster_members: dict[int, list[int]] = {}
    for idx in range(len(records)):
        cluster_members.setdefault(uf.find(idx), []).append(idx)
    record_pid: dict[int, str] = {}
    for members in cluster_members.values():
        rep_key = min(records[m].key for m in members)
        pid = _player_id(rep_key, records[members[0]].year)
        for m in members:
            record_pid[m] = pid

    # 5) Attach player_id to each frame (original row order).
    out_frames: dict[str, pl.DataFrame] = {}
    per_frame_ids: dict[str, list[str | None]] = {f: [None] * df.height for f, df in frames.items()}
    for idx, rec in enumerate(records):
        per_frame_ids[rec.frame][rec.row] = record_pid[idx]
    for fname, df in frames.items():
        out_frames[fname] = df.with_columns(
            pl.Series("player_id", per_frame_ids[fname], dtype=pl.Utf8)
        )

    # 6) Identity table: one row per canonical player (first non-null display name wins).
    id_rows: dict[str, dict[str, object]] = {}
    for idx, rec in enumerate(records):
        pid = record_pid[idx]
        if pid not in id_rows:
            id_rows[pid] = {
                "player_id": pid,
                "full_name": rec.name,
                "draft_year": rec.year,
                "birth_date": rec.bdate,
            }
        elif id_rows[pid]["birth_date"] is None and rec.bdate is not None:
            id_rows[pid]["birth_date"] = rec.bdate
    identity = pl.DataFrame(list(id_rows.values())) if id_rows else pl.DataFrame()

    return ResolutionResult(
        frames=out_frames,
        identity=identity,
        n_entities=len(cluster_members),
    )
