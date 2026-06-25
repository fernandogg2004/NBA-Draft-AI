"""Name and league normalization for entity resolution.

Names arrive in many shapes across sources ("Luka Dončić", "Doncic, Luka", "Luka Doncic Jr.").
We build a stable *match key* (accent-folded, suffix-stripped, order-normalized) used for linkage,
while preserving the original display name elsewhere.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import yaml

from nba_draft.config import REPO_ROOT

_NORM_PATH = REPO_ROOT / "config" / "normalization.yaml"
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")


@lru_cache(maxsize=1)
def _norm_tables(path: str | None = None) -> tuple[frozenset[str], dict[str, str]]:
    cfg_path = Path(path) if path else _NORM_PATH
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    suffixes = frozenset(s.lower() for s in raw.get("name_suffixes", []))
    aliases = {str(k).lower(): str(v) for k, v in (raw.get("league_aliases") or {}).items()}
    return suffixes, aliases


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize_name(name: str) -> str:
    """Lowercase, fold accents, drop punctuation, collapse whitespace; keep word order.

    "Last, First" is reordered to "First Last".
    """
    if name is None:
        raise ValueError("name must not be None.")
    text = _strip_accents(name).lower().strip()
    if "," in text:
        last, _, first = text.partition(",")
        text = f"{first.strip()} {last.strip()}"
    text = _NON_ALNUM.sub(" ", text)
    return _WS.sub(" ", text).strip()


def name_match_key(name: str) -> str:
    """Normalized name with trailing generational suffixes removed, for linkage/blocking."""
    suffixes, _ = _norm_tables()
    tokens = [t for t in normalize_name(name).split(" ") if t]
    while tokens and tokens[-1] in suffixes:
        tokens.pop()
    return " ".join(tokens)


def normalize_league(league: str) -> str | None:
    """Map a raw league string to a canonical league id, or ``None`` if unrecognized."""
    if league is None:
        return None
    _, aliases = _norm_tables()
    key = _WS.sub(" ", _strip_accents(league).lower().strip())
    return aliases.get(key)
