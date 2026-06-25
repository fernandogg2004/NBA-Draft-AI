"""Phase 2 — cleaning & integration.

Turns raw, heterogeneous per-source frames into a versioned, reproducible master dataset:
  * normalization (names, leagues) — `normalize`
  * entity resolution (one canonical player_id across sources) — `entity_resolution`
  * schema + missingness ("not measured" stays null, never zero) — `schema`, `missing`
  * LEAKAGE-SAFE imputation (fit on train, applied in folds; flags + uncertainty) — `imputation`
  * versioned master builder + manifest — `master`
"""

from nba_draft.cleaning.entity_resolution import ResolutionResult, resolve_entities
from nba_draft.cleaning.imputation import LeakageSafeImputer
from nba_draft.cleaning.normalize import (
    name_match_key,
    normalize_league,
    normalize_name,
)
from nba_draft.cleaning.schema import (
    ADVANCED_COLUMNS,
    COMBINE_COLUMNS,
    add_missing_flags,
)

__all__ = [
    "ADVANCED_COLUMNS",
    "COMBINE_COLUMNS",
    "LeakageSafeImputer",
    "ResolutionResult",
    "add_missing_flags",
    "name_match_key",
    "normalize_league",
    "normalize_name",
    "resolve_entities",
]
