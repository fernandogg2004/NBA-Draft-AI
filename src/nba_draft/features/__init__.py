"""Phase 4 — feature engineering.

Features fall into two classes, and the split is the leakage defense:
  * STATELESS (`transforms`): pure functions of a player's own pre-draft stats — efficiency,
    versatility, Combine-derived, year-over-year deltas, conference-jump / role-change flags.
    Leakage-safe by construction (inputs are all pre-draft).
  * LEARNED (`learned`): cross-player baselines — inter-league translation factors and DYNAMIC
    strength-of-schedule z-scores — implemented as fit/transform models fit on TRAIN ONLY.

`assembler.build_feature_matrix` collapses the master tables into one pre-draft row per
prospect; `assembler.assert_pre_draft_safe` guards against post-draft columns leaking in.
"""

from nba_draft.features.assembler import (
    FORBIDDEN_POST_DRAFT_COLUMNS,
    assemble_prospect_features,
    assert_pre_draft_safe,
    build_feature_matrix,
    primary_pre_draft_season,
)
from nba_draft.features.learned import LeagueSeasonContextModel
from nba_draft.features.transforms import (
    add_combine_features,
    add_stateless_features,
    sequence_features,
)

__all__ = [
    "FORBIDDEN_POST_DRAFT_COLUMNS",
    "LeagueSeasonContextModel",
    "add_combine_features",
    "add_stateless_features",
    "assemble_prospect_features",
    "assert_pre_draft_safe",
    "build_feature_matrix",
    "primary_pre_draft_season",
    "sequence_features",
]
