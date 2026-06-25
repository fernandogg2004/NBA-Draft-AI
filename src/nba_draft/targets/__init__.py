"""Phase 0 — target definitions and the label-construction contract.

This package is the single source of truth for WHAT the system predicts and HOW labels are
built from realized NBA outcomes. It contains no models — only the typed target spec and the
pure label-construction functions, so the same logic is reused by data prep, modeling, and
evaluation, and is unit-tested in isolation.

See instructions.md Phase 0 and config/targets.yaml.
"""

from nba_draft.targets.definitions import (
    OutcomeTier,
    PlayerOutcome,
    SeasonStat,
    TargetConfig,
    cumulative_value,
    is_label_resolved,
    load_target_config,
    outcome_tier,
    peak_impact,
    reached_role,
    unconditional_value,
)
from nba_draft.targets.impact import (
    add_impact_metrics,
    estimated_bpm,
    pie_rank_agreement,
    vorp,
)
from nba_draft.targets.outcomes import build_labels_frame, build_player_outcomes

__all__ = [
    "OutcomeTier",
    "PlayerOutcome",
    "SeasonStat",
    "TargetConfig",
    "add_impact_metrics",
    "build_labels_frame",
    "build_player_outcomes",
    "cumulative_value",
    "estimated_bpm",
    "is_label_resolved",
    "load_target_config",
    "outcome_tier",
    "peak_impact",
    "pie_rank_agreement",
    "reached_role",
    "unconditional_value",
    "vorp",
]
