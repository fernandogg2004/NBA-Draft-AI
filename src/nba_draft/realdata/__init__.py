"""Real-data pipeline path (nba_api only).

Composes the live pull + adapters into a runnable, honest pipeline. With only nba_api the
*real* pre-draft features are Combine measurements + draft slot (college box stats need a
source that isn't wired yet), and the *real* labels come from NBA outcomes (eBPM/VORP →
hurdle/tier labels). The join logic is pure and offline-testable; the pull step is separate.
"""

from nba_draft.realdata.age import AGE_FEATURE_COLUMNS, age_at_draft, pull_player_ages
from nba_draft.realdata.build import (
    build_real_modeling_table,
    evaluate_real_models,
    pull_real_frames,
    run_real_pipeline,
)
from nba_draft.realdata.college import COLLEGE_FEATURE_COLUMNS, link_college_features

__all__ = [
    "AGE_FEATURE_COLUMNS",
    "COLLEGE_FEATURE_COLUMNS",
    "age_at_draft",
    "build_real_modeling_table",
    "evaluate_real_models",
    "link_college_features",
    "pull_player_ages",
    "pull_real_frames",
    "run_real_pipeline",
]
