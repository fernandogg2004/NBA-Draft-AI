"""Temporal validation primitives (Phase 5).

The single most important anti-leakage machinery in the project. Built and tested BEFORE
any modeling so that no experiment can accidentally train on the future.
"""

from nba_draft.validation.pipeline import FoldPreprocessor
from nba_draft.validation.runner import (
    DataSplit,
    EvaluationReport,
    default_metrics,
    make_data_split,
    walk_forward_evaluate,
    walk_forward_hurdle_evaluate,
    walk_forward_predictions,
)
from nba_draft.validation.temporal import (
    LeakageError,
    TemporalFold,
    holdout_split,
    walk_forward_folds,
)

__all__ = [
    "DataSplit",
    "EvaluationReport",
    "FoldPreprocessor",
    "LeakageError",
    "TemporalFold",
    "default_metrics",
    "holdout_split",
    "make_data_split",
    "walk_forward_evaluate",
    "walk_forward_folds",
    "walk_forward_hurdle_evaluate",
    "walk_forward_predictions",
]
