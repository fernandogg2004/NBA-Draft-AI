"""Evaluation (Phase 7). The real use is ORDERING prospects, so ranking metrics lead;
calibration matters for the classification targets. Error-based metrics are secondary.
"""

from nba_draft.evaluation.calibration import calibration_table
from nba_draft.evaluation.comparison import compare_models, make_spec
from nba_draft.evaluation.error_analysis import largest_errors, residual_segments
from nba_draft.evaluation.metrics import (
    brier_score,
    expected_calibration_error,
    kendall_tau,
    rmse,
    spearman_corr,
    top_k_hit_rate,
)

__all__ = [
    "brier_score",
    "calibration_table",
    "compare_models",
    "expected_calibration_error",
    "kendall_tau",
    "largest_errors",
    "make_spec",
    "residual_segments",
    "rmse",
    "spearman_corr",
    "top_k_hit_rate",
]
