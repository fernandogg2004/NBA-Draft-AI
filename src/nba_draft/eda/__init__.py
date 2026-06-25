"""Phase 3 — exploratory data analysis.

Reusable, mostly-pure EDA primitives (summaries) plus an optional plotting layer and a
markdown report builder. EDA must be run on the DEVELOPMENT set only (holdout excluded) so we
never tune intuition on the test classes — `scripts/run_eda.py` enforces that.
"""

from nba_draft.eda.summaries import (
    assign_bands,
    band_counts,
    binned_relationship,
    feature_target_spearman,
    grouped_target_summary,
    missingness_by_group,
    numeric_summary,
    spearman_correlation_matrix,
)

__all__ = [
    "assign_bands",
    "band_counts",
    "binned_relationship",
    "feature_target_spearman",
    "grouped_target_summary",
    "missingness_by_group",
    "numeric_summary",
    "spearman_correlation_matrix",
]
