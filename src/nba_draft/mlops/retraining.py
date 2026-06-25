"""Retraining policy and decision.

Cadence is annual — each new draft class adds a year of labels and a new class to project, and
the college landscape shifts. We also retrain off-cycle if drift is significant. This module
encodes that policy and a simple, auditable decision function.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrainingPolicy:
    """When to retrain.

    Attributes:
        cadence: Human-readable cadence ("annual" — once per new draft class).
        drift_psi_threshold: Max-feature PSI above which we retrain off-cycle.
        min_new_labeled_years: Retrain once this many new resolved label-years are available.
    """

    cadence: str = "annual"
    drift_psi_threshold: float = 0.25
    min_new_labeled_years: int = 1


def should_retrain(
    *,
    last_trained_year: int,
    current_year: int,
    max_feature_psi: float = 0.0,
    policy: RetrainingPolicy | None = None,
) -> tuple[bool, list[str]]:
    """Decide whether to retrain, returning (decision, reasons).

    Args:
        last_trained_year: Draft year the production model was last trained through.
        current_year: The current draft year.
        max_feature_psi: Largest per-feature PSI from the latest drift report.
        policy: The retraining policy (defaults to annual + PSI 0.25).
    """
    policy = policy or RetrainingPolicy()
    reasons: list[str] = []
    if current_year - last_trained_year >= policy.min_new_labeled_years:
        reasons.append(
            f"new draft class(es) since {last_trained_year} (cadence: {policy.cadence})"
        )
    if max_feature_psi > policy.drift_psi_threshold:
        reasons.append(
            f"feature drift PSI {max_feature_psi:.2f} exceeds {policy.drift_psi_threshold}"
        )
    return (len(reasons) > 0, reasons)
