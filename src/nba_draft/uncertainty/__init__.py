"""Phase 9 — uncertainty quantification (mandatory, not optional).

The draft is a fat-tailed, high-variance problem: a point prediction is nearly useless. Every
prediction must carry its uncertainty, and for decisions we present a DISTRIBUTION of outcomes
(P(bust / rotation / starter / star / superstar)) rather than a single number — a high-ceiling,
low-floor prospect is a different decision from a safe, limited one.

Approaches provided (use whichever suits the target/data):
  * quantile    — direct quantile regression intervals (gradient boosting)
  * conformal   — split-conformal intervals with finite-sample marginal coverage guarantees
  * bayesian    — Bayesian linear predictive mean + std (fast, apt for scarce data)
  * ensemble    — bootstrap ensemble -> predictive samples -> intervals + scenario probabilities
  * scenarios   — map any predictive distribution to outcome-tier probabilities + floor/ceiling
"""

from nba_draft.uncertainty.bayesian import BayesianLinearModel
from nba_draft.uncertainty.conformal import SplitConformalRegressor
from nba_draft.uncertainty.ensemble import BootstrapEnsemble
from nba_draft.uncertainty.quantile import QuantileGBM
from nba_draft.uncertainty.scenarios import (
    ceiling_floor,
    interval_coverage,
    scenario_probabilities_from_normal,
    scenario_probabilities_from_samples,
)

__all__ = [
    "BayesianLinearModel",
    "BootstrapEnsemble",
    "QuantileGBM",
    "SplitConformalRegressor",
    "ceiling_floor",
    "interval_coverage",
    "scenario_probabilities_from_normal",
    "scenario_probabilities_from_samples",
]
