"""Phase 10 — interpretability.

Scouts and GMs will not adopt a black box. Every recommendation must be explainable so an
analyst can understand and challenge why the model values a player:
  * attribution    — SHAP global feature importance + per-prospect local explanations
  * importance     — model-agnostic permutation importance (works for any predict-able model)
  * pdp            — partial dependence: how a projection moves with one feature
  * counterfactual — "what would need to change to raise this prospect's projection"
"""

from nba_draft.interpretability.attribution import ShapExplainer, permutation_importance_table
from nba_draft.interpretability.counterfactual import (
    counterfactual_single_feature,
    greedy_counterfactual,
)
from nba_draft.interpretability.pdp import partial_dependence

__all__ = [
    "ShapExplainer",
    "counterfactual_single_feature",
    "greedy_counterfactual",
    "partial_dependence",
    "permutation_importance_table",
]
