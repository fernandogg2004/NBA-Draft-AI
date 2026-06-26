"""Models (Phase 6).

Start simple and interpretable; add complexity only if it beats the baselines under temporal
validation. Per-target families: T1 reach = classification, T2/T3 impact = regression,
T5 longevity = survival. Tuning (Optuna) and survival (lifelines) live in submodules and are
imported directly (``from nba_draft.models.tuning import tune_estimator``) to keep this package
import light and free of optional-dependency import cycles.
"""

from nba_draft.models.base import Estimator
from nba_draft.models.baseline import DraftPositionBaseline, DraftPositionEstimator
from nba_draft.models.hurdle import HurdleModel, realized_value
from nba_draft.models.zoo import (
    ProbaAdapter,
    elasticnet_regressor,
    gbm_classifier,
    gbm_regressor,
    lasso_regressor,
    logistic_classifier,
    mean_regressor,
    ridge_regressor,
)

__all__ = [
    "DraftPositionBaseline",
    "DraftPositionEstimator",
    "Estimator",
    "HurdleModel",
    "ProbaAdapter",
    "realized_value",
    "elasticnet_regressor",
    "gbm_classifier",
    "gbm_regressor",
    "lasso_regressor",
    "logistic_classifier",
    "mean_regressor",
    "ridge_regressor",
]
