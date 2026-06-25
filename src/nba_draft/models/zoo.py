"""Model zoo: factory functions returning sklearn-style estimators.

Phase 6 philosophy (instructions.md): start simple and interpretable, add complexity only if it
earns its keep under temporal validation. Linear models are scaled; tree models are not.

Gradient boosting resolves through a fallback chain — XGBoost -> LightGBM -> sklearn's
HistGradientBoosting — so the pipeline always has a working GBM even if the heavy optional
libraries are unavailable on this platform.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


# ----------------------------------------------------------------- regression
def mean_regressor() -> Any:
    """Predicts the training mean — a floor any real model must beat."""
    from sklearn.dummy import DummyRegressor

    return DummyRegressor(strategy="mean")


def ridge_regressor(alpha: float = 1.0) -> Any:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def lasso_regressor(alpha: float = 0.01) -> Any:
    from sklearn.linear_model import Lasso
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), Lasso(alpha=alpha, max_iter=10000))


def elasticnet_regressor(alpha: float = 0.01, l1_ratio: float = 0.5) -> Any:
    from sklearn.linear_model import ElasticNet
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(), ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
    )


def gbm_regressor(
    *, n_estimators: float = 300, learning_rate: float = 0.05, max_depth: float = 3, seed: int = 42
) -> Any:
    """Gradient-boosted trees via the first available backend.

    n_estimators / max_depth accept floats and are coerced to int so the same factory works
    directly as an Optuna tuning target (suggested params arrive as floats).
    """
    n_estimators = int(round(n_estimators))
    max_depth = int(round(max_depth))
    try:
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=seed,
            n_jobs=1,
        )
    except ImportError:
        pass
    try:
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=seed,
            n_jobs=1,
            verbose=-1,
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            max_iter=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
            random_state=seed,
        )


# ----------------------------------------------------------------- classification (T1 reach)
class ProbaAdapter:
    """Wrap a classifier so ``predict`` returns P(class=1) — lets the regression-style runner
    score probabilistic targets with calibration metrics (Brier/ECE)."""

    def __init__(self, classifier: Any) -> None:
        self._clf = classifier

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> ProbaAdapter:
        self._clf.fit(X, y)
        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        proba = self._clf.predict_proba(X)
        return np.asarray(proba[:, 1], dtype=np.float64)


def logistic_classifier(C: float = 1.0) -> ProbaAdapter:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return ProbaAdapter(make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000)))


def gbm_classifier(
    *, n_estimators: int = 300, learning_rate: float = 0.05, max_depth: int = 3, seed: int = 42
) -> ProbaAdapter:
    """Gradient-boosted classifier via the first available backend, as a ProbaAdapter."""
    try:
        from xgboost import XGBClassifier

        clf = XGBClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
            random_state=seed, n_jobs=1,
        )
    except ImportError:
        try:
            from lightgbm import LGBMClassifier

            clf = LGBMClassifier(
                n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                random_state=seed, n_jobs=1, verbose=-1,
            )
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingClassifier

            clf = HistGradientBoostingClassifier(
                max_iter=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                random_state=seed,
            )
    return ProbaAdapter(clf)
