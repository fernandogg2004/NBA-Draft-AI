"""Draft-board service: projections + uncertainty + fit + explanations in one place."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from nba_draft.fit import (
    CBAConfig,
    Player,
    TeamContext,
    load_cba,
    player_team_fit,
)
from nba_draft.fit.score import FitResult
from nba_draft.interpretability import ShapExplainer
from nba_draft.models.zoo import ridge_regressor
from nba_draft.uncertainty import BootstrapEnsemble
from nba_draft.validation import FoldPreprocessor, make_data_split

# Outcome-tier bands on the impact scale (BPM-like); align with config/targets in production.
TIER_EDGES: list[float] = [-1e9, -2.0, 0.0, 3.0, 6.0, 1e9]
TIER_LABELS: list[str] = ["bust", "rotation", "starter", "all_star", "superstar"]


def _scale(value: float, lo: float, hi: float) -> float:
    """Clip-and-scale a raw stat into a 0-100 skill percentile (illustrative mapping)."""
    if hi <= lo:
        return 50.0
    return float(np.clip((value - lo) / (hi - lo) * 100.0, 0.0, 100.0))


def prospect_to_player(name: str, row: dict[str, Any], impact: float) -> Player:
    """Heuristically map pre-draft features to functional skills for the fit module.

    EXPLORATORY bridge: the skill mapping is a rough illustration, not a calibrated model.
    Rim protection / perimeter defense are weakly proxied (box stats under-capture defense).
    """
    return Player(
        name=name,
        skills={
            "scoring": _scale(row.get("pts_per100", 0.0), 2, 26),
            "shooting_spacing": _scale(row.get("true_shooting", 0.5), 0.45, 0.65),
            "playmaking": _scale(row.get("ast_per100", 0.0), 0, 9),
            "rebounding": _scale(row.get("reb_per100", 0.0), 1, 14),
            "rim_protection": _scale(row.get("wingspan_in", 80.0), 76, 92),
            "perimeter_defense": 50.0,  # not captured by box features; neutral prior
        },
        impact=impact,
    )


@dataclass
class DraftBoardService:
    """Trained service powering the API and dashboard."""

    preprocessor: FoldPreprocessor
    impact_model: Any
    ensemble: BootstrapEnsemble
    feature_cols: list[str]
    background: np.ndarray
    name_col: str = "player_id"
    interval_alpha: float = 0.2
    _shap: ShapExplainer | None = None

    def _matrix(self, prospects: pl.DataFrame) -> np.ndarray:
        return self.preprocessor.transform_matrix(prospects).to_numpy()

    def rank(self, prospects: pl.DataFrame) -> pl.DataFrame:
        """Rank prospects with projection, 80% interval (floor/ceiling), and tier probabilities."""
        x = self._matrix(prospects)
        point = np.asarray(self.impact_model.predict(x), dtype=float)
        lo, hi = self.ensemble.predict_interval(x, alpha=self.interval_alpha)
        scenarios = self.ensemble.predict_scenarios(x, TIER_EDGES, TIER_LABELS)

        out = prospects.select(
            [c for c in (self.name_col, "full_name", "draft_year") if c in prospects.columns]
        ).with_columns(
            pl.Series("projected_impact", [round(float(v), 3) for v in point]),
            pl.Series("floor", [round(float(v), 3) for v in lo]),
            pl.Series("ceiling", [round(float(v), 3) for v in hi]),
        )
        for label in TIER_LABELS:
            out = out.with_columns(
                pl.Series(f"p_{label}", [round(s[label], 4) for s in scenarios])
            )
        return out.sort("projected_impact", descending=True)

    def explain(self, prospect_row: pl.DataFrame) -> tuple[pl.DataFrame, float]:
        """Local SHAP explanation for a single prospect (lazily builds the explainer)."""
        if self._shap is None:
            self._shap = ShapExplainer(self.impact_model, self.background, self.feature_cols)
        x = self._matrix(prospect_row)
        return self._shap.local_explanation(x[0])

    def fit_for_team(
        self,
        prospect_row: pl.DataFrame,
        team: TeamContext,
        *,
        pick: int,
        cba: CBAConfig | None = None,
        name: str = "Prospect",
    ) -> FitResult:
        """Score a prospect's fit with a roster (projected impact used as the VORP proxy)."""
        cba = cba or load_cba()
        x = self._matrix(prospect_row)
        projected = float(self.impact_model.predict(x)[0])
        row = prospect_row.row(0, named=True)
        player = prospect_to_player(name, dict(row), impact=projected)
        # Use projected impact as a rough per-year VORP proxy for surplus value.
        return player_team_fit(
            team, player, pick=pick, projected_vorp_per_year=max(0.0, projected), cba=cba
        )


def build_demo_service(seed: int = 42) -> tuple[DraftBoardService, pl.DataFrame]:
    """Train a demo service on the synthetic fixture; return (service, prospect_pool).

    The 'prospect pool' is the held-out (most recent) draft classes — a stand-in for the
    upcoming class to be ranked. SYNTHETIC: for wiring/demo only, no basketball meaning.
    """
    from nba_draft.data.fixtures import (
        FEATURE_COLUMNS,
        TARGET_COLUMN,
        YEAR_COLUMN,
        make_synthetic_prospects,
    )

    df = make_synthetic_prospects(seed=seed)
    split = make_data_split(df, year_col=YEAR_COLUMN, n_holdout_years=2)
    feats = list(FEATURE_COLUMNS)

    pp = FoldPreprocessor(feats).fit(split.dev)
    x_tr = pp.transform_matrix(split.dev).to_numpy()
    y_tr = split.dev[TARGET_COLUMN].to_numpy().astype(float)

    impact_model = ridge_regressor(1.0)
    impact_model.fit(x_tr, y_tr)
    ensemble = BootstrapEnsemble(lambda: ridge_regressor(1.0), n_estimators=30, seed=seed)
    ensemble.fit(x_tr, y_tr)

    service = DraftBoardService(
        preprocessor=pp,
        impact_model=impact_model,
        ensemble=ensemble,
        feature_cols=feats,
        background=x_tr,
        name_col="player_id",
    )
    # Give the pool friendly names for display.
    pool = split.holdout.with_columns(
        ("Prospect " + pl.col("player_id").cast(pl.Utf8)).alias("full_name")
    )
    return service, pool
