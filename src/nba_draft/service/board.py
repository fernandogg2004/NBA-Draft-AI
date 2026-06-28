"""Draft-board service: projections + uncertainty + fit + explanations in one place."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
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
from nba_draft.fit.financial import projected_value_usd
from nba_draft.fit.score import FitResult
from nba_draft.fit.types import SKILL_DIMS
from nba_draft.interpretability import ShapExplainer, greedy_counterfactual
from nba_draft.models.hurdle import REPLACEMENT_BPM, HurdleModel
from nba_draft.models.zoo import logistic_classifier, ridge_regressor
from nba_draft.uncertainty import BootstrapEnsemble, SplitConformalRegressor
from nba_draft.validation import FoldPreprocessor, make_data_split

# Outcome-tier bands on the impact scale (BPM-like); align with config/targets in production.
TIER_EDGES: list[float] = [-1e9, -2.0, 0.0, 3.0, 6.0, 1e9]
TIER_LABELS: list[str] = ["bust", "rotation", "starter", "all_star", "superstar"]

# Default miss rate for prediction intervals (0.2 -> 80% floor/ceiling).
DEFAULT_INTERVAL_ALPHA: float = 0.2


def _tier_for(value: float) -> str:
    """Outcome-tier label for an impact value (using TIER_EDGES bands)."""
    for label, hi in zip(TIER_LABELS, TIER_EDGES[1:], strict=True):
        if value < hi:
            return label
    return TIER_LABELS[-1]


@dataclass
class CounterfactualChange:
    """One feature move proposed to lift a prospect toward the next tier."""

    feature: str
    from_value: float
    to_value: float
    delta: float


@dataclass
class CounterfactualResult:
    """Smallest set of feature changes that would lift a prospect to the next tier.

    ``reached`` is False when no in-bounds change set hit the target within max_features.
    ``target`` is ``None`` when the prospect is already in the top tier (no change needed).
    """

    current_impact: float
    current_tier: str
    target: float | None
    target_tier: str | None
    projected_impact: float
    reached: bool
    changes: list[CounterfactualChange]


def _scale(value: float, lo: float, hi: float) -> float:
    """Clip-and-scale a raw stat into a 0-100 skill percentile (illustrative mapping)."""
    if hi <= lo:
        return 50.0
    return float(np.clip((value - lo) / (hi - lo) * 100.0, 0.0, 100.0))


def _feat(row: dict[str, Any], *names: str, default: float = 0.0) -> float:
    """First present, non-null, non-NaN value among ``names`` (else ``default``).

    Lets the skill mapping work across schemas: the synthetic fixture uses NBA-shaped per-100
    columns, while the real college table uses per-40 columns (and richer defensive box stats).
    """
    for n in names:
        v = row.get(n)
        if v is not None and v == v:  # v == v is False only for NaN
            return float(v)
    return float(default)


def prospect_to_player(name: str, row: dict[str, Any], impact: float) -> Player:
    """Heuristically map pre-draft features to functional skills for the fit module.

    EXPLORATORY bridge: the skill mapping is a rough illustration, not a calibrated model. Handles
    both the synthetic per-100 schema and the real college per-40 schema; defense uses block/steal
    rates when available (real data) and falls back to weak proxies otherwise.
    """
    per40 = row.get("pts_per40") is not None
    if per40:
        scoring = _scale(_feat(row, "pts_per40"), 5, 28)
        playmaking = _scale(_feat(row, "ast_per40"), 0, 8)
        rebounding = _scale(_feat(row, "reb_per40"), 1, 14)
    else:
        scoring = _scale(_feat(row, "pts_per100"), 2, 26)
        playmaking = _scale(_feat(row, "ast_per100"), 0, 9)
        rebounding = _scale(_feat(row, "reb_per100"), 1, 14)

    # Rim protection: block rate if present (real college), else wingspan as a weak proxy.
    if row.get("blk_per40") is not None:
        rim = _scale(_feat(row, "blk_per40"), 0, 3)
    else:
        rim = _scale(_feat(row, "wingspan_in", default=80.0), 76, 92)
    # Perimeter defense: steal rate if present, else a neutral prior (box stats miss defense).
    if row.get("stl_per40") is not None:
        perimeter = _scale(_feat(row, "stl_per40"), 0, 2.5)
    else:
        perimeter = 50.0

    return Player(
        name=name,
        skills={
            "scoring": scoring,
            "shooting_spacing": _scale(_feat(row, "true_shooting", default=0.5), 0.45, 0.65),
            "playmaking": playmaking,
            "rebounding": rebounding,
            "rim_protection": rim,
            "perimeter_defense": perimeter,
        },
        impact=impact,
    )


# Descriptive archetype = the prospect's dominant functional skill. Deterministic and
# transparent (no clustering), so it is honest about being a simple label, not a model.
_ARCHETYPE_BY_SKILL: dict[str, str] = {
    "scoring": "Scorer",
    "shooting_spacing": "Floor Spacer",
    "playmaking": "Playmaker",
    "rebounding": "Rebounder",
    "rim_protection": "Rim Protector",
    "perimeter_defense": "Perimeter Defender",
}


def archetype_from_skills(skills: dict[str, float]) -> str:
    """Label a prospect by their strongest functional skill (descriptive, EXPLORATORY)."""
    if not skills:
        return "Unspecified"
    return _ARCHETYPE_BY_SKILL[max(SKILL_DIMS, key=lambda d: skills.get(d, 0.0))]


@dataclass
class DraftBoardService:
    """Trained service powering the API and dashboard."""

    preprocessor: FoldPreprocessor
    impact_model: Any
    ensemble: BootstrapEnsemble
    feature_cols: list[str]
    background: np.ndarray
    name_col: str = "player_id"
    interval_alpha: float = DEFAULT_INTERVAL_ALPHA
    hurdle: Any | None = None      # optional HurdleModel -> survivorship-robust EV ranking
    conformal: Any | None = None   # optional SplitConformalRegressor -> calibrated floor/ceiling
    _shap: ShapExplainer | None = None

    def _matrix(self, prospects: pl.DataFrame) -> np.ndarray:
        return self.preprocessor.transform_matrix(prospects).to_numpy()

    def rank(self, prospects: pl.DataFrame) -> pl.DataFrame:
        """Rank prospects by projection + 80% interval + tier probabilities.

        If a hurdle model is attached, prospects are ranked by the survivorship-robust
        unconditional EV (and P(reach) / EV columns are added); otherwise by conditional impact.
        """
        x = self._matrix(prospects)
        point = np.asarray(self.impact_model.predict(x), dtype=float)
        # Floor/ceiling: prefer the conformal interval (finite-sample marginal coverage ≈ nominal)
        # when fit; fall back to the bootstrap ensemble (adaptive but historically overconfident).
        # Floor/ceiling AND tier scenarios: prefer the conformal layer's empirical predictive
        # distribution (calibrated coverage; honest tier spread) over the overconfident ensemble.
        if self.conformal is not None:
            lo, hi = self.conformal.predict_interval(x)
            scenarios = self.conformal.predict_scenarios(x, TIER_EDGES, TIER_LABELS)
        else:
            lo, hi = self.ensemble.predict_interval(x, alpha=self.interval_alpha)
            scenarios = self.ensemble.predict_scenarios(x, TIER_EDGES, TIER_LABELS)

        out = prospects.select(
            [c for c in (self.name_col, "full_name", "draft_year") if c in prospects.columns]
        ).with_columns(
            pl.Series("projected_impact", [round(float(v), 3) for v in point]),
            pl.Series("floor", [round(float(v), 3) for v in lo]),
            pl.Series("ceiling", [round(float(v), 3) for v in hi]),
        )
        sort_col = "projected_impact"
        if self.hurdle is not None:
            p_reach, _ = self.hurdle.predict_parts(x)
            ev = np.asarray(self.hurdle.predict(x), dtype=float)
            out = out.with_columns(
                pl.Series("p_reach", [round(float(v), 4) for v in p_reach]),
                pl.Series("projected_ev", [round(float(v), 3) for v in ev]),
            )
            sort_col = "projected_ev"
        for label in TIER_LABELS:
            out = out.with_columns(
                pl.Series(f"p_{label}", [round(s[label], 4) for s in scenarios])
            )
        return out.sort(sort_col, descending=True)

    def profile_table(self, prospects: pl.DataFrame) -> pl.DataFrame:
        """Feature-derived prospect profile: functional skills, archetype, age, wingspan.

        Skills use the same heuristic mapping as the fit module (``prospect_to_player``); the
        archetype is the dominant skill. All EXPLORATORY (the skill mapping is illustrative).
        Only columns actually present in the pool are emitted (e.g. wingspan/age may be absent
        on a minimal real table), so the API surfaces real data and omits what it doesn't have.
        """
        rows: list[dict[str, Any]] = []
        for r in prospects.iter_rows(named=True):
            player = prospect_to_player(str(r.get("full_name", "")), dict(r), impact=0.0)
            entry: dict[str, Any] = {
                self.name_col: r.get(self.name_col),
                "archetype": archetype_from_skills(player.skills),
                **{f"skill_{k}": round(v, 1) for k, v in player.skills.items()},
            }
            # age: real table uses age_at_draft; synthetic uses age.
            age = _feat(r, "age_at_draft", "age", default=float("nan"))
            if age == age:  # not NaN
                entry["age"] = round(age, 1)
            if r.get("wingspan_in") is not None:
                entry["wingspan_in"] = round(float(r["wingspan_in"]), 1)
            # Post-draft display metadata (present only on the real table) passed straight through.
            for col in ("draft_pick", "team_abbr", "team_name", "position"):
                if r.get(col) is not None:
                    entry[col] = r[col]
            rows.append(entry)
        return pl.DataFrame(rows)

    def ranked_with_profile(
        self, prospects: pl.DataFrame, *, cba: CBAConfig | None = None
    ) -> pl.DataFrame:
        """Ranked board joined with the feature-derived profile + projection/post-draft fields.

        Adds ``peak_pctile`` (percentile rank of the ranking metric within the pool; higher is
        better), ``projected_value_usd`` (team-independent $ value over the rookie window), a
        1-based ``model_rank``, and — when the actual ``draft_pick`` is known (post-draft real
        data) — ``slot_delta = draft_pick - model_rank`` (positive ⇒ the model rates the player
        higher than the league did = a "steal"). A ``headshot_url`` is built from the NBA player id
        (the frontend falls back to an icon if it 404s, e.g. for synthetic ids).
        """
        cba = cba or load_cba()
        board = self.rank(prospects)  # already sorted best-first
        rank_col = "projected_ev" if "projected_ev" in board.columns else "projected_impact"
        board = board.with_row_index("model_rank", offset=1).with_columns(
            (pl.col(rank_col).rank() / pl.len()).round(4).alias("peak_pctile"),
            pl.col("projected_impact")
            .map_elements(
                lambda v: round(projected_value_usd(max(0.0, float(v)), cba), 2),
                return_dtype=pl.Float64,
            )
            .alias("projected_value_usd"),
            pl.col("model_rank").cast(pl.Int64),
        )
        if self.name_col in board.columns:
            board = board.join(self.profile_table(prospects), on=self.name_col, how="left")
            board = board.with_columns(
                pl.format(
                    "https://cdn.nba.com/headshots/nba/latest/1040x760/{}.png",
                    pl.col(self.name_col),
                ).alias("headshot_url")
            )
            if "draft_pick" in board.columns:
                board = board.with_columns(
                    (pl.col("draft_pick") - pl.col("model_rank")).alias("slot_delta")
                )
        return board

    def explain(self, prospect_row: pl.DataFrame) -> tuple[pl.DataFrame, float]:
        """Local SHAP explanation for a single prospect (lazily builds the explainer)."""
        if self._shap is None:
            self._shap = ShapExplainer(self.impact_model, self.background, self.feature_cols)
        x = self._matrix(prospect_row)
        return self._shap.local_explanation(x[0])

    def counterfactual(
        self, prospect_row: pl.DataFrame, *, max_features: int = 3
    ) -> CounterfactualResult:
        """What feature change(s) would lift this prospect into the next outcome tier?

        Targets the lower edge of the next tier above the current projection and greedily
        searches a few features for the smallest moves that reach it. Feature bounds come from
        the 5th-95th percentiles of the training background (avoids unrealistic extremes).
        """
        x = self._matrix(prospect_row)[0]
        current = float(self.impact_model.predict(x.reshape(1, -1))[0])
        current_tier = _tier_for(current)

        # Next tier edge strictly above the current projection (skip the +inf sentinel).
        target = next((e for e in TIER_EDGES[1:-1] if e > current), None)
        if target is None:  # already top tier — nothing to change
            return CounterfactualResult(
                current_impact=round(current, 3),
                current_tier=current_tier,
                target=None,
                target_tier=None,
                projected_impact=round(current, 3),
                reached=True,
                changes=[],
            )

        lo = np.percentile(self.background, 5, axis=0)
        hi = np.percentile(self.background, 95, axis=0)
        bounds = {j: (float(lo[j]), float(hi[j])) for j in range(len(self.feature_cols))}

        moves = greedy_counterfactual(
            self.impact_model, x, target, bounds, max_features=max_features
        )
        x_new = x.copy()
        for j, v in moves.items():
            x_new[j] = v
        projected = float(self.impact_model.predict(x_new.reshape(1, -1))[0])

        changes = [
            CounterfactualChange(
                feature=self.feature_cols[j],
                from_value=round(float(x[j]), 3),
                to_value=round(float(v), 3),
                delta=round(float(v) - float(x[j]), 3),
            )
            for j, v in moves.items()
        ]
        return CounterfactualResult(
            current_impact=round(current, 3),
            current_tier=current_tier,
            target=round(float(target), 3),
            target_tier=_tier_for(target),
            projected_impact=round(projected, 3),
            reached=projected >= target,
            changes=changes,
        )

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

    # Survivorship-robust hurdle: reach (impact above replacement) × conditional impact.
    reached = (y_tr > REPLACEMENT_BPM).astype(float)
    hurdle = HurdleModel(
        reach_factory=logistic_classifier, impact_factory=lambda: ridge_regressor(1.0),
    ).fit(x_tr, reached, y_tr)
    conformal = SplitConformalRegressor(
        lambda: ridge_regressor(1.0), alpha=DEFAULT_INTERVAL_ALPHA, seed=seed
    ).fit(x_tr, y_tr)

    service = DraftBoardService(
        preprocessor=pp,
        impact_model=impact_model,
        ensemble=ensemble,
        feature_cols=feats,
        background=x_tr,
        name_col="player_id",
        hurdle=hurdle,
        conformal=conformal,
    )
    # Give the pool friendly names for display.
    pool = split.holdout.with_columns(
        ("Prospect " + pl.col("player_id").cast(pl.Utf8)).alias("full_name")
    )
    return service, pool


def build_service_from_table(
    train_table: pl.DataFrame,
    feature_cols: list[str],
    *,
    target_col: str = "peak_impact",
    reached_col: str = "reached",
    seed: int = 42,
) -> DraftBoardService:
    """Build a DraftBoardService trained on a REAL modeling table (not the synthetic fixture).

    The impact regressor + uncertainty ensemble train on reached players (non-null target). If a
    ``reached_col`` is present, a survivorship-robust hurdle is also fit over ALL prospects and the
    served board ranks by unconditional EV. The fold preprocessor is fit on the training rows.
    """
    # Leakage guard: the served feature matrix must contain no known post-draft columns (mirrors
    # the real modeling path in realdata.build.evaluate_real_models).
    from nba_draft.features import assert_pre_draft_safe

    assert_pre_draft_safe(train_table.select([c for c in feature_cols if c in train_table.columns]))
    # A finite target marks a reached prospect (polars NaN is not null, so guard both).
    has_target = pl.col(target_col).is_not_null() & pl.col(target_col).is_not_nan()
    reached_rows = train_table.filter(has_target)
    if reached_rows.height < 2:
        raise ValueError("Need at least 2 rows with a finite target to train a service.")
    has_reach = reached_col in train_table.columns
    # Fit the preprocessor on all prospects when a hurdle will be trained, else on reached rows.
    pp = FoldPreprocessor(feature_cols).fit(train_table if has_reach else reached_rows)
    x_reached = pp.transform_matrix(reached_rows).to_numpy()
    y_reached = reached_rows[target_col].to_numpy().astype(float)

    impact_model = ridge_regressor(1.0)
    impact_model.fit(x_reached, y_reached)
    ensemble = BootstrapEnsemble(lambda: ridge_regressor(1.0), n_estimators=30, seed=seed)
    ensemble.fit(x_reached, y_reached)
    # Calibrated floor/ceiling: split-conformal on the reached impacts gives ≈ nominal coverage
    # (the bootstrap interval was badly overconfident). Needs enough rows for a calibration split.
    conformal = None
    if x_reached.shape[0] >= 20:
        conformal = SplitConformalRegressor(
            lambda: ridge_regressor(1.0), alpha=DEFAULT_INTERVAL_ALPHA, seed=seed
        ).fit(x_reached, y_reached)

    hurdle = None
    if has_reach:
        x_all = pp.transform_matrix(train_table).to_numpy()
        reach01 = (pl.col(reached_col).cast(pl.Boolean) & has_target).cast(pl.Float64)
        reach_arr = train_table.select(reach01.alias("r"))["r"].to_numpy()
        impact_all = train_table[target_col].to_numpy().astype(float)
        if reach_arr.sum() >= 2:
            hurdle = HurdleModel(
                reach_factory=logistic_classifier, impact_factory=lambda: ridge_regressor(1.0),
            ).fit(x_all, reach_arr, impact_all)

    return DraftBoardService(
        preprocessor=pp,
        impact_model=impact_model,
        ensemble=ensemble,
        feature_cols=feature_cols,
        background=x_reached,
        name_col="player_id",
        hurdle=hurdle,
        conformal=conformal,
    )


def build_service_from_master(
    source: str | Path,
    *,
    pool_years: list[int] | None = None,
    seed: int = 42,
) -> tuple[DraftBoardService, pl.DataFrame]:
    """Load a persisted real modeling table + manifest -> (service, prospect_pool).

    `source` is the serving directory (or the ``serving_manifest.json``) written by
    ``realdata.build.evaluate_real_models``. The most recent draft class is held out as the pool to
    rank and the service trains on the remaining (older) classes, so the served board never trains
    on the prospects it ranks. Falls back to training on the whole table if the hold-out would leave
    too few labeled rows to fit the impact head.
    """
    path = Path(source)
    manifest_path = path if path.suffix == ".json" else path / "serving_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    table = pl.read_parquet(manifest_path.parent / manifest["table"])
    feature_cols = list(manifest["feature_cols"])
    target_col = manifest.get("target_col", "peak_impact")
    reached_col = manifest.get("reached_col", "reached")

    years = pool_years
    if years is None and "draft_year" in table.columns and table["draft_year"].n_unique() > 1:
        latest = table["draft_year"].max()
        years = [int(latest)]  # type: ignore[arg-type]  # draft_year is an integer column
    if years:
        pool = table.filter(pl.col("draft_year").is_in(years))
        train = table.filter(~pl.col("draft_year").is_in(years))
    else:
        pool, train = table, table

    finite = pl.col(target_col).is_not_null() & pl.col(target_col).is_not_nan()
    if train.filter(finite).height < 2:  # not enough signal once the pool is removed -> use all
        train = table
    service = build_service_from_table(
        train, feature_cols, target_col=target_col, reached_col=reached_col, seed=seed
    )
    if "full_name" not in pool.columns:
        pool = pool.with_columns(
            ("Prospect " + pl.col("player_id").cast(pl.Utf8)).alias("full_name")
        )
    return service, pool
