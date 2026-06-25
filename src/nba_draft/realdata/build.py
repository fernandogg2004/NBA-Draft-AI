"""Build a real modeling table and run the real pipeline (nba_api only)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from nba_draft.cleaning.schema import COMBINE_COLUMNS
from nba_draft.ingestion.parse import (
    parse_cbd_player_season,
    parse_combine,
    parse_draft_history,
    parse_player_season,
)
from nba_draft.realdata.age import AGE_FEATURE_COLUMNS, pull_player_ages
from nba_draft.realdata.college import COLLEGE_FEATURE_COLUMNS, link_college_features
from nba_draft.targets import add_impact_metrics, build_player_outcomes
from nba_draft.targets.definitions import (
    cumulative_value,
    is_label_resolved,
    load_target_config,
    outcome_tier,
    peak_impact,
    reached_role,
)
from nba_draft.targets.outcomes import season_str_to_year
from nba_draft.utils.logging import get_logger

log = get_logger("realdata.build")

COMBINE_FEATURES: list[str] = list(COMBINE_COLUMNS)
FEATURE_COLUMNS: list[str] = list(COMBINE_COLUMNS)   # default; +college features when available
PICK_COLUMN = "draft_pick"
YEAR_COLUMN = "draft_year"
TARGET_COLUMN = "peak_impact"


def _combine_season_str(draft_year: int) -> str:
    return f"{draft_year}-{str(draft_year + 1)[-2:]}"


@dataclass
class RealFrames:
    draft_history: pl.DataFrame
    combine: pl.DataFrame
    player_seasons: pl.DataFrame
    data_through_year: int
    cbd_seasons: pl.DataFrame | None = None


def pull_real_frames(
    ingester: Any,
    *,
    draft_years: list[int],
    outcome_seasons: list[str],
    cbd_ingester: Any | None = None,
) -> RealFrames:
    """Pull (cached) + parse the raw endpoints into the frames the table builder needs.

    `ingester` is an NbaStatsIngester (passed in so this stays import-light / testable).
    """
    dh = pl.concat(
        [parse_draft_history(ingester.draft_history(y)) for y in draft_years],
        how="diagonal_relaxed",
    )
    combine_parts = [
        parse_combine(ingester.draft_combine_stats(_combine_season_str(y))) for y in draft_years
    ]
    combine = pl.concat(combine_parts, how="diagonal_relaxed").unique(
        subset=["player_id"], keep="first"
    )
    seasons = [
        add_impact_metrics(
            parse_player_season(
                ingester.player_season_stats(s, "Base"),
                ingester.player_season_stats(s, "Advanced"),
                s,
            )
        )
        for s in outcome_seasons
    ]
    player_seasons = pl.concat(seasons, how="diagonal_relaxed")
    data_through = max(season_str_to_year(s) for s in outcome_seasons)

    cbd_seasons = None
    if cbd_ingester is not None:
        # A drafted player's final college season is the spring of their draft year.
        cbd_parts = [
            parse_cbd_player_season(cbd_ingester.player_season_stats(y)) for y in draft_years
        ]
        cbd_seasons = pl.concat(cbd_parts, how="diagonal_relaxed")

    return RealFrames(dh, combine, player_seasons, data_through, cbd_seasons)


def build_real_modeling_table(
    draft_history: pl.DataFrame,
    combine: pl.DataFrame,
    player_seasons: pl.DataFrame,
    *,
    data_through_year: int,
    cbd_seasons: pl.DataFrame | None = None,
    ages: pl.DataFrame | None = None,
    cfg: Any | None = None,
) -> pl.DataFrame:
    """One row per drafted player: Combine features + pick + real outcome labels + resolved flag.

    `resolved` marks whether the player's primary-window label is fully observed (not censored).
    Train only on resolved rows; for the conditional impact target keep `reached` players.
    """
    cfg = cfg or load_target_config()
    outcomes = build_player_outcomes(player_seasons, draft_history)

    label_rows: list[dict[str, object]] = []
    for pid, out in outcomes.items():
        peak = peak_impact(out, cfg)
        label_rows.append(
            {
                "player_id": pid,
                "reached": reached_role(out, cfg),
                "peak_impact": peak,
                "cumulative_value": cumulative_value(out, cfg),
                "outcome_tier": outcome_tier(out, cfg).value,
                "resolved": is_label_resolved(out, data_through_year, cfg),
            }
        )
    labels = pl.DataFrame(label_rows)

    draft = draft_history.select("player_id", YEAR_COLUMN, PICK_COLUMN, "full_name").unique(
        subset=["player_id"], keep="first"
    )
    combine_cols = [c for c in COMBINE_FEATURES if c in combine.columns]
    combine_feats = combine.select(["player_id", *combine_cols])

    table = (
        draft.join(labels, on="player_id", how="left")
        .join(combine_feats, on="player_id", how="left")
    )
    if cbd_seasons is not None:
        college = link_college_features(draft_history, cbd_seasons)
        table = table.join(college, on="player_id", how="left")
    if ages is not None:
        table = table.join(ages, on="player_id", how="left")
    return table.sort([YEAR_COLUMN, PICK_COLUMN])


@dataclass
class RealPipelineResult:
    n_drafted: int
    n_trainable: int
    comparison: list[dict[str, Any]] = field(default_factory=list)
    model_version: str = ""
    summary_path: str = ""


def run_real_pipeline(
    ingester: Any,
    *,
    draft_years: list[int],
    outcome_seasons: list[str],
    cbd_ingester: Any | None = None,
    with_age: bool = True,
    tune: bool = True,
    n_trials: int = 30,
    output_root: str | Path = "artifacts/real_pipeline",
    model_root: str | Path = "artifacts/models",
    min_train_years: int = 2,
    tracking_enabled: bool = False,
) -> RealPipelineResult:
    """End-to-end on real data: pull → labels → temporal eval → train → register.

    If `cbd_ingester` is provided, real pre-draft college production features are pulled and joined
    (the input that gives a model a genuine shot at beating the draft-position baseline).
    """
    import json

    from nba_draft.evaluation.comparison import compare_models, make_spec
    from nba_draft.mlops.registry import register_model
    from nba_draft.mlops.tracking import ExperimentTracker
    from nba_draft.models import DraftPositionEstimator, gbm_regressor, ridge_regressor
    from nba_draft.validation import FoldPreprocessor

    out = Path(output_root)
    out.mkdir(parents=True, exist_ok=True)

    frames = pull_real_frames(
        ingester, draft_years=draft_years, outcome_seasons=outcome_seasons,
        cbd_ingester=cbd_ingester,
    )
    ages = pull_player_ages(ingester, frames.draft_history) if with_age else None
    table = build_real_modeling_table(
        frames.draft_history, frames.combine, frames.player_seasons,
        data_through_year=frames.data_through_year, cbd_seasons=frames.cbd_seasons, ages=ages,
    )
    feats = (
        COMBINE_FEATURES
        + (COLLEGE_FEATURE_COLUMNS if frames.cbd_seasons is not None else [])
        + (AGE_FEATURE_COLUMNS if ages is not None else [])
    )
    # Conditional-impact training set: resolved labels with a stable peak (the hurdle's Part B).
    trainable = table.filter(
        pl.col("resolved") & pl.col("reached") & pl.col(TARGET_COLUMN).is_not_null()
    )
    log.info(
        "drafted=%d trainable(resolved&reached)=%d features=%d (college=%s)",
        table.height, trainable.height, len(feats), frames.cbd_seasons is not None,
    )

    specs = {
        "baseline_draftpos": make_spec(
            [PICK_COLUMN], lambda: DraftPositionEstimator(0),
            lambda: FoldPreprocessor([PICK_COLUMN]),
        ),
        "ridge_features": make_spec(
            feats, lambda: ridge_regressor(1.0), lambda: FoldPreprocessor(feats)
        ),
        "gbm_features": make_spec(feats, gbm_regressor, lambda: FoldPreprocessor(feats)),
    }

    # Optuna-tune the GBM inside the temporal CV; add the tuned config as another contender.
    best_params: dict[str, Any] = {}
    if tune:
        from nba_draft.models.tuning import tune_estimator

        tuning = tune_estimator(
            trainable, feature_cols=feats, target_col=TARGET_COLUMN, year_col=YEAR_COLUMN,
            build_fn=gbm_regressor,
            param_space={
                "n_estimators": ("int", 100, 600, False),
                "learning_rate": ("float", 0.01, 0.2, True),
                "max_depth": ("int", 2, 5, False),
            },
            preprocessor_factory=lambda: FoldPreprocessor(feats),
            min_train_years=min_train_years, n_trials=n_trials, seed=42,
        )
        best_params = tuning.best_params
        log.info("GBM tuned: %s (cv spearman=%.3f)", best_params, tuning.best_value)
        specs["gbm_tuned"] = make_spec(
            feats, lambda: gbm_regressor(**best_params), lambda: FoldPreprocessor(feats)
        )

    # PRODUCTION models: fuse the consensus (draft pick) WITH the public data. A deployable tool
    # uses both — the draft slot encodes scouting the box score can't, and the data adds what the
    # board under-weights. (Pick is known at draft time, so it's a valid feature; at inference you
    # score a prospect at a candidate slot.)
    prod_feats = [PICK_COLUMN, *feats]
    gbm_factory = (lambda: gbm_regressor(**best_params)) if best_params else gbm_regressor
    specs["production_ridge"] = make_spec(
        prod_feats, lambda: ridge_regressor(1.0), lambda: FoldPreprocessor(prod_feats)
    )
    specs["production_gbm"] = make_spec(
        prod_feats, gbm_factory, lambda: FoldPreprocessor(prod_feats)
    )

    with ExperimentTracker(run_name="real-pipeline", enabled=tracking_enabled) as tracker:
        comparison = compare_models(
            trainable, specs, target_col=TARGET_COLUMN, year_col=YEAR_COLUMN,
            min_train_years=min_train_years, baseline_name="baseline_draftpos",
        )
        tracker.log_metrics({"n_trainable": float(trainable.height)})

        # Register the best non-baseline model by CV Spearman, fit on all trainable using its
        # OWN feature set + preprocessor (data-only and production models use different columns).
        ranked = comparison.filter(pl.col("model") != "baseline_draftpos")
        best_name = str(ranked["model"][0])
        best_spearman = float(ranked["spearman_mean"][0])
        best_spec = specs[best_name]
        best_feats = list(best_spec["feature_cols"])
        pp = best_spec["preprocessor_factory"]().fit(trainable)
        x = pp.transform_matrix(trainable).to_numpy()
        y = trainable[TARGET_COLUMN].to_numpy().astype(float)
        model = best_spec["model_factory"]()
        model.fit(x, y)
        version = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        register_model(
            model, name="real_impact_regressor", version=version,
            metrics={"n_trainable": float(trainable.height), "cv_spearman": best_spearman},
            feature_cols=best_feats, data_version=best_name, root=model_root,
        )

        summary = {
            "created_at": datetime.now(UTC).isoformat(),
            "draft_years": draft_years,
            "outcome_seasons": outcome_seasons,
            "n_drafted": table.height,
            "n_trainable": trainable.height,
            "comparison": comparison.to_dicts(),
            "model_version": version,
            "best_model": best_name,
            "best_cv_spearman": best_spearman,
            "gbm_tuned_params": best_params,
            "n_features": len(feats),
            "college_features": frames.cbd_seasons is not None,
            "note": (
                "Real pre-draft features = Combine"
                + (" + CollegeBasketballData production" if frames.cbd_seasons is not None
                   else " + pick only (no college source)")
                + ". International prospects lack college features (imputed)."
            ),
        }
        summary_path = out / "real_run_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    return RealPipelineResult(
        n_drafted=table.height,
        n_trainable=trainable.height,
        comparison=comparison.to_dicts(),
        model_version=version,
        summary_path=str(summary_path),
    )
