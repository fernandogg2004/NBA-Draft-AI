"""Build a real modeling table and run the real pipeline (nba_api only)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from nba_draft.cleaning.schema import COMBINE_COLUMNS
from nba_draft.ingestion.parse import (
    parse_cbd_player_season,
    parse_combine,
    parse_draft_history,
    parse_euroleague_player_season,
    parse_player_season,
)
from nba_draft.realdata.age import AGE_FEATURE_COLUMNS, pull_player_ages
from nba_draft.realdata.college import COLLEGE_FEATURE_COLUMNS, link_college_features
from nba_draft.realdata.intl import INTL_FEATURE_COLUMNS, link_intl_features
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
    intl_seasons: pl.DataFrame | None = None


def pull_real_frames(
    ingester: Any,
    *,
    draft_years: list[int],
    outcome_seasons: list[str],
    cbd_ingester: Any | None = None,
    intl_ingester: Any | None = None,
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

    intl_seasons = None
    if intl_ingester is not None:
        # International prospects' final pre-draft EuroLeague season ends in draft_year (code y-1)
        # or draft_year; pull both years around each draft.
        el_years = sorted({y - 1 for y in draft_years} | set(draft_years))
        intl_parts = [
            parse_euroleague_player_season(intl_ingester.player_season_stats(y)) for y in el_years
        ]
        intl_seasons = pl.concat(intl_parts, how="diagonal_relaxed")

    return RealFrames(dh, combine, player_seasons, data_through, cbd_seasons, intl_seasons)


def build_real_modeling_table(
    draft_history: pl.DataFrame,
    combine: pl.DataFrame,
    player_seasons: pl.DataFrame,
    *,
    data_through_year: int,
    cbd_seasons: pl.DataFrame | None = None,
    intl_seasons: pl.DataFrame | None = None,
    ages: pl.DataFrame | None = None,
    honors: dict[int, tuple[int, int]] | None = None,
    cfg: Any | None = None,
) -> pl.DataFrame:
    """One row per drafted player: Combine features + pick + real outcome labels + resolved flag.

    `resolved` marks whether the player's primary-window label is fully observed (not censored).
    Train only on resolved rows; for the conditional impact target keep `reached` players.
    """
    cfg = cfg or load_target_config()
    outcomes = build_player_outcomes(player_seasons, draft_history, honors=honors)

    label_rows: list[dict[str, object]] = []
    for pid, out in outcomes.items():
        peak = peak_impact(out, cfg)
        last_year = max((s.season_year for s in out.seasons), default=None)
        # Longevity: career length (seasons) + event flag. A career is "observed ended" if the
        # last season precedes the latest data year (player no longer active); else right-censored.
        career_ended = last_year is not None and last_year < data_through_year
        label_rows.append(
            {
                "player_id": pid,
                "reached": reached_role(out, cfg),
                "peak_impact": peak,
                "cumulative_value": cumulative_value(out, cfg),
                "outcome_tier": outcome_tier(out, cfg).value,
                "resolved": is_label_resolved(out, data_through_year, cfg),
                "career_seasons": len(out.seasons),
                "career_ended": career_ended,
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
    if intl_seasons is not None:
        intl = link_intl_features(draft_history, intl_seasons)
        table = table.join(intl, on="player_id", how="left", suffix="_intl")
        # Coalesce: keep the college value where present, else fall back to the international one.
        overlap = [c for c in INTL_FEATURE_COLUMNS if f"{c}_intl" in table.columns]
        if overlap:
            table = table.with_columns(
                [pl.coalesce([pl.col(c), pl.col(f"{c}_intl")]).alias(c) for c in overlap]
            ).drop([f"{c}_intl" for c in overlap])
    if ages is not None:
        table = table.join(ages, on="player_id", how="left")
    return table.sort([YEAR_COLUMN, PICK_COLUMN])


@dataclass
class RealPipelineResult:
    n_drafted: int
    n_resolved: int
    hurdle_cv_spearman: float = float("nan")
    hurdle_holdout_spearman: float = float("nan")
    baseline_holdout_spearman: float = float("nan")
    longevity_concordance: float = float("nan")
    holdout_years: tuple[int, ...] = ()
    model_version: str = ""
    summary_path: str = ""


def _evaluate_longevity(
    dev: pl.DataFrame, holdout: pl.DataFrame, feature_cols: list[str]
) -> float:
    """Cox PH career-length model over players who reached the NBA; holdout concordance (or NaN).

    Censoring-aware (career_ended = event). Returns NaN for degenerate cases (too few players or a
    single event class), which is common on tiny synthetic data.
    """
    from nba_draft.models.survival import CoxSurvivalModel, concordance
    from nba_draft.validation import FoldPreprocessor

    dev_play = dev.filter(pl.col("career_seasons") >= 1)
    ho_play = holdout.filter(pl.col("career_seasons") >= 1)
    if dev_play.height < 10 or ho_play.height < 3 or dev_play["career_ended"].n_unique() < 2:
        return float("nan")
    pp = FoldPreprocessor(feature_cols).fit(dev_play)
    dev_t = pp.transform(dev_play).with_columns(
        pl.col("career_seasons").cast(pl.Float64).alias("duration"),
        pl.col("career_ended").cast(pl.Int64).alias("event"),
    )
    try:
        cox = CoxSurvivalModel(penalizer=0.5).fit(
            dev_t, feature_cols=feature_cols, duration_col="duration", event_col="event"
        )
        risk = cox.predict_risk(pp.transform(ho_play))
        return concordance(
            ho_play["career_seasons"].to_numpy().astype(float),
            ho_play["career_ended"].to_numpy().astype(float),
            risk,
        )
    except Exception as exc:  # noqa: BLE001 - degenerate Cox fits should not break the pipeline
        log.warning("longevity Cox failed: %s", exc)
        return float("nan")


def evaluate_real_models(
    table: pl.DataFrame,
    feature_cols: list[str],
    *,
    output_root: str | Path = "artifacts/real_pipeline",
    model_root: str | Path = "artifacts/models",
    min_train_years: int = 4,
    n_holdout_years: int = 2,
    tune: bool = True,
    n_trials: int = 30,
    tracking_enabled: bool = False,
) -> RealPipelineResult:
    """Hurdle modeling + honest evaluation on a real modeling table (offline-testable).

    Ranks ALL resolved prospects by unconditional EV (reach × impact + replacement), evaluated via
    temporal walk-forward on the DEV set and — for the unbiased headline — on an untouchable
    HOLDOUT of the most-recent classes that is never seen during tuning or CV. The draft-position
    baseline is scored on the same holdout for context. The fitted HurdleModel is registered.
    """
    import json

    from nba_draft.evaluation.metrics import spearman_corr
    from nba_draft.features import assert_pre_draft_safe
    from nba_draft.mlops.registry import register_model
    from nba_draft.mlops.tracking import ExperimentTracker
    from nba_draft.models import DraftPositionEstimator, gbm_regressor, logistic_classifier
    from nba_draft.models.hurdle import REPLACEMENT_BPM, HurdleModel, realized_value
    from nba_draft.validation import (
        FoldPreprocessor,
        make_data_split,
        walk_forward_hurdle_evaluate,
    )

    out = Path(output_root)
    out.mkdir(parents=True, exist_ok=True)

    # Leakage guard: the chosen feature columns must not include any post-draft outcome column.
    assert_pre_draft_safe(table.select([c for c in feature_cols if c in table.columns]))

    # Resolved prospects only (label fully observed). "Reached" requires a stable peak so the
    # impact head has a real value; everyone else is scored at replacement in `realized`.
    resolved = table.filter(pl.col("resolved")).with_columns(
        (pl.col("reached") & pl.col(TARGET_COLUMN).is_not_null()).cast(pl.Float64).alias("reach01"),
        pl.col(TARGET_COLUMN).alias("impact"),
    )
    resolved = resolved.with_columns(
        pl.Series(
            "realized",
            realized_value(resolved["reach01"].to_numpy(), resolved["impact"].to_numpy()),
        )
    )
    split = make_data_split(resolved, year_col=YEAR_COLUMN, n_holdout_years=n_holdout_years)
    dev, holdout = split.dev, split.holdout
    log.info(
        "resolved=%d dev=%d holdout=%d (years %s) features=%d",
        resolved.height, dev.height, holdout.height, split.holdout_years, len(feature_cols),
    )

    # Tune the impact head on DEV's reached subset only (never the holdout) — fixes selection bias.
    best_params: dict[str, Any] = {}
    dev_reached = dev.filter(pl.col("reach01") > 0.5)
    if tune and dev_reached["draft_year"].n_unique() > min_train_years:
        from nba_draft.models.tuning import tune_estimator

        best_params = tune_estimator(
            dev_reached, feature_cols=feature_cols, target_col="impact", year_col=YEAR_COLUMN,
            build_fn=gbm_regressor,
            param_space={
                "n_estimators": ("int", 100, 600, False),
                "learning_rate": ("float", 0.01, 0.2, True),
                "max_depth": ("int", 2, 5, False),
            },
            preprocessor_factory=lambda: FoldPreprocessor(feature_cols),
            min_train_years=min_train_years, n_trials=n_trials, seed=42,
        ).best_params

    def _impact_factory() -> Any:
        return gbm_regressor(**best_params) if best_params else gbm_regressor()

    with ExperimentTracker(run_name="real-hurdle", enabled=tracking_enabled) as tracker:
        # DEV walk-forward CV of the hurdle ranking.
        hurdle_cv = walk_forward_hurdle_evaluate(
            dev, feature_cols=feature_cols, reached_col="reach01", impact_col="impact",
            realized_col="realized", preprocessor_factory=lambda: FoldPreprocessor(feature_cols),
            reach_factory=logistic_classifier, impact_factory=_impact_factory,
            replacement=REPLACEMENT_BPM, min_train_years=min_train_years,
        )
        hurdle_cv_spearman = float(hurdle_cv.aggregate["spearman_mean"])

        # Fit the final hurdle on ALL dev, evaluate on the untouchable holdout (unbiased headline).
        pp = FoldPreprocessor(feature_cols).fit(dev)
        x_dev = pp.transform_matrix(dev).to_numpy()
        hurdle = HurdleModel(
            reach_factory=logistic_classifier, impact_factory=_impact_factory,
            replacement=REPLACEMENT_BPM,
        ).fit(
            x_dev,
            dev["reach01"].to_numpy().astype(float),
            dev["impact"].to_numpy().astype(float),
        )
        x_ho = pp.transform_matrix(holdout).to_numpy()
        realized_ho = holdout["realized"].to_numpy().astype(float)
        hurdle_holdout = spearman_corr(realized_ho, hurdle.predict(x_ho))

        # Draft-position baseline on the same holdout.
        base = DraftPositionEstimator(0).fit(
            dev.select([PICK_COLUMN]).to_numpy().astype(float),
            dev["realized"].to_numpy().astype(float),
        )
        base_holdout = spearman_corr(
            realized_ho, base.predict(holdout.select([PICK_COLUMN]).to_numpy().astype(float))
        )

        # LONGEVITY (career length) via Cox PH over players who reached the NBA. Censoring-aware:
        # report concordance on the holdout. Degenerate cases (too few / single event class) -> NaN.
        longevity_c = _evaluate_longevity(dev, holdout, feature_cols)

        tracker.log_metrics(
            {
                "hurdle_cv_spearman": hurdle_cv_spearman,
                "hurdle_holdout_spearman": float(hurdle_holdout),
                "baseline_holdout_spearman": float(base_holdout),
                "longevity_concordance": float(longevity_c),
            }
        )

        version = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        register_model(
            hurdle, name="real_hurdle", version=version,
            metrics={
                "hurdle_cv_spearman": hurdle_cv_spearman,
                "hurdle_holdout_spearman": float(hurdle_holdout),
                "baseline_holdout_spearman": float(base_holdout),
            },
            feature_cols=feature_cols, root=model_root,
        )

        summary = {
            "created_at": datetime.now(UTC).isoformat(),
            "n_drafted": table.height,
            "n_resolved": resolved.height,
            "holdout_years": list(split.holdout_years),
            "n_features": len(feature_cols),
            "gbm_tuned_params": best_params,
            "hurdle_cv_spearman": hurdle_cv_spearman,
            "hurdle_holdout_spearman": float(hurdle_holdout),
            "baseline_holdout_spearman": float(base_holdout),
            "longevity_concordance": float(longevity_c),
            "model_version": version,
            "note": (
                "Production ranking = survivorship-robust hurdle (reach × impact). Headline is the "
                "UNTOUCHABLE-HOLDOUT Spearman; tuning/CV never see the holdout. Longevity = Cox PH "
                "career-length concordance on the holdout."
            ),
        }
        summary_path = out / "real_run_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    return RealPipelineResult(
        n_drafted=table.height,
        n_resolved=resolved.height,
        hurdle_cv_spearman=hurdle_cv_spearman,
        hurdle_holdout_spearman=float(hurdle_holdout),
        baseline_holdout_spearman=float(base_holdout),
        longevity_concordance=float(longevity_c),
        holdout_years=split.holdout_years,
        model_version=version,
        summary_path=str(summary_path),
    )


def run_real_pipeline(
    ingester: Any,
    *,
    draft_years: list[int],
    outcome_seasons: list[str],
    cbd_ingester: Any | None = None,
    intl_ingester: Any | None = None,
    with_age: bool = True,
    tune: bool = True,
    n_trials: int = 30,
    output_root: str | Path = "artifacts/real_pipeline",
    model_root: str | Path = "artifacts/models",
    min_train_years: int = 4,
    n_holdout_years: int = 2,
    tracking_enabled: bool = False,
) -> RealPipelineResult:
    """End-to-end on real data: pull → labels → hurdle CV + holdout eval → register.

    Pull/build are env-pending (need a residential IP + `CBD_API_KEY`); the modeling/evaluation is
    delegated to `evaluate_real_models`, which is unit-tested offline. `intl_ingester` (EuroLeague)
    fills pre-draft features for international prospects who lack NCAA data.
    """
    frames = pull_real_frames(
        ingester, draft_years=draft_years, outcome_seasons=outcome_seasons,
        cbd_ingester=cbd_ingester, intl_ingester=intl_ingester,
    )
    ages = pull_player_ages(ingester, frames.draft_history) if with_age else None
    table = build_real_modeling_table(
        frames.draft_history, frames.combine, frames.player_seasons,
        data_through_year=frames.data_through_year, cbd_seasons=frames.cbd_seasons,
        intl_seasons=frames.intl_seasons, ages=ages,
    )
    # Production features fuse the draft-pick consensus WITH public data, used in BOTH hurdle heads.
    # College-named production columns are kept if EITHER NCAA or international data populates them.
    has_production = frames.cbd_seasons is not None or frames.intl_seasons is not None
    feats = (
        [PICK_COLUMN, *COMBINE_FEATURES]
        + (COLLEGE_FEATURE_COLUMNS if has_production else [])
        + (AGE_FEATURE_COLUMNS if ages is not None else [])
    )
    return evaluate_real_models(
        table, feats, output_root=output_root, model_root=model_root,
        min_train_years=min_train_years, n_holdout_years=n_holdout_years,
        tune=tune, n_trials=n_trials, tracking_enabled=tracking_enabled,
    )
