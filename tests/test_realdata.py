"""Tests for the real-data join logic + hurdle modeling (offline; pull step exercised elsewhere)."""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from nba_draft.realdata.build import (
    FEATURE_COLUMNS,
    PICK_COLUMN,
    TARGET_COLUMN,
    build_real_modeling_table,
    evaluate_real_models,
)


def _frames():
    draft_history = pl.DataFrame(
        {
            "player_id": [1, 2, 3],
            "full_name": ["Reached Star", "Bust", "Censored Recent"],
            "draft_year": [2018, 2018, 2022],
            "draft_pick": [3, 25, 5],
        }
    )
    combine = pl.DataFrame(
        {
            "player_id": [1, 3],   # player 2 never attended the Combine
            "wingspan_in": [84.0, 86.0],
            "standing_reach_in": [110.0, 112.0],
            "max_vertical_in": [35.0, 33.0],
            "lane_agility_s": [11.0, 11.5],
            "body_fat_pct": [6.0, 7.0],
        }
    )
    # player 1 plays 4 strong seasons (resolved, reached); player 3 has one recent season (censored)
    player_seasons = pl.DataFrame(
        {
            "player_id": [1, 1, 1, 1, 3],
            "season": ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23"],
            "minutes": [2000.0, 2100.0, 2200.0, 2300.0, 1500.0],
            "ebpm": [2.0, 4.0, 5.0, 3.0, 1.0],
            "vorp": [1.0, 2.0, 3.0, 2.0, 0.8],
        }
    )
    return draft_history, combine, player_seasons


def test_real_table_has_features_labels_and_resolved_flag():
    dh, combine, ps = _frames()
    table = build_real_modeling_table(dh, combine, ps, data_through_year=2023)

    assert table.height == 3
    for col in (PICK_COLUMN, "draft_year", "reached", TARGET_COLUMN, "resolved", *FEATURE_COLUMNS):
        assert col in table.columns

    p1 = table.filter(pl.col("player_id") == 1).row(0, named=True)
    assert p1["reached"] is True and p1["resolved"] is True
    assert p1["peak_impact"] is not None
    assert p1["wingspan_in"] == 84.0

    # player 2: drafted, never played -> bust, no Combine features (null, not zero)
    p2 = table.filter(pl.col("player_id") == 2).row(0, named=True)
    assert p2["reached"] is False
    assert p2["wingspan_in"] is None

    # player 3: 2022 class, window not complete by 2023 -> censored (resolved False)
    p3 = table.filter(pl.col("player_id") == 3).row(0, named=True)
    assert p3["resolved"] is False


def test_trainable_subset_excludes_bust_and_censored():
    dh, combine, ps = _frames()
    table = build_real_modeling_table(dh, combine, ps, data_through_year=2023)
    trainable = table.filter(
        pl.col("resolved") & pl.col("reached") & pl.col(TARGET_COLUMN).is_not_null()
    )
    assert trainable["player_id"].to_list() == [1]  # only the resolved, reached star


def _synthetic_real_table(seed: int = 0) -> pl.DataFrame:
    """A resolved real-shaped modeling table: features + reached/peak_impact labels across years."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    pid = 0
    for year in range(2012, 2021):  # 9 classes -> dev folds + 2-year holdout
        for _ in range(40):
            skill = float(rng.normal())
            reached = skill + rng.normal(scale=0.4) > -0.2
            peak = float(2.0 * skill + rng.normal(scale=0.5)) if reached else None
            pick_score = -skill + rng.normal(scale=0.6)
            # career length grows with skill; ~70% of careers are observed-ended (rest censored)
            career = int(np.clip(4 + 3 * skill + rng.normal(scale=1.0), 1, 18)) if reached else 0
            rows.append(
                {
                    "player_id": pid,
                    "draft_year": year,
                    "draft_pick": 1,  # filled below by rank within year
                    "_pick_score": pick_score,
                    "reached": reached,
                    "peak_impact": peak,
                    "resolved": True,
                    "career_seasons": career,
                    "career_ended": bool(reached and rng.random() < 0.7),
                    "f_skill": skill + float(rng.normal(scale=0.3)),
                    "f_noise": float(rng.normal()),
                }
            )
            pid += 1
    df = pl.DataFrame(rows)
    # draft pick = rank by pick_score within each draft year (best score -> pick 1)
    df = df.with_columns(
        (pl.col("_pick_score").rank("ordinal").over("draft_year")).cast(pl.Int64).alias("draft_pick")
    ).drop("_pick_score")
    return df


def test_evaluate_real_models_hurdle_holdout(tmp_path):
    table = _synthetic_real_table(seed=1)
    feats = [PICK_COLUMN, "f_skill", "f_noise"]
    result = evaluate_real_models(
        table, feats, output_root=tmp_path / "rp", model_root=tmp_path / "models",
        min_train_years=3, n_holdout_years=2, tune=False,
    )
    # holdout reserved (2 most recent classes) and never part of dev
    assert result.holdout_years == (2019, 2020)
    assert result.n_resolved == table.height
    # the hurdle ranking carries real signal on the holdout; baseline is finite
    assert np.isfinite(result.hurdle_holdout_spearman)
    assert result.hurdle_holdout_spearman > 0.2
    assert np.isfinite(result.baseline_holdout_spearman)
    # longevity (Cox) concordance is computed on the holdout and beats chance
    assert np.isfinite(result.longevity_concordance)
    assert result.longevity_concordance > 0.5
    # summary written + model registered
    summary = json.loads((tmp_path / "rp" / "real_run_summary.json").read_text())
    assert summary["holdout_years"] == [2019, 2020]
    from nba_draft.mlops.registry import load_model
    model = load_model("real_hurdle", "latest", root=tmp_path / "models")
    assert hasattr(model, "predict")
