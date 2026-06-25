"""Tests for Phase 4 feature engineering (stateless transforms, learned context, assembler)."""

from __future__ import annotations

import polars as pl
import pytest

from nba_draft.data.fixtures import make_feature_fixture
from nba_draft.features import (
    LeagueSeasonContextModel,
    add_combine_features,
    add_stateless_features,
    assert_pre_draft_safe,
    build_feature_matrix,
    primary_pre_draft_season,
    sequence_features,
)


# --------------------------------------------------------------- stateless transforms
def test_versatility_index_balanced_vs_specialist():
    df = pl.DataFrame(
        {
            "pts_per100": [10.0, 30.0],
            "ast_per100": [10.0, 0.0],
            "reb_per100": [10.0, 0.0],
            "true_shooting": [0.5, 0.5],
            "usage": [0.2, 0.2],
        }
    )
    out = add_stateless_features(df)
    vi = out["versatility_index"].to_list()
    assert vi[0] > vi[1]  # balanced player more versatile than pure scorer
    assert vi[0] == pytest.approx(2 / 3, abs=1e-6)  # equal thirds -> 1 - 3*(1/3)^2


def test_combine_features_sign_normalized_higher_is_better():
    combine = pl.DataFrame(
        {
            "lane_agility_s": [11.0, 12.0],
            "body_fat_pct": [6.0, 9.0],
            "max_vertical_in": [35.0, 30.0],
        }
    )
    out = add_combine_features(combine)
    # faster agility (lower seconds) -> higher score
    assert out["agility_score"][0] > out["agility_score"][1]
    # leaner (lower body fat) -> higher score
    assert out["leanness_score"][0] > out["leanness_score"][1]
    assert out["explosiveness"][0] == 35.0


# --------------------------------------------------------------- sequence features
def test_sequence_features_capture_transfer_dynamics():
    fx = make_feature_fixture()
    seq = sequence_features(fx["prospect_season"]).sort("player_id")
    p1 = seq.filter(pl.col("player_id") == "p1").row(0, named=True)
    # p1 transferred to a tougher schedule (SoS jumped) and kept efficiency
    assert p1["sos_jump"] > 0
    assert p1["efficiency_held_up"] is True
    # p2 efficiency dipped as competition rose
    p2 = seq.filter(pl.col("player_id") == "p2").row(0, named=True)
    assert p2["ts_change"] < 0
    assert p2["efficiency_held_up"] is False


def test_single_season_player_has_null_deltas():
    fx = make_feature_fixture()
    seq = sequence_features(fx["prospect_season"])
    p3 = seq.filter(pl.col("player_id") == "p3").row(0, named=True)
    assert p3["n_pre_draft_seasons"] == 1
    assert p3["sos_jump"] is None
    assert p3["efficiency_held_up"] is None


# --------------------------------------------------------------- learned context model
def test_translation_makes_leagues_comparable():
    fx = make_feature_fixture()
    ps = fx["prospect_season"]
    model = LeagueSeasonContextModel().fit(ps)
    out = model.transform(ps)
    assert "pts_per100_translated" in out.columns
    assert "sos_z" in out.columns
    # reference-league (ncaa) rows: translated values are populated on the reference scale
    ncaa = out.filter(pl.col("league_id") == "ncaa")
    assert ncaa["pts_per100_translated"].drop_nulls().len() == ncaa.height


def test_dynamic_sos_z_centered_within_league_season():
    fx = make_feature_fixture()
    ps = fx["prospect_season"]
    out = LeagueSeasonContextModel().fit(ps).transform(ps)
    # NCAA 2017 has multiple players; their sos_z should be finite and roughly centered
    grp = out.filter((pl.col("league_id") == "ncaa") & (pl.col("season") == 2017))
    zvals = grp["sos_z"].drop_nulls().to_list()
    assert len(zvals) >= 2
    assert abs(sum(zvals) / len(zvals)) < 1e-6  # mean ~ 0 within the (league, season) cell


def test_context_model_is_leakage_safe_fit_on_train_only():
    fx = make_feature_fixture()
    ps = fx["prospect_season"]
    train = ps.filter(pl.col("draft_year") < 2018)   # only p6
    test = ps.filter(pl.col("draft_year") == 2018)
    model = LeagueSeasonContextModel().fit(train)
    out = model.transform(test)
    # transform must succeed using only train-derived baselines (global fallback), no errors
    assert "pts_per100_translated" in out.columns
    assert out.height == test.height


def test_context_requires_fit():
    with pytest.raises(RuntimeError):
        LeagueSeasonContextModel().transform(pl.DataFrame({"league_id": ["ncaa"]}))


# --------------------------------------------------------------- assembler + leakage guard
def test_primary_season_is_most_recent():
    fx = make_feature_fixture()
    primary = primary_pre_draft_season(fx["prospect_season"])
    p1 = primary.filter(pl.col("player_id") == "p1").row(0, named=True)
    assert p1["season"] == 2017  # p1's later (high-major) season


def test_build_feature_matrix_one_row_per_prospect():
    fx = make_feature_fixture()
    ps = fx["prospect_season"]
    model = LeagueSeasonContextModel().fit(ps)
    fm = build_feature_matrix(ps, fx["combine"], model, identity=fx["identity"])
    assert fm.height == ps["player_id"].n_unique()
    for col in ("versatility_index", "pts_per100_translated", "sos_z", "sos_jump", "agility_score"):
        assert col in fm.columns
    # international players lack Combine data -> nulls (not zero), to be imputed later
    p4 = fm.filter(pl.col("player_id") == "p4").row(0, named=True)
    assert p4["agility_score"] is None


def test_leakage_guard_rejects_post_draft_columns():
    df = pl.DataFrame({"player_id": ["p1"], "nba_impact": [3.0]})
    with pytest.raises(ValueError):
        assert_pre_draft_safe(df)
