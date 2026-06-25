"""Tests for Phase 2 cleaning & integration."""

from __future__ import annotations

import json

import polars as pl
import pytest

from nba_draft.cleaning import (
    LeakageSafeImputer,
    name_match_key,
    normalize_league,
    normalize_name,
    resolve_entities,
)
from nba_draft.cleaning.master import build_master
from nba_draft.cleaning.schema import impute_sd_name, missing_flag_name
from nba_draft.data.fixtures import make_multisource_fixture


# ------------------------------------------------------------------ normalization
def test_normalize_name_folds_accents_and_order():
    assert normalize_name("Luka Dončić") == "luka doncic"
    assert normalize_name("Young, Trae") == "trae young"
    assert normalize_name("D'Angelo Russell") == "d angelo russell"


def test_match_key_strips_generational_suffix():
    assert name_match_key("Marvin Bagley III") == "marvin bagley"
    assert name_match_key("Jaren Jackson Jr.") == "jaren jackson"
    # suffix-stripped key matches the plain name
    assert name_match_key("Marvin Bagley III") == name_match_key("Marvin Bagley")


def test_normalize_league_maps_aliases():
    assert normalize_league("NCAA Division I") == "ncaa"
    assert normalize_league("G-League") == "gleague"
    assert normalize_league("EuroLega") == "euroleague"
    assert normalize_league("Some Random League") is None


# ------------------------------------------------------------------ entity resolution
def test_resolution_unifies_name_variants_across_sources():
    frames = make_multisource_fixture()
    res = resolve_entities(frames)
    assert res.n_entities == 6  # Trae, Bagley, Ayton, Wendell, Luka, Bogdan

    # Trae Young appears in college + combine under different spellings -> same id
    college = res.frames["college_stats"]
    combine = res.frames["combine"]
    trae_college = college.filter(pl.col("full_name") == "Young, Trae")["player_id"][0]
    trae_combine = combine.filter(pl.col("full_name") == "Trae Young")["player_id"][0]
    assert trae_college == trae_combine

    # Suffix variant links too
    bagley_c = college.filter(pl.col("full_name") == "Marvin Bagley III")["player_id"][0]
    bagley_k = combine.filter(pl.col("full_name") == "Marvin Bagley")["player_id"][0]
    assert bagley_c == bagley_k


def test_resolution_is_deterministic():
    frames = make_multisource_fixture()
    a = resolve_entities(frames).frames["college_stats"]["player_id"].to_list()
    b = resolve_entities(frames).frames["college_stats"]["player_id"].to_list()
    assert a == b


def test_fuzzy_merge_respects_birthdate_guard():
    frames = {
        "a": pl.DataFrame(
            {"full_name": ["Jon Smith"], "draft_year": [2020], "birth_date": ["2000-01-01"]}
        ),
        "b": pl.DataFrame(
            {"full_name": ["Jonn Smith"], "draft_year": [2020], "birth_date": ["1999-05-05"]}
        ),
    }
    # near-identical names but incompatible birthdates -> must NOT merge
    res = resolve_entities(frames)
    assert res.n_entities == 2


def test_different_draft_years_never_merge():
    frames = {
        "a": pl.DataFrame({"full_name": ["Chris Paul"], "draft_year": [2005]}),
        "b": pl.DataFrame({"full_name": ["Chris Paul"], "draft_year": [2018]}),
    }
    assert resolve_entities(frames, birth_date_col=None).n_entities == 2


# ------------------------------------------------------------------ imputation (leakage-safe)
def _impute_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "league_id": ["ncaa", "ncaa", "euroleague", "euroleague"],
            "true_shooting": [0.60, 0.50, None, 0.55],  # one missing in euroleague
            "usage": [0.30, 0.20, 0.25, None],
        }
    )


def test_imputer_fills_from_comparable_league_and_flags():
    df = _impute_frame()
    imp = LeakageSafeImputer(columns=("true_shooting", "usage"), group_col="league_id")
    out = imp.fit_transform(df)
    # euroleague true_shooting was null -> filled with euroleague mean (only 0.55 observed)
    filled = out["true_shooting"].to_list()
    assert filled[2] == pytest.approx(0.55)
    flags = out[missing_flag_name("true_shooting")].to_list()
    assert flags == [False, False, True, False]
    # observed cells carry zero imputation uncertainty
    sds = out[impute_sd_name("true_shooting")].to_list()
    assert sds[0] == 0.0 and sds[1] == 0.0


def test_imputer_is_leakage_safe_uses_only_train_stats():
    train = pl.DataFrame({"league_id": ["ncaa", "ncaa"], "usage": [0.40, 0.20]})  # mean 0.30
    test = pl.DataFrame({"league_id": ["ncaa"], "usage": [None]})
    imp = LeakageSafeImputer(columns=("usage",), group_col="league_id").fit(train)
    out = imp.transform(test)
    # filled from TRAIN mean (0.30), not from anything in the test frame
    assert out["usage"][0] == pytest.approx(0.30)
    assert out[missing_flag_name("usage")][0] is True


def test_imputer_falls_back_to_global_for_unknown_group():
    train = pl.DataFrame({"league_id": ["ncaa", "ncaa"], "usage": [0.40, 0.20]})
    test = pl.DataFrame({"league_id": ["nbl"], "usage": [None]})  # unseen group
    out = LeakageSafeImputer(columns=("usage",), group_col="league_id").fit(train).transform(test)
    assert out["usage"][0] == pytest.approx(0.30)  # global train mean


def test_imputer_requires_fit():
    with pytest.raises(RuntimeError):
        LeakageSafeImputer(columns=("usage",)).transform(pl.DataFrame({"usage": [None]}))


# ------------------------------------------------------------------ master dataset
def test_build_master_produces_versioned_tables(tmp_path):
    frames = make_multisource_fixture()
    season = {"college_stats": frames["college_stats"], "intl_stats": frames["intl_stats"]}
    master = build_master(season, {"combine": frames["combine"]}, output_root=tmp_path)

    # three tables + manifest written under the version dir
    assert (master.root / "manifest.json").exists()
    assert (master.root / "identity.parquet").exists()
    assert master.identity.height == 6

    # international rows keep advanced metrics as NULL (not zero) and flagged imputed-eligible
    luka = master.prospect_season.filter(pl.col("full_name") == "Luka Dončić")
    assert luka["true_shooting"][0] is None
    assert luka[missing_flag_name("true_shooting")][0] is True

    # combine missing measurement stays null + flagged
    bagley = master.combine.filter(pl.col("full_name") == "Marvin Bagley")
    assert bagley["lane_agility_s"][0] is None
    assert bagley[missing_flag_name("lane_agility_s")][0] is True

    # league normalized to canonical id
    assert set(master.prospect_season["league_id"].to_list()) == {"ncaa", "euroleague"}


def test_build_master_version_is_deterministic(tmp_path):
    frames = make_multisource_fixture()
    season = {"college_stats": frames["college_stats"], "intl_stats": frames["intl_stats"]}
    v1 = build_master(season, {"combine": frames["combine"]}, output_root=tmp_path / "a").version
    v2 = build_master(season, {"combine": frames["combine"]}, output_root=tmp_path / "b").version
    assert v1 == v2

    manifest = json.loads((tmp_path / "a" / v1 / "manifest.json").read_text())
    assert manifest["n_entities"] == 6
    assert "leakage-safe" in manifest["imputation"]
