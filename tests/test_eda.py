"""Tests for Phase 3 EDA summaries and report assembly."""

from __future__ import annotations

import math

import polars as pl
import pytest

from nba_draft.data.fixtures import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    make_synthetic_prospects,
)
from nba_draft.eda.report import build_eda_report, df_to_markdown
from nba_draft.eda.summaries import (
    assign_bands,
    band_counts,
    binned_relationship,
    feature_target_spearman,
    grouped_target_summary,
    missingness_by_group,
    numeric_summary,
    spearman_correlation_matrix,
)


def test_numeric_summary_reports_missingness():
    df = pl.DataFrame({"a": [1.0, 2.0, None, 4.0]})
    out = numeric_summary(df, ["a"])
    row = out.to_dicts()[0]
    assert row["n"] == 4
    assert row["n_missing"] == 1
    assert row["missing_rate"] == pytest.approx(0.25)
    assert row["median"] == pytest.approx(2.0)


def test_missingness_by_group_flags_international_gaps():
    # a frame with league + an advanced column that's null for intl
    df = pl.DataFrame(
        {
            "league_id": ["ncaa", "ncaa", "euroleague", "euroleague"],
            "bpm_college": [9.0, 8.0, None, None],
        }
    )
    out = missingness_by_group(df, ["bpm_college"], "league_id").sort("league_id")
    rates = dict(zip(out["league_id"], out["bpm_college"], strict=True))
    assert rates["euroleague"] == pytest.approx(1.0)  # fully missing for intl
    assert rates["ncaa"] == pytest.approx(0.0)


def test_spearman_matrix_diagonal_is_one():
    df = make_synthetic_prospects(seed=3)
    corr = spearman_correlation_matrix(df, FEATURE_COLUMNS)
    cols = corr["column"].to_list()
    mat = corr.select(cols).to_numpy()
    for i in range(len(cols)):
        assert mat[i, i] == pytest.approx(1.0)


def test_feature_target_spearman_ranks_by_abs_strength():
    df = make_synthetic_prospects(seed=3)
    out = feature_target_spearman(df, FEATURE_COLUMNS, TARGET_COLUMN)
    vals = out["spearman_with_target"].to_list()
    rhos = [abs(v) for v in vals if v is not None and not math.isnan(v)]
    assert rhos == sorted(rhos, reverse=True)  # sorted by |rho| desc


def test_assign_bands_and_base_rates():
    df = pl.DataFrame({"x": [-5.0, -1.0, 1.0, 4.0, 7.0, None]})
    edges = [-1e9, -2.0, 0.0, 3.0, 6.0, 1e9]
    labels = ["bust", "rotation", "starter", "all_star", "superstar"]
    banded = assign_bands(df, "x", edges, labels)
    bands = banded["x_band"].to_list()
    assert bands == ["bust", "rotation", "starter", "all_star", "superstar", None]
    counts = band_counts(banded.drop_nulls("x_band"), "x_band", order=labels)
    # ordered per the tier order, base rates sum to 1
    assert counts["x_band"].to_list() == labels
    assert counts["base_rate"].sum() == pytest.approx(1.0)


def test_binned_relationship_shapes():
    df = make_synthetic_prospects(seed=3)
    rel = binned_relationship(df, "age", TARGET_COLUMN, n_bins=4)
    assert rel.height <= 4
    assert set(rel.columns) == {"bin_lo", "bin_hi", "n", "mean_target"}


def test_grouped_target_summary():
    df = make_synthetic_prospects(seed=3)
    out = grouped_target_summary(df, "league_tier", TARGET_COLUMN)
    assert "mean_target" in out.columns
    assert out.height == df["league_tier"].n_unique()


def test_assign_bands_validates_labels():
    with pytest.raises(ValueError):
        assign_bands(pl.DataFrame({"x": [1.0]}), "x", [0.0, 1.0, 2.0], ["only_one"])


def test_df_to_markdown_renders_table():
    md = df_to_markdown(pl.DataFrame({"a": [1], "b": ["x"]}))
    assert md.startswith("| a | b |")
    assert "| 1 | x |" in md


def test_build_eda_report_smoke():
    df = make_synthetic_prospects(seed=3)
    report = build_eda_report(
        df,
        features=FEATURE_COLUMNS,
        target=TARGET_COLUMN,
        group_col="league_tier",
        tier_edges=[-1e9, -2.0, 0.0, 3.0, 6.0, 1e9],
        tier_labels=["bust", "rotation", "starter", "all_star", "superstar"],
    )
    assert "# EDA Report" in report
    assert "SYNTHETIC" in report
    assert "development set only" in report
    for section in ("Distributions", "Missingness by league", "base rates"):
        assert section in report
