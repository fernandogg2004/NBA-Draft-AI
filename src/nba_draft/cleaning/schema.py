"""Canonical master-dataset schema and missingness handling.

Column groups make explicit which metrics are commonly absent for international/low-data
prospects (ADVANCED, COMBINE). The integration layer must keep these as ``null`` when not
measured — distinct from a true zero — so the imputer (not the loader) decides how to fill them.
"""

from __future__ import annotations

import polars as pl

# Identity / context columns present for (almost) every prospect.
IDENTITY_COLUMNS: tuple[str, ...] = (
    "player_id",
    "full_name",
    "draft_year",
    "birth_date",
    "position",
)

# Basic box production — generally available from any source, including basic FIBA stats.
BASIC_COLUMNS: tuple[str, ...] = (
    "age",
    "pts_per100",
    "ast_per100",
    "reb_per100",
)

# Advanced metrics — rich for NCAA (KenPom/Torvik-style), often MISSING for international.
ADVANCED_COLUMNS: tuple[str, ...] = (
    "true_shooting",
    "usage",
    "bpm_college",
    "strength_of_schedule",
)

# Draft Combine measurements — missing for prospects who did not test at the Combine.
COMBINE_COLUMNS: tuple[str, ...] = (
    "wingspan_in",
    "standing_reach_in",
    "max_vertical_in",
    "lane_agility_s",
    "body_fat_pct",
)

# Columns that may legitimately be absent and must be imputed (with flags), never zero-filled.
IMPUTABLE_COLUMNS: tuple[str, ...] = ADVANCED_COLUMNS + COMBINE_COLUMNS


def missing_flag_name(column: str) -> str:
    """Flag column marking whether `column` was imputed (True) vs observed (False)."""
    return f"{column}_imputed"


def impute_sd_name(column: str) -> str:
    """Per-cell imputation-uncertainty column (0 for observed cells)."""
    return f"{column}_impute_sd"


def add_missing_flags(
    df: pl.DataFrame,
    columns: tuple[str, ...] = IMPUTABLE_COLUMNS,
) -> pl.DataFrame:
    """Add a boolean ``<col>_imputed`` flag per column, seeded from current null-ness.

    Run at integration time so the flag captures "not measured" BEFORE any imputation. The
    imputer later sets the flag True only for cells it actually fills.
    """
    present = [c for c in columns if c in df.columns]
    return df.with_columns(
        [pl.col(c).is_null().alias(missing_flag_name(c)) for c in present]
    )
