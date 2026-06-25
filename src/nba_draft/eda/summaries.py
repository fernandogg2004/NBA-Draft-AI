"""Tabular EDA primitives (pure functions over Polars frames).

Designed for the draft problem's specific questions: distributions + missingness, rank
correlations (robust to the fat tails), how age and league level relate to success, and base
rates per outcome tier. Rank-based (Spearman) correlation is used throughout because the
targets are heavy-tailed and we care about ordering.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from nba_draft.evaluation.metrics import spearman_corr


def numeric_summary(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Per-column count, missing rate, mean, std, and quartiles.

    Missing rate is reported explicitly so "not measured" is visible (domain risk #9).
    """
    rows: list[dict[str, object]] = []
    n = df.height
    for c in columns:
        if c not in df.columns:
            continue
        s = df[c]
        non_null = s.drop_nulls()
        k = non_null.len()
        rows.append(
            {
                "column": c,
                "n": n,
                "n_missing": n - k,
                "missing_rate": (n - k) / n if n else float("nan"),
                "mean": float(non_null.mean()) if k else float("nan"),  # type: ignore[arg-type]
                "std": float(non_null.std()) if k > 1 else float("nan"),  # type: ignore[arg-type]
                "min": float(non_null.min()) if k else float("nan"),  # type: ignore[arg-type]
                "q25": float(non_null.quantile(0.25)) if k else float("nan"),  # type: ignore[arg-type]
                "median": float(non_null.median()) if k else float("nan"),  # type: ignore[arg-type]
                "q75": float(non_null.quantile(0.75)) if k else float("nan"),  # type: ignore[arg-type]
                "max": float(non_null.max()) if k else float("nan"),  # type: ignore[arg-type]
            }
        )
    return pl.DataFrame(rows)


def missingness_by_group(
    df: pl.DataFrame, columns: list[str], group_col: str
) -> pl.DataFrame:
    """Missing rate of each column within each group — the core bias check.

    Reveals e.g. that advanced metrics are missing far more often for international leagues,
    so the model/imputer must not confound "missing" with "bad" (domain risk #9).
    """
    present = [c for c in columns if c in df.columns and c != group_col]
    agg = [pl.col(c).is_null().mean().alias(c) for c in present]
    return df.group_by(group_col).agg(pl.len().alias("n"), *agg).sort(group_col)


def spearman_correlation_matrix(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Pairwise Spearman correlation, computed on each pair's mutually-complete rows."""
    cols = [c for c in columns if c in df.columns]
    data = {c: df[c].to_numpy().astype(float) for c in cols}
    out: dict[str, list[str] | list[float]] = {"column": cols}
    for cj in cols:
        col_vals: list[float] = []
        for ci in cols:
            xi, xj = data[ci], data[cj]
            mask = ~np.isnan(xi) & ~np.isnan(xj)
            if mask.sum() < 2:
                col_vals.append(float("nan"))
            else:
                col_vals.append(round(spearman_corr(xi[mask], xj[mask]), 4))
        out[cj] = col_vals
    return pl.DataFrame(out)


def feature_target_spearman(
    df: pl.DataFrame, features: list[str], target: str
) -> pl.DataFrame:
    """Spearman correlation of each feature with the target, ranked by absolute strength.

    The actionable "which pre-draft signals order prospects" table.
    """
    if target not in df.columns:
        raise ValueError(f"target {target!r} not in frame.")
    yt = df[target].to_numpy().astype(float)
    rows: list[dict[str, object]] = []
    for f in features:
        if f not in df.columns or f == target:
            continue
        xf = df[f].to_numpy().astype(float)
        mask = ~np.isnan(xf) & ~np.isnan(yt)
        rho = spearman_corr(xf[mask], yt[mask]) if mask.sum() >= 2 else float("nan")
        rows.append({"feature": f, "spearman_with_target": round(rho, 4), "n": int(mask.sum())})
    return (
        pl.DataFrame(rows)
        .with_columns(pl.col("spearman_with_target").abs().alias("abs_rho"))
        .sort("abs_rho", descending=True, nulls_last=True)
        .drop("abs_rho")
    )


def assign_bands(
    df: pl.DataFrame, column: str, edges: list[float], labels: list[str]
) -> pl.DataFrame:
    """Add a categorical band column from numeric `column` using half-open [lo, hi) edges.

    ``len(labels)`` must be ``len(edges) - 1``. Values below the first / above the last edge
    fall into the first / last band respectively.
    """
    if len(labels) != len(edges) - 1:
        raise ValueError("labels must have exactly len(edges)-1 entries.")
    vals = df[column].to_numpy().astype(float)
    # np.digitize on interior edges -> band index in [0, len(labels)-1]
    idx = np.clip(np.digitize(vals, edges[1:-1], right=False), 0, len(labels) - 1)
    band = [labels[i] if not np.isnan(v) else None for v, i in zip(vals, idx, strict=True)]
    return df.with_columns(pl.Series(f"{column}_band", band, dtype=pl.Utf8))


def band_counts(df: pl.DataFrame, band_col: str, order: list[str] | None = None) -> pl.DataFrame:
    """Counts and proportions per band — i.e. base rates for an outcome tiering."""
    counts = df.group_by(band_col).agg(pl.len().alias("count"))
    total = df.height
    counts = counts.with_columns((pl.col("count") / total).alias("base_rate"))
    if order:
        rank = {name: i for i, name in enumerate(order)}
        counts = (
            counts.with_columns(
                pl.col(band_col).replace_strict(rank, default=len(order)).alias("_o")
            )
            .sort("_o")
            .drop("_o")
        )
    else:
        counts = counts.sort(band_col)
    return counts


def grouped_target_summary(
    df: pl.DataFrame, group_col: str, target: str
) -> pl.DataFrame:
    """Mean/median target by a categorical group (e.g. league level vs success)."""
    return (
        df.group_by(group_col)
        .agg(
            pl.len().alias("n"),
            pl.col(target).mean().alias("mean_target"),
            pl.col(target).median().alias("median_target"),
        )
        .sort(group_col)
    )


def binned_relationship(
    df: pl.DataFrame, feature: str, target: str, n_bins: int = 5
) -> pl.DataFrame:
    """Mean target across quantile bins of a numeric feature (e.g. age vs success).

    Uses quantile edges so each bin holds a comparable number of prospects.
    """
    x = df[feature].to_numpy().astype(float)
    mask = ~np.isnan(x)
    if mask.sum() < n_bins:
        raise ValueError(f"Not enough non-null values in {feature!r} for {n_bins} bins.")
    edges = np.unique(np.quantile(x[mask], np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.digitize(x, edges[1:-1], right=False), 0, len(edges) - 2)
    bin_ids = [int(i) if not np.isnan(v) else None for v, i in zip(x, idx, strict=True)]
    binned = df.with_columns(pl.Series("_bin", bin_ids)).drop_nulls("_bin")
    lo = {i: float(edges[i]) for i in range(len(edges) - 1)}
    hi = {i: float(edges[i + 1]) for i in range(len(edges) - 1)}
    return (
        binned.group_by("_bin")
        .agg(pl.len().alias("n"), pl.col(target).mean().alias("mean_target"))
        .sort("_bin")
        .with_columns(
            pl.col("_bin").replace_strict(lo, default=None).alias("bin_lo"),
            pl.col("_bin").replace_strict(hi, default=None).alias("bin_hi"),
        )
        .select("bin_lo", "bin_hi", "n", "mean_target")
    )
