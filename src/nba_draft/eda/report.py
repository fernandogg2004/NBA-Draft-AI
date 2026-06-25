"""Assemble EDA summaries into a single markdown report with actionable findings."""

from __future__ import annotations

import polars as pl

from nba_draft.eda.summaries import (
    assign_bands,
    band_counts,
    binned_relationship,
    feature_target_spearman,
    grouped_target_summary,
    missingness_by_group,
    numeric_summary,
)


def df_to_markdown(df: pl.DataFrame, max_rows: int = 50) -> str:
    """Render a Polars DataFrame as a GitHub-flavored markdown table."""
    if df.is_empty():
        return "_(empty)_"
    head = df.head(max_rows)
    cols = head.columns
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in head.iter_rows():
        cells = ["" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v)) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    if df.height > max_rows:
        lines.append(f"\n_… {df.height - max_rows} more rows_")
    return "\n".join(lines)


def build_eda_report(
    df: pl.DataFrame,
    *,
    features: list[str],
    target: str,
    group_col: str,
    tier_edges: list[float],
    tier_labels: list[str],
    age_col: str = "age",
    is_synthetic: bool = True,
) -> str:
    """Build a markdown EDA report covering the draft problem's key questions."""
    parts: list[str] = ["# EDA Report\n"]
    if is_synthetic:
        parts.append(
            "> ⚠️ **SYNTHETIC data** — for pipeline/tooling verification only. No basketball "
            "conclusions. Re-run on the real master dataset for real findings.\n"
        )
    parts.append(
        "> Computed on the **development set only** (holdout draft classes excluded) to avoid "
        "tuning intuition on the test set.\n"
    )

    parts.append("## 1. Distributions & missingness\n")
    parts.append(df_to_markdown(numeric_summary(df, [*features, target])))

    parts.append("\n\n## 2. Missingness by league (bias check)\n")
    parts.append(
        "_High missingness of advanced metrics for weaker/international leagues is expected; "
        "the imputer must not treat it as poor performance (risk #9)._\n"
    )
    parts.append(df_to_markdown(missingness_by_group(df, features, group_col)))

    parts.append("\n\n## 3. Feature → target rank correlation (actionable ordering signal)\n")
    parts.append(df_to_markdown(feature_target_spearman(df, features, target)))

    parts.append(f"\n\n## 4. {age_col.title()} vs success (binned)\n")
    parts.append(df_to_markdown(binned_relationship(df, age_col, target, n_bins=5)))

    parts.append("\n\n## 5. League level vs success\n")
    parts.append(df_to_markdown(grouped_target_summary(df, group_col, target)))

    parts.append("\n\n## 6. Outcome-tier base rates\n")
    banded = assign_bands(df, target, tier_edges, tier_labels)
    parts.append(df_to_markdown(band_counts(banded, f"{target}_band", order=tier_labels)))

    return "\n".join(parts) + "\n"
