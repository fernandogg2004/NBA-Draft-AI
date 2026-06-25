"""Optional plotting layer (matplotlib, Agg backend). Requires the ``eda`` extra.

Kept separate and lazily-imported so the core EDA + test suite never depend on matplotlib.
Each function writes a PNG and returns its path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nba_draft.eda.summaries import binned_relationship, spearman_correlation_matrix


def _ensure_mpl() -> Any:
    import matplotlib

    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    return plt


def plot_histograms(df: pl.DataFrame, columns: list[str], out: str | Path) -> Path:
    plt = _ensure_mpl()
    cols = [c for c in columns if c in df.columns]
    ncols = min(3, len(cols)) or 1
    nrows = (len(cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    for ax, c in zip(axes.flat, cols, strict=False):
        vals = df[c].drop_nulls().to_numpy()
        ax.hist(vals, bins=20, color="#4C72B0")
        ax.set_title(c)
    for ax in list(axes.flat)[len(cols):]:
        ax.axis("off")
    fig.tight_layout()
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_correlation_heatmap(df: pl.DataFrame, columns: list[str], out: str | Path) -> Path:
    plt = _ensure_mpl()
    corr = spearman_correlation_matrix(df, columns)
    cols = corr["column"].to_list()
    mat = corr.select(cols).to_numpy()
    fig, ax = plt.subplots(figsize=(1 + len(cols), 1 + len(cols)))
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(cols)), cols, rotation=90)
    ax.set_yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_binned_relationship(
    df: pl.DataFrame, feature: str, target: str, out: str | Path, n_bins: int = 5
) -> Path:
    plt = _ensure_mpl()
    rel = binned_relationship(df, feature, target, n_bins=n_bins)
    centers = ((rel["bin_lo"] + rel["bin_hi"]) / 2).to_numpy()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(centers, rel["mean_target"].to_numpy(), marker="o", color="#C44E52")
    ax.set_xlabel(feature)
    ax.set_ylabel(f"mean {target}")
    ax.set_title(f"{feature} vs {target}")
    fig.tight_layout()
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
