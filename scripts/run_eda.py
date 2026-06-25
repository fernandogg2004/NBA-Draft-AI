"""Generate the Phase 3 EDA report (and plots if matplotlib is installed).

Runs on the DEVELOPMENT set only (holdout draft classes removed). On the synthetic fixture
this verifies the EDA tooling end-to-end; point it at the real master dataset for real findings.

    python scripts/run_eda.py
"""

from __future__ import annotations

from nba_draft.config import load_config
from nba_draft.data.fixtures import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    YEAR_COLUMN,
    make_synthetic_prospects,
)
from nba_draft.eda.report import build_eda_report
from nba_draft.utils.logging import get_logger
from nba_draft.validation.temporal import holdout_split

log = get_logger("run_eda")

# Illustrative tier bands on the (synthetic) impact metric; real bands come from config/targets.
TIER_EDGES = [-1e9, -2.0, 0.0, 3.0, 6.0, 1e9]
TIER_LABELS = ["bust", "rotation", "starter", "all_star", "superstar"]


def main() -> None:
    cfg = load_config()
    df = make_synthetic_prospects(seed=cfg.seed)

    # Restrict to the development set — never EDA the holdout classes.
    dev_idx, _ = holdout_split(df[YEAR_COLUMN].to_numpy(), cfg.validation.n_holdout_years)
    dev = df[dev_idx.tolist()]
    log.info("EDA on development set: %d rows (holdout excluded).", dev.height)

    report = build_eda_report(
        dev,
        features=FEATURE_COLUMNS,
        target=TARGET_COLUMN,
        group_col="league_tier",
        tier_edges=TIER_EDGES,
        tier_labels=TIER_LABELS,
        age_col="age",
        is_synthetic=True,
    )

    out_dir = cfg.path("artifacts") / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "eda_report.md"
    report_path.write_text(report, encoding="utf-8")
    log.info("Wrote %s", report_path)

    try:
        from nba_draft.eda.plots import (
            plot_binned_relationship,
            plot_correlation_heatmap,
            plot_histograms,
        )

        all_cols = [*FEATURE_COLUMNS, TARGET_COLUMN]
        plot_histograms(dev, all_cols, out_dir / "histograms.png")
        plot_correlation_heatmap(dev, all_cols, out_dir / "correlations.png")
        plot_binned_relationship(dev, "age", TARGET_COLUMN, out_dir / "age_vs_target.png")
        log.info("Wrote plots to %s", out_dir)
    except ImportError:
        log.warning("matplotlib not installed (extra 'eda'); skipped plots. Tables written.")


if __name__ == "__main__":
    main()
