"""Milestone 0 end-to-end smoke run.

Runs the full skeleton on the synthetic fixture:
  1. load + validate config
  2. carve an untouchable temporal holdout
  3. walk-forward validate the draft-position baseline on the development set
  4. report ranking + error metrics per fold and in aggregate

This proves the architecture is wired and leakage-safe BEFORE real data exists. Run with:
    python scripts/reproduce.py
"""

from __future__ import annotations

import numpy as np

from nba_draft.config import load_config
from nba_draft.data.fixtures import (
    PICK_COLUMN,
    TARGET_COLUMN,
    YEAR_COLUMN,
    make_synthetic_prospects,
)
from nba_draft.evaluation.metrics import rmse, spearman_corr, top_k_hit_rate
from nba_draft.models.baseline import DraftPositionBaseline
from nba_draft.utils.logging import get_logger
from nba_draft.utils.seeds import set_global_seed
from nba_draft.validation.temporal import holdout_split, walk_forward_folds

log = get_logger("reproduce")


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg.seed)
    log.info("Config loaded. seed=%d horizon=%d yrs", cfg.seed, cfg.horizon.primary_window_years)

    df = make_synthetic_prospects(seed=cfg.seed)
    log.info("SYNTHETIC fixture: %d prospects across %d draft years (NOT real data).",
             df.height, df[YEAR_COLUMN].n_unique())

    years = df[YEAR_COLUMN].to_numpy()
    picks = df[PICK_COLUMN].to_numpy()
    target = df[TARGET_COLUMN].to_numpy()

    # 1) Untouchable holdout (most-recent classes). Reported but NOT used for any tuning.
    dev_idx, holdout_idx = holdout_split(years, cfg.validation.n_holdout_years)
    log.info("Holdout reserved: dev=%d rows, holdout=%d rows (years %s held out).",
             dev_idx.size, holdout_idx.size,
             sorted(set(years[holdout_idx].tolist())))

    # 2) Walk-forward validation on the development set.
    dev_years = years[dev_idx]
    dev_picks = picks[dev_idx]
    dev_target = target[dev_idx]

    fold_spearman: list[float] = []
    fold_top10: list[float] = []
    fold_rmse: list[float] = []

    for i, fold in enumerate(
        walk_forward_folds(
            dev_years,
            min_train_years=cfg.validation.min_train_years,
            val_horizon_years=cfg.validation.val_horizon_years,
            step_years=cfg.validation.step_years,
            expanding=cfg.validation.expanding,
        )
    ):
        model = DraftPositionBaseline().fit(dev_picks[fold.train_idx], dev_target[fold.train_idx])
        pred = model.predict(dev_picks[fold.val_idx])
        yt = dev_target[fold.val_idx]

        sp = spearman_corr(yt, pred)
        hk = top_k_hit_rate(yt, pred, k=10)
        er = rmse(yt, pred)
        fold_spearman.append(sp)
        fold_top10.append(hk)
        fold_rmse.append(er)
        log.info(
            "fold %d | train years %s -> val %s | spearman=%.3f top10=%.2f rmse=%.3f",
            i, fold.train_years, fold.val_years, sp, hk, er,
        )

    log.info("=" * 70)
    log.info(
        "BASELINE (draft position) aggregate over %d folds | "
        "spearman=%.3f+/-%.3f  top10=%.2f+/-%.2f  rmse=%.3f+/-%.3f",
        len(fold_spearman),
        float(np.mean(fold_spearman)), float(np.std(fold_spearman)),
        float(np.mean(fold_top10)), float(np.std(fold_top10)),
        float(np.mean(fold_rmse)), float(np.std(fold_rmse)),
    )
    log.info("Milestone 0 OK: config -> temporal split -> baseline -> metrics, leakage-guarded.")


if __name__ == "__main__":
    main()
