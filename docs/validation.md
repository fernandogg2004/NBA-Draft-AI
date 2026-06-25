# Validation protocol (Phase 5)

The single rule everything else serves: **we never train on the future.** The real task is
predicting NBA outcomes for prospects who have not debuted, so all validation is temporal.
This document is the contract Phase 6+ modeling must follow.

## 1. The untouchable test set

`make_data_split(df, n_holdout_years=N)` carves off the **most-recent N draft classes** as a
holdout. It is *never* used for feature design, hyperparameter tuning, model selection, or
EDA — only for a single final evaluation at the very end. Everything else (development set) is
where walk-forward validation happens.

```
[ 2010 ... 2021 | 2022 2023 ]
   development     holdout (locked)
```

## 2. Walk-forward folds

`walk_forward_folds` produces folds where **every training draft year is strictly earlier than
every validation year**. With an expanding window, training grows to include all prior classes;
each fold validates on the next `val_horizon_years`. Every fold self-checks for leakage and
raises `LeakageError` rather than returning a leaky split.

We split on the *group* (draft year), never on rows — all prospects in a class stay together,
because same-class players share era/context that would otherwise leak.

## 3. The fold preprocessing firewall

Any step that **learns parameters from data** must be fit on the train fold only. Stateless
features (`features.assemble_prospect_features`) are computed once upstream; everything learned
is bundled in `FoldPreprocessor` and fit per fold:

```
FoldPreprocessor.fit(train):
    1. LeagueSeasonContextModel  — inter-league translation + dynamic SoS   (train baselines)
    2. LeakageSafeImputer        — comparable-league fill + flags + sd       (train stats)
    3. median backfill           — guarantees no nulls reach the estimator   (train medians)
FoldPreprocessor.transform(val): apply the SAME fitted parameters
```

Fitting any of these on combined train+val would leak validation distributions into training.
Tests assert the preprocessor uses train-only statistics.

## 4. The runner

`walk_forward_evaluate(df, feature_cols, target_col, model_factory, preprocessor_factory, ...)`
ties it together: per fold it fits the preprocessor on train, transforms train+val, fits a fresh
estimator, predicts val, and scores. It refuses to proceed if any null survives preprocessing.
Metrics default to ranking-first (Spearman, top-10 hit rate) plus RMSE — because the product
ranks prospects.

## 5. Reporting

`EvaluationReport` holds per-fold metrics and mean/std aggregates. Phase 7 will add the
draft-position baseline as the comparison every model must beat under this exact protocol.
