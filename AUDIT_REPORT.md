# 🔍 Audit Report — NBA Draft AI

> Completeness, dead-code, manual-action, functional-correctness, and ML-integrity audit per
> `AUDIT.md`. Every finding is verified against source (`file:line`), not against
> `PROJECT_STATE.md`. Runnable checks were executed; the real (nba_api + `CBD_API_KEY`) path was
> audited statically and marked **verification pending (environment)**. **No code was modified.**
> Audit date: 2026-06-26.

---

## 1. Executive summary

**Traffic light: 🟡 YELLOW** (functional and high-quality engineering; a few real
methodological/wiring gaps and stale claims).

The repo is genuinely functional: **162 tests pass, `ruff` and `mypy --strict` are clean, and the
synthetic pipeline runs end-to-end** (reproduced in §8). The **per-fold anti-leakage machinery is
real, not just declared** — preprocessing/imputation/context-model are provably fit on the train
fold only (`runner.py:127-136`, `pipeline.py:47-58`), and the temporal split guards are invoked and
tested. Code hygiene is excellent: **no TODO/FIXME/stub/NotImplementedError in `src`, no leaked
absolute paths**, 91% test coverage. However, the audit surfaced several substantive findings the
documentation hides: the **hurdle / anti-survivorship design (the project's headline defense) is
implemented as components but never wired into a ranking pipeline**; **no automated pipeline ever
evaluates on the "untouchable holdout"**; the **real pipeline tunes and reports on the same folds**
(optimism bias on the headline 0.336); **`params.yaml` is not read by any code**; and a **stated
gotcha is now false** (pyarrow *is* installed, and the dashboard depends on it via an undeclared
transitive dep). None are Critical; two are High.

---

## 2. Findings by category & severity

### A) Completeness & integration

**🔴 A1 — High — The hurdle / anti-survivorship structure is not wired end-to-end.**
The central methodological defense (`EV = P(reach)·E(impact|reach) + (1−P)·replacement`) exists only
as isolated parts:
- `unconditional_value` (`src/nba_draft/targets/definitions.py`) is **called only in
  `tests/test_targets.py:132`** — never in any pipeline/script/service.
- The reach classifiers `logistic_classifier`/`gbm_classifier` (`src/nba_draft/models/zoo.py:106,114`)
  are used only in the `run_evaluation.py:100` calibration *demo* and tests.
- The real pipeline computes the `reached` label (`src/nba_draft/realdata/build.py:124`) but trains
  **only the conditional impact regressor** on `trainable` (reached players)
  (`build.py:205,273`) and ranks by that. P(reach) is never trained, never combined.

*Why it matters:* the production ranking is *conditional on reaching* — exactly the survivorship
conditioning the hurdle was designed to remove for the final ranking. The components all exist; they
are just not assembled.
*Action:* add a reach-probability head to the pipeline and combine via `unconditional_value` to rank
by unconditional EV.

**🔴 A3 — High — Real pipeline tunes and evaluates on the same walk-forward folds (optimism bias).**
In `src/nba_draft/realdata/build.py`, `tune_estimator(...)` selects GBM hyperparameters by maximizing
the walk-forward metric on `trainable` (`build.py:229`), then `compare_models(...)` reports the
`gbm_tuned` Spearman on the **same** `trainable` folds (`build.py:260`). No nested CV, no held-out
test.
*Why it matters:* the headline real result (`gbm_tuned 0.336`) is measured on the folds its
hyperparameters were chosen to optimize → upward-biased. The `models/tuning.py:5` docstring even
claims "the untouchable holdout is reserved for the single final evaluation" — but no such evaluation
exists (see A2).
*Action:* reserve a final holdout (or nested CV) and report the tuned model there.

**🟠 A2 — Medium — The "untouchable holdout" is never used for a final evaluation.**
`make_data_split` produces `dev`/`holdout`, but the holdout is consumed only as (a) the drift "new
class" (`mlops/pipeline.py:142`) and (b) the demo prospect pool to rank (`service/board.py:155`). No
pipeline ever scores a trained model on it. `run_real_pipeline` doesn't call `make_data_split` at all.
*Action:* add a final holdout evaluation step, or correct the docs that claim one happens.

**🟠 A4 — Medium — `params.yaml` is not read by any code.**
The only reference is a docstring (`src/nba_draft/mlops/__init__.py:10`); the pipeline reads
`config/config.yaml` via `load_config`, and `model.alpha` is hardcoded as `ridge_regressor(1.0)`. So
`dvc.yaml`'s `params:` tracking is cosmetic — editing `params.yaml` changes nothing.
*Action:* load `params.yaml` in `run_pipeline` (single source of truth) or delete it and document
`config/*.yaml` as authoritative.

**🟡 A5 — Low — Config keys defined/validated but never consumed.**
`evaluation.top_k` (`config.py:44`, validated at `:48` but no consumer; `default_metrics` hardcodes
top-10), `horizon.include_peak` (`config.py:31`), `validation.draft_year_column` (`config.py:35`),
and `targets.horizon.secondary_window_years` (`targets/definitions.py:44`, the "secondary 1–7yr
window" is documented but never computed).
*Action:* wire or remove; removes false config surface.

**🟡 A6 — Low — The real path has no leakage guard.**
`assert_pre_draft_safe` is invoked only on the synthetic feature path (`assembler.py:86,108`);
`realdata/build.py` builds its modeling table manually without it. No actual leak today (the `feats`
list excludes target columns), but there is no automated guard against a future mistake.
*Action:* call `assert_pre_draft_safe` on the real feature matrix too.

### B) Dead / unused code
See the dedicated table in §3.

### C) Pending work / stubs — ✅ essentially clean
- **No** `TODO`/`FIXME`/`XXX`/`HACK`/`NotImplementedError`/`placeholder`/`stub` in `src` (the only
  match is sklearn's `DummyRegressor`, `zoo.py:22` — legitimate).
- **No** absolute machine paths leaking into code (all `C:\Users` hits are in `AUDIT.md`).
- Half-built features (documented but not implemented): the **secondary 1–7yr window** (A5), the
  **hurdle assembly** (A1), and **longevity/survival** (orphaned — see §3). These are the real
  "pending work," surfaced by integration analysis rather than text markers.

### D) Manual actions
See §4.

### E) Functional correctness — ✅ strong
- All runnable checks pass (see §8). The synthetic pipeline runs end-to-end with no manual editing.
- **Test-quality sample (meaningful assertions, not vacuous):** `test_temporal.py` asserts
  `max(train_years) < min(val_years)` per fold; `test_cleaning.py::test_imputer_is_leakage_safe_uses_only_train_stats`
  asserts the fill equals the *train* mean; `test_uncertainty.py` asserts conformal coverage ≥
  nominal; `test_nba_adapters.py` asserts eBPM is league-centered (~0) and matches PIE ranking. These
  assert real behavior.
- **Coverage 91%.** Critical ML logic well-covered (imputation 88%, temporal 88%, targets 98%,
  entity-resolution 97%, features/learned 96%). Low-coverage modules are the
  **real-path/optional/fallback** code not runnable offline: `realdata/build.py` 46%,
  `mlops/tracking.py` 55% (MLflow-enabled branch), `models/zoo.py` 56% (LightGBM/sklearn fallback +
  classifier branches), `ingestion/college_bb_data.py` 76% — all **verification pending (environment)**.

### F) ML-specific integrity
- **🟢 Per-fold anti-leakage is correct and tested.** `runner.py:127-136` fits the preprocessor on the
  train fold, then transforms train/val separately; `pipeline.py:47-58` fits context model + imputer +
  medians on train only; the temporal splitters raise `LeakageError` on any overlap. This is the most
  important thing and it holds up.
- **🟠 Methodological completeness gaps:** A1 (hurdle), A2 (no final holdout), A3 (tune+report same
  folds).
- **🟠 Stale "gotcha": pyarrow.** PROJECT_STATE claims "pyarrow is NOT installed" — **false**:
  `pyarrow 24.0.0` is present. The dashboard relies on `polars.to_pandas()`
  (`dashboard/streamlit_app.py:86,114`), which works *only* because pyarrow is present **as an
  undeclared transitive dependency** (not in the `app` extra). If a future resolver drops it, the
  dashboard breaks. *Action:* declare `pyarrow` in the `app` extra (or switch those two calls to
  `to_dicts()`/native), and fix the PROJECT_STATE claim. (The survival model's manual pandas build,
  `survival.py:17`, is harmless either way but its stated rationale is now moot.)
- **🟢 MLflow degrades gracefully** (`tracking.py:38-43` no-op when disabled / ImportError) and uses
  the SQLite backend (`:52`) — confirmed.
- **Reproducibility:** seeds set via `set_global_seed` in the synthetic pipeline; master-version
  content hash is deterministic (proven by `test_build_master_version_is_deterministic`). But
  `params.yaml` is cosmetic (A4), and `run_real_pipeline` hardcodes the tuning seed rather than
  threading config — minor.

---

## 3. Dead / unused code

| Symbol | Location | Classification | Recommendation |
|---|---|---|---|
| `_default_fetcher` | `ingestion/http.py:49` | **Dead** (orphaned after refactor to `make_requests_fetcher`; zero callers) | **Remove** |
| `FileCache.has` | `ingestion/cache.py:35` | Unused (no callers) | Remove, or keep as deliberate API |
| `unconditional_value` | `targets/definitions.py` | **Orphaned** — the hurdle combiner, never wired (A1) | **Wire in** (high value) |
| `CoxSurvivalModel.predict_risk`, `concordance` | `models/survival.py:54,63` | **Orphaned** — T5 longevity built + tested, never in any pipeline/service | Wire a longevity path, or mark as library API |
| `logistic_classifier`/`gbm_classifier` | `models/zoo.py:106,114` | Orphaned for production (demo/test only) — part of A1 | Wire into the reach head |
| `LeakageSafeImputer.fit_transform` | `cleaning/imputation.py:111` | Convenience, test-only | Keep or remove |
| `LeagueSeasonContextModel.fit_transform` | `features/learned.py:132` | Convenience, no callers | Keep or remove |
| `IDENTITY_COLUMNS`, `BASIC_COLUMNS` | `cleaning/schema.py:13,22` | Unused constants (other groups are used) | Remove or keep as documentation |
| `pymc` | `pyproject.toml` `[models]` | **Unused dependency** (DEP002; zero imports — Bayesian uses sklearn `BayesianRidge`) | **Remove from extra** (heavy; slows CI) |
| `mapie` | `pyproject.toml` `[explain]` | **Unused dependency** (DEP002; conformal is hand-rolled) | **Remove from extra** |
| `global_importance` | `interpretability/attribution.py:47` | Deliberate API (tested; not surfaced in app) | Keep (consider surfacing in dashboard) |
| `make_feature_fixture`, `is_good_fit` | fixtures / `fit/synergy.py` | Test infra / deliberate API | Keep |

*Note:* deptry's other DEP002 hits (`uvicorn`, `dvc`, `pytest`, `pytest-cov`, `ruff`, `mypy`, `httpx`)
are CLI/runtime/dev tools invoked outside imports — **legitimate, not findings**. Its 184 DEP003 hits
are a false positive (it misreads the package's own `nba_draft` self-imports).

---

## 4. Manual actions detected

| Action | Verdict | How |
|---|---|---|
| `CBD_API_KEY` env var for college data | **Unavoidable** (user secret) | Already env-based ✓ |
| `nba_api` run on a residential IP (cloud IPs banned) | **Unavoidable** (provider constraint) | Documented; cannot be automated |
| **Install extras to get green tests** | **Automatable (docs gap)** — `pip install -e ".[dev]"` (README:261) + `pytest` (README:345) **fails**: ~5 test files need `models`/`explain`/`app` (xgboost, shap, fastapi). CI installs all extras; the README doesn't tell a human to. | Document `pip install -e ".[dev,models,explain,app]"` as the test prerequisite (Medium DX finding) |
| `git init && dvc init` before `dvc repro` | Expected for DVC | Documented; fine |
| Scripts run end-to-end with no code editing | ✅ Confirmed for the synthetic scripts | — |

---

## 5. Improvement ideas

- **Robustness/DX:** declare `pyarrow` in `[app]` (or drop `.to_pandas()`); remove `pymc`/`mapie`
  from extras to cut CI install time; add a `[test]` extra (= dev+models+explain+app) and document it;
  fix the PROJECT_STATE pyarrow claim.
- **Model quality:** implement the hurdle ranking (A1) — likely the single biggest quality lever; add
  a true held-out evaluation (A2/A3) so reported numbers are unbiased; wire `secondary_window_years`
  or remove it.
- **ML rigor:** apply `assert_pre_draft_safe` on the real path (A6); thread seeds/`params.yaml`
  through `run_real_pipeline` for full reproducibility.
- **CI:** cache the heavy install; add a coverage gate; consider running `ruff`+`mypy` as a fast
  separate job from the heavy test job.

---

## 6. Logical next steps (prioritized, effort × impact)

1. **Wire the hurdle** (reach classifier + `unconditional_value`) into `run_real_pipeline` and the
   service — *High impact, Medium effort* (validates A1; the components exist).
2. **Add a final holdout evaluation** and stop tuning-on-eval-folds — *High impact, Low-Medium effort*
   (A2/A3).
3. **Fix dependency/doc drift**: pyarrow declaration + PROJECT_STATE correction; drop `pymc`/`mapie`;
   document the test-extras — *Medium impact, Low effort* (PROJECT_STATE "next steps" omit all of
   these).
4. **Make `params.yaml` real** or remove it — *Medium impact, Low effort* (A4).
5. Remove the genuinely dead `_default_fetcher`/`FileCache.has` and unused constants — *Low impact,
   Low effort*.

*(PROJECT_STATE §8's suggestions — honors source, international features, reach classifier, branch
protection — remain valid; #1 above is essentially their "reach classifier" but is more urgent than
presented because the hurdle is the headline design and currently unexecuted.)*

---

## 7. What else can be implemented

- **Reach classifier in production** closing the hurdle (overlaps #1) — defeats survivorship in the
  final ranking, the system's stated purpose.
- **All-Star/All-NBA honors source** → real top-tier labels (today they fall back to BPM bands).
- **International pre-draft features** (EuroLeague feeds) → coverage for non-NCAA prospects.
- **A real-data service** (point `build_demo_service` at the real master) + a longevity (survival)
  output that is currently orphaned.
- **Repo hardening:** branch protection requiring CI, Dependabot, a coverage badge/gate.

---

## 8. Verification run

| Check | Command | Result |
|---|---|---|
| Tests | `pytest` | ✅ **162 passed** (~24s), offline |
| Lint | `ruff check src tests scripts api dashboard` | ✅ **All checks passed!** |
| Types | `mypy` | ✅ **no issues in 79 files** |
| Synthetic pipeline | `python scripts/run_pipeline.py` | ✅ **Pipeline OK** (master `c21f504ae67d`; ridge 0.665 > baseline 0.545; drift none) |
| Dead code | `vulture src` | leads triaged (§3) |
| Deps | `deptry .` | DEP002 → `pymc`, `mapie` unused; no missing deps (0 DEP001) |
| Coverage | `coverage run -m pytest` | **91%** total; critical ML 88–100% |

**Left unverified — verification pending (environment):** the entire **real path**
(`scripts/run_real_pipeline.py`, `scripts/run_ingest.py`, `scripts/verify_cbd.py`,
`realdata/build.py`, `ingestion/nba_stats.py` + `college_bb_data.py` live calls) requires a residential
IP and `CBD_API_KEY`; audited **statically** only. The PROJECT_STATE real-result claims
(**0.336 vs 0.233**, eBPM–PIE ρ≈0.92, "0/389 trainable null ages") are therefore **not reproduced
here** — and the 0.336 additionally carries the A3 optimism caveat. The MLflow-enabled tracking branch
and the LightGBM/sklearn GBM fallbacks were not exercised (offline suite uses XGBoost + disabled
tracking).
