# NBA Draft AI — Decision-Support System

[![CI](https://github.com/fernandogg2004/NBA-Draft-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/fernandogg2004/NBA-Draft-AI/actions/workflows/ci.yml)

A **human-in-the-loop** decision-support system for NBA draft decisions. For each prospect it:

1. **Projects** future NBA production / trajectory,
2. **Estimates** contribution to winning,
3. **Evaluates fit** with a *specific* roster and system of play, and
4. **Quantifies the uncertainty** of every prediction.

It is a tool to *augment* scouting, not replace it. The guiding principle throughout is
**validity over spectacle**: an honest, well-validated baseline beats a flashy, poorly-evaluated
model. Every design decision that touches a known draft-modeling trap (data leakage,
survivorship bias, opportunity confound, era effects, small samples, fat tails) is handled
explicitly and flagged in code.

---

## Table of contents

- [Headline result](#headline-result)
- [How it works](#how-it-works)
- [Data](#data)
- [What the model predicts](#what-the-model-predicts)
- [The fit module](#the-fit-module)
- [Uncertainty & interpretability](#uncertainty--interpretability)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Quality & testing](#quality--testing)
- [Limitations & honest caveats](#limitations--honest-caveats)
- [Documentation](#documentation)

---

## Headline result

On **real data** (2011–2020 draft classes, 389 trainable players), the **production model** — a
tuned gradient-boosting model that fuses the **draft-position consensus** with **public pre-draft
data** (college production + Combine measurements + age-at-draft) — **beats the draft-position
baseline** at ranking prospects by realized NBA impact:

| model | features | Spearman ↑ | RMSE ↓ |
|---|---|---|---|
| **production (tuned GBM)** | draft pick + college + age + combine | **0.336** | **0.697** |
| production (ridge) | draft pick + college + age + combine | 0.314 | 0.704 |
| data-only (tuned GBM) | college + age + combine (no pick) | 0.302 | 0.704 |
| data-only (ridge) | college + age + combine (no pick) | 0.302 | 0.709 |
| **draft-position baseline** | draft pick only | 0.233 | 0.700 |

The progression is the story:

> **consensus alone (0.233) → public data alone (0.302) → consensus + data (0.336)**

A model built only from public box data already **matches** the collective judgment of every NBA
front office; fusing that data **with** the draft consensus beats either source alone. Age was
the single most valuable added feature, exactly as draft-modeling literature predicts.

Target = peak impact (eBPM) over the first 4 NBA seasons, *conditional on reaching* a rotation
role. Evaluation is strictly temporal (train on older classes, test on newer ones). These numbers
are **indicative, not precise** — see [caveats](#limitations--honest-caveats).

---

## How it works

The system walks the full ML lifecycle. The hard part of draft modeling isn't the model — it's
avoiding the inferential traps that make results look great and be worthless. So the architecture
is built defense-first.

### 1. Strictly temporal validation (anti-leakage core)

The real task is predicting the future of players who haven't debuted, so **all validation is
temporal**:

- `holdout_split` reserves the most-recent draft classes as an **untouchable test set** (never
  used for tuning, feature design, or EDA).
- `walk_forward_folds` yields folds where **every training draft year is strictly earlier than
  every validation year**; each fold self-checks and raises rather than returning a leaky split.
- Splitting is by **draft year (group)**, never by row — same-class players share era/context
  that would otherwise leak.

### 2. The leakage firewall (`FoldPreprocessor`)

Features split into two classes, and the split *is* the defense:

- **Stateless** features (a player's own pre-draft stats) are leakage-safe by construction.
- **Learned** steps — inter-league translation, dynamic strength-of-schedule, comparable-league
  imputation, scaling — are bundled into a `FoldPreprocessor` that is **fit on the training fold
  only** and applied to validation/test. Fitting any of these on combined data would leak the
  future.

A guard (`assert_pre_draft_safe`) rejects any known post-draft column that tries to enter the
feature matrix.

### 3. The hurdle target (anti-survivorship)

We only have NBA stats for players who actually played, so modeling "NBA impact" naively
conditions on the outcome. The spine is a two-part / Heckman-style structure:

```
expected value = P(reach role) · E(impact | reached) + (1 − P(reach role)) · replacement_level
```

`P(reach)` is fit over **all** prospects (it learns who washes out); the conditional impact is fit
only on players with enough minutes for a stable estimate, and labeled honestly as conditional.

### 4. Feature engineering

- **Stateless**: age- and possession-adjusted production, true shooting, usage, versatility /
  archetype indicators, year-over-year deltas, and **Transfer-Portal/NIL-era** signals
  (conference jump, role change, whether efficiency held up as competition rose).
- **Learned (fold-fit)**: inter-league translation factors (so a EuroLeague line and an NCAA line
  are comparable) and **dynamic strength-of-schedule** standardized within `(league, season)` —
  never treated as a static player attribute.
- **Combine** measurements, sign-normalized so higher = better.

### 5. Modeling

Simple and interpretable first, complexity only if it earns its keep under temporal validation:

- Regularized regression (ridge / lasso / elastic-net), the **draft-position baseline to beat**,
  gradient boosting (XGBoost → LightGBM → scikit-learn fallback), and survival analysis
  (lifelines Cox PH) for career longevity.
- Hyperparameters tuned with **Optuna inside the temporal CV** (so tuning can't leak), optimizing
  the mean walk-forward ranking metric.

### 6. Evaluation

The product *orders* prospects, so **ranking metrics lead** (Spearman, Kendall, top-k hit rate),
with **calibration** (Brier / ECE) for the classification targets, an honest comparison against
the draft-position baseline, and **error analysis** (which player types the model fails on, e.g.
weaker-league prospects) using out-of-fold predictions.

---

## Data

The system prefers APIs and open datasets, respects each source's Terms of Service and
`robots.txt`, and rate-limits, caches, and records provenance (source, URL, timestamp, license,
SHA-256) for every artifact. Source vetting is recorded in `config/sources.yaml`.

| Source | Use | Status |
|---|---|---|
| **stats.nba.com** via `nba_api` | NBA outcomes (draft history, player seasons), Combine measurements, birthdates | ✅ primary — **run locally** (cloud IPs get banned) |
| **CollegeBasketballData.com** | NCAA pre-draft production (per-40 stats, true shooting, usage, ratings, PORPAG, win shares) | ✅ free API key (Bearer auth) |
| Basketball-Reference / Bart Torvik / KenPom | — | ❌ excluded (their data paths are disallowed by `robots.txt` / paid) |

**A key consequence:** BPM/VORP are Basketball-Reference's metrics and can't be scraped, so the
system **computes its own impact metric** from `nba_api` raw box + advanced data:

- **eBPM** — a transparent, league-centered box plus-minus *proxy* (documented heuristic weights,
  not BBRef's RAPM-calibrated BPM). It is validated against the official NBA **PIE** metric at a
  Spearman of **≈0.92**, so rankings are trustworthy even though absolute magnitudes are approximate.
- **VORP** — the exact standard formula, `VORP = (eBPM + 2.0) · minutes / (240 · 82)`.

`age-at-draft` is computed from `CommonPlayerInfo` birthdates (years to the late-June draft day).

Real pre-draft **features** available today are therefore: **college production (NCAA)** + **Combine
measurements** + **age** + **draft pick**. International prospects (EuroLeague/NBL) lack college
features and rely on the imputer. The whole pipeline also runs **fixture-first** on deterministic
synthetic data, so the architecture is verifiable without any data access.

---

## What the model predicts

Targets are defined declaratively in `config/targets.yaml` and built by a tested
label-construction contract (`src/nba_draft/targets/`):

- **Reach probability** — does the prospect become a rotation player within the horizon? (the
  hurdle gate, fit over all prospects)
- **Peak impact** — best-season eBPM over the first 4 NBA seasons (ceiling / talent).
- **Cumulative value** — Σ VORP over the window (realized contribution to winning).
- **Outcome-tier distribution** — P(bust / rotation / starter / all-star / superstar), the
  decision-facing view (a high-ceiling/low-floor prospect reads differently from a safe one).
- **Longevity** — career length via censoring-aware survival analysis.
- **Real Surplus Value** — projected $ value minus the cheap rookie-scale cost (see fit module).

**Horizon:** debut-anchored and capped — the window starts at a player's first NBA season; a
prospect must reach the NBA within ~4 years of being drafted, else they count as a non-reach
(so draft-and-stash internationals aren't unfairly marked busts). Recent, unfinished classes are
treated as **right-censored**, not as low outcomes.

---

## The fit module

A player's value depends on the team that drafts them. This is the most differentiating — and
most exploratory — part, so its outputs are explicitly flagged as higher-variance.

- **Archetypes** — functional player profiles via clustering on style skills (not positions).
- **Roster need** — which skills the current roster lacks (supply-based).
- **Synergy vs redundancy** — does a prospect fill scarce needs or duplicate strengths?
- **Lineup Net-Rating simulation** — concrete and actionable: estimates how the most-used
  lineup's Net Rating changes if its weakest link is replaced by the rookie, e.g. *"Net Rating
  goes from +9.3 to +14.4 (+5.1 per 100)."*
- **Financial fit (CBA / aprons)** — **Real Surplus Value** modulated by cap pressure: a cheap
  rookie contract is most valuable to a team over the second apron. CBA figures are
  **parameterized per season** in `config/cba_rules.yaml` (2025-26 verified; 2026-27 projected and
  flagged as an assumption) because the rules change yearly.

---

## Uncertainty & interpretability

**Uncertainty is mandatory, not optional** (the draft is fat-tailed):

- Prediction intervals via **quantile regression**, **split-conformal** (finite-sample coverage),
  **Bayesian** linear predictive std, and **bootstrap ensembles**.
- Per-prospect **scenario distributions** over outcome tiers, plus floor/ceiling — so the tool
  reflects boom-bust vs safe-but-limited as different decisions.

**Interpretability** so scouts can challenge every recommendation:

- **SHAP** (global importance + additive per-prospect explanations), model-agnostic **permutation
  importance**, **partial dependence**, and **counterfactuals** ("what would need to change to
  raise this projection?").

---

## Project structure

```
config/            central, versioned configuration (config, targets, sources, CBA, leagues)
src/nba_draft/
  config.py        typed config loader
  ingestion/       source registry, caching/rate-limit/retry/robots, provenance, parsers
  cleaning/        normalization, entity resolution, imputation, versioned master dataset
  targets/         target definitions + label construction + eBPM/VORP + outcome assembly
  features/        stateless transforms + learned context model + assembler (leakage guard)
  validation/      temporal splits, fold preprocessor, walk-forward runner
  models/          baselines, model zoo, Optuna tuning, survival
  evaluation/      ranking/calibration metrics, error analysis, baseline comparison
  fit/             archetypes, need, synergy, lineup sim, financial (CBA) — player-team fit
  uncertainty/     quantile, conformal, bayesian, ensemble, scenario distributions
  interpretability/ SHAP, permutation importance, PDP, counterfactuals
  service/         draft-board service (the logic the API and dashboard share)
  realdata/        real nba_api + CollegeBasketballData pipeline (pull → labels → train)
  mlops/           MLflow tracking, model registry, drift (PSI), retraining, pipeline
api/               FastAPI service
dashboard/         Streamlit GM dashboard
scripts/           one-command entrypoints
tests/             pytest suite (162 tests)
docs/              design decisions, assumptions/limitations, usage, validation, MLOps
```

---

## Installation

Requires **Python ≥ 3.11** (developed on 3.14; CI runs 3.12).

```bash
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1
# Git Bash / macOS / Linux:  source .venv/Scripts/activate   (or .venv/bin/activate)

pip install -e ".[dev]"          # core library + dev tooling (ruff, mypy, pytest)
```

Optional dependency groups (install what you need):

| extra | pulls in | for |
|---|---|---|
| `models` | xgboost, lightgbm, lifelines, pymc, optuna | gradient boosting, survival, tuning |
| `explain` | shap, mapie | interpretability |
| `app` | fastapi, uvicorn, streamlit, plotly | API + dashboard |
| `mlops` | mlflow, dvc | tracking + data versioning |
| `ingest` | requests, nba_api | the live local data pull |
| `eda` | matplotlib | EDA plots |

```bash
pip install -e ".[test]"                                      # everything the test suite needs
pip install -e ".[dev,models,explain,app,mlops,ingest,eda]"   # everything
```

> **To get green tests from zero:** `pip install -e ".[test]"` then `pytest`. The bare `[dev]`
> extra installs only tooling, so several test files (which import xgboost/shap/fastapi) would fail.

---

## Usage

### Run analyses (synthetic demo — runs out of the box)

```bash
python scripts/reproduce.py        # smoke test: temporal split → baseline → metrics
python scripts/run_eda.py          # → artifacts/eda/eda_report.md (+ plots with [eda])
python scripts/run_modeling.py     # → artifacts/modeling/comparison.md
python scripts/run_evaluation.py   # → artifacts/evaluation/evaluation_report.md
python scripts/run_pipeline.py     # full MLOps pipeline (the one reproducible command)
```

### Serve the app

```bash
uvicorn api.main:app --reload                 # API docs at http://localhost:8000/docs
streamlit run dashboard/streamlit_app.py      # GM dashboard at http://localhost:8501
```

API endpoints: `GET /health`, `GET /prospects` (ranked board), `GET /explain/{player_id}`,
`POST /fit` (submit a roster + cap situation → fit score + narrative).

### Run on real data

The live pull must run on a **personal machine** (stats.nba.com blocks datacenter IPs).

```bash
pip install -e ".[ingest,models,explain]"

# 1. (optional) college features — get a free key at https://collegebasketballdata.com/key
setx CBD_API_KEY "your-key"        # Windows; or export CBD_API_KEY=... in bash

# 2. end-to-end real pipeline: pull → eBPM/VORP labels → temporal CV → tune → register
python scripts/run_real_pipeline.py
```

Other entrypoints: `scripts/run_ingest.py` (raw NBA pull only), `scripts/verify_cbd.py` (verify
the college API key + endpoints). To deploy a real-data service, point
`service.build_demo_service` at the rebuilt master dataset.

### Reproduce from scratch with DVC

```bash
git init && dvc init && dvc repro    # runs the pipeline stage defined in dvc.yaml
```

---

## Configuration

All run-controlling knobs live in `config/*.yaml`, validated on load — no magic constants in code:

- `config.yaml` — seeds, prediction horizon, temporal-validation scheme, evaluation k's.
- `targets.yaml` — target / outcome-tier definitions and thresholds.
- `sources.yaml` — data-source registry with license + ToS/robots review and enable gates.
- `cba_rules.yaml` — salary cap / aprons / rookie scale (per season; projections flagged).
- `leagues.yaml`, `normalization.yaml` — league tiers and name/league normalization tables.
- `params.yaml` — DVC-tracked parameters mirroring the above for the pipeline graph.

---

## Quality & testing

```bash
pytest                                          # 162 tests
ruff check src tests scripts api dashboard      # lint
mypy                                            # strict type checking
```

**162 tests** pass; `ruff` and `mypy --strict` are clean; GitHub Actions CI runs the full gate on
every push and pull request. No test makes a network call (external services are injected/faked),
so the suite is fully deterministic and CI-safe.

---

## Limitations & honest caveats

Being explicit about the limits is part of the deliverable:

- **eBPM is a proxy, not BBRef's BPM 2.0.** Rankings are reliable (ρ≈0.92 vs PIE); treat absolute
  magnitudes as indicative and recalibrate before any high-stakes use.
- **Box-score metrics under-credit defense** — defensive specialists are under-served by the
  impact projection.
- **NCAA-centric.** International prospects lack college features and lean on imputed values.
- **Small samples.** A few hundred trainable players over a limited history caps model complexity
  and makes rare-tier (all-star/superstar) probabilities inherently uncertain. The **direction**
  of results is robust; exact magnitudes are indicative.
- **The fit module is exploratory** — the lineup Net-Rating model is a transparent proxy, not a
  calibrated RAPM model, and the CBA `$`-per-win conversion is a flagged assumption.
- **~9% of drafted players have no fetchable birthdate** — these are all players who never reached
  the NBA (no `CommonPlayerInfo` record), so the impact model's age data is complete; they would be
  imputed for a reach-classifier.

---

## Documentation

- [Design decisions](docs/design_decisions.md) — why the system is built the way it is
- [Assumptions & limitations](docs/assumptions_and_limitations.md)
- [Usage guide](docs/usage_guide.md) — install, run, real-data setup
- [Validation protocol](docs/validation.md) — the temporal-validation contract
- [MLOps](docs/mlops.md) — tracking, registry, drift, retraining

---

*Built phase-by-phase with verifiable milestones; see `instructions.md` for the original
specification and domain-risk checklist.*
