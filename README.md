# NBA Draft AI — draft decision support with leakage-safe evaluation

[![CI](https://github.com/fernandogg2004/NBA-Draft-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/fernandogg2004/NBA-Draft-AI/actions/workflows/ci.yml)

A decision-support system that ranks NBA draft prospects by expected NBA value, attaches calibrated
uncertainty to every projection and explains each recommendation — evaluated on a locked temporal
holdout so that the reported performance is honest.

## The problem

Draft models are easy to make look good and hard to make useful. Typical failure modes are data
leakage (using information that only exists after the draft), survivorship bias (NBA statistics
only exist for players who made it), era effects and very small samples. This project builds the
whole pipeline — ingestion, labels, features, models, uncertainty, serving — around those traps,
and asks a precise question: **do public pre-draft data add ranking signal beyond the draft order
itself, which already aggregates the judgment of 30 front offices?**

## At a glance

| | |
|---|---|
| **Approach** | Two-part *hurdle* target, `P(reach rotation) · E(impact \| reached) + (1 − P) · replacement`, fit with ridge / gradient boosting; split-conformal prediction intervals; calibrated outcome-tier classifier; Cox model for career length; SHAP and counterfactual explanations |
| **Data** | Public: `stats.nba.com` (via `nba_api`), CollegeBasketballData.com API, EuroLeague live API. Draft classes 2005–2020 for training and evaluation; the 2025 class as the projected board |
| **Evaluation** | Strictly temporal: walk-forward CV on 2005–2018, **locked holdout = 2019 + 2020 draft classes**, never used for tuning or model selection |
| **Main result** | The served model **matches but does not beat** the draft-position baseline on the holdout (Spearman 0.469 vs 0.516). Uncertainty is calibrated: 80% intervals cover ~78–81% of outcomes |
| **Status** | Functional research prototype (library + FastAPI + Streamlit + React front end). Not deployed |

## Results

All figures below come from the project's own development log (`IMPROVEMENT_LOG.md` and
`NBA_READINESS_REPORT.md`, commit [`a0132a5`](https://github.com/fernandogg2004/NBA-Draft-AI/commit/a0132a5),
later removed from the tree). They were produced on real data with `scripts/run_real_pipeline.py`,
whose output artifacts are not committed. They are **consistent with the code but have not been
independently re-run** for this README (the data pull requires a residential IP; see
*Reproduction*).

Locked holdout: draft classes 2019–2020 (≈115 resolved prospects, 78 of whom reached a rotation role).

| Metric | Result | Protocol / notes |
|---|---|---|
| Ranking of realized value — served hurdle model | Spearman **0.469** | Holdout; trained on 2005–2018; draft pick included as a feature |
| Ranking of realized value — draft-position baseline | Spearman **0.516** | Same holdout; ranks by pick number |
| Same model without the draft pick ("scouting-only") | Spearman 0.263 | Same holdout; earlier run trained on 2011–2018 |
| 80% prediction-interval coverage | **0.78–0.81** | Split-conformal, calibrated inside the training fold (0.346 before conformal) |
| Outcome-tier probabilities, multiclass ECE | **0.111** | Against honors-aware tiers (bust → superstar); 0.390 before the tier classifier |
| Reach-a-rotation-role probability, ECE | 0.097 | Holdout |
| Career length (Cox PH), concordance | 0.659 | Holdout; 2011–2020 sample |

**How to read this.** With ~115 holdout players, a difference of 0.05 in Spearman is within
sampling noise, so the honest conclusion is *"matches the consensus"*, not *"loses to it"* or
*"beats it"*. The negative result was stress-tested: doubling the training sample (594 → 954
resolved players by adding 2005–2010) raised walk-forward CV from 0.43 to 0.59 but left the
holdout unchanged, and adding high-school recruiting rankings added no signal. The evidence points
to a structural ceiling of public box-score / combine / age features rather than a modelling defect.

> An earlier version of this README reported that the model *beat* the draft-position baseline
> (Spearman 0.336 vs 0.233). Those numbers came from a conditional-impact run evaluated without the
> locked holdout and are superseded by the results above.

## How it works

1. **Temporal validation as the core contract.** `holdout_split` locks the most recent classes;
   `walk_forward_folds` yields folds in which every training draft year precedes every validation
   year and raises if a split would leak. Splits are by draft year, never by row.
2. **Leakage firewall.** Stateless features (a player's own pre-draft numbers) are safe by
   construction. Learned steps — inter-league translation, strength-of-schedule standardisation,
   imputation, scaling — live in a `FoldPreprocessor` fit on the training fold only.
   `assert_pre_draft_safe` rejects known post-draft columns.
3. **Hurdle target against survivorship bias.** `P(reach)` is fit on *all* drafted players, so it
   learns who washes out; conditional impact is fit only on players with stable minutes.
4. **Own impact metric.** Basketball-Reference BPM cannot be scraped (robots.txt), so the project
   computes **eBPM**, a documented box plus-minus proxy built from `nba_api` box/advanced data, and
   VORP from it. Its rank agreement with the NBA's PIE metric is checked by `pie_rank_agreement`
   (reported ρ ≈ 0.92 during development; not re-run here).
5. **Models.** Draft-position baseline, regularised linear models, gradient boosting
   (XGBoost/LightGBM with a scikit-learn fallback) tuned with Optuna *inside* the temporal CV, a
   multinomial tier classifier and Cox PH for longevity.
6. **Uncertainty and explanations.** Split-conformal intervals and tier probabilities, SHAP
   attributions, permutation importance, partial dependence and counterfactuals ("smallest change
   that moves this prospect up a tier").
7. **Team fit (exploratory).** Archetype clustering, roster need and redundancy, a transparent
   lineup Net-Rating proxy and a CBA/apron-aware surplus-value calculation
   (`config/cba_rules.yaml`; 2026-27 values flagged as projections). Not validated against outcomes.

## Data and provenance

| Source | Used for | Access |
|---|---|---|
| [stats.nba.com](https://www.nba.com/stats) via [`nba_api`](https://github.com/swar/nba_api) | Draft history, player seasons (labels), Draft Combine, birthdates, awards | Public; blocks datacenter IPs, so pulls must run on a personal machine |
| [CollegeBasketballData.com](https://collegebasketballdata.com) | NCAA pre-draft production | Free API key (`CBD_API_KEY`) |
| [EuroLeague live API](https://live.euroleague.net) | International pre-draft features | Public; opt-in with `NBA_DRAFT_AI_INTL=1` |
| Basketball-Reference, Bart Torvik, KenPom | — | Deliberately excluded (robots.txt / paid) |

Every pulled artifact is cached with provenance (source, URL, timestamp, licence note, SHA-256);
source vetting is recorded in `config/sources.yaml`. **No raw or processed data are committed.**
The pipeline also runs on a deterministic **synthetic fixture** (`nba_draft.data.fixtures`) so the
architecture can be exercised without network access — synthetic results are labelled as such and
say nothing about real-world performance.

Label horizon: a player's first four NBA seasons, debut-anchored; prospects who do not reach the
NBA within ~4 years of the draft count as non-reach; unfinished classes are treated as censored.

## Limitations

- **No edge over the draft consensus** on the holdout; the "steals and reaches" view in the app
  lists model–consensus disagreements, not a validated source of value.
- **Small holdout** (two classes, ~115 players): rankings and calibration numbers carry wide
  sampling error.
- **eBPM is a proxy**, not BBRef BPM 2.0, and box-score metrics under-credit defence.
- **NCAA-centric features**; international prospects rely heavily on imputation.
- **Headline numbers not re-run here** — they depend on a local data pull and uncommitted artifacts.
- **The fit module is exploratory**; the $-per-win conversion is an explicit assumption.
- ~8% of players fail the per-player bio fetch in `nba_api` and get an imputed age.

## Reproduction

Requires Python ≥ 3.11 (CI uses 3.12).

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[test]"        # library + everything the test suite imports
pytest                          # 197 tests, no network access
ruff check src tests scripts api dashboard
mypy
```

Synthetic end-to-end runs (no data access needed):

```bash
python scripts/reproduce.py        # temporal split -> baseline -> metrics (smoke test)
python scripts/run_modeling.py     # model comparison on the fixture
python scripts/run_pipeline.py     # full MLOps pipeline on the fixture
```

Real-data pipeline (personal machine; produces the results above):

```bash
pip install -e ".[ingest,models,explain]"
export CBD_API_KEY=...                # optional: college features
python scripts/run_real_pipeline.py   # pull -> labels -> temporal CV -> tune -> holdout report
```

The run logs the served-model and baseline holdout Spearman and writes a summary under
`artifacts/real_pipeline/`. Serving:

```bash
uvicorn api.main:app --reload                 # API docs at http://localhost:8000/docs
streamlit run dashboard/streamlit_app.py      # analyst dashboard
cd frontend && npm install && npm run dev     # React front end (proxies /api to :8000)
```

Checked for this README on 2026-10-26 (Python 3.12, Linux): `pip install -e ".[test]"`, `pytest`
(197 passed), `ruff` (clean) and `scripts/reproduce.py` (runs). `mypy` reports one type error in
`models/tuning.py` with the current Optuna release. The real-data pipeline was not run.

## Repository structure

```
src/nba_draft/
  ingestion/  cleaning/  realdata/     data pull, caching, provenance, entity resolution
  targets/  features/                  labels (eBPM/VORP, tiers), stateless + fold-fit features
  validation/  evaluation/             temporal splits, walk-forward runner, metrics, error analysis
  models/  uncertainty/                baselines, GBM/ridge, hurdle, tiers, survival, conformal
  interpretability/  fit/  service/    SHAP/counterfactuals, team fit, draft-board service
  mlops/                               MLflow tracking, registry, drift (PSI), retraining
api/  dashboard/  frontend/            FastAPI, Streamlit, React
config/                                all run-controlling parameters (validated on load)
scripts/  tests/  docs/                entry points, test suite, design and validation notes
```

Design rationale and the validation contract: [`docs/design_decisions.md`](docs/design_decisions.md),
[`docs/validation.md`](docs/validation.md).

## Technologies

Python, Polars, scikit-learn, XGBoost/LightGBM, Optuna, lifelines, SHAP, conformal prediction,
FastAPI, Streamlit, React/TypeScript, MLflow, DVC, pytest, ruff, mypy, GitHub Actions.

## Author

**Fernando González González** — BSc Mathematics, Universidade de Santiago de Compostela.
[LinkedIn](https://www.linkedin.com/in/fernando-gonzalez-gonzalez-math/) ·
[GitHub](https://github.com/fernandogg2004)
