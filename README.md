# NBA Draft AI — Decision-Support System

A human-in-the-loop decision-support tool for NBA draft decisions. For each prospect it aims
to: (1) project NBA production/trajectory, (2) estimate contribution to winning, (3) evaluate
fit with a *specific* roster & system, and (4) quantify the uncertainty of every prediction.

> **Status:** All 12 phases built and tested end-to-end on a **synthetic fixture** (no real
> basketball data yet — see the live-pull steps in the usage guide). 143 tests, `ruff` + `mypy
> --strict` clean. Validity over spectacle: see `instructions.md` for the full spec and domain risks.
>
> **Docs:** [design decisions](docs/design_decisions.md) · [assumptions & limitations](docs/assumptions_and_limitations.md)
> · [usage guide](docs/usage_guide.md) · [validation protocol](docs/validation.md) · [MLOps](docs/mlops.md)

## Why the skeleton looks the way it does

The hardest part of draft modeling is *not* the model — it's avoiding the inference traps that
make results look great and be worthless. So the first thing built is the defense:

- **Temporal validation only** (`src/nba_draft/validation/`): train on older draft classes,
  validate on newer ones. Random splits leak the future. Every fold is leakage-guarded and
  raises rather than returning a leaky split.
- **Untouchable holdout**: the most-recent draft classes are reserved and never used for tuning.
- **Draft-position baseline** (`src/nba_draft/models/baseline.py`): the number every model must
  beat under temporal validation.
- **Ranking-first evaluation** (`src/nba_draft/evaluation/`): the tool orders prospects, so
  Spearman / Kendall / top-k hit-rate lead; calibration covers the classification targets.

## Project layout

```
config/        central, versioned configuration (config, sources, CBA rules, leagues)
src/nba_draft/ library code (ingestion, cleaning, features, validation, models, eval, fit, ...)
api/           FastAPI service (Phase 11)
dashboard/     Streamlit app (Phase 11)
tests/         pytest — cleaning, features, validation, etc.
scripts/       one-command entrypoints
data/          DVC-tracked; contents gitignored
```

## Quickstart

Requires Python ≥ 3.11.

```bash
# create + activate a virtual environment
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# Git Bash / macOS / Linux:  source .venv/Scripts/activate  (or .venv/bin/activate)

pip install -e ".[dev]"      # core + dev tooling

pytest                       # run the test suite
python scripts/reproduce.py  # Milestone 0 end-to-end smoke run
```

With `make` available you can use `make dev`, `make test`, `make reproduce` instead.

## Run the app (Phase 11)

```bash
pip install -e ".[app]"                                  # FastAPI + Streamlit + Plotly

# API (serves projections, fit, explanations):
uvicorn api.main:app --reload                            # http://localhost:8000/docs

# GM dashboard (roster + system -> ranked board, fit, uncertainty, explanations):
streamlit run dashboard/streamlit_app.py                 # http://localhost:8501
```

Both are wired to a demo service trained on the synthetic fixture so they run out of the box;
swap `service.build_demo_service` for a service built on the real master dataset in production.

## Configuration

All run-controlling knobs live in `config/*.yaml`, loaded and type-checked via
`src/nba_draft/config.py`. No magic constants in code.

- `config.yaml` — seeds, horizon, temporal-validation scheme, evaluation k's.
- `sources.yaml` — Phase 1 source registry (provenance/license/ToS; gated until reviewed).
- `cba_rules.yaml` — CBA/apron/rookie-scale params; **placeholders**, verified per season.
- `leagues.yaml` — league tiers + inter-league translation priors (estimated in Phase 4).

## Roadmap

Phases 0–12 per `instructions.md`. `find-skills` is consulted at the start of each phase.
Next up after Milestone 0 sign-off: **Phase 0 — target definition** (impact metric + horizon
+ classification ladder + Real Surplus Value), then **Phase 1 — data acquisition**.
