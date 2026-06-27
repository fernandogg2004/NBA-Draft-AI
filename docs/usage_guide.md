# Usage guide

## Install

```bash
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1   |  Git Bash:  source .venv/Scripts/activate
pip install -e ".[dev]"                  # core + tooling
# optional, per task:
pip install -e ".[models,explain,app,mlops]"   # boosting/survival, SHAP, API/dashboard, MLflow/DVC
pip install -e ".[ingest]"               # nba_api + requests (run the live pull locally)
```

## Everyday commands

```bash
pytest                         # full test suite
ruff check src tests scripts api dashboard
mypy
python scripts/reproduce.py    # Milestone-0 smoke: split -> baseline -> metrics
```

## Generate analyses (synthetic demo)

```bash
python scripts/run_eda.py          # -> artifacts/eda/eda_report.md (+ plots with [eda] extra)
python scripts/run_modeling.py     # -> artifacts/modeling/comparison.md
python scripts/run_evaluation.py   # -> artifacts/evaluation/evaluation_report.md
python scripts/run_pipeline.py     # full MLOps pipeline (the one command)
```

## Serve the app

```bash
uvicorn api.main:app --reload                 # API docs at http://localhost:8000/docs
streamlit run dashboard/streamlit_app.py      # GM dashboard at http://localhost:8501
```

By default both serve the **synthetic demo** so they run with no data or secrets. To serve a **real
board**, point `NBA_DRAFT_AI_MASTER` at the `serving/` directory written by the real pipeline (see
below) — the API/dashboard then build a `DraftBoardService` from the persisted modeling table,
hold the most recent draft class out as the prospect pool, and rank it by unconditional EV. A bad
path fails closed to the demo (the API logs it) rather than crashing:

```bash
NBA_DRAFT_AI_MASTER=artifacts/real_pipeline/serving uvicorn api.main:app
NBA_DRAFT_AI_MASTER=artifacts/real_pipeline/serving streamlit run dashboard/streamlit_app.py
```

API endpoints: `GET /health`, `GET /prospects`, `GET /explain/{player_id}`, `POST /fit`.

The served board is ranked by the **survivorship-robust hurdle** when one is attached (the demo
service and any real table with a `reached` column): each prospect carries `p_reach` and
`projected_ev` (`EV = P(reach)·E(impact|reached) + (1−P(reach))·replacement`) and the board sorts by
`projected_ev` rather than conditional `projected_impact`. The dashboard surfaces both.

## Use real data

1. `pip install -e ".[ingest]"` and pull locally (cloud IPs get banned):
   ```python
   from nba_draft.ingestion.nba_stats import NbaStatsIngester
   ing = NbaStatsIngester("data/raw")
   ing.draft_history(2025); ing.draft_combine_stats("2025"); ing.player_season_stats("2024-25")
   ```
2. Build the master dataset with `nba_draft.cleaning.master.build_master(...)`.
3. Run `python scripts/run_real_pipeline.py` — it persists a `serving/` directory
   (`modeling_table.parquet` + `serving_manifest.json`) under `artifacts/real_pipeline/`.
4. Serve it by setting `NBA_DRAFT_AI_MASTER=artifacts/real_pipeline/serving` (see *Serve the app*).

## Add real college pre-draft features (CollegeBasketballData.com)

This is the high-value input the nba_api-only path lacks. Steps:

1. **Get a free API key:** https://collegebasketballdata.com/key (email signup). Obtaining a key
   means accepting their terms — use politely (the client self-limits to ~20 req/min).
2. **Set it** (then open a new shell):
   ```bash
   setx CBD_API_KEY "your-key"        # Windows; or `export CBD_API_KEY=...` in bash
   ```
3. **Verify key + endpoints + schema:**
   ```bash
   python scripts/verify_cbd.py --season 2024
   ```
   This makes one small authenticated call per endpoint and prints the field names. Share that
   output and the parser is written against the real schema (then the college features are joined
   into the modeling table by entity-resolving drafted players to their college via name + school).

`config/sources.yaml::college_bb_data` holds the verified base URL / auth / endpoints; the
ingester (`CollegeBasketballDataIngester`) caches + rate-limits + records provenance like the
nba_api one.

## Add international pre-draft features (EuroLeague)

For prospects who never played NCAA, EuroLeague production fills the same feature slots (a subset —
fewer advanced metrics). Needs the `ingest` extra (`euroleague-api`), no key:

```bash
python scripts/verify_euroleague.py --season 2017   # prints live schema + canonical columns
NBA_DRAFT_AI_INTL=1 python scripts/run_real_pipeline.py   # enables the EuroLeague ingester
```

EuroLeague column names vary by endpoint/version; the parser resolves them case-insensitively and
`verify_euroleague.py` confirms the mapping (extend `_EL_CANDIDATES` in `ingestion/parse.py` if a
stat is missing). International features are coalesced into the college-named columns, so a prospect
gets NCAA *or* EuroLeague values; gaps are imputed.

## Configuration

All knobs live in `config/*.yaml` (validated by `nba_draft.config` / per-module loaders):
`config.yaml` (seeds, horizon, validation), `targets.yaml` (target/tier definitions),
`sources.yaml` (data sources + ToS), `cba_rules.yaml` (CBA/aprons/rookie scale), `leagues.yaml`,
`normalization.yaml`. DVC params mirror these in `params.yaml`.

## Reproduce from scratch with DVC

```bash
git init && dvc init && dvc repro     # runs the pipeline stage defined in dvc.yaml
```
