"""FastAPI service serving draft projections, fit, and explanations.

Thin layer over `nba_draft.service` (which holds the logic and is unit-tested directly). The
demo service is trained on the synthetic fixture at startup so the API runs out of the box;
swap `build_demo_service` for a service built on the real master dataset in production.

Run locally:  uvicorn api.main:app --reload
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import polars as pl
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from nba_draft.fit import Player, TeamContext, load_cba
from nba_draft.service import (
    DraftBoardService,
    build_demo_service,
    build_service_from_master,
)
from nba_draft.utils.logging import get_logger

log = get_logger("api")

app = FastAPI(
    title="NBA Draft AI",
    version="0.1.0",
    description="Decision-support: projections, uncertainty, fit, and explanations.",
)

# Allow the React frontend (Vite dev server / any deployed origin) to call the API
# directly. In dev the Vite proxy makes requests same-origin, so this mainly covers
# running the SPA against the API without the proxy. Override via NBA_DRAFT_AI_CORS
# (comma-separated origins); defaults to localhost dev ports.
_cors_env = os.environ.get("NBA_DRAFT_AI_CORS")
_cors_origins = (
    [o.strip() for o in _cors_env.split(",") if o.strip()]
    if _cors_env
    else ["http://localhost:5173", "http://127.0.0.1:5173"]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def _service_and_pool() -> tuple[DraftBoardService, pl.DataFrame]:
    """Serve a real board if NBA_DRAFT_AI_MASTER points to a persisted serving dir/manifest.

    Set ``NBA_DRAFT_AI_MASTER`` to the ``serving/`` directory (or its ``serving_manifest.json``)
    written by ``scripts/run_real_pipeline.py``; otherwise the synthetic demo service is served so
    the API runs out of the box. A bad path fails closed to the demo (logged), never crashing boot.
    """
    master = os.environ.get("NBA_DRAFT_AI_MASTER")
    if master:
        try:
            service, pool = build_service_from_master(master)
            log.info("Serving REAL board from %s (%d prospects).", master, pool.height)
            return service, pool
        except Exception:  # noqa: BLE001 - never let a bad path break the service; fall back
            log.exception("Failed to load real master from %s; falling back to demo.", master)
    return build_demo_service()


# ----------------------------------------------------------------- schemas
class RosterPlayer(BaseModel):
    name: str
    skills: dict[str, float] = Field(default_factory=dict)
    impact: float = 0.0
    salary_usd: float = 0.0


class FitRequest(BaseModel):
    prospect_player_id: int
    roster: list[RosterPlayer]
    team_total_salary_usd: float = 0.0
    pick: int = Field(ge=1)
    season: str | None = None


# ----------------------------------------------------------------- endpoints
@app.get("/meta")
def meta() -> dict[str, Any]:
    """Real serving metadata for the UI (no fabricated version strings).

    Reports whether a REAL board (from NBA_DRAFT_AI_MASTER) or the synthetic demo is served, the
    pool size, the draft class(es) projected, and — for the real path — the trained model version
    and feature count read from the serving manifest.
    """
    _, pool = _service_and_pool()
    master = os.environ.get("NBA_DRAFT_AI_MASTER")
    info: dict[str, Any] = {"mode": "demo", "n_prospects": int(pool.height)}
    if "draft_year" in pool.columns and pool.height:
        info["draft_years"] = sorted({int(y) for y in pool["draft_year"].to_list()})
    if master:
        try:
            mp = Path(master)
            mp = mp if mp.suffix == ".json" else mp / "serving_manifest.json"
            manifest = json.loads(mp.read_text(encoding="utf-8"))
            info["mode"] = "real"
            info["model_version"] = manifest.get("model_version")
            info["n_features"] = len(manifest.get("feature_cols", []))
        except Exception:  # noqa: BLE001 - meta is best-effort; never break on a bad manifest
            log.exception("Could not read serving manifest for /meta")
    return info


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/prospects")
def prospects(limit: int = 30) -> list[dict[str, Any]]:
    """Ranked draft board for the demo pool, enriched with a per-prospect profile.

    With the survivorship-robust hurdle attached, rows carry ``p_reach`` + ``projected_ev`` and
    the board is ranked by unconditional EV; otherwise by conditional ``projected_impact``. Each
    row also has the 80% interval (floor/ceiling), outcome-tier probabilities, a feature-derived
    profile (functional ``skill_*`` ratings, ``archetype``, ``age``, ``wingspan_in``), and two
    projection-derived fields (``peak_pctile``, ``projected_value_usd``).
    """
    service, pool = _service_and_pool()
    board = service.ranked_with_profile(pool).head(limit)
    return board.to_dicts()


@app.get("/explain/{player_id}")
def explain(player_id: int) -> dict[str, Any]:
    """Local SHAP explanation for one prospect in the demo pool."""
    service, pool = _service_and_pool()
    row = pool.filter(pl.col("player_id") == player_id)
    if row.height == 0:
        raise HTTPException(status_code=404, detail=f"player_id {player_id} not in pool")
    table, base = service.explain(row)
    return {"player_id": player_id, "base_value": base, "contributions": table.to_dicts()}


@app.get("/counterfactual/{player_id}")
def counterfactual(player_id: int, max_features: int = 3) -> dict[str, Any]:
    """What feature change(s) would lift this prospect into the next outcome tier?

    Greedy, in-bounds counterfactual search over the projection model. ``reached`` is False if
    no change set within ``max_features`` hits the target; ``target`` is null if already top-tier.
    """
    service, pool = _service_and_pool()
    row = pool.filter(pl.col("player_id") == player_id)
    if row.height == 0:
        raise HTTPException(status_code=404, detail=f"player_id {player_id} not in pool")
    cf = service.counterfactual(row, max_features=max_features)
    return {
        "player_id": player_id,
        "current_impact": cf.current_impact,
        "current_tier": cf.current_tier,
        "target": cf.target,
        "target_tier": cf.target_tier,
        "projected_impact": cf.projected_impact,
        "reached": cf.reached,
        "changes": [
            {
                "feature": c.feature,
                "from_value": c.from_value,
                "to_value": c.to_value,
                "delta": c.delta,
            }
            for c in cf.changes
        ],
    }


@app.post("/fit")
def fit(req: FitRequest) -> dict[str, Any]:
    """Score a prospect's fit with a submitted roster + cap situation."""
    service, pool = _service_and_pool()
    row = pool.filter(pl.col("player_id") == req.prospect_player_id)
    if row.height == 0:
        raise HTTPException(status_code=404, detail="prospect not in pool")
    team = TeamContext(
        roster=[Player(p.name, p.skills, p.impact, p.salary_usd) for p in req.roster],
        total_salary_usd=req.team_total_salary_usd,
    )
    cba = load_cba(season=req.season) if req.season else load_cba()
    result = service.fit_for_team(
        row, team, pick=req.pick, cba=cba, name=f"P{req.prospect_player_id}"
    )
    return {
        "overall": result.overall,
        "basketball_fit": result.basketball_fit,
        "financial_fit": result.financial_fit,
        "rsv_usd": result.rsv_usd,
        "rsv_modulated_usd": result.rsv_modulated_usd,
        "apron_label": result.apron_label,
        # Synergy sub-scores (0..1): how the prospect fills needs vs. duplicates strengths.
        "synergy_complementarity": result.synergy.complementarity,
        "synergy_redundancy": result.synergy.redundancy,
        "synergy_net": result.synergy.net,
        # Lineup Net-Rating simulation: base -> with-prospect (replacing the weakest link).
        "lineup_before": result.lineup.before,
        "lineup_after": result.lineup.after,
        "lineup_delta": result.lineup.delta,
        "lineup_replaced": result.lineup.replaced,
        "narrative": result.narrative,
        "exploratory": result.exploratory,
        "assumptions": result.assumptions,
    }
