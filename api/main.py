"""FastAPI service serving draft projections, fit, and explanations.

Thin layer over `nba_draft.service` (which holds the logic and is unit-tested directly). The
demo service is trained on the synthetic fixture at startup so the API runs out of the box;
swap `build_demo_service` for a service built on the real master dataset in production.

Run locally:  uvicorn api.main:app --reload
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import polars as pl
from fastapi import FastAPI, HTTPException
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
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/prospects")
def prospects(limit: int = 30) -> list[dict[str, Any]]:
    """Ranked draft board for the demo pool.

    With the survivorship-robust hurdle attached, rows carry ``p_reach`` + ``projected_ev`` and
    the board is ranked by unconditional EV; otherwise by conditional ``projected_impact``. Each
    row also has the 80% interval (floor/ceiling) and outcome-tier probabilities.
    """
    service, pool = _service_and_pool()
    board = service.rank(pool).head(limit)
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
        "lineup_delta": result.lineup.delta,
        "narrative": result.narrative,
        "exploratory": result.exploratory,
        "assumptions": result.assumptions,
    }
