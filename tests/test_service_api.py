"""Tests for Phase 11 deployment: service layer + FastAPI endpoints (offline TestClient)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from nba_draft.fit import Player, TeamContext, load_cba
from nba_draft.service import build_demo_service, prospect_to_player
from nba_draft.service.board import TIER_LABELS


@pytest.fixture(scope="module")
def service_and_pool():
    return build_demo_service(seed=1)


# ----------------------------------------------------------------- service
def test_rank_returns_projection_interval_and_scenarios(service_and_pool):
    service, pool = service_and_pool
    board = service.rank(pool)
    assert board.height == pool.height
    for col in ("projected_impact", "floor", "ceiling", *[f"p_{t}" for t in TIER_LABELS]):
        assert col in board.columns
    # sorted by projection desc
    proj = board["projected_impact"].to_list()
    assert proj == sorted(proj, reverse=True)
    # floor <= ceiling for every prospect
    assert (board["floor"] <= board["ceiling"]).all()
    # scenario probabilities sum to ~1 per prospect
    probs = board.select([f"p_{t}" for t in TIER_LABELS]).to_numpy().sum(axis=1)
    assert all(abs(p - 1.0) < 1e-6 for p in probs)


def test_explain_is_additive(service_and_pool):
    service, pool = service_and_pool
    row = pool.head(1)
    table, base = service.explain(row)
    recon = base + float(table["shap_value"].sum())
    x = service.preprocessor.transform_matrix(row).to_numpy()
    pred = float(service.impact_model.predict(x)[0])
    assert recon == pytest.approx(pred, abs=0.2)


def test_fit_for_team_uses_apron_pressure(service_and_pool):
    service, pool = service_and_pool
    row = pool.head(1)
    cba = load_cba(season="2025-26")
    roster = [
        Player(f"S{i}", {"scoring": 80, "shooting_spacing": 78}, impact=1.5) for i in range(5)
    ]
    over2 = TeamContext(roster=roster, total_salary_usd=cba.second_apron_usd + 1)
    res = service.fit_for_team(row, over2, pick=8, cba=cba)
    assert res.exploratory is True
    assert "second apron" in res.apron_label


def test_prospect_to_player_maps_skills_in_range():
    feats = {"pts_per100": 20, "true_shooting": 0.6, "ast_per100": 5}
    p = prospect_to_player("X", feats, impact=3.0)
    assert p.impact == 3.0
    for v in p.skills.values():
        assert 0.0 <= v <= 100.0


# ----------------------------------------------------------------- API
@pytest.fixture(scope="module")
def client():
    from api.main import app

    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_prospects_endpoint_returns_ranked_board(client):
    r = client.get("/prospects?limit=5")
    assert r.status_code == 200
    body = r.json()
    assert len(body) <= 5
    assert "projected_impact" in body[0]
    projs = [b["projected_impact"] for b in body]
    assert projs == sorted(projs, reverse=True)


def test_explain_endpoint_and_404(client):
    # a valid player id from the pool
    _, pool = build_demo_service()
    pid = int(pool["player_id"][0])
    r = client.get(f"/explain/{pid}")
    assert r.status_code == 200
    assert "contributions" in r.json()
    # unknown id -> 404
    assert client.get("/explain/99999999").status_code == 404


def test_fit_endpoint(client):
    _, pool = build_demo_service()
    pid = int(pool["player_id"][0])
    payload = {
        "prospect_player_id": pid,
        "roster": [{"name": "S1", "skills": {"scoring": 80}, "impact": 2.0}],
        "team_total_salary_usd": 210_000_000,
        "pick": 8,
        "season": "2025-26",
    }
    r = client.post("/fit", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "narrative" in body and body["exploratory"] is True
    assert "Net Rating goes from" in body["narrative"]
