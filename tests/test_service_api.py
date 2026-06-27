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
    cols = ("projected_impact", "floor", "ceiling", *[f"p_{t}" for t in TIER_LABELS])
    for col in cols:
        assert col in board.columns
    # demo service attaches a hurdle -> EV columns present and board sorted by unconditional EV
    assert "p_reach" in board.columns and "projected_ev" in board.columns
    ev = board["projected_ev"].to_list()
    assert ev == sorted(ev, reverse=True)
    assert ((board["p_reach"] >= 0.0) & (board["p_reach"] <= 1.0)).all()
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
    # demo service ranks by survivorship-robust EV
    assert "projected_ev" in body[0] and "p_reach" in body[0]
    evs = [b["projected_ev"] for b in body]
    assert evs == sorted(evs, reverse=True)


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


# ----------------------------------------------------------------- real-data service builder
def test_build_service_from_table_ranks_real_prospects():
    import numpy as np
    import polars as pl

    from nba_draft.service import build_service_from_table

    rng = np.random.default_rng(0)
    n = 60
    skill = rng.normal(size=n)
    table = pl.DataFrame(
        {
            "player_id": list(range(n)),
            "full_name": [f"P{i}" for i in range(n)],
            "f_skill": skill + rng.normal(scale=0.2, size=n),
            "f_noise": rng.normal(size=n),
            "peak_impact": 2.0 * skill + rng.normal(scale=0.3, size=n),
        }
    )
    feats = ["f_skill", "f_noise"]
    service = build_service_from_table(table, feats, target_col="peak_impact")
    board = service.rank(table)
    assert board.height == n
    for col in ("projected_impact", "floor", "ceiling"):
        assert col in board.columns
    # ranking carries signal: projection correlates with the latent skill
    merged = board.join(table.select("player_id", "f_skill"), on="player_id")
    corr = np.corrcoef(merged["projected_impact"].to_numpy(), merged["f_skill"].to_numpy())[0, 1]
    assert corr > 0.5
    # no reach column -> conditional-impact ranking, no hurdle attached
    assert service.hurdle is None
    proj = board["projected_impact"].to_list()
    assert proj == sorted(proj, reverse=True)


def test_build_service_from_table_attaches_hurdle_when_reach_present():
    import numpy as np
    import polars as pl

    from nba_draft.service import build_service_from_table

    rng = np.random.default_rng(1)
    n = 120
    skill = rng.normal(size=n)
    # reach probability rises with skill; impact only observed for reached players
    reached = (skill + rng.normal(scale=0.5, size=n)) > 0.0
    peak = np.where(reached, 2.0 * skill + rng.normal(scale=0.3, size=n), np.nan)
    table = pl.DataFrame(
        {
            "player_id": list(range(n)),
            "full_name": [f"P{i}" for i in range(n)],
            "f_skill": skill + rng.normal(scale=0.2, size=n),
            "f_noise": rng.normal(size=n),
            "reached": reached.tolist(),
            "peak_impact": peak.tolist(),
        }
    )
    service = build_service_from_table(table, ["f_skill", "f_noise"], target_col="peak_impact")
    assert service.hurdle is not None
    board = service.rank(table)
    assert "p_reach" in board.columns and "projected_ev" in board.columns
    ev = board["projected_ev"].to_list()
    assert ev == sorted(ev, reverse=True)
    # higher skill -> higher reach probability
    merged = board.join(table.select("player_id", "f_skill"), on="player_id")
    corr = np.corrcoef(merged["p_reach"].to_numpy(), merged["f_skill"].to_numpy())[0, 1]
    assert corr > 0.4


def test_build_service_from_table_rejects_post_draft_feature():
    import numpy as np
    import polars as pl

    from nba_draft.service import build_service_from_table

    rng = np.random.default_rng(2)
    n = 20
    table = pl.DataFrame(
        {
            "player_id": list(range(n)),
            "full_name": [f"P{i}" for i in range(n)],
            "f_skill": rng.normal(size=n),
            "peak_impact": rng.normal(size=n),  # post-draft label, must not be a feature
        }
    )
    # using a known post-draft column as a feature must trip the leakage guard
    with pytest.raises(ValueError, match="leakage"):
        build_service_from_table(table, ["f_skill", "peak_impact"], target_col="peak_impact")


def test_build_service_from_master_round_trip(tmp_path):
    import json

    import numpy as np
    import polars as pl

    from nba_draft.service import build_service_from_master

    rng = np.random.default_rng(3)
    rows = []
    for year in (2018, 2019, 2020, 2021):
        for _ in range(25):
            skill = rng.normal()
            reached = (skill + rng.normal(scale=0.5)) > 0.0
            rows.append(
                {
                    "player_id": len(rows),
                    "full_name": f"P{len(rows)}",
                    "draft_year": year,
                    "f_skill": skill + rng.normal(scale=0.2),
                    "reached": bool(reached),
                    "peak_impact": (2.0 * skill if reached else None),
                }
            )
    table = pl.DataFrame(rows)
    serving = tmp_path / "serving"
    serving.mkdir()
    table.write_parquet(serving / "modeling_table.parquet")
    (serving / "serving_manifest.json").write_text(
        json.dumps(
            {
                "table": "modeling_table.parquet",
                "feature_cols": ["f_skill"],
                "target_col": "peak_impact",
                "reached_col": "reached",
            }
        ),
        encoding="utf-8",
    )

    service, pool = build_service_from_master(serving)
    # latest class is held out as the pool; service trains on the rest -> hurdle attached
    assert pool["draft_year"].unique().to_list() == [2021]
    assert service.hurdle is not None
    board = service.rank(pool)
    assert board.height == pool.height
    assert "projected_ev" in board.columns and "p_reach" in board.columns
    # accepting a manifest path directly works too
    service2, _ = build_service_from_master(serving / "serving_manifest.json")
    assert service2.hurdle is not None
