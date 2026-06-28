"""Tests for Phase 11 deployment: service layer + FastAPI endpoints (offline TestClient)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from nba_draft.fit import SKILL_DIMS, Player, TeamContext, load_cba
from nba_draft.service import build_demo_service, prospect_to_player
from nba_draft.service.board import _ARCHETYPE_BY_SKILL, TIER_LABELS

_ARCHETYPE_LABELS = list(_ARCHETYPE_BY_SKILL.values())


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


def test_conformal_interval_is_attached_and_calibrated(service_and_pool):
    """Floor/ceiling come from the split-conformal layer (finite-sample marginal coverage),
    not the historically-overconfident bootstrap. Its signature is a constant-width interval."""
    import numpy as np

    service, pool = service_and_pool
    assert service.conformal is not None
    board = service.rank(pool)
    floor = board["floor"].to_numpy()
    ceiling = board["ceiling"].to_numpy()
    point = board["projected_impact"].to_numpy()
    # point estimate lies inside the interval...
    assert bool(np.all((point >= floor - 1e-6) & (point <= ceiling + 1e-6)))
    # ...and the width is constant (= 2*qhat), the distinguishing mark of conformal vs ensemble
    # (tolerance covers the 3-decimal rounding applied to floor/ceiling in rank()).
    widths = ceiling - floor
    assert float(widths.max() - widths.min()) < 1e-2
    assert float(widths.mean()) > 0.0


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


def test_ranked_with_profile_adds_real_profile_fields(service_and_pool):
    service, pool = service_and_pool
    board = service.ranked_with_profile(pool)
    assert board.height == pool.height
    skill_cols = [f"skill_{d}" for d in SKILL_DIMS]
    extra = ("archetype", "age", "wingspan_in", "peak_pctile", "projected_value_usd")
    for col in (*skill_cols, *extra):
        assert col in board.columns
    # skills are 0-100 percentiles
    skills = board.select(skill_cols).to_numpy()
    assert skills.min() >= 0.0 and skills.max() <= 100.0
    # peak percentile is a real 0..1 rank; the top prospect is at the max
    assert ((board["peak_pctile"] > 0.0) & (board["peak_pctile"] <= 1.0)).all()
    assert board.sort("projected_impact", descending=True)["peak_pctile"][0] == pytest.approx(1.0)
    # projected $ value is non-negative and the archetype is one of the known labels
    assert (board["projected_value_usd"] >= 0.0).all()
    assert set(board["archetype"].unique()).issubset(set(_ARCHETYPE_LABELS))


def test_ranked_with_profile_real_shaped_pool():
    """The enrichment must work on the REAL schema: college per-40, age_at_draft, and an
    unlabeled post-draft pool carrying actual pick/team — producing steals/reaches + photos."""
    import numpy as np
    import polars as pl

    from nba_draft.service import build_service_from_table

    rng = np.random.default_rng(0)
    n = 40
    skill = rng.normal(size=n)
    peak = [float(2 * skill[i] + rng.normal(0, 0.3)) if skill[i] > -0.3 else None for i in range(n)]
    train = pl.DataFrame(
        {
            "player_id": list(range(n)),
            "full_name": [f"P{i}" for i in range(n)],
            "draft_year": [2018 + (i % 3) for i in range(n)],
            "pts_per40": np.clip(15 + 4 * skill, 3, 30),
            "ast_per40": np.clip(3 + skill, 0, 8),
            "reb_per40": np.clip(6 + skill, 1, 14),
            "blk_per40": np.clip(0.8 + 0.3 * skill, 0, 3),
            "stl_per40": np.clip(1.0 + 0.2 * skill, 0, 2.5),
            "true_shooting": np.clip(0.55 + 0.03 * skill, 0.45, 0.65),
            "wingspan_in": np.clip(82 + 1.5 * skill, 76, 92),
            "age_at_draft": np.clip(20 - 0.3 * skill, 18, 23),
            "reached": [bool(s > -0.3) for s in skill],
        }
    ).with_columns(pl.Series("peak_impact", peak, dtype=pl.Float64))
    feats = ["pts_per40", "ast_per40", "reb_per40", "blk_per40", "stl_per40",
             "true_shooting", "wingspan_in", "age_at_draft"]
    service = build_service_from_table(
        train, feats, target_col="peak_impact", reached_col="reached"
    )

    pool = pl.DataFrame(
        {
            "player_id": [900, 901, 902],
            "full_name": ["Rookie A", "Rookie B", "Rookie C"],
            "draft_year": [2026, 2026, 2026],
            "draft_pick": [1, 2, 15],
            "team_abbr": ["ATL", "WAS", "LAL"],
            "position": ["G", "F", "C"],
            "pts_per40": [24.0, 18.0, 12.0], "ast_per40": [6.0, 3.0, 1.0],
            "reb_per40": [5.0, 7.0, 11.0], "blk_per40": [0.3, 1.2, 2.4],
            "stl_per40": [2.0, 1.0, 0.4], "true_shooting": [0.60, 0.56, 0.62],
            "wingspan_in": [80.0, 84.0, 89.0], "age_at_draft": [19.2, 20.1, 18.7],
        }
    )
    board = service.ranked_with_profile(pool)
    # Skills are derived from per-40 (non-zero) and age comes from age_at_draft.
    assert board.select([f"skill_{d}" for d in SKILL_DIMS]).to_numpy().sum() > 0
    assert board["age"].null_count() == 0
    # model_rank is a dense 1..n ranking; slot_delta = draft_pick - model_rank.
    assert sorted(board["model_rank"].to_list()) == [1, 2, 3]
    rows = {r["player_id"]: r for r in board.to_dicts()}
    for r in rows.values():
        assert r["slot_delta"] == r["draft_pick"] - r["model_rank"]
        assert r["team_abbr"] in {"ATL", "WAS", "LAL"}
        assert r["headshot_url"].endswith(f"{r['player_id']}.png")


def test_counterfactual_lifts_a_low_prospect_toward_next_tier(service_and_pool):
    import polars as pl

    service, pool = service_and_pool
    board = service.rank(pool)
    # the weakest prospect should have a reachable next-tier target
    low_name = board.sort("projected_impact")["full_name"][0]
    row = pool.filter(pl.col("full_name") == low_name)
    cf = service.counterfactual(row)
    assert cf.target is not None  # not already top-tier
    assert cf.target_tier is not None
    # the proposed changes move the projection upward toward the target
    assert cf.projected_impact >= cf.current_impact
    assert all(c.feature in service.feature_cols for c in cf.changes)
    if cf.reached:
        assert cf.projected_impact >= cf.target


def test_counterfactual_top_tier_needs_no_change(service_and_pool):
    import polars as pl

    service, pool = service_and_pool
    board = service.rank(pool)
    top_name = board.sort("projected_impact", descending=True)["full_name"][0]
    row = pool.filter(pl.col("full_name") == top_name)
    cf = service.counterfactual(row)
    if cf.current_tier == TIER_LABELS[-1]:
        assert cf.target is None and cf.reached and cf.changes == []


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


def test_meta_reports_real_serving_metadata(client):
    # The default test app serves the synthetic demo (no NBA_DRAFT_AI_MASTER set).
    r = client.get("/meta")
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] in {"demo", "real"}
    assert body["n_prospects"] > 0  # real count from the served pool, not a fabricated string


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
    # synergy sub-scores + lineup before/after/replaced are exposed
    for key in (
        "synergy_complementarity",
        "synergy_redundancy",
        "synergy_net",
        "lineup_before",
        "lineup_after",
        "lineup_replaced",
    ):
        assert key in body
    # delta is consistent with before/after
    delta = body["lineup_after"] - body["lineup_before"]
    assert delta == pytest.approx(body["lineup_delta"], abs=0.01)


def test_counterfactual_endpoint_and_404(client):
    _, pool = build_demo_service()
    pid = int(pool["player_id"][0])
    r = client.get(f"/counterfactual/{pid}")
    assert r.status_code == 200
    body = r.json()
    assert body["player_id"] == pid
    assert "current_tier" in body and "changes" in body
    assert isinstance(body["changes"], list)
    # unknown id -> 404
    assert client.get("/counterfactual/99999999").status_code == 404


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


def test_build_service_from_master_serves_unlabeled_latest_class(tmp_path):
    """The 2026 use-case: an unlabeled latest class (no outcomes yet) is the projection pool and is
    excluded from training, which uses only the older resolved classes."""
    import json

    import numpy as np
    import polars as pl

    from nba_draft.service import build_service_from_master

    rng = np.random.default_rng(7)
    rows = []
    for year in (2018, 2019, 2020):  # resolved training classes
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
    # The current (2026) class: pre-draft features only, NO outcome labels.
    for _ in range(15):
        rows.append(
            {
                "player_id": len(rows),
                "full_name": f"Rookie{len(rows)}",
                "draft_year": 2026,
                "f_skill": rng.normal(),
                "reached": None,
                "peak_impact": None,
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
    # Only the unlabeled 2026 class is served as the pool...
    assert pool["draft_year"].unique().to_list() == [2026]
    assert pool["peak_impact"].null_count() == pool.height  # no outcomes
    # ...and the board still ranks it (trained on the older labeled classes).
    board = service.ranked_with_profile(pool)
    assert board.height == pool.height
    assert "projected_impact" in board.columns and board["model_rank"].n_unique() == pool.height
