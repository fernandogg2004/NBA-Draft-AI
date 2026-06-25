"""Deterministic SYNTHETIC fixture for pipeline verification.

This is NOT real data and must never be used for any basketball conclusion. Its only job is
to exercise the end-to-end skeleton (temporal split -> baseline -> evaluation) so the
architecture is verifiable before real data acquisition (Phase 1) is complete.

Design choices that make it a meaningful smoke test:
  * A latent "talent" drives the NBA-impact target with heavy-tailed noise (busts & steals).
  * Draft pick is correlated with latent talent but noisily, so the draft-position baseline
    is informative yet beatable — exactly the real setup.
  * Pre-draft features (age, per-100 production, efficiency, wingspan, league tier) carry
    signal about talent, available BEFORE the draft (no leakage).
"""

from __future__ import annotations

import numpy as np
import polars as pl

# Columns a downstream consumer can rely on from the fixture.
FEATURE_COLUMNS = [
    "age",
    "pts_per100",
    "ast_per100",
    "reb_per100",
    "true_shooting",
    "usage",
    "wingspan_in",
    "league_tier",
]
TARGET_COLUMN = "nba_impact"  # stand-in for a BPM/VORP-style first-4-years impact metric
PICK_COLUMN = "draft_pick"
YEAR_COLUMN = "draft_year"
ID_COLUMN = "player_id"


def make_synthetic_prospects(
    *,
    start_year: int = 2010,
    n_years: int = 14,
    per_year: int = 60,
    seed: int = 42,
) -> pl.DataFrame:
    """Generate a deterministic synthetic prospect table spanning several draft classes.

    Args:
        start_year: First draft year.
        n_years: Number of consecutive draft classes.
        per_year: Prospects per class.
        seed: RNG seed for reproducibility.

    Returns:
        A Polars DataFrame with id, draft_year, pre-draft features, draft_pick, and target.
    """
    rng = np.random.default_rng(seed)
    rows = []
    pid = 0
    for k in range(n_years):
        year = start_year + k
        # Latent talent ~ standard normal; mild era drift so classes differ slightly.
        talent = rng.normal(0.1 * k * 0.0, 1.0, size=per_year)  # era drift kept ~0 for now

        age = np.clip(rng.normal(19.5, 1.3, per_year), 17.5, 24.0)
        # Younger producers are more impressive: talent gets an age-adjusted boost downstream.
        pts = np.clip(12 + 4 * talent - 1.2 * (age - 19.5) + rng.normal(0, 2, per_year), 2, 35)
        ast = np.clip(3 + 1.5 * talent + rng.normal(0, 1, per_year), 0, 12)
        reb = np.clip(6 + 1.2 * talent + rng.normal(0, 1.5, per_year), 1, 16)
        ts = np.clip(0.55 + 0.04 * talent + rng.normal(0, 0.03, per_year), 0.40, 0.72)
        usage = np.clip(0.22 + 0.05 * talent + rng.normal(0, 0.03, per_year), 0.10, 0.40)
        wing = np.clip(rng.normal(82 + 1.5 * talent, 2.5, per_year), 72, 95)
        tier = rng.integers(1, 4, per_year)  # 1=strong .. 3=weak league

        # Draft pick: better talent tends to go earlier, but noisily (scouting error).
        pick_score = -talent + rng.normal(0, 0.6, per_year)
        order = np.argsort(pick_score)
        pick = np.empty(per_year, dtype=np.int64)
        pick[order] = np.arange(1, per_year + 1)

        # Target: impact driven by talent + youth bonus + heavy-tailed noise (fat tails).
        youth_bonus = 0.4 * (19.5 - age)
        noise = rng.standard_t(df=3, size=per_year) * 1.2  # t-dist -> busts & steals
        impact = 2.0 * talent + youth_bonus + noise

        for i in range(per_year):
            rows.append(
                {
                    ID_COLUMN: pid,
                    YEAR_COLUMN: int(year),
                    "age": float(age[i]),
                    "pts_per100": float(pts[i]),
                    "ast_per100": float(ast[i]),
                    "reb_per100": float(reb[i]),
                    "true_shooting": float(ts[i]),
                    "usage": float(usage[i]),
                    "wingspan_in": float(wing[i]),
                    "league_tier": int(tier[i]),
                    PICK_COLUMN: int(pick[i]),
                    TARGET_COLUMN: float(impact[i]),
                }
            )
            pid += 1

    return pl.DataFrame(rows)


def make_multisource_fixture() -> dict[str, pl.DataFrame]:
    """SYNTHETIC multi-source frames for Phase 2 (entity resolution + integration).

    Deliberately includes the messiness real integration must survive:
      * name variants across sources ("Young, Trae" vs "Trae Young"; suffix "III"; accents)
      * Combine measurements present for only some prospects (not-measured -> null)
      * international rows with ONLY basic stats (advanced metrics absent, not zero)

    Returns a dict with keys: 'college_stats', 'intl_stats' (season frames) and
    'combine' (measurements). Expected canonical entities: 6.
    """
    college_stats = pl.DataFrame(
        [
            # name (variant), draft_yr, league, season, age, pts, ast, reb, ts, usg, bpm, sos
            ("Young, Trae", 2018, "NCAA", 2017, 19.1, 27.4, 8.7, 3.9, 0.595, 0.36, 11.2, 8.1),
            ("Marvin Bagley III", 2018, "NCAA", 2017, 19.0, 21.0, 1.5, 11.1, 0.612, 0.30, 9.4, 7.5),
            ("Deandre Ayton", 2018, "NCAA", 2017, 19.6, 20.1, 1.6, 11.6, 0.638, 0.28, 8.8, 7.9),
        ],
        schema=[
            "full_name", "draft_year", "league", "season", "age",
            "pts_per100", "ast_per100", "reb_per100",
            "true_shooting", "usage", "bpm_college", "strength_of_schedule",
        ],
        orient="row",
    )

    # International rows: ONLY basic stats — advanced columns are simply absent.
    intl_stats = pl.DataFrame(
        [
            ("Luka Dončić", 2018, "EuroLeague", 2017, 19.0, 21.2, 6.0, 5.3),
            ("Bogdan Bogdanović", 2014, "EuroLeague", 2013, 21.5, 18.0, 4.1, 3.4),
        ],
        schema=[
            "full_name", "draft_year", "league", "season", "age",
            "pts_per100", "ast_per100", "reb_per100",
        ],
        orient="row",
    )

    # Combine: 'Trae Young' / 'Marvin Bagley' (no suffix) / 'Deandre Ayton' link to college;
    # 'Wendell Carter' is combine-only -> its own entity. Some measurements missing (null).
    combine = pl.DataFrame(
        [
            ("Trae Young", 2018, 76.5, 100.5, 33.0, 11.1, 7.0),
            ("Marvin Bagley", 2018, 87.0, 113.0, 35.5, None, 6.5),
            ("Deandre Ayton", 2018, 87.25, 114.0, None, 11.5, 8.0),
            ("Wendell Carter", 2018, 83.5, 110.0, 30.0, 11.8, 9.0),
        ],
        schema=[
            "full_name", "draft_year",
            "wingspan_in", "standing_reach_in", "max_vertical_in",
            "lane_agility_s", "body_fat_pct",
        ],
        orient="row",
    )

    return {"college_stats": college_stats, "intl_stats": intl_stats, "combine": combine}


def make_feature_fixture() -> dict[str, pl.DataFrame]:
    """SYNTHETIC master-shaped tables for Phase 4 feature engineering.

    Player ids are given directly (entity resolution already done). Includes:
      * multi-season NCAA players (a Transfer-Portal case: mid-major -> high-major, SoS jumps)
      * an international single-season player with NULL strength_of_schedule (sparse case)
    Returns {'prospect_season', 'combine', 'identity'}.
    """
    prospect_season = pl.DataFrame(
        [
            # player_id, draft_year, league_id, season, age, pts, ast, reb, ts, usage, sos
            # p1: transfer — 2016 mid-major (low SoS) -> 2017 high-major (high SoS), eff. holds
            ("p1", 2018, "ncaa", 2016, 18.5, 18.0, 3.0, 5.0, 0.560, 0.24, 2.0),
            ("p1", 2018, "ncaa", 2017, 19.5, 22.0, 4.0, 5.5, 0.585, 0.31, 8.5),
            # p2: stayed, role grew (usage up), efficiency dipped as competition rose
            ("p2", 2018, "ncaa", 2016, 18.8, 10.0, 5.0, 3.0, 0.575, 0.18, 7.0),
            ("p2", 2018, "ncaa", 2017, 19.8, 17.0, 6.5, 3.4, 0.545, 0.29, 9.0),
            # p3: one-and-done high-major, single season
            ("p3", 2018, "ncaa", 2017, 19.1, 20.0, 2.0, 11.0, 0.620, 0.27, 8.0),
            # p4: international, single season, SoS unknown (null)
            ("p4", 2018, "euroleague", 2017, 19.0, 14.0, 5.5, 4.0, 0.560, 0.23, None),
            # p5: international, single season, basic-ish, null SoS
            ("p5", 2018, "euroleague", 2017, 20.2, 12.0, 3.0, 6.0, 0.540, 0.21, None),
            # p6: prior-class NCAA player (for temporal train/val separation in tests)
            ("p6", 2017, "ncaa", 2016, 19.4, 16.0, 4.5, 5.0, 0.565, 0.25, 7.5),
        ],
        schema=[
            "player_id", "draft_year", "league_id", "season", "age",
            "pts_per100", "ast_per100", "reb_per100",
            "true_shooting", "usage", "strength_of_schedule",
        ],
        orient="row",
    )

    combine = pl.DataFrame(
        [
            ("p1", 78.0, 102.0, 36.0, 11.0, 6.5),
            ("p2", 80.5, 105.0, 34.0, 11.2, 7.0),
            ("p3", 87.0, 113.0, 31.0, 11.6, 8.5),
            # p4, p5 (international) did not test at the Combine -> absent
            ("p6", 81.0, 106.0, 35.0, 11.1, 7.5),
        ],
        schema=[
            "player_id", "wingspan_in", "standing_reach_in",
            "max_vertical_in", "lane_agility_s", "body_fat_pct",
        ],
        orient="row",
    )

    identity = pl.DataFrame(
        {
            "player_id": ["p1", "p2", "p3", "p4", "p5", "p6"],
            "full_name": ["P One", "P Two", "P Three", "P Four", "P Five", "P Six"],
            "draft_year": [2018, 2018, 2018, 2018, 2018, 2017],
        }
    )

    return {"prospect_season": prospect_season, "combine": combine, "identity": identity}
