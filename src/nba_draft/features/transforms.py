"""Stateless feature transforms (pure functions of pre-draft stats).

No cross-player fitting happens here, so these cannot leak: each output depends only on the
prospect's own pre-draft inputs. Justification for each family is in the docstrings.
"""

from __future__ import annotations

import polars as pl

_EPS = 1e-9


def add_stateless_features(df: pl.DataFrame) -> pl.DataFrame:
    """Per-row production-style features.

    Families & rationale:
      * versatility_index — Gini-Simpson balance of pts/ast/reb shares (1 = perfectly balanced,
        0 = one-dimensional). Captures all-around vs specialist profiles for archetyping.
      * playmaking_share — assists relative to scoring+playmaking load (creation vs finishing).
      * scoring_load — points share of total box production (on-ball scoring tilt).
    Efficiency (true_shooting) and usage are passed through from the source as-is.
    """
    pts, ast, reb = pl.col("pts_per100"), pl.col("ast_per100"), pl.col("reb_per100")
    total = pts + ast + reb + _EPS
    p_pts, p_ast, p_reb = pts / total, ast / total, reb / total
    versatility = 1.0 - (p_pts**2 + p_ast**2 + p_reb**2)
    return df.with_columns(
        versatility.alias("versatility_index"),
        (ast / (pts + ast + _EPS)).alias("playmaking_share"),
        p_pts.alias("scoring_load"),
    )


def add_combine_features(combine: pl.DataFrame) -> pl.DataFrame:
    """Sign-normalize Combine measurements so 'higher = better' for every athletic feature.

    Lane agility time and body fat are inverted (faster / leaner is better). Length and
    explosiveness pass through. Stays null where not measured (imputation handles it later).
    """
    out = combine
    if "lane_agility_s" in out.columns:
        out = out.with_columns((-pl.col("lane_agility_s")).alias("agility_score"))
    if "body_fat_pct" in out.columns:
        out = out.with_columns((-pl.col("body_fat_pct")).alias("leanness_score"))
    if "max_vertical_in" in out.columns:
        out = out.with_columns(pl.col("max_vertical_in").alias("explosiveness"))
    return out


def sequence_features(
    prospect_season: pl.DataFrame,
    *,
    player_col: str = "player_id",
    season_col: str = "season",
) -> pl.DataFrame:
    """Per-player year-over-year features from a prospect's ordered pre-draft seasons.

    Captures the Transfer-Portal / NIL-era dynamics the spec calls out:
      * sos_jump            — change in strength_of_schedule (rising competition is SIGNAL)
      * usage_change        — role change (secondary -> primary option)
      * ts_change           — did efficiency hold as the level rose?
      * efficiency_held_up  — True if TS did not fall while SoS rose (the key robustness signal)
      * pts_yoy_delta       — raw scoring improvement
      * n_pre_draft_seasons — sample depth (more seasons = more stable read)

    Single-season players get null deltas (flagged via n_pre_draft_seasons == 1) — absence of a
    delta is not zero improvement.
    """
    rows: list[dict[str, object]] = []
    for pid, g in prospect_season.group_by(player_col, maintain_order=True):
        player_id = pid[0] if isinstance(pid, tuple) else pid
        gg = g.sort(season_col)
        n = gg.height
        rec: dict[str, object] = {player_col: player_id, "n_pre_draft_seasons": n}
        if n >= 2:
            last, prev = gg.row(-1, named=True), gg.row(-2, named=True)

            def _delta(key: str) -> float | None:
                a, b = last[key], prev[key]  # noqa: B023 - intentional closure over last/prev
                return None if a is None or b is None else float(a) - float(b)

            sos_jump = _delta("strength_of_schedule")
            ts_change = _delta("true_shooting")
            rec.update(
                {
                    "sos_jump": sos_jump,
                    "usage_change": _delta("usage"),
                    "ts_change": ts_change,
                    "pts_yoy_delta": _delta("pts_per100"),
                    "efficiency_held_up": (
                        None
                        if sos_jump is None or ts_change is None
                        else bool(sos_jump > 0 and ts_change >= 0)
                    ),
                }
            )
        else:
            rec.update(
                {
                    "sos_jump": None,
                    "usage_change": None,
                    "ts_change": None,
                    "pts_yoy_delta": None,
                    "efficiency_held_up": None,
                }
            )
        rows.append(rec)
    return pl.DataFrame(
        rows,
        schema={
            player_col: pl.Utf8,
            "n_pre_draft_seasons": pl.Int64,
            "sos_jump": pl.Float64,
            "usage_change": pl.Float64,
            "ts_change": pl.Float64,
            "pts_yoy_delta": pl.Float64,
            "efficiency_held_up": pl.Boolean,
        },
    )
