"""NBA Draft AI — GM dashboard (Phase 11).

A GM picks their cap situation, roster style, and draft slot, and gets a ranked board with
projections, uncertainty intervals, the outcome-tier distribution, player-team fit (lineup Net
Rating + Real Surplus Value), and a SHAP explanation for the selected prospect.

Run:  streamlit run dashboard/streamlit_app.py
The demo is trained on the SYNTHETIC fixture — for wiring/UX only, no basketball meaning.
"""

from __future__ import annotations

import plotly.express as px
import polars as pl
import streamlit as st

from nba_draft.fit import Player, TeamContext, load_cba
from nba_draft.service import build_demo_service
from nba_draft.service.board import TIER_LABELS

st.set_page_config(page_title="NBA Draft AI", page_icon="🏀", layout="wide")


@st.cache_resource
def get_service():  # type: ignore[no-untyped-def]
    """Train (or load) the draft-board service once per session."""
    return build_demo_service()


ROSTER_PRESETS = {
    "Guard-heavy (needs size/defense)": dict(
        scoring=82, shooting_spacing=80, playmaking=70, rebounding=30,
        rim_protection=20, perimeter_defense=40,
    ),
    "Wing-heavy (needs playmaking/rim)": dict(
        scoring=70, shooting_spacing=72, playmaking=35, rebounding=45,
        rim_protection=35, perimeter_defense=65,
    ),
    "Balanced contender": dict(
        scoring=65, shooting_spacing=62, playmaking=58, rebounding=58,
        rim_protection=60, perimeter_defense=60,
    ),
}


def _preset_roster(skills: dict[str, float]) -> list[Player]:
    return [Player(f"Starter {i+1}", dict(skills), impact=1.5 + 0.3 * i) for i in range(5)]


def main() -> None:
    service, pool = get_service()

    st.title("🏀 NBA Draft AI — Decision Support")
    st.caption(
        "Projections • uncertainty • fit • explanations. "
        "⚠️ Demo on SYNTHETIC data — wiring/UX only."
    )

    # ---- sidebar: the GM's situation ----
    with st.sidebar:
        st.header("Your team")
        preset = st.selectbox("Roster style", list(ROSTER_PRESETS))
        cap_situation = st.selectbox(
            "Cap situation",
            ["below tax", "over luxury tax", "over first apron", "over second apron"],
            index=3,
        )
        pick = st.number_input("Your draft pick", min_value=1, max_value=60, value=8)
        st.divider()
        cba = load_cba()
        st.caption(f"CBA season {cba.season}" + ("" if cba.verified else " (projection)"))

    roster = _preset_roster(ROSTER_PRESETS[preset])
    salary_by_situation = {
        "below tax": 100_000_000,
        "over luxury tax": float(cba.luxury_tax_usd) + 1,
        "over first apron": float(cba.first_apron_usd) + 1,
        "over second apron": float(cba.second_apron_usd) + 1,
    }
    team = TeamContext(roster=roster, total_salary_usd=salary_by_situation[cap_situation])

    # ---- ranked board ----
    board = service.rank(pool)
    ranked_by_ev = "projected_ev" in board.columns
    with st.container(border=True):
        st.subheader(
            "Draft board (ranked by unconditional EV)"
            if ranked_by_ev
            else "Draft board (ranked by projected impact)"
        )
        if ranked_by_ev:
            st.caption(
                "EV = P(reach) · E(impact | reached) + (1 − P(reach)) · replacement — "
                "survivorship-robust over all prospects."
            )
        st.dataframe(board.to_pandas(), hide_index=True, width="stretch")

    # ---- per-prospect detail ----
    names = board["full_name"].to_list()
    chosen = st.selectbox("Inspect a prospect", names)
    row = pool.filter(pl.col("full_name") == chosen)
    brow = board.filter(pl.col("full_name") == chosen).row(0, named=True)

    with st.container(horizontal=True):
        if ranked_by_ev:
            st.metric("Projected EV", f"{brow['projected_ev']:.2f}", border=True)
            st.metric("P(reach)", f"{brow['p_reach']:.0%}", border=True)
        st.metric("Projected impact", f"{brow['projected_impact']:.2f}", border=True)
        st.metric("Floor (P10)", f"{brow['floor']:.2f}", border=True)
        st.metric("Ceiling (P90)", f"{brow['ceiling']:.2f}", border=True)

    col1, col2 = st.columns(2)
    with col1, st.container(border=True):
        st.markdown("**Outcome-tier distribution**")
        probs = {t: brow[f"p_{t}"] for t in TIER_LABELS}
        fig = px.bar(
            x=list(probs), y=list(probs.values()),
            labels={"x": "tier", "y": "probability"},
        )
        st.plotly_chart(fig, width="stretch")

    with col2, st.container(border=True):
        st.markdown("**Why (SHAP, top contributions)**")
        try:
            contrib, base = service.explain(row)
            top = contrib.head(6)
            figs = px.bar(top.to_pandas(), x="shap_value", y="feature", orientation="h")
            st.plotly_chart(figs, width="stretch")
            st.caption(f"base value {base:.2f}")
        except Exception as exc:  # noqa: BLE001 - dashboard should degrade gracefully
            st.warning(f"Explanation unavailable: {exc}")

    # ---- fit with this team ----
    with st.container(border=True):
        st.subheader("Fit with your team")
        result = service.fit_for_team(row, team, pick=int(pick), cba=cba, name=chosen)
        a, b, c = st.columns(3)
        a.metric("Overall fit", f"{result.overall:.0f}/100")
        b.metric("Lineup Net Rating Δ", f"{result.lineup.delta:+.1f}")
        c.metric("Surplus value (eff.)", f"${result.rsv_modulated_usd/1e6:.1f}M")
        st.info(result.narrative)
        if result.assumptions:
            st.caption("Assumptions: " + " ".join(result.assumptions))


if __name__ == "__main__":
    main()
