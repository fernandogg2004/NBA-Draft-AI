"""Tests for Phase 8 fit modeling: archetypes, need, synergy, lineup sim, financial, score."""

from __future__ import annotations

import pytest

from nba_draft.fit import (
    ArchetypeModel,
    Player,
    TeamContext,
    apron_pressure_multiplier,
    functional_need,
    lineup_upgrade,
    load_cba,
    player_team_fit,
    real_surplus_value,
    rookie_cost_total,
    simulate_net_rating,
    synergy_score,
)


def _scorer(name: str, impact: float = 2.0) -> Player:
    return Player(
        name=name,
        skills={
            "scoring": 80, "shooting_spacing": 75, "playmaking": 60,
            "rebounding": 30, "rim_protection": 20, "perimeter_defense": 35,
        },
        impact=impact,
    )


def _rim_protector(name: str, impact: float = 3.0) -> Player:
    return Player(
        name=name,
        skills={
            "scoring": 35, "shooting_spacing": 25, "playmaking": 20,
            "rebounding": 85, "rim_protection": 90, "perimeter_defense": 70,
        },
        impact=impact,
    )


def _roster_of_scorers() -> list[Player]:
    return [_scorer(f"S{i}", impact=1.5 + 0.1 * i) for i in range(5)]


# ----------------------------------------------------------------- archetypes
def test_archetypes_separate_styles():
    players = _roster_of_scorers() + [_rim_protector(f"R{i}") for i in range(4)]
    model = ArchetypeModel(n_archetypes=2, seed=0).fit(players)
    labels = model.predict(players)
    # the scorers and rim protectors should land in different clusters
    assert labels[0] != labels[-1]


def test_archetypes_need_enough_players():
    with pytest.raises(ValueError):
        ArchetypeModel(n_archetypes=5).fit([_scorer("only")])


# ----------------------------------------------------------------- need + synergy
def test_functional_need_flags_missing_rim_protection():
    needs = functional_need(_roster_of_scorers())
    # roster of scorers lacks rim protection -> high need there, ~zero for scoring
    assert needs["rim_protection"] > needs["scoring"]
    assert needs["scoring"] == pytest.approx(0.0)


def test_synergy_rewards_filling_need_over_redundancy():
    roster = _roster_of_scorers()
    filler = _rim_protector("Filler")
    duplicate = _scorer("Duplicate")
    s_fill = synergy_score(roster, filler)
    s_dupe = synergy_score(roster, duplicate)
    assert s_fill.net > s_dupe.net
    assert s_fill.is_good_fit
    # the duplicate scorer mostly piles onto an existing strength
    assert s_dupe.redundancy > s_fill.redundancy


# ----------------------------------------------------------------- lineup simulation
def test_lineup_upgrade_replaces_weakest_and_changes_net_rating():
    roster = _roster_of_scorers()
    weakest = min(roster, key=lambda p: p.impact)
    rookie = _rim_protector("Rookie", impact=4.0)  # higher impact + fills rim need
    sim = lineup_upgrade(roster, rookie)
    assert sim.replaced == weakest.name
    assert sim.after > sim.before        # better player + better balance raises Net Rating
    assert sim.delta == pytest.approx(round(sim.after - sim.before, 2))
    assert "Net Rating goes from" in sim.narrative


def test_simulate_net_rating_rewards_spacing():
    # two lineups identical except for spacing (rim protection held adequate & equal)
    def _p(name: str, spacing: float) -> Player:
        return Player(
            name=name,
            skills={"shooting_spacing": spacing, "rim_protection": 60},
            impact=2.0,
        )

    spaced = [_p(f"A{i}", 80) for i in range(5)]
    cramped = [_p(f"B{i}", 30) for i in range(5)]
    # equal impact and rim protection -> spacing term decides
    assert simulate_net_rating(spaced) > simulate_net_rating(cramped)


# ----------------------------------------------------------------- financial
def test_cba_loads_and_flags_projection_assumption():
    cba = load_cba()  # default season 2026-27 (projected)
    assert cba.season == "2026-27"
    assert cba.verified is False
    assert any("PROJECTION" in a for a in cba.assumptions)


def test_verified_season_has_no_projection_flag():
    cba = load_cba(season="2025-26")
    assert cba.verified is True
    assert not any("PROJECTION" in a for a in cba.assumptions)


def test_rookie_cost_decreases_with_pick():
    cba = load_cba(season="2025-26")
    assert rookie_cost_total(1, cba) > rookie_cost_total(14, cba) > rookie_cost_total(30, cba)


def test_real_surplus_value_positive_for_cheap_star():
    cba = load_cba(season="2025-26")
    # a 3.0 VORP/yr player projects to far more than a late-lottery rookie deal
    rsv = real_surplus_value(3.0, pick=10, cba=cba)
    assert rsv > 0


def test_apron_multiplier_higher_when_strangled():
    cba = load_cba(season="2025-26")
    below, _ = apron_pressure_multiplier(100_000_000, cba)
    over2, label = apron_pressure_multiplier(cba.second_apron_usd + 1, cba)
    assert over2 > below
    assert "second apron" in label


# ----------------------------------------------------------------- combined score
def test_player_team_fit_combines_components_with_caveats():
    team = TeamContext(roster=_roster_of_scorers(), total_salary_usd=210_000_000)  # over 2nd apron
    cba = load_cba(season="2025-26")
    rookie = _rim_protector("Wonder Rookie", impact=4.0)
    result = player_team_fit(
        team, rookie, pick=8, projected_vorp_per_year=3.0, cba=cba
    )
    assert result.exploratory is True
    assert 0 <= result.overall <= 100
    assert result.rsv_modulated_usd > result.rsv_usd        # apron pressure amplifies surplus
    assert "Net Rating goes from" in result.narrative
    assert "EXPLORATORY" in result.narrative
