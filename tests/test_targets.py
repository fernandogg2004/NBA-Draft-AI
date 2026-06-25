"""Tests for the Phase 0 target definitions and label-construction contract."""

from __future__ import annotations

import pytest

from nba_draft.targets import (
    OutcomeTier,
    PlayerOutcome,
    SeasonStat,
    cumulative_value,
    is_label_resolved,
    load_target_config,
    outcome_tier,
    peak_impact,
    reached_role,
    unconditional_value,
)


@pytest.fixture(scope="module")
def cfg():
    return load_target_config()


def _seasons(start: int, specs: list[tuple[float, float, float]]) -> tuple[SeasonStat, ...]:
    """Build consecutive seasons from (minutes, bpm, vorp) specs starting at `start`."""
    return tuple(
        SeasonStat(season_year=start + i, minutes=m, bpm=b, vorp=v)
        for i, (m, b, v) in enumerate(specs)
    )


def test_config_loads_and_has_bust_fallback(cfg):
    assert cfg.horizon.primary_window_years == 4
    assert any(t.name == "bust" for t in cfg.tiers)


def test_never_debuted_is_non_reach_and_bust(cfg):
    out = PlayerOutcome(draft_year=2018, debut_year=None)
    assert reached_role(out, cfg) is False
    assert peak_impact(out, cfg) is None
    assert cumulative_value(out, cfg) == 0.0
    assert outcome_tier(out, cfg) is OutcomeTier.BUST


def test_debut_beyond_cap_is_non_reach(cfg):
    # debut 5 years after draft exceeds the 4-year cap -> treated as never reaching
    out = PlayerOutcome(
        draft_year=2015,
        debut_year=2020,
        seasons=_seasons(2020, [(2000, 5.0, 4.0)]),
    )
    assert reached_role(out, cfg) is False
    assert outcome_tier(out, cfg) is OutcomeTier.BUST


def test_stash_player_within_cap_counts(cfg):
    # debut 2 years after draft (draft-and-stash) is within cap and window anchors on debut
    out = PlayerOutcome(
        draft_year=2015,
        debut_year=2017,
        seasons=_seasons(2017, [(1500, 1.0, 2.0), (1500, 2.0, 3.0)]),
    )
    assert reached_role(out, cfg) is True
    assert outcome_tier(out, cfg) is OutcomeTier.STARTER


def test_reach_requires_minimum_minutes(cfg):
    # debuted but logged few minutes across the window -> did not reach
    out = PlayerOutcome(
        draft_year=2018,
        debut_year=2018,
        seasons=_seasons(2018, [(200, -1.0, 0.0), (300, 0.0, 0.1)]),
    )
    assert reached_role(out, cfg) is False
    assert outcome_tier(out, cfg) is OutcomeTier.BUST


def test_peak_is_mean_of_top_two_qualifying_seasons(cfg):
    out = PlayerOutcome(
        draft_year=2018,
        debut_year=2018,
        seasons=_seasons(2018, [(1000, 1.0, 2.0), (1000, 5.0, 4.0), (1000, 3.0, 3.0)]),
    )
    # top-2 BPM = 5.0 and 3.0 -> mean 4.0
    assert peak_impact(out, cfg) == pytest.approx(4.0)


def test_peak_ignores_low_minute_seasons(cfg):
    # the 10.0 BPM season is below stable_season_minutes and must be ignored
    out = PlayerOutcome(
        draft_year=2018,
        debut_year=2018,
        seasons=_seasons(2018, [(1000, 2.0, 2.0), (100, 10.0, 0.1)]),
    )
    assert peak_impact(out, cfg) == pytest.approx(2.0)


def test_window_truncates_to_primary_years(cfg):
    # 5 seasons but only the first 4 (window) count toward cumulative VORP
    out = PlayerOutcome(
        draft_year=2018,
        debut_year=2018,
        seasons=_seasons(2018, [(1000, 1.0, 1.0)] * 5),
    )
    assert cumulative_value(out, cfg) == pytest.approx(4.0)


def test_tier_uses_honors_for_top_even_if_bpm_modest(cfg):
    # an All-Star selection promotes despite a sub-+3 peak BPM
    out = PlayerOutcome(
        draft_year=2017,
        debut_year=2017,
        seasons=_seasons(2017, [(2000, 2.5, 4.0)]),
        all_star_count=1,
    )
    assert outcome_tier(out, cfg) is OutcomeTier.ALL_STAR


def test_tier_superstar_by_bpm_band(cfg):
    out = PlayerOutcome(
        draft_year=2017,
        debut_year=2017,
        seasons=_seasons(2017, [(2000, 7.0, 6.0), (2000, 6.5, 6.0)]),
    )
    assert outcome_tier(out, cfg) is OutcomeTier.SUPERSTAR


def test_unconditional_value_blends_with_replacement(cfg):
    # high conditional value but low reach probability -> pulled toward replacement
    ev = unconditional_value(0.25, 8.0, replacement=cfg.replacement.bpm)
    assert ev == pytest.approx(0.25 * 8.0 + 0.75 * (-2.0))


def test_unconditional_value_rejects_bad_probability(cfg):
    with pytest.raises(ValueError):
        unconditional_value(1.5, 3.0, replacement=0.0)


def test_label_resolution_and_censoring(cfg):
    debuted = PlayerOutcome(
        draft_year=2018,
        debut_year=2018,
        seasons=_seasons(2018, [(1000, 1.0, 1.0)] * 4),
    )
    # window ends 2021; resolved once data reaches 2021, censored before
    assert is_label_resolved(debuted, data_through_year=2021, cfg=cfg) is True
    assert is_label_resolved(debuted, data_through_year=2020, cfg=cfg) is False

    # not yet debuted: censored until the debut cap passes, then a definitive non-reach
    pending = PlayerOutcome(draft_year=2024, debut_year=None)
    assert is_label_resolved(pending, data_through_year=2026, cfg=cfg) is False
    assert is_label_resolved(pending, data_through_year=2028, cfg=cfg) is True
