"""Combined player-team fit score: basketball fit + lineup impact + financial surplus.

Transparent and component-wise by design. The overall score blends a normalized basketball-fit
component (synergy + lineup Net-Rating delta) with a financial component (apron-modulated Real
Surplus Value). It is explicitly flagged as exploratory and carries its assumptions, because fit
is higher-variance and less validated than the individual projections.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nba_draft.fit.financial import (
    CBAConfig,
    apron_pressure_multiplier,
    real_surplus_value,
)
from nba_draft.fit.lineup import LineupSim, LineupWeights, lineup_upgrade
from nba_draft.fit.synergy import SynergyScore, synergy_score
from nba_draft.fit.types import Player, TeamContext


@dataclass
class FitResult:
    synergy: SynergyScore
    lineup: LineupSim
    rsv_usd: float
    rsv_modulated_usd: float
    apron_label: str
    basketball_fit: float       # 0-100-ish normalized basketball fit
    financial_fit: float        # 0-100-ish normalized financial fit
    overall: float              # blended 0-100-ish
    narrative: str
    exploratory: bool = True
    assumptions: list[str] = field(default_factory=list)


def _norm_rsv_to_score(rsv_usd: float, scale_usd: float = 30_000_000.0) -> float:
    """Map RSV ($) to ~0-100 via a saturating transform; $0 surplus -> 50."""
    import math

    return 50.0 * (1.0 + math.tanh(rsv_usd / scale_usd))


def player_team_fit(
    team: TeamContext,
    prospect: Player,
    *,
    pick: int,
    projected_vorp_per_year: float,
    cba: CBAConfig,
    weights: LineupWeights | None = None,
    basketball_weight: float = 0.6,
) -> FitResult:
    """Score a prospect's fit with a specific team.

    Args:
        team: The drafting team's roster + total salary.
        prospect: The prospect as a :class:`Player` (skills + projected impact).
        pick: Draft slot (drives rookie-scale cost).
        projected_vorp_per_year: The prospect's projected per-year VORP (from the models).
        cba: Loaded CBA parameters for the relevant season.
        basketball_weight: Blend weight for basketball vs financial fit (0..1).

    Returns:
        A component-wise :class:`FitResult` with a GM-readable narrative and assumption flags.
    """
    syn = synergy_score(team.roster, prospect)
    lineup = lineup_upgrade(team.lineup_by_usage(), prospect, weights)

    rsv = real_surplus_value(projected_vorp_per_year, pick, cba)
    mult, apron_label = apron_pressure_multiplier(team.total_salary_usd, cba)
    rsv_mod = round(rsv * mult, 2)

    # Normalize components to ~0-100. Basketball fit: synergy.net (~0-1) + lineup delta (per 100).
    basketball_fit = max(0.0, min(100.0, 50.0 + 100.0 * syn.net + 5.0 * lineup.delta))
    financial_fit = _norm_rsv_to_score(rsv_mod)
    overall = round(basketball_weight * basketball_fit + (1 - basketball_weight) * financial_fit, 1)

    narrative = (
        f"{lineup.narrative} "
        f"Synergy net {syn.net:+.2f} (fills needs {syn.complementarity:.2f}, "
        f"overlap {syn.redundancy:.2f}). "
        f"Real Surplus Value ${rsv:,.0f} over the rookie deal, "
        f"x{mult:g} ({apron_label}) -> ${rsv_mod:,.0f} effective. "
        f"Overall fit {overall:.0f}/100 (EXPLORATORY)."
    )

    return FitResult(
        synergy=syn,
        lineup=lineup,
        rsv_usd=rsv,
        rsv_modulated_usd=rsv_mod,
        apron_label=apron_label,
        basketball_fit=round(basketball_fit, 1),
        financial_fit=round(financial_fit, 1),
        overall=overall,
        narrative=narrative,
        exploratory=True,
        assumptions=cba.assumptions,
    )
