"""Skill synergy vs redundancy: does a prospect fill scarce needs or duplicate strengths?"""

from __future__ import annotations

from dataclasses import dataclass

from nba_draft.fit.need import functional_need
from nba_draft.fit.types import SKILL_DIMS, Player


@dataclass(frozen=True)
class SynergyScore:
    complementarity: float   # how much the prospect addresses team needs
    redundancy: float        # how much the prospect overlaps with existing strengths
    net: float               # complementarity - redundancy_weight * redundancy

    @property
    def is_good_fit(self) -> bool:
        return self.net > 0


def synergy_score(
    roster: list[Player],
    prospect: Player,
    *,
    redundancy_weight: float = 0.5,
    target_level: float = 60.0,
) -> SynergyScore:
    """Score how a prospect's skills mesh with the roster.

    complementarity = Σ need[skill] · prospect_skill[skill]   (filling gaps is rewarded)
    redundancy      = Σ surplus[skill] · prospect_skill[skill] (piling onto strengths is discounted)
    where surplus = max(0, supply - target_level). Both are scaled to roughly 0-100.
    """
    needs = functional_need(roster, target_level=target_level)
    # Supply/surplus per skill (mirror of need): surplus where the team is already strong.
    surplus: dict[str, float] = {}
    for d in SKILL_DIMS:
        vals = sorted((p.skills.get(d, 0.0) for p in roster), reverse=True)
        supply = (vals[0] + vals[1]) / 2 if len(vals) >= 2 else (vals[0] if vals else 0.0)
        surplus[d] = max(0.0, supply - target_level)

    p = prospect.skills
    comp = sum(needs[d] * p.get(d, 0.0) for d in SKILL_DIMS) / (100.0 * len(SKILL_DIMS))
    redun = sum(surplus[d] * p.get(d, 0.0) for d in SKILL_DIMS) / (100.0 * len(SKILL_DIMS))
    return SynergyScore(
        complementarity=round(comp, 4),
        redundancy=round(redun, 4),
        net=round(comp - redundancy_weight * redun, 4),
    )
