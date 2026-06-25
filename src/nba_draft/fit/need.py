"""Functional roster need: what skills the team lacks.

Need is defined by SUPPLY: for each skill, how well the roster's best providers cover it. A team
with no rim protector has high rim-protection need even if it has five good scorers.
"""

from __future__ import annotations

import numpy as np

from nba_draft.fit.types import SKILL_DIMS, Player


def functional_need(
    roster: list[Player], *, target_level: float = 60.0, top_providers: int = 2
) -> dict[str, float]:
    """Per-skill need in [0, target_level].

    Supply for a skill = mean of the roster's top ``top_providers`` values in that skill.
    Need = max(0, target_level - supply): zero once the team adequately covers a skill.

    Args:
        roster: Current roster.
        target_level: The coverage level above which a skill is considered satisfied.
        top_providers: How many best providers define supply (a skill needs only a few specialists).
    """
    if not roster:
        return {d: target_level for d in SKILL_DIMS}
    needs: dict[str, float] = {}
    for d in SKILL_DIMS:
        vals = sorted((p.skills.get(d, 0.0) for p in roster), reverse=True)
        supply = float(np.mean(vals[:top_providers])) if vals else 0.0
        needs[d] = max(0.0, target_level - supply)
    return needs
