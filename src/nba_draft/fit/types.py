"""Core types for fit modeling: functional skill dimensions and the Player record."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

# Functional skill dimensions (0-100 scale, ~percentile vs NBA peers). Functional, not positional.
SKILL_DIMS: tuple[str, ...] = (
    "scoring",
    "shooting_spacing",
    "playmaking",
    "rebounding",
    "rim_protection",
    "perimeter_defense",
)


@dataclass(frozen=True)
class Player:
    """A player as seen by the fit module.

    Attributes:
        name: Display name.
        skills: Map of each SKILL_DIM -> 0-100 value (percentile-like).
        impact: Projected net impact per 100 possessions (BPM-like). Lineup Net Rating is
            approximated as the sum of lineup members' impact, since BPM is constructed so a
            lineup's BPMs sum to ~its efficiency differential.
        salary_usd: Current/known salary (0 for an undrafted prospect).
        archetype: Optional assigned archetype label.
    """

    name: str
    skills: dict[str, float]
    impact: float = 0.0
    salary_usd: float = 0.0
    archetype: str | None = None

    def skill_vector(self) -> NDArray[np.float64]:
        """Skills as a vector in SKILL_DIMS order (missing dims -> 0)."""
        return np.array([float(self.skills.get(d, 0.0)) for d in SKILL_DIMS], dtype=np.float64)


@dataclass
class TeamContext:
    """The drafting team's situation for financial fit."""

    roster: list[Player] = field(default_factory=list)
    total_salary_usd: float = 0.0

    def lineup_by_usage(self, n: int = 5) -> list[Player]:
        """Most-used lineup proxy: top-n roster players by impact (stand-in for minutes)."""
        return sorted(self.roster, key=lambda p: p.impact, reverse=True)[:n]
