"""Phase 8 — player-team FIT modeling.

The most differentiating and most exploratory part: a player's value depends on the team that
drafts them. Components (each transparent and individually testable):
  * archetypes      — functional player profiles via clustering on style skills
  * need            — what the current roster lacks (functional, not positional)
  * synergy         — complementarity minus redundancy vs the roster
  * lineup          — projected Net Rating change if the rookie replaces the weakest link
  * financial       — Real Surplus Value under the CBA/aprons (cheap rookie deal = scarce asset)
  * score           — combined player-team fit, with explicit uncertainty/assumption flags

Everything here is more exploratory and higher-variance than the individual projections; outputs
carry that caveat and list their assumptions (CBA projections, $-per-win conversion).
"""

from nba_draft.fit.archetypes import ArchetypeModel, skills_matrix
from nba_draft.fit.financial import (
    CBAConfig,
    apron_pressure_multiplier,
    load_cba,
    real_surplus_value,
    rookie_cost_total,
)
from nba_draft.fit.lineup import LineupSim, LineupWeights, lineup_upgrade, simulate_net_rating
from nba_draft.fit.need import functional_need
from nba_draft.fit.score import FitResult, player_team_fit
from nba_draft.fit.synergy import SynergyScore, synergy_score
from nba_draft.fit.types import SKILL_DIMS, Player, TeamContext

__all__ = [
    "SKILL_DIMS",
    "ArchetypeModel",
    "CBAConfig",
    "FitResult",
    "LineupSim",
    "LineupWeights",
    "Player",
    "SynergyScore",
    "TeamContext",
    "apron_pressure_multiplier",
    "functional_need",
    "lineup_upgrade",
    "load_cba",
    "player_team_fit",
    "real_surplus_value",
    "rookie_cost_total",
    "simulate_net_rating",
    "skills_matrix",
    "synergy_score",
]
