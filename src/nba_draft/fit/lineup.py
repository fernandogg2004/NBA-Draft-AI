"""Lineup Net Rating simulation: the concrete, actionable fit output.

Instructions.md demands more than an abstract fit score: estimate how a lineup's Net Rating
(points per 100 possessions) changes if its weakest link is replaced by the rookie.

Transparent model (clearly a simplified proxy, not a calibrated RAPM lineup model):
    Net Rating ≈ Σ player.impact            # BPM-like impacts sum to ~efficiency differential
               + spacing_term               # reward adequate floor spacing, penalize cramped floors
               + rim_term                    # penalize a lineup with no rim protection
The spacing/rim terms encode the synergy intuition that basketball is played in fives.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nba_draft.fit.types import Player


@dataclass(frozen=True)
class LineupWeights:
    spacing_coef: float = 0.06
    spacing_threshold: float = 55.0    # mean shooting_spacing below this cramps the floor
    rim_coef: float = 0.05
    rim_threshold: float = 50.0        # best rim_protection below this leaves the rim exposed


@dataclass(frozen=True)
class LineupSim:
    before: float
    after: float
    delta: float
    replaced: str
    narrative: str


def _spacing_term(lineup: list[Player], w: LineupWeights) -> float:
    mean_space = float(np.mean([p.skills.get("shooting_spacing", 0.0) for p in lineup]))
    return w.spacing_coef * (mean_space - w.spacing_threshold)


def _rim_term(lineup: list[Player], w: LineupWeights) -> float:
    best_rim = max((p.skills.get("rim_protection", 0.0) for p in lineup), default=0.0)
    return w.rim_coef * (best_rim - w.rim_threshold)


def simulate_net_rating(lineup: list[Player], weights: LineupWeights | None = None) -> float:
    """Approximate a lineup's Net Rating (points per 100 possessions)."""
    w = weights or LineupWeights()
    base = float(np.sum([p.impact for p in lineup]))
    return round(base + _spacing_term(lineup, w) + _rim_term(lineup, w), 2)


def lineup_upgrade(
    lineup: list[Player],
    prospect: Player,
    weights: LineupWeights | None = None,
) -> LineupSim:
    """Replace the lineup's weakest link (lowest impact) with the prospect; report the change.

    Returns before/after Net Rating, the delta, who was replaced, and a GM-readable narrative.
    """
    if not lineup:
        raise ValueError("lineup must contain at least one player.")
    w = weights or LineupWeights()
    weakest = min(lineup, key=lambda p: p.impact)
    new_lineup = [prospect if p is weakest else p for p in lineup]

    before = simulate_net_rating(lineup, w)
    after = simulate_net_rating(new_lineup, w)
    delta = round(after - before, 2)

    space_before = float(np.mean([p.skills.get("shooting_spacing", 0.0) for p in lineup]))
    space_after = float(np.mean([p.skills.get("shooting_spacing", 0.0) for p in new_lineup]))
    space_change = space_after - space_before
    way = "improves" if space_change >= 0 else "declines"
    narrative = (
        f"Replacing {weakest.name} (weakest link) with {prospect.name}: "
        f"lineup spacing {way} by {abs(space_change):.0f} pts of percentile, "
        f"Net Rating goes from {before:+.1f} to {after:+.1f} ({delta:+.1f} per 100)."
    )
    return LineupSim(
        before=before, after=after, delta=delta, replaced=weakest.name, narrative=narrative
    )
