"""Typed target definitions and pure label-construction functions (Phase 0).

These functions turn a player's *realized* NBA outcomes into the labels the system predicts.
They are deliberately pure and data-source-agnostic: real ingestion (Phase 1/2) will populate
:class:`PlayerOutcome` records, and the same functions produce labels here and in evaluation.

Key design choices (confirmed 2026-06-25), see config/targets.yaml:
  * Hurdle structure: T1 reach-probability gates conditional impact, defeating survivorship.
  * Impact spine = box-score metrics: peak BPM (ceiling), cumulative VORP (realized value).
  * Tiers = hybrid: honors for top tiers, BPM bands below.
  * Horizon = debut-anchored, capped: must reach the NBA within `debut_cap_years` of draft.
  * Right-censoring of recent classes is explicit (`is_label_resolved`), never treated as zero.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from nba_draft.config import REPO_ROOT

DEFAULT_TARGETS_PATH = REPO_ROOT / "config" / "targets.yaml"


class OutcomeTier(enum.Enum):
    """Decision-facing outcome tiers (T4). Ordinal, worst -> best."""

    BUST = "bust"
    ROTATION = "rotation"
    STARTER = "starter"
    ALL_STAR = "all_star"
    SUPERSTAR = "superstar"


# --------------------------------------------------------------------------------------
# Config models
# --------------------------------------------------------------------------------------
class HorizonSpec(BaseModel):
    primary_window_years: int = Field(gt=0)
    debut_cap_years: int = Field(ge=0)


class ReachSpec(BaseModel):
    min_window_minutes: float = Field(ge=0)


class PeakImpactSpec(BaseModel):
    metric: str = "bpm"
    stable_season_minutes: float = Field(ge=0)
    top_n_seasons: int = Field(gt=0)


class CumulativeValueSpec(BaseModel):
    metric: str = "vorp"


class ReplacementSpec(BaseModel):
    bpm: float
    vorp: float


class TierSpec(BaseModel):
    name: str
    requires_reach: bool
    honors: list[str] = Field(default_factory=list)
    peak_bpm_min: float | None = None


class TargetConfig(BaseModel):
    horizon: HorizonSpec
    reach: ReachSpec
    peak_impact: PeakImpactSpec
    cumulative_value: CumulativeValueSpec
    replacement: ReplacementSpec
    tiers: list[TierSpec]

    @model_validator(mode="after")
    def _validate_tiers(self) -> TargetConfig:
        names = {t.name for t in self.tiers}
        valid = {t.value for t in OutcomeTier}
        unknown = names - valid
        if unknown:
            raise ValueError(f"Unknown tier name(s) in config: {sorted(unknown)}")
        if "bust" not in names:
            raise ValueError("tiers must include a 'bust' fallback tier.")
        return self


def load_target_config(path: str | Path | None = None) -> TargetConfig:
    """Load and validate config/targets.yaml into a :class:`TargetConfig`."""
    cfg_path = Path(path) if path is not None else DEFAULT_TARGETS_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Targets config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return TargetConfig(**raw)


# --------------------------------------------------------------------------------------
# Realized-outcome input records (populated from real data in Phase 1/2)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SeasonStat:
    """One realized NBA season for a player. Seasons are identified by their starting year
    (e.g. the 2021-22 season is ``season_year=2021``)."""

    season_year: int
    minutes: float
    bpm: float
    vorp: float


@dataclass(frozen=True)
class PlayerOutcome:
    """A prospect's realized NBA outcome, as known at label-construction time.

    Attributes:
        draft_year: Year the player was drafted.
        debut_year: Starting year of the first NBA season played, or ``None`` if never debuted.
        seasons: All realized NBA seasons (any order).
        all_star_count: Number of All-Star selections (within career; tiering uses > 0).
        all_nba_count: Number of All-NBA selections.
    """

    draft_year: int
    debut_year: int | None
    seasons: tuple[SeasonStat, ...] = ()
    all_star_count: int = 0
    all_nba_count: int = 0


# --------------------------------------------------------------------------------------
# Label construction
# --------------------------------------------------------------------------------------
def _debut_is_valid(outcome: PlayerOutcome, cfg: TargetConfig) -> bool:
    """True if the player debuted within the cap (debut-anchored, capped horizon)."""
    if outcome.debut_year is None:
        return False
    offset = outcome.debut_year - outcome.draft_year
    return 0 <= offset <= cfg.horizon.debut_cap_years


def _primary_window_seasons(outcome: PlayerOutcome, cfg: TargetConfig) -> list[SeasonStat]:
    """Seasons falling in the first ``primary_window_years`` from a valid debut.

    Returns an empty list if the debut is invalid (never reached within the cap).
    """
    if not _debut_is_valid(outcome, cfg):
        return []
    assert outcome.debut_year is not None  # for type-checkers; guaranteed by _debut_is_valid
    lo = outcome.debut_year
    hi = outcome.debut_year + cfg.horizon.primary_window_years  # exclusive
    return [s for s in outcome.seasons if lo <= s.season_year < hi]


def reached_role(outcome: PlayerOutcome, cfg: TargetConfig) -> bool:
    """T1 label: did the player reach a rotation role within the primary window?

    Requires a valid (capped) debut and at least ``reach.min_window_minutes`` total minutes
    across the window. Defined over ALL prospects so the hurdle model learns who washes out.
    """
    window = _primary_window_seasons(outcome, cfg)
    total_minutes = sum(s.minutes for s in window)
    return total_minutes >= cfg.reach.min_window_minutes


def peak_impact(outcome: PlayerOutcome, cfg: TargetConfig) -> float | None:
    """T2 label: peak impact (mean of the best N qualifying seasons' BPM) within the window.

    Conditional on minutes: only seasons with >= ``stable_season_minutes`` count, so the
    estimate is stable. Returns ``None`` when no season qualifies (not estimable) — such
    players are handled by the hurdle, not assigned a fabricated impact.
    """
    window = _primary_window_seasons(outcome, cfg)
    qualifying = [s for s in window if s.minutes >= cfg.peak_impact.stable_season_minutes]
    if not qualifying:
        return None
    metric = cfg.peak_impact.metric
    values = sorted((getattr(s, metric) for s in qualifying), reverse=True)
    top = values[: cfg.peak_impact.top_n_seasons]
    return float(sum(top) / len(top))


def cumulative_value(outcome: PlayerOutcome, cfg: TargetConfig) -> float:
    """T3 label: cumulative realized value (sum of VORP) across the primary window.

    VORP already weights minutes, so non-reach players naturally accrue little. Returns 0.0
    when there are no window seasons.
    """
    window = _primary_window_seasons(outcome, cfg)
    metric = cfg.cumulative_value.metric
    return float(sum(getattr(s, metric) for s in window))


def outcome_tier(outcome: PlayerOutcome, cfg: TargetConfig) -> OutcomeTier:
    """T4 label: hybrid outcome tier (honors for top tiers, BPM bands below).

    Tiers are evaluated top-down as ordered in config; the first match wins. A non-fallback
    tier matches when the player reached a role AND (an honors condition holds OR peak BPM
    clears the tier's lower bound). The 'bust' fallback always matches last.
    """
    reached = reached_role(outcome, cfg)
    peak = peak_impact(outcome, cfg)
    honors_present = {
        "all_star": outcome.all_star_count > 0,
        "all_nba": outcome.all_nba_count > 0,
    }
    for tier in cfg.tiers:
        if tier.name == "bust":
            continue  # fallback handled after the loop
        if tier.requires_reach and (not reached or peak is None):
            continue
        honors_match = any(honors_present.get(h, False) for h in tier.honors)
        band_match = (
            tier.peak_bpm_min is not None and peak is not None and peak >= tier.peak_bpm_min
        )
        if honors_match or band_match:
            return OutcomeTier(tier.name)
    return OutcomeTier.BUST


def unconditional_value(
    p_reach: float,
    conditional_value: float,
    *,
    replacement: float,
) -> float:
    """Combine the hurdle parts into an honest, survivorship-robust expected value.

        EV = p_reach * conditional_value + (1 - p_reach) * replacement

    Args:
        p_reach: Probability the prospect reaches a role (T1).
        conditional_value: Projected value *given* they reach (T2 or T3).
        replacement: Replacement-level value for the non-reach branch (from config).
    """
    if not 0.0 <= p_reach <= 1.0:
        raise ValueError("p_reach must be in [0, 1].")
    return p_reach * conditional_value + (1.0 - p_reach) * replacement


def is_label_resolved(
    outcome: PlayerOutcome,
    data_through_year: int,
    cfg: TargetConfig,
) -> bool:
    """Is this prospect's primary-window label fully observed (not right-censored)?

    Recent classes are censored, not low-outcome (domain risk #1 corollary). A label is
    resolved when either:
      * the player debuted (within cap) AND the full primary window has elapsed, or
      * the player has not debuted AND the debut cap has already passed (definitive non-reach).
    Otherwise it is censored: the player could still debut or complete their window.

    Args:
        outcome: The (possibly partial) realized outcome.
        data_through_year: Starting year of the most recent season for which we have data.
    """
    if _debut_is_valid(outcome, cfg):
        assert outcome.debut_year is not None
        window_end = outcome.debut_year + cfg.horizon.primary_window_years - 1
        return data_through_year >= window_end
    # Not (yet) a valid debut: resolved only once the cap has definitively passed.
    return (data_through_year - outcome.draft_year) >= cfg.horizon.debut_cap_years
