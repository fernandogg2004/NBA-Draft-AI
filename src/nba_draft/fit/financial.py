"""Financial fit under the CBA and aprons (Real Surplus Value).

A rookie-scale contract is a contender's most valuable resource: cheap, fixed by the CBA. Real
Surplus Value = projected on-court value in $ minus the rookie-scale cost. A team strangled by
the second apron values that surplus far more than a rebuilder, so apron pressure modulates it.

CBA figures live in config/cba_rules.yaml and are season-parameterized; projected seasons are
flagged so outputs can label them as assumptions. The $-per-win conversion is itself an
assumption (see config valuation block).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from nba_draft.config import REPO_ROOT

DEFAULT_CBA_PATH = REPO_ROOT / "config" / "cba_rules.yaml"


@dataclass(frozen=True)
class CBAConfig:
    season: str
    verified: bool
    salary_cap_usd: float
    luxury_tax_usd: float
    first_apron_usd: float
    second_apron_usd: float
    rookie_scale_first_year_usd: dict[int, float]
    wins_per_vorp: float
    dollars_per_win_usd: float
    rookie_contract_years: int
    valuation_is_assumption: bool

    @property
    def assumptions(self) -> list[str]:
        notes: list[str] = []
        if not self.verified:
            notes.append(f"CBA figures for {self.season} are PROJECTIONS, not official.")
        if self.valuation_is_assumption:
            notes.append(
                f"$-per-win conversion is a rule-of-thumb "
                f"({self.wins_per_vorp} wins/VORP, ${self.dollars_per_win_usd:,.0f}/win)."
            )
        return notes


def load_cba(path: str | Path | None = None, *, season: str | None = None) -> CBAConfig:
    """Load CBA parameters for a season (defaults to the file's ``default_season``)."""
    cfg_path = Path(path) if path else DEFAULT_CBA_PATH
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    season = season or raw["default_season"]
    s = raw["seasons"][season]
    val = raw["valuation"]
    rookie_scale = {int(k): float(v) for k, v in raw["rookie_scale_first_year_usd"].items()}
    return CBAConfig(
        season=season,
        verified=bool(s.get("verified", False)),
        salary_cap_usd=float(s["salary_cap_usd"]),
        luxury_tax_usd=float(s["luxury_tax_usd"]),
        first_apron_usd=float(s["first_apron_usd"]),
        second_apron_usd=float(s["second_apron_usd"]),
        rookie_scale_first_year_usd=rookie_scale,
        wins_per_vorp=float(val["wins_per_vorp"]),
        dollars_per_win_usd=float(val["dollars_per_win_usd"]),
        rookie_contract_years=int(val["rookie_contract_years"]),
        valuation_is_assumption=bool(val.get("assumption", True)),
    )


def rookie_first_year(pick: int, cba: CBAConfig) -> float:
    """First-year rookie-scale salary for a pick; extrapolates (decay) beyond the known table."""
    if pick < 1:
        raise ValueError("pick must be >= 1.")
    table = cba.rookie_scale_first_year_usd
    if pick in table:
        return table[pick]
    last_pick = max(table)
    # Exponential decay past the last known pick (rough; second-round picks are near-minimum).
    return table[last_pick] * (0.93 ** (pick - last_pick))


def rookie_cost_total(pick: int, cba: CBAConfig, *, raise_per_year: float = 0.05) -> float:
    """Approximate total rookie-scale cost across the control window (with annual raises)."""
    first = rookie_first_year(pick, cba)
    years = cba.rookie_contract_years
    return round(sum(first * (1 + raise_per_year) ** k for k in range(years)), 2)


def projected_value_usd(vorp_per_year: float, cba: CBAConfig) -> float:
    """Convert a per-year VORP projection into total $ value over the rookie window."""
    wins = vorp_per_year * cba.wins_per_vorp
    return round(wins * cba.dollars_per_win_usd * cba.rookie_contract_years, 2)


def real_surplus_value(vorp_per_year: float, pick: int, cba: CBAConfig) -> float:
    """Real Surplus Value = projected $ value − rookie-scale cost over the window."""
    return round(projected_value_usd(vorp_per_year, cba) - rookie_cost_total(pick, cba), 2)


def apron_pressure_multiplier(team_total_salary_usd: float, cba: CBAConfig) -> tuple[float, str]:
    """How much more a cheap surplus asset is worth given the team's cap situation.

    Returns (multiplier, label). Over the second apron, surplus is most precious.
    """
    if team_total_salary_usd >= cba.second_apron_usd:
        return 1.5, "over second apron"
    if team_total_salary_usd >= cba.first_apron_usd:
        return 1.3, "over first apron"
    if team_total_salary_usd >= cba.luxury_tax_usd:
        return 1.15, "over luxury tax"
    return 1.0, "below tax"
