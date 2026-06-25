# Assumptions & known limitations

Being honest about the limits is part of the deliverable. Read this before trusting any number.

## Data status
- **The shipped demo runs on a SYNTHETIC fixture.** Every number in the demo dashboard/API/reports
  is for wiring and UX verification only and has **no basketball meaning**. Real findings require
  running the live ingestion (below) and rebuilding the master dataset.

## Sources (verified 2026-06-25; see `config/sources.yaml`)
- **Basketball-Reference, Bart Torvik, KenPom are NOT scraped** — their data paths are disallowed by
  robots.txt / ToS. We compute BPM/VORP ourselves from `nba_api` raw box + team data via the public
  BPM formula; **our values will differ slightly from BBRef's published ones** (validate on a sample).
- **`nba_api` (stats.nba.com)** is unofficial/undocumented; cloud IPs get banned, so the pull is
  **local-only**. Run it yourself: `pip install -e ".[ingest]"` then call `NbaStatsIngester`.
- **CollegeBasketballData.com** (college) and **EuroLeague feeds** (international) are disabled
  pending an API key / ToU confirmation.

## Modeling assumptions
- **Box-score metrics under-credit defense and are role-sensitive** — defensive specialists will be
  under-served by the impact projection. Flagged, not hidden.
- **Small, heterogeneous samples** (dozens of prospects × a limited history) cap model complexity and
  make rare-tier (All-Star/superstar) probabilities inherently uncertain.
- **Inter-league translation factors** are learned from limited overlap and are approximate.

## Fit module (most exploratory)
- The **lineup Net-Rating model is a transparent proxy** (Σ player impact + spacing/rim synergy
  terms), not a calibrated RAPM lineup model. Treat deltas as directional.
- The **features→skills mapping** (`prospect_to_player`) is a rough heuristic; perimeter defense is a
  neutral prior because box features don't capture it.
- Fit outputs are flagged `exploratory=True`.

## CBA / financial assumptions (`config/cba_rules.yaml`)
- **2025-26 figures are official; 2026-27 are PROJECTIONS** (flagged) and change in-season.
- **Rookie scale** picks 1-10 verified, 11-14 approximate, beyond extrapolated.
- **$-per-win conversion** (2.7 wins/VORP, $3.5M/win) is a rule-of-thumb assumption surfaced in
  every fit result; tune against the actual free-agent market.

## Scope not yet covered
- Full hierarchical PyMC model (a fast BayesianRidge stands in for Bayesian uncertainty).
- Real entity-resolution at scale (current approach is conservative blocking + guarded fuzzy match).
- Finer-grained DVC stages (one orchestrated stage today; split when real ingestion writes interim data).
