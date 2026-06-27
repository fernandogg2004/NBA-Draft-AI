/**
 * Types mirroring the FastAPI responses in api/main.py.
 *
 * The board rows come straight from DraftBoardService.rank(): every row has the
 * projection + interval + the five outcome-tier probabilities, and (when the
 * survivorship-robust hurdle is attached) p_reach + projected_ev.
 */

/** Outcome tiers, in order, as emitted by the model (board.py TIER_LABELS). */
export const TIER_KEYS = [
  "bust",
  "rotation",
  "starter",
  "all_star",
  "superstar",
] as const;
export type TierKey = (typeof TIER_KEYS)[number];

/** Display labels matching the Apex design ("Bust / Role / Starter / Star / Super"). */
export const TIER_LABELS: Record<TierKey, string> = {
  bust: "Bust",
  rotation: "Role",
  starter: "Starter",
  all_star: "Star",
  superstar: "Super",
};

export interface ProspectRow {
  player_id: number;
  full_name: string;
  draft_year?: number;
  projected_impact: number;
  floor: number;
  ceiling: number;
  // Present only when the hurdle model is attached (the default demo service has it).
  p_reach?: number;
  projected_ev?: number;
  // Outcome-tier probabilities: p_bust, p_rotation, p_starter, p_all_star, p_superstar.
  p_bust: number;
  p_rotation: number;
  p_starter: number;
  p_all_star: number;
  p_superstar: number;
  // Allow extra columns the real-data service may attach without breaking typing.
  [key: string]: number | string | undefined;
}

export interface ShapContribution {
  feature: string;
  shap_value: number;
}

export interface Explanation {
  player_id: number;
  base_value: number;
  contributions: ShapContribution[];
}

export interface RosterPlayer {
  name: string;
  skills: Record<string, number>;
  impact: number;
  salary_usd: number;
}

export interface FitRequest {
  prospect_player_id: number;
  roster: RosterPlayer[];
  team_total_salary_usd: number;
  pick: number;
  season?: string | null;
}

export interface FitResult {
  overall: number;
  basketball_fit: number;
  financial_fit: number;
  rsv_usd: number;
  rsv_modulated_usd: number;
  apron_label: string;
  // Synergy sub-scores (exploratory raw scores — relative, not 0..100).
  synergy_complementarity: number;
  synergy_redundancy: number;
  synergy_net: number;
  // Lineup Net-Rating simulation.
  lineup_before: number;
  lineup_after: number;
  lineup_delta: number;
  lineup_replaced: string;
  narrative: string;
  exploratory: boolean;
  assumptions: string[];
}

export interface CounterfactualChange {
  feature: string;
  from_value: number;
  to_value: number;
  delta: number;
}

export interface Counterfactual {
  player_id: number;
  current_impact: number;
  current_tier: string;
  /** null when already in the top tier (no change needed). */
  target: number | null;
  target_tier: string | null;
  projected_impact: number;
  reached: boolean;
  changes: CounterfactualChange[];
}
