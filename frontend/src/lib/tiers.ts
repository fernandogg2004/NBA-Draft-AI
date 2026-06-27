import { TIER_KEYS, TIER_LABELS, type ProspectRow, type TierKey } from "./types";

/** Hex colors for the 5-tier outcome scale (mirrors tailwind `tier.*`). */
export const TIER_COLORS: Record<TierKey, string> = {
  bust: "#ef4444",
  rotation: "#7d8694",
  starter: "#ff6a2c",
  star: "#4f8cff", // not used directly; see note below
  super: "#ffb59b",
} as unknown as Record<TierKey, string>;

// Map the model's tier keys to colors. (TierKey is bust|rotation|starter|all_star|superstar.)
export const TIER_COLOR: Record<TierKey, string> = {
  bust: "#ef4444",
  rotation: "#7d8694",
  starter: "#ff6a2c",
  all_star: "#4f8cff",
  superstar: "#ffb59b",
};

export interface TierSlice {
  key: TierKey;
  label: string;
  prob: number;
  color: string;
}

/** Pull the 5 tier probabilities off a board row in display order. */
export function tierSlices(row: ProspectRow): TierSlice[] {
  return TIER_KEYS.map((key) => ({
    key,
    label: TIER_LABELS[key],
    prob: Number(row[`p_${key}`] ?? 0),
    color: TIER_COLOR[key],
  }));
}

/** Best EV ranking metric: unconditional EV if present, else conditional impact. */
export function rankingValue(row: ProspectRow): number {
  return row.projected_ev ?? row.projected_impact;
}
