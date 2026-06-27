/**
 * Roster + cap presets, mirroring dashboard/streamlit_app.py so the React app's
 * /fit requests produce the same results as the existing Streamlit dashboard.
 *
 * These are demo team contexts; the real product would let a GM build a roster.
 */
import type { RosterPlayer } from "./types";

export interface RosterPreset {
  label: string;
  skills: Record<string, number>;
}

export const ROSTER_PRESETS: RosterPreset[] = [
  {
    label: "Guard-heavy (needs size/defense)",
    skills: {
      scoring: 82,
      shooting_spacing: 80,
      playmaking: 70,
      rebounding: 30,
      rim_protection: 20,
      perimeter_defense: 40,
    },
  },
  {
    label: "Wing-heavy (needs playmaking/rim)",
    skills: {
      scoring: 70,
      shooting_spacing: 72,
      playmaking: 35,
      rebounding: 45,
      rim_protection: 35,
      perimeter_defense: 65,
    },
  },
  {
    label: "Balanced contender",
    skills: {
      scoring: 65,
      shooting_spacing: 62,
      playmaking: 58,
      rebounding: 58,
      rim_protection: 60,
      perimeter_defense: 60,
    },
  },
];

export const CAP_SITUATIONS = [
  "below tax",
  "over luxury tax",
  "over first apron",
  "over second apron",
] as const;
export type CapSituation = (typeof CAP_SITUATIONS)[number];

/**
 * Approximate total-salary stand-ins per cap situation (the Streamlit app keys
 * these off live CBA thresholds; here we use round demo figures, with the apron
 * tiers above the 2024-25 thresholds so the API classifies them correctly).
 */
export const SALARY_BY_SITUATION: Record<CapSituation, number> = {
  "below tax": 100_000_000,
  "over luxury tax": 172_000_000,
  "over first apron": 179_000_000,
  "over second apron": 190_000_000,
};

export function presetRoster(skills: Record<string, number>): RosterPlayer[] {
  return Array.from({ length: 5 }, (_, i) => ({
    name: `Starter ${i + 1}`,
    skills: { ...skills },
    impact: 1.5 + 0.3 * i,
    salary_usd: 0,
  }));
}
