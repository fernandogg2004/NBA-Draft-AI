/** Small display formatters shared across screens. */

/** Signed fixed-precision number, e.g. +3.4 / -1.2. */
export const signed = (v: number, digits = 1): string =>
  `${v >= 0 ? "+" : ""}${v.toFixed(digits)}`;

/** USD in millions, e.g. +$12.4M. */
export const usdM = (v: number, signedOut = true): string => {
  const m = v / 1e6;
  const sign = signedOut && m >= 0 ? "+" : m < 0 ? "-" : "";
  return `${sign}$${Math.abs(m).toFixed(1)}M`;
};

/** Percent from a 0..1 probability, e.g. 0.942 -> "94.2%". */
export const pct = (v: number, digits = 1): string => `${(v * 100).toFixed(digits)}%`;

/** Round a probability to an integer percent for compact bars. */
export const pctInt = (v: number): string => `${Math.round(v * 100)}%`;

/** Turn snake_case feature names into Title Case for labels. */
export const humanize = (s: string): string =>
  s
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .replace(/\bPct\b/i, "%")
    .replace(/\bTs\b/i, "TS");
