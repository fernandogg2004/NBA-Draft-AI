import {
  PolarAngleAxis,
  PolarGrid,
  Radar,
  RadarChart,
  ResponsiveContainer,
} from "recharts";
import { pctInt } from "../lib/format";
import { type TierSlice } from "../lib/tiers";

/**
 * Outcome-tier distribution as a thin stacked bar (the design's mini-bar / the
 * larger detail bar). Widths are the tier probabilities.
 */
export function TierBar({
  slices,
  height = 6,
  showLabels = false,
}: {
  slices: TierSlice[];
  height?: number;
  showLabels?: boolean;
}) {
  return (
    <div className="w-full">
      <div
        className="flex w-full overflow-hidden rounded-full bg-surface-container-highest"
        style={{ height }}
      >
        {slices.map((s) => (
          <div
            key={s.key}
            style={{ width: `${s.prob * 100}%`, backgroundColor: s.color, opacity: 0.85 }}
            title={`${s.label}: ${pctInt(s.prob)}`}
          />
        ))}
      </div>
      {showLabels && (
        <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1">
          {slices.map((s) => (
            <div key={s.key} className="flex items-center gap-1.5">
              <span
                className="inline-block h-1.5 w-1.5 rounded-full"
                style={{ backgroundColor: s.color }}
              />
              <span className="font-label-caps text-[10px] text-on-surface-variant">
                {s.label}
              </span>
              <span className="font-data-tabular text-[11px] text-on-surface">{pctInt(s.prob)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/**
 * Floor → point → ceiling range bar. The colored band spans floor..ceiling,
 * with an orange tick at the point estimate, normalized over [domainMin, domainMax].
 */
export function RangeBar({
  floor,
  point,
  ceiling,
  domainMin,
  domainMax,
}: {
  floor: number;
  point: number;
  ceiling: number;
  domainMin: number;
  domainMax: number;
}) {
  const span = Math.max(1e-6, domainMax - domainMin);
  const pos = (v: number) => Math.min(100, Math.max(0, ((v - domainMin) / span) * 100));
  const left = pos(floor);
  const right = pos(ceiling);
  return (
    <div className="flex items-center gap-2">
      <span className="w-8 text-right font-data-tabular text-[11px] text-on-surface-variant">
        {floor.toFixed(1)}
      </span>
      <div className="relative h-2 flex-1 overflow-hidden rounded-full bg-surface-container-highest">
        <div
          className="absolute h-full rounded-full bg-secondary/50"
          style={{ left: `${left}%`, width: `${Math.max(2, right - left)}%` }}
        />
        <div
          className="absolute h-full w-1 bg-brand-orange"
          style={{ left: `${pos(point)}%` }}
          title={`Projection ${point.toFixed(1)}`}
        />
      </div>
      <span className="w-8 font-data-tabular text-[11px] text-on-surface-variant">
        {ceiling.toFixed(1)}
      </span>
    </div>
  );
}

export interface ShapItem {
  label: string;
  value: number;
}

/**
 * Horizontal SHAP contribution bars, diverging from a center axis: positive
 * contributions in orange (right), negative in blue (left). Thin (8px) per spec.
 */
export function ShapBars({ items }: { items: ShapItem[] }) {
  const max = Math.max(1e-6, ...items.map((i) => Math.abs(i.value)));
  return (
    <div className="space-y-3">
      {items.map((it) => {
        const w = (Math.abs(it.value) / max) * 50; // half-width percent
        const positive = it.value >= 0;
        return (
          <div key={it.label} className="flex items-center gap-3">
            <span className="w-36 shrink-0 truncate font-body-sm text-[13px] text-on-surface-variant">
              {it.label}
            </span>
            <div className="relative h-2 flex-1">
              {/* center line */}
              <div className="absolute left-1/2 top-0 h-full w-px bg-outline-variant" />
              <div
                className="absolute top-0 h-full rounded-full"
                style={{
                  width: `${w}%`,
                  left: positive ? "50%" : `${50 - w}%`,
                  backgroundColor: positive ? "#ff6a2c" : "#4f8cff",
                  opacity: 0.9,
                }}
              />
            </div>
            <span
              className={`w-12 text-right font-data-tabular text-[12px] ${
                positive ? "text-brand-orange" : "text-brand-blue"
              }`}
            >
              {it.value >= 0 ? "+" : ""}
              {it.value.toFixed(1)}
            </span>
          </div>
        );
      })}
    </div>
  );
}

export interface RadarSeries {
  name: string;
  color: string;
  values: Record<string, number>;
}

/**
 * Skill radar. Each series is a named profile over the same axes; current player
 * uses an orange stroke over a translucent blue fill per the design spec.
 */
export function SkillRadar({
  axes,
  series,
  height = 260,
}: {
  axes: string[];
  series: RadarSeries[];
  height?: number;
}) {
  const data = axes.map((axis) => {
    const row: Record<string, number | string> = { axis };
    series.forEach((s) => (row[s.name] = s.values[axis] ?? 0));
    return row;
  });
  return (
    <ResponsiveContainer width="100%" height={height}>
      <RadarChart data={data} outerRadius="72%">
        <PolarGrid stroke="#32353a" />
        <PolarAngleAxis
          dataKey="axis"
          tick={{ fill: "#a98a7f", fontSize: 11, fontFamily: "Inter" }}
        />
        {series.map((s) => (
          <Radar
            key={s.name}
            name={s.name}
            dataKey={s.name}
            stroke={s.color}
            fill={s.color}
            fillOpacity={0.25}
            strokeWidth={2}
          />
        ))}
      </RadarChart>
    </ResponsiveContainer>
  );
}

/** Circular composite-fit gauge (e.g. 88 / Elite). */
export function Gauge({
  value,
  max = 100,
  label,
  size = 160,
}: {
  value: number;
  max?: number;
  label?: string;
  size?: number;
}) {
  const r = size / 2 - 10;
  const circ = 2 * Math.PI * r;
  const frac = Math.min(1, Math.max(0, value / max));
  return (
    <div className="relative flex items-center justify-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle cx={size / 2} cy={size / 2} r={r} stroke="#32353a" strokeWidth={10} fill="none" />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          stroke="#ff6a2c"
          strokeWidth={10}
          fill="none"
          strokeLinecap="round"
          strokeDasharray={circ}
          strokeDashoffset={circ * (1 - frac)}
        />
      </svg>
      <div className="absolute flex flex-col items-center">
        <span className="font-display-num text-[36px] font-bold leading-none text-on-surface">
          {Math.round(value)}
        </span>
        {label && (
          <span className="mt-1 font-label-caps text-label-caps text-brand-orange">{label}</span>
        )}
      </div>
    </div>
  );
}

/** Thin labeled progress bar (archetype / need / synergy sub-scores). */
export function ScoreBar({
  label,
  value,
  max = 100,
}: {
  label: string;
  value: number;
  max?: number;
}) {
  const frac = Math.min(1, Math.max(0, value / max));
  return (
    <div>
      <div className="mb-1 flex items-center justify-between">
        <span className="font-body-sm text-[13px] text-on-surface-variant">{label}</span>
        <span className="font-data-tabular text-[12px] text-on-surface">
          {Math.round(value)}
          <span className="text-on-surface-variant">/{max}</span>
        </span>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-surface-container-highest">
        <div
          className="h-full rounded-full bg-brand-orange"
          style={{ width: `${frac * 100}%`, opacity: 0.9 }}
        />
      </div>
    </div>
  );
}
