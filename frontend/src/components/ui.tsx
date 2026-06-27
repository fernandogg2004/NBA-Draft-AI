import type { ReactNode } from "react";
import { Icon } from "./Icon";

/** Card: the tonal-mid container with a hairline border (10px radius). */
export function Card({
  children,
  className = "",
  padded = true,
}: {
  children: ReactNode;
  className?: string;
  padded?: boolean;
}) {
  return (
    <div
      className={`rounded-lg border border-outline-variant bg-surface-container ${
        padded ? "p-4" : ""
      } ${className}`}
    >
      {children}
    </div>
  );
}

/** All-caps section label, used above cards and table headers. */
export function SectionLabel({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <p className={`font-label-caps text-label-caps text-on-surface-variant ${className}`}>
      {children}
    </p>
  );
}

/** Small bordered chip (archetype tags etc.). */
export function Chip({
  children,
  tone = "neutral",
  className = "",
}: {
  children: ReactNode;
  tone?: "neutral" | "muted" | "primary";
  className?: string;
}) {
  const tones = {
    neutral:
      "bg-surface-container-highest border-outline-variant text-on-surface",
    muted: "border-outline-variant text-muted-pill",
    primary: "border-primary/30 bg-primary/10 text-primary",
  } as const;
  return (
    <span
      className={`inline-block rounded border px-2 py-0.5 font-label-caps text-[10px] ${tones[tone]} ${className}`}
    >
      {children}
    </span>
  );
}

/** A labeled metric tile. */
export function MetricTile({
  label,
  value,
  sub,
  accent = false,
}: {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
  accent?: boolean;
}) {
  return (
    <div className="rounded-lg border border-outline-variant bg-surface-container-low p-4">
      <SectionLabel>{label}</SectionLabel>
      <div
        className={`mt-1 font-display-num ${
          accent ? "text-brand-orange" : "text-on-surface"
        } text-[28px] font-bold leading-none`}
      >
        {value}
      </div>
      {sub && <div className="mt-1 font-body-sm text-[12px] text-on-surface-variant">{sub}</div>}
    </div>
  );
}

/** A placeholder value for design-only fields with no backing data yet. */
export function Placeholder({ label = "n/a" }: { label?: string }) {
  return (
    <span
      title="Not available from the model yet — placeholder"
      className="font-data-tabular text-on-surface-variant/60"
    >
      {label}
    </span>
  );
}

export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex items-center gap-3 p-8 text-on-surface-variant">
      <Icon name="progress_activity" className="animate-spin" size={20} />
      <span className="font-body-sm">{label}</span>
    </div>
  );
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <Card className="flex items-center justify-between border-error/40 bg-error-container/10">
      <div className="flex items-center gap-3">
        <Icon name="error" className="text-error" size={22} />
        <div>
          <p className="font-body-sm font-semibold text-on-surface">Could not load data</p>
          <p className="font-body-sm text-[13px] text-on-surface-variant">{message}</p>
          <p className="mt-1 font-label-caps text-[10px] text-on-surface-variant">
            Is the API running? Try: uvicorn api.main:app --reload
          </p>
        </div>
      </div>
      {onRetry && (
        <button
          onClick={onRetry}
          className="rounded-md border border-outline-variant px-3 py-1.5 font-label-caps text-label-caps text-on-surface hover:bg-surface-variant"
        >
          Retry
        </button>
      )}
    </Card>
  );
}

/** Page heading row used at the top of each screen. */
export function PageHeader({
  title,
  subtitle,
  actions,
}: {
  title: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <div className="flex items-start justify-between gap-4">
      <div>
        <h1 className="font-headline-lg text-headline-lg font-semibold text-on-surface">{title}</h1>
        {subtitle && (
          <p className="mt-1 font-body-sm text-body-sm text-on-surface-variant">{subtitle}</p>
        )}
      </div>
      {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </div>
  );
}
