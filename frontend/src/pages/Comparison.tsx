import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api } from "../lib/api";
import { useAsync } from "../lib/useAsync";
import { pct, signed, usdM } from "../lib/format";
import { rankingValue, tierSlices } from "../lib/tiers";
import { prospectSkills, SKILL_DIMS, type ProspectRow } from "../lib/types";
import { Icon } from "../components/Icon";
import { Card, Chip, ErrorState, Loading, Placeholder, SectionLabel } from "../components/ui";
import { SkillRadar, TierBar } from "../components/charts";
import { ProspectSelect } from "../components/ProspectSelect";

export function Comparison() {
  const [params, setParams] = useSearchParams();
  const board = useAsync(() => api.prospects(60), []);
  const rows = board.data ?? [];

  const [a, setA] = useState<number | null>(null);
  const [b, setB] = useState<number | null>(null);

  // Seed selections from ?a=&b= or the top two prospects.
  useEffect(() => {
    if (!rows.length) return;
    const qa = params.get("a");
    const qb = params.get("b");
    setA(qa ? Number(qa) : rows[0].player_id);
    setB(qb ? Number(qb) : (rows[1]?.player_id ?? rows[0].player_id));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [board.data]);

  const rowA = rows.find((r) => r.player_id === a);
  const rowB = rows.find((r) => r.player_id === b);

  const setSel = (which: "a" | "b", id: number) => {
    if (which === "a") setA(id);
    else setB(id);
    const next = new URLSearchParams(params);
    next.set(which, String(id));
    setParams(next, { replace: true });
  };

  if (board.loading) return <Loading label="Loading prospects…" />;
  if (board.error) return <ErrorState message={board.error} onRetry={board.reload} />;
  if (!rowA || !rowB) return <ErrorState message="Need two prospects to compare." />;

  const skillsA = prospectSkills(rowA);
  const skillsB = prospectSkills(rowB);
  const evA = rankingValue(rowA);
  const evB = rankingValue(rowB);
  const winner = evA >= evB ? rowA : rowB;
  const margin = Math.abs(evA - evB);

  return (
    <>
      <h1 className="font-headline-lg text-headline-lg font-semibold text-on-surface">
        Prospect Comparison
      </h1>

      {/* ---- Recommendation banner ---- */}
      <Card className="flex items-center justify-between border-brand-orange/40 bg-primary/5">
        <div className="flex items-center gap-3">
          <Icon name="emoji_events" size={24} className="text-brand-orange" />
          <div>
            <SectionLabel>Decision Support Recommendation</SectionLabel>
            <p className="font-headline-sm text-[18px] font-semibold text-on-surface">
              Best value: {winner.full_name}
            </p>
          </div>
        </div>
        <div className="text-right">
          <SectionLabel>{rowA.projected_ev != null ? "EV edge" : "Impact edge"}</SectionLabel>
          <p className="font-display-num text-[24px] font-bold text-brand-orange">
            {signed(margin)}
          </p>
        </div>
      </Card>

      {/* ---- Side-by-side ---- */}
      <div className="grid grid-cols-[160px_1fr_1fr] gap-px overflow-hidden rounded-lg border border-outline-variant bg-outline-variant/40">
        {/* Header row */}
        <HeaderCell>
          <SectionLabel>Metric</SectionLabel>
        </HeaderCell>
        {[
          { row: rowA, which: "a" as const },
          { row: rowB, which: "b" as const },
        ].map(({ row, which }) => (
          <div key={which} className="bg-surface-container p-4 text-center">
            <div className="mx-auto mb-2 flex h-12 w-12 items-center justify-center rounded-full border border-outline-variant bg-surface-container-highest">
              <Icon name="person" size={28} className="text-on-surface-variant" />
            </div>
            <ProspectSelect
              prospects={rows}
              value={row.player_id}
              onChange={(id) => setSel(which, id)}
            />
            {row.player_id === winner.player_id && (
              <Chip tone="primary" className="mt-2">
                Projected Leader
              </Chip>
            )}
          </div>
        ))}

        <MetricRow
          label={rowA.projected_ev != null ? "Unconditional EV" : "Projected Impact"}
          a={signed(evA)}
          b={signed(evB)}
          aWin={evA >= evB}
        />
        <MetricRow
          label="Projected Impact"
          a={signed(rowA.projected_impact)}
          b={signed(rowB.projected_impact)}
          aWin={rowA.projected_impact >= rowB.projected_impact}
        />
        <MetricRow
          label="P(NBA Reach)"
          a={rowA.p_reach != null ? pct(rowA.p_reach) : "—"}
          b={rowB.p_reach != null ? pct(rowB.p_reach) : "—"}
          aWin={(rowA.p_reach ?? 0) >= (rowB.p_reach ?? 0)}
        />
        <MetricRow
          label="Floor / Ceiling"
          a={`${rowA.floor.toFixed(1)} / ${rowA.ceiling.toFixed(1)}`}
          b={`${rowB.floor.toFixed(1)} / ${rowB.ceiling.toFixed(1)}`}
        />
        <MetricRow
          label="Projected Value (rookie window)"
          a={rowA.projected_value_usd != null ? usdM(rowA.projected_value_usd, false) : "—"}
          b={rowB.projected_value_usd != null ? usdM(rowB.projected_value_usd, false) : "—"}
          aWin={(rowA.projected_value_usd ?? 0) >= (rowB.projected_value_usd ?? 0)}
        />
        <MetricRow
          label="Archetype"
          a={rowA.archetype ?? "—"}
          b={rowB.archetype ?? "—"}
        />
        <MetricRow
          label="System Fit Score"
          a={<Placeholder label="open Team Fit" />}
          b={<Placeholder label="open Team Fit" />}
        />

        {/* Outcome distributions */}
        <HeaderCell>
          <SectionLabel>Outcome Dist.</SectionLabel>
        </HeaderCell>
        <DistCell row={rowA} />
        <DistCell row={rowB} />
      </div>

      {/* ---- Radar overlay (feature-derived) ---- */}
      {skillsA && skillsB && (
        <Card>
          <div className="mb-2 flex items-center justify-between">
            <SectionLabel>Skill Radar Comparison</SectionLabel>
            <Chip tone="muted">exploratory mapping</Chip>
          </div>
          <SkillRadar
            height={320}
            axes={SKILL_DIMS.map((d) => d.label)}
            series={[
              { name: rowA.full_name, color: "#ff6a2c", values: skillsA },
              { name: rowB.full_name, color: "#4f8cff", values: skillsB },
            ]}
          />
        </Card>
      )}
    </>
  );
}

function HeaderCell({ children }: { children: React.ReactNode }) {
  return <div className="flex items-center bg-surface-container-highest p-4">{children}</div>;
}

function MetricRow({
  label,
  a,
  b,
  aWin,
}: {
  label: string;
  a: React.ReactNode;
  b: React.ReactNode;
  aWin?: boolean;
}) {
  return (
    <>
      <div className="flex items-center bg-surface p-4">
        <span className="font-body-sm text-[13px] text-on-surface-variant">{label}</span>
      </div>
      <ValueCell win={aWin === true}>{a}</ValueCell>
      <ValueCell win={aWin === false}>{b}</ValueCell>
    </>
  );
}

function ValueCell({ children, win }: { children: React.ReactNode; win?: boolean }) {
  return (
    <div className="bg-surface p-4 text-center">
      <span
        className={`font-data-tabular text-[15px] ${
          win ? "font-bold text-brand-orange" : "text-on-surface"
        }`}
      >
        {children}
      </span>
    </div>
  );
}

function DistCell({ row }: { row: ProspectRow }) {
  return (
    <div className="bg-surface p-4">
      <TierBar slices={tierSlices(row)} height={10} />
    </div>
  );
}
