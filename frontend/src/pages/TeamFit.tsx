import { useEffect, useMemo, useState } from "react";
import { api } from "../lib/api";
import { useAsync } from "../lib/useAsync";
import { signed, usdM } from "../lib/format";
import {
  CAP_SITUATIONS,
  ROSTER_PRESETS,
  SALARY_BY_SITUATION,
  presetRoster,
  type CapSituation,
} from "../lib/presets";
import { Icon } from "../components/Icon";
import { Card, Chip, ErrorState, Loading, SectionLabel } from "../components/ui";
import { Gauge, ScoreBar } from "../components/charts";
import { ProspectSelect } from "../components/ProspectSelect";

function fitTier(overall: number): string {
  if (overall >= 85) return "ELITE";
  if (overall >= 70) return "STRONG";
  if (overall >= 55) return "SOLID";
  return "MARGINAL";
}

export function TeamFit() {
  const board = useAsync(() => api.prospects(60), []);
  const rows = board.data ?? [];

  const [prospectId, setProspectId] = useState<number | null>(null);
  const [presetIdx, setPresetIdx] = useState(0);
  const [cap, setCap] = useState<CapSituation>("over first apron");
  const [pick, setPick] = useState(4);

  useEffect(() => {
    if (prospectId == null && rows.length) setProspectId(rows[0].player_id);
  }, [rows, prospectId]);

  const roster = useMemo(() => presetRoster(ROSTER_PRESETS[presetIdx].skills), [presetIdx]);

  const fit = useAsync(() => {
    if (prospectId == null) return Promise.reject(new Error("no prospect"));
    return api.fit({
      prospect_player_id: prospectId,
      roster,
      team_total_salary_usd: SALARY_BY_SITUATION[cap],
      pick,
    });
  }, [prospectId, presetIdx, cap, pick]);

  const row = rows.find((r) => r.player_id === prospectId);

  return (
    <>
      {/* ---- Header + controls ---- */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <SectionLabel>Simulations · Team Fit &amp; Simulator</SectionLabel>
          <h1 className="font-headline-lg text-headline-lg font-semibold text-on-surface">
            {row ? row.full_name : "Prospect"} <span className="text-on-surface-variant">· Target</span>
          </h1>
        </div>
        <button className="flex items-center gap-2 rounded-md bg-primary-container px-4 py-2 font-label-caps text-label-caps text-on-primary-container hover:opacity-90">
          <Icon name="save" size={16} /> Save Scenario
        </button>
      </div>

      <Card className="flex flex-wrap items-end gap-4">
        {rows.length > 0 && (
          <div>
            <label className="mb-1 block font-label-caps text-label-caps text-on-surface-variant">
              Prospect
            </label>
            <ProspectSelect
              prospects={rows}
              value={prospectId}
              onChange={setProspectId}
            />
          </div>
        )}
        <Control label="Roster Style">
          <select
            value={presetIdx}
            onChange={(e) => setPresetIdx(Number(e.target.value))}
            className="rounded border border-outline-variant bg-surface px-3 py-1.5 font-body-sm text-body-sm text-on-surface focus:border-primary focus:outline-none"
          >
            {ROSTER_PRESETS.map((p, i) => (
              <option key={p.label} value={i}>
                {p.label}
              </option>
            ))}
          </select>
        </Control>
        <Control label="Cap Situation">
          <select
            value={cap}
            onChange={(e) => setCap(e.target.value as CapSituation)}
            className="rounded border border-outline-variant bg-surface px-3 py-1.5 font-body-sm text-body-sm text-on-surface focus:border-primary focus:outline-none"
          >
            {CAP_SITUATIONS.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </Control>
        <Control label="Draft Pick">
          <input
            type="number"
            min={1}
            max={60}
            value={pick}
            onChange={(e) => setPick(Math.min(60, Math.max(1, Number(e.target.value))))}
            className="w-20 rounded border border-outline-variant bg-surface px-3 py-1.5 font-data-tabular text-body-sm text-on-surface focus:border-primary focus:outline-none"
          />
        </Control>
      </Card>

      {board.error && <ErrorState message={board.error} onRetry={board.reload} />}
      {fit.loading && <Loading label="Scoring fit…" />}
      {fit.error && !fit.loading && <ErrorState message={fit.error} onRetry={fit.reload} />}

      {fit.data && (
        <>
          <div className="grid grid-cols-1 gap-card-gap lg:grid-cols-3">
            {/* Composite fit gauge */}
            <Card className="flex flex-col items-center justify-center">
              <SectionLabel className="mb-2">Composite Fit</SectionLabel>
              <Gauge value={fit.data.overall} label={`TIER: ${fitTier(fit.data.overall)}`} />
            </Card>

            {/* Sub-scores */}
            <Card className="lg:col-span-2">
              <SectionLabel className="mb-4">Fit Breakdown</SectionLabel>
              <div className="space-y-4">
                <ScoreBar label="Basketball Fit" value={fit.data.basketball_fit} />
                <ScoreBar label="Financial Fit" value={fit.data.financial_fit} />
              </div>

              <div className="mt-5">
                <div className="mb-2 flex items-center gap-2">
                  <SectionLabel>Synergy Sub-scores</SectionLabel>
                  <Chip tone="muted">exploratory · relative</Chip>
                </div>
                <SynergyBars
                  complementarity={fit.data.synergy_complementarity}
                  net={fit.data.synergy_net}
                  redundancy={fit.data.synergy_redundancy}
                />
              </div>

              <div className="mt-4 flex items-center gap-2">
                <SectionLabel>Cap Status:</SectionLabel>
                <Chip tone="primary">{fit.data.apron_label}</Chip>
              </div>
            </Card>
          </div>

          <div className="grid grid-cols-1 gap-card-gap lg:grid-cols-3">
            {/* Franchise context + roster */}
            <Card>
              <SectionLabel className="mb-3">Franchise Context</SectionLabel>
              <div className="mb-4 grid grid-cols-2 gap-3">
                <div className="rounded-md border border-outline-variant bg-surface-container-low p-3">
                  <SectionLabel>Draft Slot</SectionLabel>
                  <div className="font-display-num text-[28px] font-bold text-on-surface">
                    #{pick}
                  </div>
                </div>
                <div className="rounded-md border border-outline-variant bg-surface-container-low p-3">
                  <SectionLabel>Team Salary</SectionLabel>
                  <div className="font-display-num text-[20px] font-bold text-on-surface">
                    {usdM(SALARY_BY_SITUATION[cap], false)}
                  </div>
                </div>
              </div>
              <SectionLabel className="mb-2">Current Roster (preset)</SectionLabel>
              <ul className="divide-y divide-outline-variant/50">
                {roster.map((p) => (
                  <li key={p.name} className="flex items-center justify-between py-1.5">
                    <span className="font-body-sm text-[13px] text-on-surface">{p.name}</span>
                    <span className="font-data-tabular text-[12px] text-on-surface-variant">
                      impact {p.impact.toFixed(1)}
                    </span>
                  </li>
                ))}
              </ul>
            </Card>

            {/* Lineup net rating */}
            <Card className="lg:col-span-2">
              <SectionLabel className="mb-3">Lineup Net-Rating Simulation</SectionLabel>
              <div className="flex items-center justify-around rounded-lg border border-outline-variant bg-surface-container-low p-6">
                <div className="text-center">
                  <SectionLabel>Base Lineup</SectionLabel>
                  <div className="font-display-num text-[28px] font-bold text-on-surface-variant">
                    {signed(fit.data.lineup_before)}
                  </div>
                </div>
                <div className="flex flex-col items-center">
                  <Icon name="arrow_forward" size={28} className="text-on-surface-variant" />
                  <span
                    className={`font-data-tabular text-[12px] ${
                      fit.data.lineup_delta >= 0 ? "text-brand-orange" : "text-brand-blue"
                    }`}
                  >
                    {signed(fit.data.lineup_delta)}
                  </span>
                </div>
                <div className="text-center">
                  <SectionLabel>With Prospect</SectionLabel>
                  <div
                    className={`font-display-num text-[40px] font-bold ${
                      fit.data.lineup_after >= fit.data.lineup_before
                        ? "text-brand-orange"
                        : "text-brand-blue"
                    }`}
                  >
                    {signed(fit.data.lineup_after)}
                  </div>
                </div>
              </div>
              <p className="mt-2 text-center font-label-caps text-[10px] text-on-surface-variant">
                Replacing weakest link: {fit.data.lineup_replaced}
              </p>

              {/* GM narrative */}
              <div className="mt-4 rounded-md border-l-2 border-brand-orange bg-surface-container-low p-3">
                <div className="mb-1 flex items-center gap-2">
                  <Icon name="format_quote" size={16} className="text-brand-orange" />
                  <SectionLabel>GM Narrative</SectionLabel>
                </div>
                <p className="font-body-sm text-[13px] leading-relaxed text-on-surface">
                  {fit.data.narrative}
                </p>
              </div>
            </Card>
          </div>

          {/* Financial value projection */}
          <Card>
            <SectionLabel className="mb-3">Financial Value Projection</SectionLabel>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
              <FinRow label="Real Surplus Value" value={usdM(fit.data.rsv_usd)} />
              <FinRow label="Surplus (apron-modulated)" value={usdM(fit.data.rsv_modulated_usd)} accent />
              <FinRow
                label="Overall Fit"
                value={`${Math.round(fit.data.overall)}/100`}
              />
            </div>
            {fit.data.assumptions.length > 0 && (
              <p className="mt-3 font-label-caps text-[10px] text-on-surface-variant">
                Assumptions: {fit.data.assumptions.join(" · ")}
              </p>
            )}
          </Card>
        </>
      )}
    </>
  );
}

function Control({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="mb-1 block font-label-caps text-label-caps text-on-surface-variant">
        {label}
      </label>
      {children}
    </div>
  );
}

/**
 * The three synergy components on a shared relative scale (they're exploratory raw
 * scores, not 0..100). Need + Net in orange, Overlap in blue (overlap is a discount).
 */
function SynergyBars({
  complementarity,
  net,
  redundancy,
}: {
  complementarity: number;
  net: number;
  redundancy: number;
}) {
  const max = Math.max(1e-6, complementarity, net, redundancy);
  const rows = [
    { label: "Functional Need (fills gaps)", value: complementarity, color: "#ff6a2c" },
    { label: "Roster Synergy (net)", value: net, color: "#ff6a2c" },
    { label: "Overlap (redundancy)", value: redundancy, color: "#4f8cff" },
  ];
  return (
    <div className="space-y-3">
      {rows.map((r) => (
        <div key={r.label}>
          <div className="mb-1 flex items-center justify-between">
            <span className="font-body-sm text-[13px] text-on-surface-variant">{r.label}</span>
            <span className="font-data-tabular text-[12px] text-on-surface">
              {r.value.toFixed(2)}
            </span>
          </div>
          <div className="h-2 w-full overflow-hidden rounded-full bg-surface-container-highest">
            <div
              className="h-full rounded-full"
              style={{
                width: `${Math.max(0, Math.min(100, (r.value / max) * 100))}%`,
                backgroundColor: r.color,
                opacity: 0.9,
              }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

function FinRow({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className="rounded-md border border-outline-variant bg-surface-container-low p-4">
      <SectionLabel>{label}</SectionLabel>
      <div
        className={`mt-1 font-display-num text-[24px] font-bold ${
          accent ? "text-brand-orange" : "text-on-surface"
        }`}
      >
        {value}
      </div>
    </div>
  );
}
