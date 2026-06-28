import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../lib/api";
import { useAsync } from "../lib/useAsync";
import { tierSlices, rankingValue } from "../lib/tiers";
import {
  CAP_SITUATIONS,
  ROSTER_PRESETS,
  SALARY_BY_SITUATION,
  presetRoster,
  type CapSituation,
} from "../lib/presets";
import type { ProspectRow } from "../lib/types";
import { Icon } from "../components/Icon";
import {
  Card,
  Chip,
  ErrorState,
  Headshot,
  Loading,
  Placeholder,
  SectionLabel,
  StealReachChip,
} from "../components/ui";
import { RangeBar, TierBar } from "../components/charts";

export function DraftBoard() {
  const navigate = useNavigate();
  const { data, loading, error, reload } = useAsync(() => api.prospects(60), []);
  const { data: meta } = useAsync(() => api.meta(), []);

  // Roster-style fit controls. "Target Team" is a roster STYLE (no real NBA rosters in the data),
  // so the reorder is honest: fit to an explicit roster context, not a fabricated team roster.
  const [rosterIdx, setRosterIdx] = useState(0);
  const [cap, setCap] = useState<CapSituation>("over first apron");
  const [pick, setPick] = useState(14);
  const [fitRows, setFitRows] = useState<ProspectRow[] | null>(null);
  const [fitLoading, setFitLoading] = useState(false);
  const [fitError, setFitError] = useState<string | null>(null);

  async function updateBoard() {
    setFitLoading(true);
    setFitError(null);
    try {
      const rows = await api.boardFit({
        roster: presetRoster(ROSTER_PRESETS[rosterIdx].skills),
        team_total_salary_usd: SALARY_BY_SITUATION[cap],
        pick,
        limit: 60,
      });
      setFitRows(rows);
    } catch (e) {
      setFitError(e instanceof Error ? e.message : String(e));
    } finally {
      setFitLoading(false);
    }
  }

  // Board shown: fit-ranked when a fit query is active, else the talent-EV board.
  const rows = fitRows ?? data;
  const fitMode = fitRows !== null;

  // Shared domain for the floor/ceiling range bars so rows are comparable.
  const domain = useMemo(() => {
    if (!rows?.length) return { min: -5, max: 10 };
    const lo = Math.min(...rows.map((r) => r.floor));
    const hi = Math.max(...rows.map((r) => r.ceiling));
    return { min: lo, max: hi };
  }, [rows]);

  const rankedByEv = !!data?.[0]?.projected_ev;
  // The class being projected, taken from the served pool (latest draft year present).
  const draftYear = data?.length
    ? Math.max(...data.map((r) => r.draft_year ?? 0)) || null
    : null;

  return (
    <>
      {/* ---- Controls + consensus strip ---- */}
      <div className="flex flex-col items-start justify-between gap-4 xl:flex-row xl:items-end">
        <Card className="flex w-full flex-wrap items-end gap-4 xl:w-auto">
          <Field label="Roster Style">
            <select
              value={rosterIdx}
              onChange={(e) => setRosterIdx(Number(e.target.value))}
              className="rounded border border-outline-variant bg-surface px-3 py-1.5 font-body-sm text-body-sm text-on-surface focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
            >
              {ROSTER_PRESETS.map((p, i) => (
                <option key={p.label} value={i}>
                  {p.label}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Pick Slot">
            <input
              type="number"
              min={1}
              max={60}
              value={pick}
              onChange={(e) => setPick(Math.min(60, Math.max(1, Number(e.target.value))))}
              className="w-20 rounded border border-outline-variant bg-surface px-3 py-1.5 font-data-tabular text-body-sm text-on-surface focus:border-primary focus:outline-none"
            />
          </Field>
          <Field label="Cap Situation">
            <select
              value={cap}
              onChange={(e) => setCap(e.target.value as CapSituation)}
              className="rounded border border-outline-variant bg-surface px-3 py-1.5 font-body-sm text-body-sm text-on-surface focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
            >
              {CAP_SITUATIONS.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </Field>
          <button
            onClick={updateBoard}
            disabled={fitLoading}
            className="ml-auto rounded bg-primary-container px-4 py-2 font-label-caps text-label-caps text-on-primary-container hover:opacity-90 disabled:opacity-50 xl:ml-2"
          >
            {fitLoading ? "Ranking…" : "Update Board"}
          </button>
          {fitMode && (
            <button
              onClick={() => setFitRows(null)}
              className="rounded border border-outline-variant px-3 py-2 font-label-caps text-label-caps text-on-surface-variant hover:bg-surface-variant"
            >
              Clear (talent EV)
            </button>
          )}
        </Card>

        <Card className="flex w-full flex-1 items-center justify-between">
          <div className="flex items-center gap-3">
            <Icon name="insights" className="text-brand-orange" size={28} />
            <div>
              <h2 className="font-headline-sm text-[18px] font-semibold text-on-surface">
                {draftYear ? `${draftYear} Draft Board` : "Algorithmic Consensus"}
              </h2>
              <p className="mt-0.5 font-body-sm text-on-surface-variant">
                {fitMode
                  ? `Ranked by fit to a ${ROSTER_PRESETS[rosterIdx].label} roster — lineup Net-Rating Δ + synergy + apron surplus (exploratory).`
                  : rankedByEv
                    ? "Ranked by survivorship-robust unconditional EV — P(reach) × impact + replacement."
                    : "Ranked by conditional projected impact."}
              </p>
            </div>
          </div>
          <Chip tone="primary" className="hidden sm:inline-block">
            {meta
              ? meta.mode === "real"
                ? `Model ${meta.model_version ?? "real"} · ${meta.n_features ?? "?"} feats`
                : "Synthetic demo"
              : "…"}
          </Chip>
        </Card>
      </div>

      {/* ---- Table ---- */}
      <div className="flex flex-1 flex-col overflow-hidden rounded-lg border border-outline-variant bg-surface">
        {(loading || fitLoading) && (
          <Loading label={fitLoading ? "Ranking by team fit…" : "Loading draft board…"} />
        )}
        {error && (
          <div className="p-4">
            <ErrorState message={error} onRetry={reload} />
          </div>
        )}
        {fitError && (
          <div className="p-4">
            <ErrorState message={fitError} onRetry={updateBoard} />
          </div>
        )}
        {rows && (
          <>
            <div className="overflow-x-auto">
              <table className="w-full border-collapse text-left">
                <thead className="sticky top-0 z-10 border-b border-outline-variant bg-surface-container-highest">
                  <tr>
                    <Th className="w-12">Rnk</Th>
                    <Th className="w-64">Prospect</Th>
                    <Th className="hidden sm:table-cell">Archetype</Th>
                    <Th className="hidden text-center lg:table-cell">
                      {rankedByEv ? "EV (Flr–Ceil)" : "Impact (Flr–Ceil)"}
                    </Th>
                    <Th className="hidden w-32 xl:table-cell">Outcome Dist.</Th>
                    {fitMode && <Th className="text-right">Team Fit</Th>}
                    <Th className="text-right">Actual Pick</Th>
                    <Th className="text-right">Draft Result</Th>
                    <Th className="text-right">Actions</Th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-outline-variant/50">
                  {rows.map((row, i) => {
                    const slices = tierSlices(row);
                    return (
                      <tr
                        key={row.player_id}
                        onClick={() => navigate(`/prospect/${row.player_id}`)}
                        className="group h-table-row-height cursor-pointer transition-colors hover:bg-surface-variant/30"
                      >
                        <td className="p-3 font-data-tabular text-data-tabular text-on-surface">
                          {String(i + 1).padStart(2, "0")}
                        </td>
                        <td className="p-3">
                          <div className="flex items-center gap-3">
                            <Headshot url={row.headshot_url} size={32} alt={row.full_name} />
                            <div>
                              <div className="font-body-sm font-semibold text-on-surface">
                                {row.full_name}
                              </div>
                              <div className="mt-0.5 font-label-caps text-[9px] text-on-surface-variant">
                                {[
                                  row.team_abbr ?? (row.draft_year ? `Class ${row.draft_year}` : null),
                                  row.position,
                                  row.age != null ? `${row.age.toFixed(1)}y` : null,
                                ]
                                  .filter(Boolean)
                                  .join(" • ") || "Prospect pool"}
                              </div>
                            </div>
                          </div>
                        </td>
                        <td className="hidden p-3 sm:table-cell">
                          {row.archetype ? <Chip>{row.archetype}</Chip> : <Placeholder label="—" />}
                        </td>
                        <td className="hidden p-3 lg:table-cell">
                          <div className="mx-auto max-w-[160px]">
                            <RangeBar
                              floor={row.floor}
                              point={rankingValue(row)}
                              ceiling={row.ceiling}
                              domainMin={domain.min}
                              domainMax={domain.max}
                            />
                          </div>
                        </td>
                        <td className="hidden p-3 xl:table-cell">
                          <TierBar slices={slices} height={6} />
                        </td>
                        {fitMode && (
                          <td className="p-3 text-right">
                            {row.fit_overall != null ? (
                              <div className="flex items-center justify-end gap-2">
                                <span
                                  className="font-data-tabular text-body-sm font-bold text-brand-orange"
                                  title={
                                    row.fit_lineup_delta != null
                                      ? `Lineup NetRtg Δ ${row.fit_lineup_delta >= 0 ? "+" : ""}${row.fit_lineup_delta.toFixed(1)}`
                                      : undefined
                                  }
                                >
                                  {Math.round(row.fit_overall)}
                                </span>
                                <span className="text-[10px] text-on-surface-variant">/100</span>
                              </div>
                            ) : (
                              <Placeholder label="—" />
                            )}
                          </td>
                        )}
                        <td className="p-3 text-right font-data-tabular text-body-sm text-on-surface">
                          {row.draft_pick != null ? `#${row.draft_pick}` : <Placeholder label="—" />}
                        </td>
                        <td className="p-3 text-right">
                          {row.slot_delta != null ? (
                            <StealReachChip slotDelta={row.slot_delta} />
                          ) : (
                            <Placeholder label="—" />
                          )}
                        </td>
                        <td className="p-3 text-right">
                          <div className="flex items-center justify-end gap-1 opacity-0 transition-opacity group-hover:opacity-100">
                            <RowAction
                              icon="psychology"
                              title="Explain"
                              onClick={(e) => {
                                e.stopPropagation();
                                navigate(`/explain/${row.player_id}`);
                              }}
                            />
                            <RowAction
                              icon="compare_arrows"
                              title="Compare"
                              onClick={(e) => {
                                e.stopPropagation();
                                navigate(`/compare?a=${row.player_id}`);
                              }}
                            />
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            <div className="mt-auto flex items-center justify-between border-t border-outline-variant bg-surface-container-lowest p-3">
              <SectionLabel>
                Showing 1–{rows.length} of {rows.length} Prospects
                {fitMode ? ` · by fit to ${ROSTER_PRESETS[rosterIdx].label}` : ""}
              </SectionLabel>
              <p className="font-label-caps text-[10px] text-on-surface-variant">
                {fitMode
                  ? "Team Fit = lineup Net-Rating Δ + synergy + apron surplus for the chosen roster style (exploratory; no real NBA rosters in the data)."
                  : "Steal/Reach = model rank vs. actual pick (exploratory — the model matches, not beats, the draft consensus)."}{" "}
                <button
                  className="text-brand-orange hover:underline"
                  onClick={() => navigate("/team-fit")}
                >
                  Team Fit →
                </button>
              </p>
            </div>
          </>
        )}
      </div>
    </>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="mb-1 block font-label-caps text-label-caps text-on-surface-variant">
        {label}
      </label>
      {children}
    </div>
  );
}

function Th({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <th
      className={`p-3 font-label-caps text-label-caps font-medium text-on-surface-variant ${className}`}
    >
      {children}
    </th>
  );
}

function RowAction({
  icon,
  title,
  onClick,
}: {
  icon: string;
  title: string;
  onClick: (e: React.MouseEvent) => void;
}) {
  return (
    <button
      onClick={onClick}
      title={title}
      className="rounded p-1 text-on-surface-variant hover:bg-surface-container hover:text-primary"
    >
      <Icon name={icon} size={18} />
    </button>
  );
}
