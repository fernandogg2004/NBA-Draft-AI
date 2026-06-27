import { useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../lib/api";
import { useAsync } from "../lib/useAsync";
import { tierSlices, rankingValue } from "../lib/tiers";
import { Icon } from "../components/Icon";
import { Card, Chip, ErrorState, Loading, Placeholder, SectionLabel } from "../components/ui";
import { RangeBar, TierBar } from "../components/charts";

export function DraftBoard() {
  const navigate = useNavigate();
  const { data, loading, error, reload } = useAsync(() => api.prospects(60), []);

  // Shared domain for the floor/ceiling range bars so rows are comparable.
  const domain = useMemo(() => {
    if (!data?.length) return { min: -5, max: 10 };
    const lo = Math.min(...data.map((r) => r.floor));
    const hi = Math.max(...data.map((r) => r.ceiling));
    return { min: lo, max: hi };
  }, [data]);

  const rankedByEv = !!data?.[0]?.projected_ev;

  return (
    <>
      {/* ---- Controls + consensus strip ---- */}
      <div className="flex flex-col items-start justify-between gap-4 xl:flex-row xl:items-end">
        <Card className="flex w-full flex-wrap items-end gap-4 xl:w-auto">
          <Field label="Target Team">
            <Select options={["GSW", "BOS", "OKC", "SAS"]} />
          </Field>
          <Field label="Pick Slot">
            <Select options={["#8", "#14", "#15", "#20"]} />
          </Field>
          <Field label="Cap Situation">
            <Select options={["Over 1st Apron", "Over 2nd Apron", "Below Tax"]} />
          </Field>
          <button
            onClick={reload}
            className="ml-auto rounded bg-primary-container px-4 py-2 font-label-caps text-label-caps text-on-primary-container hover:opacity-90 xl:ml-2"
          >
            Update Board
          </button>
        </Card>

        <Card className="flex w-full flex-1 items-center justify-between">
          <div className="flex items-center gap-3">
            <Icon name="insights" className="text-brand-orange" size={28} />
            <div>
              <h2 className="font-headline-sm text-[18px] font-semibold text-on-surface">
                Algorithmic Consensus
              </h2>
              <p className="mt-0.5 font-body-sm text-on-surface-variant">
                {rankedByEv
                  ? "Ranked by survivorship-robust unconditional EV — P(reach) × impact + replacement."
                  : "Ranked by conditional projected impact."}
              </p>
            </div>
          </div>
          <Chip tone="primary" className="hidden sm:inline-block">
            Model V2.4 Active
          </Chip>
        </Card>
      </div>

      {/* ---- Table ---- */}
      <div className="flex flex-1 flex-col overflow-hidden rounded-lg border border-outline-variant bg-surface">
        {loading && <Loading label="Loading draft board…" />}
        {error && (
          <div className="p-4">
            <ErrorState message={error} onRetry={reload} />
          </div>
        )}
        {data && (
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
                    <Th className="text-right">Fit Score</Th>
                    <Th className="text-right">Real Surplus</Th>
                    <Th className="text-right">Actions</Th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-outline-variant/50">
                  {data.map((row, i) => {
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
                            <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-outline-variant bg-surface-container-highest">
                              <Icon name="person" size={20} className="text-on-surface-variant" />
                            </div>
                            <div>
                              <div className="font-body-sm font-semibold text-on-surface">
                                {row.full_name}
                              </div>
                              <div className="mt-0.5 font-label-caps text-[9px] text-on-surface-variant">
                                {row.draft_year ? `Class ${row.draft_year}` : "Prospect pool"}
                              </div>
                            </div>
                          </div>
                        </td>
                        <td className="hidden p-3 sm:table-cell">
                          <Placeholder label="—" />
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
                        <td className="p-3 text-right">
                          <Placeholder label="—" />
                        </td>
                        <td className="p-3 text-right">
                          <Placeholder label="—" />
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
                Showing 1–{data.length} of {data.length} Prospects
              </SectionLabel>
              <p className="font-label-caps text-[10px] text-on-surface-variant">
                Fit Score &amp; Real Surplus are team-specific — open{" "}
                <button
                  className="text-brand-orange hover:underline"
                  onClick={() => navigate("/team-fit")}
                >
                  Team Fit
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

function Select({ options }: { options: string[] }) {
  return (
    <select className="rounded border border-outline-variant bg-surface px-3 py-1.5 font-body-sm text-body-sm text-on-surface focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary">
      {options.map((o) => (
        <option key={o}>{o}</option>
      ))}
    </select>
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
