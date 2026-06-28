import { useEffect, useMemo } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api } from "../lib/api";
import { useAsync } from "../lib/useAsync";
import { pct, signed, usdM } from "../lib/format";
import { tierSlices } from "../lib/tiers";
import { prospectSkills } from "../lib/types";
import { Icon } from "../components/Icon";
import {
  Card,
  Chip,
  ErrorState,
  Headshot,
  Loading,
  MetricTile,
  Placeholder,
  SectionLabel,
} from "../components/ui";
import { SkillRadar, TierBar } from "../components/charts";
import { ProspectSelect } from "../components/ProspectSelect";

export function ProspectDetail() {
  const { playerId } = useParams();
  const navigate = useNavigate();
  const { data, loading, error, reload } = useAsync(() => api.prospects(60), []);

  const selectedId = playerId ? Number(playerId) : null;
  const rows = data ?? [];
  const row = useMemo(
    () => rows.find((r) => r.player_id === selectedId) ?? rows[0],
    [rows, selectedId],
  );

  // Land on the top prospect if none chosen.
  useEffect(() => {
    if (!playerId && rows.length) navigate(`/prospect/${rows[0].player_id}`, { replace: true });
  }, [playerId, rows, navigate]);

  if (loading) return <Loading label="Loading prospect…" />;
  if (error) return <ErrorState message={error} onRetry={reload} />;
  if (!row) return <ErrorState message="No prospects available." onRetry={reload} />;

  const slices = tierSlices(row);
  const skills = prospectSkills(row);
  const halfInterval = (row.ceiling - row.floor) / 2;

  return (
    <>
      <div className="flex items-center justify-between">
        <SectionLabel>Draft Decision Support · Prospect Detail</SectionLabel>
        <ProspectSelect
          prospects={rows}
          value={row.player_id}
          onChange={(id) => navigate(`/prospect/${id}`)}
        />
      </div>

      {/* ---- Hero ---- */}
      <Card className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-4">
          <Headshot url={row.headshot_url} size={64} alt={row.full_name} />
          <div>
            <h1 className="font-headline-lg text-headline-lg font-semibold text-on-surface">
              {row.full_name}
            </h1>
            {row.draft_pick != null && (
              <p className="mt-0.5 font-label-caps text-label-caps text-brand-orange">
                Drafted #{row.draft_pick}
                {row.team_abbr ? ` · ${row.team_abbr}` : ""}
              </p>
            )}
            <div className="mt-2 flex flex-wrap items-center gap-2">
              {row.archetype && <Chip tone="primary">{row.archetype}</Chip>}
              {row.position && <Chip>POS · {row.position}</Chip>}
              <Chip>
                AGE · {row.age != null ? row.age.toFixed(1) : <Placeholder label="—" />}
              </Chip>
              <Chip>
                WINGSPAN ·{" "}
                {row.wingspan_in != null ? `${row.wingspan_in.toFixed(1)}"` : <Placeholder label="—" />}
              </Chip>
            </div>
          </div>
        </div>
        <div className="text-right">
          <SectionLabel>Projected Impact (WAR/BPM)</SectionLabel>
          <div className="font-display-num text-display-num font-bold text-brand-orange">
            {row.projected_impact.toFixed(1)}
          </div>
          <p className="font-data-tabular text-[12px] text-on-surface-variant">
            ± {halfInterval.toFixed(1)} (80% interval)
          </p>
          <button
            onClick={() => navigate(`/explain/${row.player_id}`)}
            className="mt-2 inline-flex items-center gap-1 font-label-caps text-label-caps text-brand-orange hover:underline"
          >
            <Icon name="lightbulb" size={14} /> Explain this projection
          </button>
        </div>
      </Card>

      <div className="grid grid-cols-1 gap-card-gap lg:grid-cols-3">
        {/* ---- Career projection ---- */}
        <Card className="lg:col-span-2">
          <SectionLabel className="mb-3">Career Projection</SectionLabel>
          <div className="grid grid-cols-2 gap-4">
            <MetricTile
              label="Prob. of NBA Reach"
              value={row.p_reach != null ? pct(row.p_reach) : "—"}
              accent
            />
            <MetricTile
              label={row.projected_ev != null ? "Unconditional EV" : "Projected Impact"}
              value={signed(row.projected_ev ?? row.projected_impact)}
            />
            <MetricTile
              label="Peak Impact"
              value={row.peak_pctile != null ? `Top ${Math.max(1, Math.round((1 - row.peak_pctile) * 100))}%` : "—"}
              sub="percentile in pool"
            />
            <MetricTile
              label="Cumulative Value"
              value={row.projected_value_usd != null ? usdM(row.projected_value_usd, false) : "$ —"}
              sub="over rookie window (assumption)"
            />
          </div>
        </Card>

        {/* ---- Skill radar (feature-derived) ---- */}
        <Card>
          <div className="mb-2 flex items-center justify-between">
            <SectionLabel>Skill Profile</SectionLabel>
            <Chip tone="muted">exploratory mapping</Chip>
          </div>
          {skills ? (
            <SkillRadar
              axes={Object.keys(skills)}
              series={[{ name: row.full_name, color: "#ff6a2c", values: skills }]}
            />
          ) : (
            <div className="p-8 text-center font-body-sm text-on-surface-variant">
              Skill profile unavailable for this prospect.
            </div>
          )}
          <p className="mt-1 text-center font-label-caps text-[10px] text-on-surface-variant">
            Functional skills derived from pre-draft box stats (0–100 percentile).
          </p>
        </Card>
      </div>

      <div className="grid grid-cols-1 gap-card-gap lg:grid-cols-2">
        {/* ---- Outcome distribution (real) ---- */}
        <Card>
          <SectionLabel className="mb-3">Outcome Distribution</SectionLabel>
          <TierBar slices={slices} height={28} />
          <div className="mt-3">
            <TierBar slices={slices} height={0} showLabels />
          </div>
          <p className="mt-2 font-data-tabular text-[12px] text-on-surface-variant">
            Floor {row.floor.toFixed(1)} · Ceiling {row.ceiling.toFixed(1)}
          </p>
        </Card>

        {/* ---- Combine (real measurements; null = did not test) ---- */}
        <Card>
          <SectionLabel className="mb-3">Combine Data</SectionLabel>
          <dl className="divide-y divide-outline-variant/50">
            {(
              [
                ["Wingspan", row.wingspan_in, '"'],
                ["Standing Reach", row.standing_reach_in, '"'],
                ["Max Vertical", row.max_vertical_in, '"'],
                ["Lane Agility", row.lane_agility_s, "s"],
                ["Body Fat", row.body_fat_pct, "%"],
              ] as [string, number | undefined, string][]
            ).map(([label, value, unit]) => (
              <div key={label} className="flex items-center justify-between py-2">
                <dt className="font-body-sm text-[13px] text-on-surface-variant">{label}</dt>
                <dd className="font-data-tabular text-[13px] text-on-surface">
                  {value != null ? `${value.toFixed(1)}${unit}` : <Placeholder label="not measured" />}
                </dd>
              </div>
            ))}
          </dl>
          <p className="mt-1 font-label-caps text-[10px] text-on-surface-variant">
            From the NBA Draft Combine; prospects who didn’t test show “not measured”.
          </p>
        </Card>
      </div>
    </>
  );
}
