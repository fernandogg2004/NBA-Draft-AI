import { useEffect, useMemo } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api } from "../lib/api";
import { useAsync } from "../lib/useAsync";
import { pct, signed } from "../lib/format";
import { tierSlices } from "../lib/tiers";
import { Icon } from "../components/Icon";
import {
  Card,
  Chip,
  ErrorState,
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
          <div className="flex h-16 w-16 items-center justify-center rounded-full border border-outline-variant bg-surface-container-highest">
            <Icon name="person" size={36} className="text-on-surface-variant" />
          </div>
          <div>
            <h1 className="font-headline-lg text-headline-lg font-semibold text-on-surface">
              {row.full_name}
            </h1>
            <div className="mt-2 flex flex-wrap items-center gap-2">
              <Chip>POS · <Placeholder label="—" /></Chip>
              <Chip>MEAS · <Placeholder label="—" /></Chip>
              <Chip tone="muted">ARCHETYPE · placeholder</Chip>
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
              value={<Placeholder label="—" />}
              sub="model field pending"
            />
            <MetricTile
              label="Cumulative Value"
              value={<Placeholder label="$ —" />}
              sub="model field pending"
            />
          </div>
        </Card>

        {/* ---- Skill radar ---- */}
        <Card>
          <div className="mb-2 flex items-center justify-between">
            <SectionLabel>Skill Profile</SectionLabel>
            <Chip tone="muted">illustrative</Chip>
          </div>
          <SkillRadar
            axes={["Scoring", "Shooting", "Playmaking", "Rebounding", "Rim Prot.", "Per. Def."]}
            series={[
              {
                name: row.full_name,
                color: "#ff6a2c",
                values: {
                  Scoring: 70,
                  Shooting: 62,
                  Playmaking: 58,
                  Rebounding: 55,
                  "Rim Prot.": 48,
                  "Per. Def.": 60,
                },
              },
            ]}
          />
          <p className="mt-1 text-center font-label-caps text-[10px] text-on-surface-variant">
            Skill ratings not yet exposed by the model — placeholder shape.
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

        {/* ---- Combine (placeholder) ---- */}
        <Card>
          <SectionLabel className="mb-3">Combine Data</SectionLabel>
          <dl className="divide-y divide-outline-variant/50">
            {["Wingspan", "Standing Vertical", "Max Vertical", "Lane Agility"].map((k) => (
              <div key={k} className="flex items-center justify-between py-2">
                <dt className="font-body-sm text-[13px] text-on-surface-variant">{k}</dt>
                <dd>
                  <Placeholder label="—" />
                </dd>
              </div>
            ))}
          </dl>
          <p className="mt-1 font-label-caps text-[10px] text-on-surface-variant">
            Combine measurements not in the model — placeholder.
          </p>
        </Card>
      </div>
    </>
  );
}
