import { useEffect, useMemo } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api } from "../lib/api";
import { useAsync } from "../lib/useAsync";
import { humanize } from "../lib/format";
import { Icon } from "../components/Icon";
import { Card, Chip, ErrorState, Loading, SectionLabel } from "../components/ui";
import { ShapBars, type ShapItem } from "../components/charts";
import { ProspectSelect } from "../components/ProspectSelect";

export function Explainability() {
  const { playerId } = useParams();
  const navigate = useNavigate();

  const board = useAsync(() => api.prospects(60), []);
  const rows = board.data ?? [];
  const selectedId = playerId ? Number(playerId) : (rows[0]?.player_id ?? null);

  useEffect(() => {
    if (!playerId && rows.length) navigate(`/explain/${rows[0].player_id}`, { replace: true });
  }, [playerId, rows, navigate]);

  const expl = useAsync(
    () => (selectedId != null ? api.explain(selectedId) : Promise.reject(new Error("no prospect"))),
    [selectedId],
  );

  const cf = useAsync(
    () =>
      selectedId != null ? api.counterfactual(selectedId) : Promise.reject(new Error("no prospect")),
    [selectedId],
  );

  const row = rows.find((r) => r.player_id === selectedId);

  // SHAP contributions sorted by magnitude → bars + ranked importance.
  const items: ShapItem[] = useMemo(() => {
    const c = expl.data?.contributions ?? [];
    return [...c]
      .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
      .slice(0, 8)
      .map((x) => ({ label: humanize(x.feature), value: x.shap_value }));
  }, [expl.data]);

  const top = items[0];
  const topNeg = [...items].reverse().find((i) => i.value < 0);

  return (
    <>
      <div className="flex items-center justify-between">
        <SectionLabel>Reports · Model Explainability</SectionLabel>
        {rows.length > 0 && (
          <ProspectSelect
            prospects={rows}
            value={selectedId}
            onChange={(id) => navigate(`/explain/${id}`)}
          />
        )}
      </div>

      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="font-headline-lg text-headline-lg font-semibold text-on-surface">
            Why This Ranking?
          </h1>
          <p className="mt-1 font-body-sm text-body-sm text-on-surface-variant">
            {row ? row.full_name : "Prospect"} · local SHAP attribution
          </p>
        </div>
        <button className="flex items-center gap-2 rounded-md border border-outline-variant bg-surface-container-high px-3 py-1.5 font-label-caps text-label-caps text-on-surface hover:bg-surface-variant">
          <Icon name="download" size={16} /> Export Report
        </button>
      </div>

      {board.error && <ErrorState message={board.error} onRetry={board.reload} />}

      {/* ---- Executive summary (derived from SHAP) ---- */}
      <Card>
        <div className="mb-2 flex items-center gap-2">
          <Icon name="summarize" size={18} className="text-brand-orange" />
          <SectionLabel>Executive Summary</SectionLabel>
        </div>
        {expl.loading && <Loading label="Computing attribution…" />}
        {expl.error && <ErrorState message={expl.error} onRetry={expl.reload} />}
        {expl.data && top && (
          <p className="font-body-lg text-body-lg leading-relaxed text-on-surface">
            The model projects{" "}
            <span className="font-semibold text-brand-orange">{row?.full_name}</span> from a base
            value of{" "}
            <span className="font-data-tabular">{expl.data.base_value.toFixed(2)}</span>. The single
            largest driver is{" "}
            <span className="font-semibold text-brand-orange">{top.label}</span> (
            {top.value >= 0 ? "+" : ""}
            {top.value.toFixed(2)}).{" "}
            {topNeg
              ? `The main dampening factor is ${topNeg.label} (${topNeg.value.toFixed(2)}).`
              : "No material negative contributors stand out."}
          </p>
        )}
      </Card>

      <div className="grid grid-cols-1 gap-card-gap lg:grid-cols-3">
        {/* ---- SHAP bars (real) ---- */}
        <Card className="lg:col-span-2">
          <div className="mb-4 flex items-center justify-between">
            <SectionLabel>SHAP Value Contributions</SectionLabel>
            <div className="flex items-center gap-3">
              <Legend color="#ff6a2c" label="Positive" />
              <Legend color="#4f8cff" label="Negative" />
            </div>
          </div>
          {expl.data ? (
            <ShapBars items={items} />
          ) : (
            !expl.loading && <p className="text-on-surface-variant">No attribution available.</p>
          )}
        </Card>

        {/* ---- Ranked importance (derived from local |SHAP|) ---- */}
        <Card>
          <SectionLabel className="mb-3">Top Local Drivers</SectionLabel>
          <ol className="space-y-2">
            {items.map((it, i) => (
              <li key={it.label} className="flex items-center justify-between">
                <span className="flex items-center gap-2 font-body-sm text-[13px] text-on-surface">
                  <span className="font-data-tabular text-on-surface-variant">{i + 1}.</span>
                  {it.label}
                </span>
                <span className="font-data-tabular text-[12px] text-on-surface-variant">
                  {Math.round((Math.abs(it.value) / Math.abs(items[0]?.value || 1)) * 100)}
                </span>
              </li>
            ))}
          </ol>
          <p className="mt-3 font-label-caps text-[10px] text-on-surface-variant">
            Ranked by |SHAP| for this prospect (local, not dataset-global).
          </p>
        </Card>
      </div>

      {/* ---- Counterfactual (real) ---- */}
      <Card>
        <div className="mb-2 flex items-center justify-between">
          <SectionLabel>Counterfactual Analysis</SectionLabel>
          {cf.data?.target_tier && (
            <Chip tone="primary">
              → {cf.data.target_tier} tier (≥ {cf.data.target?.toFixed(1)})
            </Chip>
          )}
        </div>
        {cf.loading && <Loading label="Searching counterfactuals…" />}
        {cf.error && <ErrorState message={cf.error} onRetry={cf.reload} />}
        {cf.data && (
          <>
            {cf.data.target == null ? (
              <p className="font-body-sm text-on-surface">
                Already in the top tier ({cf.data.current_tier}) — no change needed to advance.
              </p>
            ) : (
              <>
                <p className="font-body-sm text-on-surface-variant">
                  Smallest feature change(s) to lift {row?.full_name ?? "this prospect"} from{" "}
                  <span className="text-on-surface">{cf.data.current_tier}</span> toward{" "}
                  <span className="text-brand-orange">{cf.data.target_tier}</span> — projection{" "}
                  {cf.data.current_impact.toFixed(2)} →{" "}
                  <span className="font-data-tabular text-on-surface">
                    {cf.data.projected_impact.toFixed(2)}
                  </span>
                  {cf.data.reached ? "" : " (target not reachable within 3 changes)"}.
                </p>
                <div className="mt-3 flex flex-wrap gap-2">
                  {cf.data.changes.map((c) => (
                    <div
                      key={c.feature}
                      className="inline-flex items-center gap-2 rounded-md border border-outline-variant bg-surface-container-low px-3 py-2"
                    >
                      <Icon
                        name={c.delta >= 0 ? "trending_up" : "trending_down"}
                        size={18}
                        className={c.delta >= 0 ? "text-brand-orange" : "text-brand-blue"}
                      />
                      <span className="font-body-sm text-[13px] text-on-surface">
                        {humanize(c.feature)}
                      </span>
                      <span className="font-data-tabular text-[12px] text-on-surface-variant">
                        {c.from_value.toFixed(1)} →{" "}
                        <span className="text-on-surface">{c.to_value.toFixed(1)}</span>
                      </span>
                    </div>
                  ))}
                  {cf.data.changes.length === 0 && (
                    <p className="font-body-sm text-on-surface-variant">
                      No single in-bounds change set reaches the next tier.
                    </p>
                  )}
                </div>
                <p className="mt-3 font-label-caps text-[10px] text-on-surface-variant">
                  Greedy search over the projection model; feature values in model (preprocessed)
                  units.
                </p>
              </>
            )}
          </>
        )}
      </Card>
    </>
  );
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <span className="inline-block h-2 w-2 rounded-full" style={{ backgroundColor: color }} />
      <span className="font-label-caps text-[10px] text-on-surface-variant">{label}</span>
    </span>
  );
}
