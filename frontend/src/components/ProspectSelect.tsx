import type { ProspectRow } from "../lib/types";
import { Icon } from "./Icon";

/** Dropdown to choose a prospect from the ranked board. */
export function ProspectSelect({
  prospects,
  value,
  onChange,
  label = "Prospect",
}: {
  prospects: ProspectRow[];
  value: number | null;
  onChange: (playerId: number) => void;
  label?: string;
}) {
  return (
    <div className="flex items-center gap-2">
      <Icon name="person_search" size={18} className="text-on-surface-variant" />
      <label className="sr-only">{label}</label>
      <select
        value={value ?? ""}
        onChange={(e) => onChange(Number(e.target.value))}
        className="min-w-[200px] rounded-md border border-outline-variant bg-surface-container-high px-3 py-1.5 font-body-sm text-body-sm text-on-surface focus:border-primary focus:outline-none"
      >
        {prospects.map((p, i) => (
          <option key={p.player_id} value={p.player_id}>
            {String(i + 1).padStart(2, "0")} · {p.full_name}
          </option>
        ))}
      </select>
    </div>
  );
}
