/**
 * Typed client for the NBA Draft AI FastAPI service.
 *
 * In dev, requests go to /api/* and Vite proxies them to the uvicorn server
 * (see vite.config.ts). In production, set VITE_API_BASE to the API origin.
 */
import type {
  Counterfactual,
  Explanation,
  FitRequest,
  FitResult,
  ProspectRow,
} from "./types";

const BASE = (import.meta.env.VITE_API_BASE ?? "/api").replace(/\/$/, "");

class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch {
      /* non-JSON error body; keep statusText */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

export interface Meta {
  mode: "real" | "demo";
  n_prospects: number;
  draft_years?: number[];
  model_version?: string;
  n_features?: number;
}

export const api = {
  health: () => request<{ status: string }>("/health"),

  meta: () => request<Meta>("/meta"),

  prospects: (limit = 60) =>
    request<ProspectRow[]>(`/prospects?limit=${encodeURIComponent(limit)}`),

  explain: (playerId: number) =>
    request<Explanation>(`/explain/${encodeURIComponent(playerId)}`),

  counterfactual: (playerId: number, maxFeatures = 3) =>
    request<Counterfactual>(
      `/counterfactual/${encodeURIComponent(playerId)}?max_features=${maxFeatures}`,
    ),

  fit: (body: FitRequest) =>
    request<FitResult>("/fit", {
      method: "POST",
      body: JSON.stringify(body),
    }),
};

export { ApiError };
