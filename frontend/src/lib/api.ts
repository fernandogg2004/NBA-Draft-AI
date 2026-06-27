/**
 * Typed client for the NBA Draft AI FastAPI service.
 *
 * In dev, requests go to /api/* and Vite proxies them to the uvicorn server
 * (see vite.config.ts). In production, set VITE_API_BASE to the API origin.
 */
import type { Explanation, FitRequest, FitResult, ProspectRow } from "./types";

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

export const api = {
  health: () => request<{ status: string }>("/health"),

  prospects: (limit = 60) =>
    request<ProspectRow[]>(`/prospects?limit=${encodeURIComponent(limit)}`),

  explain: (playerId: number) =>
    request<Explanation>(`/explain/${encodeURIComponent(playerId)}`),

  fit: (body: FitRequest) =>
    request<FitResult>("/fit", {
      method: "POST",
      body: JSON.stringify(body),
    }),
};

export { ApiError };
