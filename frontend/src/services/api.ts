/**
 * API service — thin wrappers around fetch for each backend endpoint.
 *
 * In development Vite proxies `/api` → `http://localhost:8000`.
 * In production FastAPI serves both the API and the built frontend
 * on the same port, so relative URLs work everywhere.
 */

import type {
  HealthResponse,
  ResultsResponse,
  SimResetRequest,
  SimResetResponse,
  SimStepRequest,
  SimStepResponse,
  SimRunRequest,
  SimRunResponse,
} from "../types/api";

/* ------------------------------------------------------------------ */

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(
      (body as { detail?: string }).detail ??
        `Request failed: ${res.status} ${res.statusText}`
    );
  }
  return res.json() as Promise<T>;
}

/* ------------------------------------------------------------------ */

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/api/health");
}

export function getResults(): Promise<ResultsResponse> {
  return request<ResultsResponse>("/api/results");
}

export function resetSimulation(
  body: SimResetRequest = {}
): Promise<SimResetResponse> {
  return request<SimResetResponse>("/api/simulation/reset", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function stepSimulation(
  body: SimStepRequest = {}
): Promise<SimStepResponse> {
  return request<SimStepResponse>("/api/simulation/step", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function runSimulation(
  body: SimRunRequest = {}
): Promise<SimRunResponse> {
  return request<SimRunResponse>("/api/simulation/run", {
    method: "POST",
    body: JSON.stringify(body),
  });
}
