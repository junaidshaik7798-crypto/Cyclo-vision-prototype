import type {
  AnalysisResult,
  DataSource,
  DemoSample,
  DatasetsOverview,
  HealthStatus,
  IBTrACSDatasetResponse,
  IBTrACSStatus,
  IBTrACSStormsResponse,
  ReferenceDatasetResponse,
} from "./types";

const BASE = "/api";

// A dataset panel must not stay empty because the backend was mid-startup (or
// the IBTrACS warm-up was still running) when the page first loaded, so
// idempotent requests are retried a few times with a small backoff.
const MAX_ATTEMPTS = 3;
const RETRY_BACKOFF_MS = 700;
// Ordinary requests are fast; dataset reads are allowed to be slower because a
// large payload (the complete 342-storm record) legitimately takes a moment on
// a cold cache. The backend itself never waits for the NOAA download, so this
// only covers the local parse of the archive.
const REQUEST_TIMEOUT_MS = 20_000;
const DATASET_TIMEOUT_MS = 60_000;

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

async function fetchWithTimeout(
  url: string,
  options?: RequestInit,
  timeoutMs: number = REQUEST_TIMEOUT_MS
): Promise<Response> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    window.clearTimeout(timer);
  }
}

type AttemptResult<T> =
  | { ok: true; value: T }
  | { ok: false; error: Error; retryable: boolean };

async function attempt<T>(
  url: string,
  options?: RequestInit,
  timeoutMs: number = REQUEST_TIMEOUT_MS
): Promise<AttemptResult<T>> {
  try {
    const res = await fetchWithTimeout(
      `${BASE}${url}`,
      {
        cache: "no-store",
        ...options,
      },
      timeoutMs
    );
    if (res.ok) {
      return { ok: true, value: (await res.json()) as T };
    }
    const text = await res.text().catch(() => "");
    const error = new Error(`${url}: ${text || `HTTP ${res.status}`}`);
    // 5xx is usually transient (backend restarting); 4xx will not improve.
    return { ok: false, error, retryable: res.status >= 500 };
  } catch (err) {
    const error =
      err instanceof Error && err.name === "AbortError"
        ? new Error(`${url}: timed out after ${timeoutMs / 1000}s`)
        : err instanceof Error
          ? err
          : new Error(String(err));
    return { ok: false, error, retryable: true };
  }
}

async function request<T>(
  url: string,
  options?: RequestInit,
  timeoutMs: number = REQUEST_TIMEOUT_MS
): Promise<T> {
  const method = (options?.method ?? "GET").toUpperCase();
  // Only idempotent requests are retried: replaying POST /analyze would run
  // the pipeline twice on a slow-but-healthy backend.
  const maxAttempts = method === "GET" ? MAX_ATTEMPTS : 1;
  let lastError: Error = new Error(`${url}: request failed`);

  for (let n = 1; n <= maxAttempts; n += 1) {
    const result = await attempt<T>(url, options, timeoutMs);
    if (result.ok) return result.value;

    lastError = result.error;
    if (!result.retryable || n === maxAttempts) break;
    await delay(RETRY_BACKOFF_MS * n);
  }
  throw lastError;
}

export async function getHealth(): Promise<HealthStatus> {
  return request<HealthStatus>("/health");
}

export async function getDataSources(): Promise<{ sources: DataSource[] }> {
  return request("/data-sources");
}

export async function getDemoSamples(): Promise<{ samples: DemoSample[] }> {
  return request("/analyze/samples");
}

export async function getReferenceDataset(): Promise<ReferenceDatasetResponse> {
  return request("/reference-dataset");
}

// ---------------------------------------------------------------------------
// Dataset bundle
// ---------------------------------------------------------------------------

/**
 * Every dataset panel in a single request. The backend degrades per dataset
 * and always answers 200, so one slow source can never blank the dashboard --
 * which is what used to trigger the "retry" banner.
 */
export async function getDatasetsOverview(
  recentLimit = 10,
  intenseLimit = 10
): Promise<DatasetsOverview> {
  return request<DatasetsOverview>(
    `/datasets/overview?recent_limit=${recentLimit}&intense_limit=${intenseLimit}`,
    undefined,
    DATASET_TIMEOUT_MS
  );
}

export interface DatasetQuery {
  limit?: number;
  offset?: number;
  sort?: string;
  basin?: string;
  category?: string;
  min_wind?: number;
  year_min?: number;
  year_max?: number;
  search?: string;
}

/** The complete observed record with every attribute and every filter. */
export async function getIBTrACSDataset(
  query: DatasetQuery = {}
): Promise<IBTrACSDatasetResponse> {
  const params = new URLSearchParams();
  Object.entries(query).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      params.set(key, String(value));
    }
  });
  const qs = params.toString();
  return request<IBTrACSDatasetResponse>(
    `/ibtracs/dataset${qs ? `?${qs}` : ""}`,
    undefined,
    DATASET_TIMEOUT_MS
  );
}

// ---------------------------------------------------------------------------
// Live IBTrACS (real NOAA best-track archive)
// ---------------------------------------------------------------------------

export async function getIBTrACSStatus(): Promise<IBTrACSStatus> {
  return request<IBTrACSStatus>("/ibtracs/status");
}

export async function getIBTrACSRecent(
  limit = 25
): Promise<IBTrACSStormsResponse> {
  return request<IBTrACSStormsResponse>(`/ibtracs/recent?limit=${limit}`);
}

export async function getIBTrACSIntense(
  limit = 25
): Promise<IBTrACSStormsResponse> {
  return request<IBTrACSStormsResponse>(`/ibtracs/intense?limit=${limit}`);
}

export async function analyzeDemo(
  sampleId?: string
): Promise<AnalysisResult> {
  const body = { source: "demo" };
  const url = sampleId ? `/analyze/demo/${sampleId}` : "/analyze/demo";
  return request<AnalysisResult>(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function uploadImage(file: File): Promise<AnalysisResult> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("source", "upload");
  return request<AnalysisResult>("/analyze", {
    method: "POST",
    body: formData,
  });
}