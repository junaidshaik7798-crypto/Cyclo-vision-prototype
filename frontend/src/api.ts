import type {
  AnalysisResult,
  DataSource,
  DemoSample,
  HealthStatus,
  IBTrACSStatus,
  IBTrACSStormsResponse,
  ReferenceDatasetResponse,
} from "./types";

const BASE = "/api";

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${url}`, options);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
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