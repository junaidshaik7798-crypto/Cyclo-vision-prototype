export interface HealthStatus {
  status: string;
  app: string;
  version: string;
  ml_mode: string;
  model_available: boolean;
  database: string;
  message: string;
}

export interface DataSource {
  id: string;
  name: string;
  satellite_type: string;
  description: string;
  status: string;
  availability: string;
  last_updated: string;
  region: string;
}

export interface DemoSample {
  id: string;
  name: string;
  description: string;
  category: string;
  filename: string;
  file_size_kb: number;
}

export interface TrackPoint {
  label: string;
  hours: number;
  lat: number;
  lon: number;
  cone_km: number;
}

export interface RiskFactor {
  factor: string;
  impact: string;
}

export interface Explainability {
  type: string;
  note: string;
  /** Base64-encoded PNG of the attention/activation overlay (P1-2). */
  heatmap_png_b64: string | null;
}

export interface AnalysisResult {
  cyclone_detected: boolean;
  classification: string;
  class_index: number;
  confidence: number;
  estimated_wind_speed_knots: number;
  estimated_pressure_hpa: number;
  intensity_category: string;
  risk_level: string;
  risk_factors: RiskFactor[];
  inference_mode: string;
  calibration_source?: string;
  reference_cyclone: string | null;
  reference_year?: number | null;
  explainability: Explainability;
  track: TrackPoint[];
  center: { lat: number; lon: number };
  source?: string;
  image_name?: string;
  demo_sample?: {
    id: string;
    name: string;
    description: string;
    category: string;
    filename: string;
    file_size_kb: number;
  };
}

export interface ReferenceCyclone {
  name: string;
  year: number;
  basin: string;
  category: string;
  max_wind_knots: number;
  min_pressure_hpa: number;
  landfall_lat: number;
  landfall_lon: number;
  track_direction_deg: number;
  forward_speed_knots: number;
  intensity_index: number;
  notes: string;
}

export interface ReferenceDatasetResponse {
  dataset: ReferenceCyclone[];
  summary: {
    total_events: number;
    year_range: string;
    basins: Record<string, number>;
    categories: string[];
    strongest_wind_knots: number;
    lowest_pressure_hpa: number;
    updated: string;
  };
}
// ---------------------------------------------------------------------------
// Live IBTrACS (real NOAA best-track archive)
// ---------------------------------------------------------------------------

export interface IBTrACSStorm {
  sid: string;
  name: string;
  year: number;
  basin: string;
  category: string;
  max_wind_knots: number;
  /** null for pre-satellite-era storms with no pressure record. */
  min_pressure_hpa: number | null;
  genesis_lat: number;
  genesis_lon: number;
  landfall_lat: number;
  landfall_lon: number;
  track_direction_deg: number;
  forward_speed_knots: number;
  intensity_index: number;
  notes: string;
}

export interface IBTrACSStormsResponse {
  source: string;
  storms: IBTrACSStorm[];
  /** "ready" | "warming" | "degraded" -- see IBTrACSStatus. */
  state?: string;
  records?: number;
}

export interface IBTrACSStatus {
  source: string;
  /**
   * "ready"    -> the real observed record is loaded,
   * "warming"  -> the download/parse is still running (show what we have),
   * "degraded" -> no local copy yet, the curated anchors are being served.
   */
  state: string;
  loading: boolean;
  records: number;
  year_range: string;
  cache_file: string;
  cache_present: boolean;
  cache_size_mb: number;
  cache_age_hours: number | null;
  cache_fresh: boolean;
  cache_ttl_hours: number;
  sidecar_file?: string;
  sidecar_present?: boolean;
  downloaded_at: string | null;
  last_error: string | null;
  url: string;
  license: string;
  records_by_basin?: Record<string, number>;
  records_by_category?: Record<string, number>;
  strongest_wind_knots?: number | null;
  lowest_pressure_hpa?: number | null;
  attribute_count?: number;
}

/** One row of the IMD intensity scale derived from the shared wind bands. */
export interface WindBand {
  class_index: number;
  min_wind_knots: number;
  category: string;
}

/** Field-level documentation of the live dataset payload. */
export interface DatasetAttribute {
  field: string;
  label: string;
  source: string;
  description: string;
}

export interface IBTrACSSummary {
  state: string;
  loading: boolean;
  source: string;
  records: number;
  named_records: number;
  records_with_pressure: number;
  year_range: string;
  first_year: number | null;
  last_year: number | null;
  strongest_wind_knots: number | null;
  lowest_pressure_hpa: number | null;
  mean_peak_wind_knots: number | null;
  basins: Record<string, number>;
  categories: Record<string, number>;
  records_by_basin: Record<string, number>;
  records_by_category: Record<string, number>;
  attribute_count: number;
  generated_at: string;
}

/** Complete observed record: every storm, every attribute, plus context. */
export interface IBTrACSDatasetResponse {
  state: string;
  loading: boolean;
  source: string;
  total: number;
  matched: number;
  returned: number;
  offset: number;
  sort: string;
  sort_orders: string[];
  storms: IBTrACSStorm[];
  summary: IBTrACSSummary;
  scale: WindBand[];
  attributes: DatasetAttribute[];
}

/** Single-request payload that hydrates every dataset panel on the page. */
export interface DatasetsOverview {
  generated_at: string;
  errors: Record<string, string>;
  samples: DemoSample[];
  sources: DataSource[];
  reference: ReferenceDatasetResponse;
  ibtracs: {
    status: IBTrACSStatus | null;
    summary: IBTrACSSummary | null;
    scale: WindBand[];
    attributes: DatasetAttribute[];
    dataset: IBTrACSDatasetResponse | null;
    recent?: IBTrACSStorm[];
    intense?: IBTrACSStorm[];
  };
}