
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  getHealth,
  getDatasetsOverview,
  getDataSources,
  getDemoSamples,
  getReferenceDataset,
  getIBTrACSStatus,
  getIBTrACSRecent,
  getIBTrACSIntense,
  getIBTrACSDataset,
  analyzeDemo,
  uploadImage,
} from "./api";
import {
  ScanSearch,
  Satellite,
  LibraryBig,
  SatelliteDish,
  UploadCloud,
  CloudLightning,
  RefreshCw,
  Database,
  Clock3,
  AlertTriangle,
  Search,
  Download,
  Info,
  Loader2,
} from "lucide-react";
import type {
  AnalysisResult,
  DataSource,
  DatasetAttribute,
  DatasetsOverview,
  DemoSample,
  HealthStatus,
  IBTrACSDatasetResponse,
  IBTrACSSummary,
  IBTrACSStatus,
  IBTrACSStorm,
  ReferenceDatasetResponse,
  WindBand,
} from "./types";

type TabKey = "analyze" | "live" | "reference" | "sources";

type DatasetKey = "samples" | "sources" | "reference" | "ibtracs";
type DatasetErrors = Partial<Record<DatasetKey, string>>;

// Dataset auto-retry policy: transient failures (backend still booting, a slow
// IBTrACS warm-up, a dropped packet) are retried with a linear backoff, capped
// so a genuinely offline backend does not spin forever. Recovery is automatic:
// the user never has to press a Retry button.
const MAX_AUTO_RETRIES = 5;
const AUTO_RETRY_DELAY_MS = 4000;
const HEALTH_POLL_MS = 30000;
// While the archive is still being read/downloaded by the backend the panels
// already show the local copy, so a short poll is enough to pick up the final
// record. Bounded, so a permanently offline backend stops polling.
const WARM_POLL_MS = 3000;
const MAX_WARM_POLLS = 40;

function formatTimestamp(value: string | number | Date | null | undefined): string {
  if (!value) return "Not available";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
}

function formatEstimate(value: number, suffix: string, decimals = 1): string {
  return Number.isFinite(value) ? `${value.toFixed(decimals)} ${suffix}` : "--";
}

/** "ready" | "warming" | "degraded" -> label shown on the status chips. */
function datasetStateLabel(state: string | undefined): string {
  switch (state) {
    case "ready":
      return "Loaded";
    case "warming":
      return "Loading archive…";
    case "degraded":
      return "Bundled records";
    default:
      return "Starting";
  }
}

const SORT_LABELS: Record<string, string> = {
  recent: "Newest first",
  oldest: "Oldest first",
  intense: "Strongest wind",
  weakest: "Weakest wind",
  name: "Storm name (A-Z)",
  pressure: "Lowest pressure",
};

/** Summary tiles rendered for the live dataset (one per reported statistic). */
const SUMMARY_TILES: { key: string; label: string }[] = [
  { key: "records", label: "Observed storms" },
  { key: "year_range", label: "Year range" },
  { key: "named_records", label: "Named storms" },
  { key: "records_with_pressure", label: "Records with pressure" },
  { key: "strongest_wind_knots", label: "Strongest wind (kt)" },
  { key: "lowest_pressure_hpa", label: "Lowest pressure (hPa)" },
  { key: "mean_peak_wind_knots", label: "Mean peak wind (kt)" },
  { key: "attribute_count", label: "Dataset attributes" },
];

/**
 * Field documentation used until the API payload (which carries the
 * authoritative list, including its source column) arrives. Keeping a copy
 * here means the panel can always explain every column it renders.
 */
const FALLBACK_ATTRIBUTES: DatasetAttribute[] = [
  { field: "sid", label: "Storm ID (SID)", source: "IBTrACS SID", description: "Unique storm identifier." },
  { field: "name", label: "Storm name", source: "IBTrACS NAME", description: "Official name, or UNNAMED." },
  { field: "year", label: "Season (year)", source: "IBTrACS SEASON", description: "Season of the storm." },
  { field: "basin", label: "Basin", source: "IBTrACS BASIN", description: "Reporting basin of the track." },
  { field: "category", label: "IMD category", source: "derived from USA_WIND", description: "IMD band of the peak wind." },
  { field: "max_wind_knots", label: "Peak wind", source: "IBTrACS USA_WIND", description: "Peak 1-minute sustained wind (kt)." },
  { field: "min_pressure_hpa", label: "Minimum pressure", source: "IBTrACS USA_PRES", description: "Lowest central pressure (hPa)." },
  { field: "genesis_lat", label: "Genesis latitude", source: "IBTrACS LAT", description: "First valid track point (deg N)." },
  { field: "genesis_lon", label: "Genesis longitude", source: "IBTrACS LON", description: "First valid track point (deg E)." },
  { field: "landfall_lat", label: "Final latitude", source: "IBTrACS LAT", description: "Last valid track point (deg N)." },
  { field: "landfall_lon", label: "Final longitude", source: "IBTrACS LON", description: "Last valid track point (deg E)." },
  { field: "track_direction_deg", label: "Track bearing", source: "derived", description: "Great-circle bearing (deg)." },
  { field: "forward_speed_knots", label: "Forward speed", source: "derived", description: "Implied translation speed (kt)." },
  { field: "intensity_index", label: "Intensity index", source: "derived from USA_WIND", description: "Engine class index 0-6." },
  { field: "notes", label: "Notes", source: "generated", description: "Provenance note for the record." },
];

/** Render one dataset attribute exactly as documented by the backend. */
function formatAttribute(field: string, storm: IBTrACSStorm): string {
  const value = (storm as unknown as Record<string, unknown>)[field];
  if (value === null || value === undefined || value === "") return "n/a";
  if (typeof value === "number") {
    if (field.endsWith("_lat") || field.endsWith("_lon")) return value.toFixed(2);
    if (field.endsWith("_knots") || field.endsWith("_deg")) return value.toFixed(1);
    return String(value);
  }
  return String(value);
}

/** Trigger a client-side download of a text payload (CSV/JSON export). */
function downloadTextFile(filename: string, text: string, mime: string): void {
  const blob = new Blob([text], { type: `${mime};charset=utf-8` });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

function stormsToCsv(storms: IBTrACSStorm[]): string {
  if (storms.length === 0) return "";
  const columns = Object.keys(storms[0]) as (keyof IBTrACSStorm)[];
  const escape = (value: unknown) => {
    const text = value === null || value === undefined ? "" : String(value);
    return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  };
  const rows = storms.map((storm) =>
    columns.map((column) => escape(storm[column])).join(",")
  );
  return [columns.join(","), ...rows].join("\n");
}

/** One summary tile from either payload (`summary` wins over `status`). */
function formatSummaryValue(
  key: string,
  summary: IBTrACSSummary | null,
  status: IBTrACSStatus | null
): string {
  const source: Record<string, unknown> = {
    ...(status ?? {}),
    ...(summary ?? {}),
  } as Record<string, unknown>;
  const value = source[key];
  if (value === null || value === undefined || value === "") {
    if (key === "records" && status?.records) return String(status.records);
    return "—";
  }
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(1);
  }
  return String(value);
}

/** "NI: 273 · WP: 69" for the basin/category breakdowns. */
function formatCounts(
  primary: Record<string, number> | undefined,
  fallback: Record<string, number> | undefined
): string {
  const counts = primary ?? fallback;
  if (!counts) return "";
  return Object.entries(counts)
    .map(([label, count]) => `${label}: ${count}`)
    .join(" · ");
}

export default function App() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [overview, setOverview] = useState<DatasetsOverview | null>(null);
  const [samples, setSamples] = useState<DemoSample[]>([]);
  const [sources, setSources] = useState<DataSource[]>([]);
  const [refData, setRefData] = useState<ReferenceDatasetResponse | null>(null);
  const [ibtracsStatus, setIbtracsStatus] = useState<IBTrACSStatus | null>(null);
  const [ibtracsDataset, setIbtracsDataset] = useState<IBTrACSDatasetResponse | null>(null);
  const [ibtracsRecent, setIbtracsRecent] = useState<IBTrACSStorm[]>([]);
  const [ibtracsIntense, setIbtracsIntense] = useState<IBTrACSStorm[]>([]);
  const [selectedSample, setSelectedSample] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<TabKey>("analyze");
  const [datasetErrors, setDatasetErrors] = useState<DatasetErrors>({});
  const [datasetNotices, setDatasetNotices] = useState<Record<string, string>>({});
  const [datasetsFetchedAt, setDatasetsFetchedAt] = useState<Date | null>(null);
  const [refreshingDatasets, setRefreshingDatasets] = useState(false);
  const [showDatasetDetails, setShowDatasetDetails] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  // Resilience bookkeeping: a transient backend hiccup must never leave a
  // dataset panel permanently empty, so failures are retried automatically.
  const inFlightRef = useRef(false);
  const retryAttemptRef = useRef(0);
  const warmPollRef = useRef(0);
  const datasetErrorsRef = useRef<DatasetErrors>({});

  useEffect(() => {
    datasetErrorsRef.current = datasetErrors;
  }, [datasetErrors]);

  const loadHealth = useCallback(async (): Promise<boolean> => {
    try {
      const h = await getHealth();
      setHealth(h);
      return true;
    } catch {
      setHealth(null);
      return false;
    }
  }, []);

  /** The live-record state reported by the backend. */
  const ibtracsState = ibtracsStatus?.state ?? ibtracsDataset?.state ?? "cold";
  const ibtracsWarming = ibtracsState === "warming";

  // The complete observed record, with the one-shot overview payload as a
  // fallback so the panel renders even if only one of the two requests landed.
  const liveDataset = ibtracsDataset ?? overview?.ibtracs?.dataset ?? null;
  const liveSummary: IBTrACSSummary | null =
    liveDataset?.summary ?? overview?.ibtracs?.summary ?? null;
  const liveScale: WindBand[] = liveDataset?.scale ?? overview?.ibtracs?.scale ?? [];
  const liveAttributes: DatasetAttribute[] =
    liveDataset?.attributes ?? overview?.ibtracs?.attributes ?? [];

  /**
   * Pull every dataset. One request (`/datasets/overview`) fills the whole
   * dashboard; the individual endpoints are only used when that call itself
   * fails, so a single slow source can no longer blank the other panels.
   */
  const loadData = useCallback(async () => {
    // Guard against overlapping runs (manual refresh + auto-retry + health
    // recovery probe firing together would duplicate every request).
    if (inFlightRef.current) return;
    inFlightRef.current = true;
    setRefreshingDatasets(true);
    const errors: DatasetErrors = {};
    let loaded = false;
    try {
      try {
        const data = await getDatasetsOverview(10, 10);
        setOverview(data);
        setSamples(data.samples ?? []);
        setSources(data.sources ?? []);
        setRefData(data.reference ?? null);
        setIbtracsStatus(data.ibtracs?.status ?? null);
        setIbtracsDataset(data.ibtracs?.dataset ?? null);
        if (data.ibtracs?.recent) setIbtracsRecent(data.ibtracs.recent);
        if (data.ibtracs?.intense) setIbtracsIntense(data.ibtracs.intense);
        setDatasetNotices(data.errors ?? {});
        loaded = true;
      } catch {
        // Fallback path: the individual endpoints, each degrading on its own.
        const [sampleResult, sourceResult, referenceResult, statusResult, datasetResult] =
          await Promise.allSettled([
            getDemoSamples(),
            getDataSources(),
            getReferenceDataset(),
            getIBTrACSStatus(),
            getIBTrACSDataset(),
          ]);

        if (sampleResult.status === "fulfilled") {
          setSamples(sampleResult.value.samples ?? []);
          loaded = true;
        } else {
          errors.samples = sampleResult.reason instanceof Error
            ? sampleResult.reason.message
            : "Could not load demo samples";
        }

        if (sourceResult.status === "fulfilled") {
          setSources(sourceResult.value.sources ?? []);
          loaded = true;
        } else {
          errors.sources = sourceResult.reason instanceof Error
            ? sourceResult.reason.message
            : "Could not load data sources";
        }

        if (referenceResult.status === "fulfilled") {
          setRefData(referenceResult.value);
          loaded = true;
        } else {
          errors.reference = referenceResult.reason instanceof Error
            ? referenceResult.reason.message
            : "Could not load reference data";
        }

        if (statusResult.status === "fulfilled") {
          setIbtracsStatus(statusResult.value);
          loaded = true;
        } else {
          errors.ibtracs = statusResult.reason instanceof Error
            ? statusResult.reason.message
            : "Could not load IBTrACS status";
        }

        if (datasetResult.status === "fulfilled") {
          setIbtracsDataset(datasetResult.value);
          loaded = true;
        } else if (!errors.ibtracs) {
          errors.ibtracs = datasetResult.reason instanceof Error
            ? datasetResult.reason.message
            : "Could not load the observed record";
        }

        if (statusResult.status === "fulfilled") {
          const [recentResult, intenseResult] = await Promise.allSettled([
            getIBTrACSRecent(10),
            getIBTrACSIntense(10),
          ]);
          if (recentResult.status === "fulfilled") {
            setIbtracsRecent(recentResult.value.storms ?? []);
          }
          if (intenseResult.status === "fulfilled") {
            setIbtracsIntense(intenseResult.value.storms ?? []);
          }
        }
      }

      setDatasetErrors(errors);
      // Only claim a fetch time when at least one dataset actually arrived,
      // otherwise the control strip would read "Fetched ..." over empty panels.
      if (loaded) setDatasetsFetchedAt(new Date());
      if (Object.keys(errors).length === 0) retryAttemptRef.current = 0;
    } finally {
      inFlightRef.current = false;
      setRefreshingDatasets(false);
    }
  }, []);

  /**
   * Cheap re-read of the live record only: used while the backend is still
   * warming the archive and by the recovery probe. It never reports an error
   * to the user -- the panels keep showing the data they already have.
   */
  const refreshLiveRecord = useCallback(async () => {
    if (inFlightRef.current) return;
    inFlightRef.current = true;
    try {
      const [dataset, status] = await Promise.allSettled([
        getIBTrACSDataset(),
        getIBTrACSStatus(),
      ]);
      if (dataset.status === "fulfilled") setIbtracsDataset(dataset.value);
      if (status.status === "fulfilled") setIbtracsStatus(status.value);
      if (dataset.status === "fulfilled" || status.status === "fulfilled") {
        setDatasetsFetchedAt(new Date());
      }
    } finally {
      inFlightRef.current = false;
    }
  }, []);

  // Automatic retry with linear backoff. The counter is reset by a fully
  // successful fetch (above) or by the health recovery probe (below), so an
  // outage simply keeps the automatic recovery loop running.
  useEffect(() => {
    if (Object.keys(datasetErrors).length === 0) {
      retryAttemptRef.current = 0;
      return;
    }
    if (retryAttemptRef.current >= MAX_AUTO_RETRIES) return;
    const timer = window.setTimeout(() => {
      retryAttemptRef.current += 1;
      loadData();
    }, AUTO_RETRY_DELAY_MS * (retryAttemptRef.current + 1));
    return () => window.clearTimeout(timer);
  }, [datasetErrors, loadData]);

  // While the archive is warming, poll briefly for the final record. The
  // panels already render the local copy, so this is silent by design.
  useEffect(() => {
    if (!ibtracsWarming || warmPollRef.current >= MAX_WARM_POLLS) return;
    const timer = window.setTimeout(() => {
      warmPollRef.current += 1;
      refreshLiveRecord();
    }, WARM_POLL_MS);
    return () => window.clearTimeout(timer);
  }, [ibtracsWarming, ibtracsDataset, refreshLiveRecord]);

  useEffect(() => {
    loadHealth();
    loadData();
    // The health poll doubles as a recovery probe: when the API comes back
    // after an outage, the datasets are fetched again immediately instead of
    // waiting for the user to notice.
    const interval = window.setInterval(async () => {
      const healthy = await loadHealth();
      if (!healthy) return;
      if (Object.keys(datasetErrorsRef.current).length > 0) {
        retryAttemptRef.current = 0;
        warmPollRef.current = 0;
        loadData();
      }
    }, HEALTH_POLL_MS);
    return () => window.clearInterval(interval);
  }, [loadHealth, loadData]);

  const runAnalysis = useCallback(
    async (sampleId?: string) => {
      setLoading(true);
      setError(null);
      setResult(null);
      setImageUrl(null);
      try {
        const r = await analyzeDemo(sampleId);
        setResult(r);
        const filename = r.demo_sample?.filename;
        if (filename) {
          setImageUrl(`/data/demo/${filename}`);
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : "Analysis failed");
      } finally {
        setLoading(false);
      }
    },
    []
  );
  const handlePickSample = (id: string) => {
    setSelectedSample(id);
    runAnalysis(id);
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setImageUrl(URL.createObjectURL(file));
    try {
      const r = await uploadImage(file);
      setResult(r);
      setSelectedSample(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setLoading(false);
      e.target.value = "";
    }
  };

  return (
    <div className="app">
      <div className="aurora" aria-hidden="true" />
      <div className="aurora-grid" aria-hidden="true" />
      <nav className="navbar">
        <div className="logo">
          <div className="spinner" />
          <span>CYCLO-VISION</span>
          <span style={{ fontSize: "0.8rem", color: "#64748b", fontWeight: 400 }}>
            Cyclone Intelligence
          </span>
        </div>
        <div style={{ display: "flex", gap: "8px" }}>
          {health && (
            <>
              <span className="status-pill">
                <span
                  style={{
                    width: 8,
                    height: 8,
                    borderRadius: "50%",
                    background: health.status === "ok" ? "#4ade80" : "#f87171",
                  }}
                />
                {health.ml_mode === "demo" ? "Demo Mode" : "Model Mode"}
              </span>
              <span
                className={`database-pill ${
                  health.database === "connected" ? "" : "offline"
                }`}
              >
                DB: {health.database}
              </span>
            </>
          )}
        </div>
      </nav>

      <div className="container">
        <header className="hero">
          <div className="hero-kicker"><span className="live-dot" /> OPERATIONS DATA DESK</div>
          <h1>Cyclone Intelligence Dashboard</h1>
          <p>
            Upload a satellite image or pick a bundled storm sample — the
            pipeline segments the storm, classifies its intensity, calibrates
            wind and pressure against the observed NOAA record, and projects a
            forecast track with an uncertainty cone.
          </p>
        </header>
        <section className="dashboard-strip" aria-label="Dataset status">
          <div className="strip-title"><Database size={17} /> Dataset control room</div>
          <div className="strip-items">
            <span className="dataset-state">Samples: {samples.length}</span>
            <span className="dataset-state">Reference events: {refData?.dataset.length ?? 0}</span>
            <span className="dataset-state">
              IBTrACS records: {ibtracsStatus?.records ?? ibtracsDataset?.total ?? 0}
            </span>
            <span className="dataset-state">Sources: {sources.length}</span>
            <span className={`dataset-state ${ibtracsWarming ? "busy" : ibtracsState === "degraded" ? "warn" : ""}`}>
              {datasetStateLabel(ibtracsState)}
            </span>
          </div>
          <div className="strip-time"><Clock3 size={14} /> Fetched {formatTimestamp(datasetsFetchedAt)}</div>
          <button className="icon-button" onClick={loadData} disabled={refreshingDatasets} title="Refresh all datasets" aria-label="Refresh all datasets">
            <RefreshCw size={16} className={refreshingDatasets ? "spin-icon" : ""} />
          </button>
        </section>
        {/* Informational only: the panels below always render whatever arrived,
            and the client keeps retrying in the background by itself. */}
        {(ibtracsWarming || ibtracsState === "degraded" || Object.keys(datasetNotices).length > 0) && (
          <div className={`dataset-alert ${ibtracsState === "degraded" ? "" : "info"}`}>
            {ibtracsWarming ? (
              <Loader2 size={17} className="spin-icon" />
            ) : ibtracsState === "degraded" ? (
              <AlertTriangle size={17} />
            ) : (
              <Info size={17} />
            )}
            <span>
              {ibtracsWarming
                ? "Reading the live NOAA archive — the panels below already show every record that is available locally and update automatically."
                : ibtracsState === "degraded"
                  ? "Live archive unreachable right now: the bundled curated records are shown and the app keeps retrying quietly in the background."
                  : "Some datasets reported a recoverable issue; every panel below still shows the data it has."}
              {Object.keys(datasetNotices).length > 0 && (
                <button
                  type="button"
                  className="link-button"
                  onClick={() => setShowDatasetDetails((v) => !v)}
                >
                  {showDatasetDetails ? "Hide details" : "Show details"}
                </button>
              )}
              {showDatasetDetails && (
                <small className="dataset-error-list">
                  {Object.entries({ ...datasetNotices, ...datasetErrors })
                    .map(([key, message]) => `${key}: ${message}`)
                    .join(" | ")}
                </small>
              )}
            </span>
          </div>
        )}
        <div className="tabs">
          <button
            className={`tab-btn ${tab === "analyze" ? "active" : ""}`}
            onClick={() => setTab("analyze")}
          >
            <ScanSearch size={16} /> Analyze
          </button>
          <button
            className={`tab-btn ${tab === "live" ? "active" : ""}`}
            onClick={() => setTab("live")}
          >
            <Satellite size={16} /> Live IBTrACS ({ibtracsStatus?.records ?? 0})
          </button>
          <button
            className={`tab-btn ${tab === "reference" ? "active" : ""}`}
            onClick={() => setTab("reference")}
          >
            <LibraryBig size={16} /> Reference Dataset (
            {refData?.dataset.length ?? 0})
          </button>
          <button
            className={`tab-btn ${tab === "sources" ? "active" : ""}`}
            onClick={() => setTab("sources")}
          >
            <SatelliteDish size={16} /> Data Sources
          </button>
        </div>
{tab === "analyze" && (
          <div className="grid">
            {/* LEFT: sample picker */}
            <div className="rise-in">
              <div className="card">
                <h3>Demo Satellite Samples</h3>
                {samples.length === 0 && (
                  <p style={{ color: "#94a3b8", fontSize: "0.85rem" }}>
                    No demo samples found.
                  </p>
                )}
                {samples.map((s) => (
                  <div
                    key={s.id}
                    className={`demo-sample ${
                      selectedSample === s.id ? "active" : ""
                    }`}
                    onClick={() => handlePickSample(s.id)}
                  >
                    <div className="name">
                      {s.name}
                      <span className="source-chip" style={{ marginLeft: 8 }}>
                        {s.category}
                      </span>
                    </div>
                    <div className="desc">{s.description}</div>
                  </div>
                ))}

                <div className="upload-area" onClick={handleUploadClick}>
                  <UploadCloud
                    size={28}
                    style={{ marginBottom: 8, color: "#38bdf8" }}
                  />
                  <div style={{ fontSize: "0.9rem", color: "#94a3b8" }}>
                    Click to upload satellite image
                  </div>
                </div>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".png,.jpg,.jpeg,.tif,.tiff"
                  onChange={handleFileChange}
                />
              </div>

              {refData && (
                <div className="card mt-16">
                  <h3>Reference Database</h3>
                  <div style={{ fontSize: "0.85rem", color: "#94a3b8" }}>
                    {refData.summary.total_events} historical cyclone events (
                    {refData.summary.year_range})
                    <br />
                    Basis:{" "}
                    {Object.entries(refData.summary.basins)
                      .map(([b, n]) => `${b}: ${n}`)
                      .join(" · ")}
                  </div>
                </div>
              )}
            </div>

            {/* RIGHT: results */}
            <div className="rise-in" style={{ animationDelay: "0.1s" }}>
              {loading && (
                <div className="card">
                  <div className="loading">
                    <span className="loader-spin" />
                    Analyzing satellite image...
                  </div>
                </div>
              )}

              {error && <div className="error-box">{error}</div>}

              {result && !loading && (
                <ResultPanel result={result} imageUrl={imageUrl} />
              )}

              {!result && !loading && !error && (
                <div className="card">
                  <h3>Analysis Results</h3>
                  <div
                    style={{
                      color: "#94a3b8",
                      textAlign: "center",
                      padding: "40px 20px",
                    }}
                  >
                    <CloudLightning
                      size={44}
                      style={{ marginBottom: 12, color: "#38bdf8" }}
                    />
                    <p>Select a demo sample or upload a satellite image</p>
                    <p style={{ fontSize: "0.8rem", marginTop: 8 }}>
                      The AI will classify the cyclone, estimate wind speed,
                      pressure, risk, and generate a forecast track.
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Every tab renders unconditionally: a panel shows the data it has
            (or an explicit empty state) instead of disappearing when a
            request is still in flight. */}
        {tab === "live" && (
          <LivePanel
            status={ibtracsStatus}
            summary={liveSummary}
            dataset={liveDataset}
            scale={liveScale}
            attributes={liveAttributes}
            recent={ibtracsRecent}
            intense={ibtracsIntense}
            fetchedAt={datasetsFetchedAt}
            stale={Boolean(datasetErrors.ibtracs)}
          />
        )}
        {tab === "reference" && (
          <ReferencePanel data={refData} fetchedAt={datasetsFetchedAt} />
        )}
        {tab === "sources" && (
          <SourcesPanel sources={sources} fetchedAt={datasetsFetchedAt} />
        )}
      </div>
    </div>
  );
}
function ResultPanel({
  result,
  imageUrl,
}: {
  result: AnalysisResult;
  imageUrl: string | null;
}) {
  const riskClass = (result.risk_level || "low").toLowerCase();

  // P1-2: the backend now returns a base64 PNG, so the client no longer
  // has to rasterise a 1-2 MB pixel array through a canvas.
  const heatmapDataUrl = buildHeatmapDataUrl(
    result.explainability?.heatmap_png_b64,
  );

  return (
    <div className="card">
      <div className="result-header">
        <h3 style={{ margin: 0 }}>Analysis Result</h3>
        <span
          className={`status-badge ${
            result.cyclone_detected ? "detected" : "not-detected"
          }`}
        >
          {result.cyclone_detected ? "CYCLONE DETECTED" : "NO CYCLONE"}
        </span>
        <span className="status-badge demo">
          {result.inference_mode === "demo" ? "DEMO MODE" : "ML MODEL"}
        </span>
      </div>

      {imageUrl && (
        <div className="image-preview">
          <img src={imageUrl} alt="Satellite input" />
        </div>
      )}

      <div className="metrics">
        <div className="metric-box">
          <div className="label">Classification</div>
          <div className="value" style={{ fontSize: "1rem" }}>
            {result.classification}
          </div>
          <div className="sub">{result.intensity_category}</div>
        </div>
        <div className="metric-box">
          <div className="label">Confidence</div>
          <div className="value">{(result.confidence * 100).toFixed(1)}%</div>
          <div className="confidence-bar">
            <span style={{ width: `${Math.min(result.confidence * 100, 100)}%` }} />
          </div>
        </div>
        <div className="metric-box">
          <div className="label">Wind Speed</div>
          <div className="value">
            {formatEstimate(result.estimated_wind_speed_knots, "kt")}
          </div>
          <div className="sub">
            {Number.isFinite(result.estimated_wind_speed_knots)
              ? `≈ ${(result.estimated_wind_speed_knots * 1.852).toFixed(0)} km/h`
              : "Estimate unavailable"}
          </div>
        </div>
        <div className="metric-box">
          <div className="label">Pressure</div>
          <div className="value">
            {formatEstimate(result.estimated_pressure_hpa, "hPa")}
          </div>
        </div>
      </div>

      <div className={`risk-card ${riskClass}`}>
        <div className="flex">
          <div>
            <div
              style={{
                fontSize: "0.72rem",
                textTransform: "uppercase",
                opacity: 0.7,
              }}
            >
              Risk Level
            </div>
            <div style={{ fontSize: "1.2rem", fontWeight: 700 }}>
              {result.risk_level}
            </div>
          </div>
          <div style={{ textAlign: "right", fontSize: "0.8rem", opacity: 0.7 }}>
            Center: {result.center.lat.toFixed(2)}°N,{" "}
            {result.center.lon.toFixed(2)}°E
          </div>
        </div>
        {result.risk_factors.length > 0 && (
          <ul style={{ marginTop: 10, fontSize: "0.8rem", listStyle: "none" }}>
            {result.risk_factors.map((f, i) => (
              <li key={i} style={{ padding: "2px 0" }}>
                <b>{f.factor}:</b> {f.impact}
              </li>
            ))}
          </ul>
        )}
      </div>

      {result.reference_cyclone && (
        <div className="reference-note">
          {"\u{1F4DA}"} Calibrated against observed storm:{" "}
          <b>{result.reference_cyclone}</b>
          {result.reference_year ? ` (${result.reference_year})` : ""} from{" "}
          {result.calibration_source === "ibtracs-live"
            ? "the live NOAA IBTrACS archive"
            : "the bundled reference dataset"}
        </div>
      )}

      <TrackMap points={result.track} />

      {heatmapDataUrl && (
        <div className="mt-16">
          <h3 style={{ marginBottom: 8 }}>Explainability Heatmap</h3>
          <div className="heatmap-preview">
            <img src={heatmapDataUrl} alt="Attention heatmap" />
          </div>
          <div style={{ fontSize: "0.75rem", color: "#64748b", marginTop: 4 }}>
            Type: {result.explainability.type}
          </div>
        </div>
      )}
    </div>
  );
}
function TrackMap({
  points,
}: {
  points: { label: string; lat: number; lon: number; cone_km: number }[];
}) {
  if (!points || points.length === 0) return null;

  const lats = points.map((p) => p.lat);
  const lons = points.map((p) => p.lon);
  const minLat = Math.min(...lats) - 2;
  const maxLat = Math.max(...lats) + 2;
  const minLon = Math.min(...lons) - 2;
  const maxLon = Math.max(...lons) + 2;

  const x = (lon: number) => ((lon - minLon) / (maxLon - minLon)) * 100;
  const y = (lat: number) => ((maxLat - lat) / (maxLat - minLat)) * 100;

  return (
    <div className="track-map">
      <svg viewBox="0 0 400 240" style={{ width: "100%", height: "auto" }}>
        <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
          <path
            d="M 40 0 L 0 0 0 40"
            fill="none"
            stroke="rgba(56,189,248,0.08)"
            strokeWidth="0.5"
          />
        </pattern>
        <rect width="400" height="240" fill="url(#grid)" />
        <polyline
          className="track-path"
          points={points.map((p) => `${x(p.lon)},${y(p.lat)}`).join(" ")}
          fill="none"
          stroke="#38bdf8"
          strokeWidth="2"
          strokeLinejoin="round"
        />
        {points.map((p, i) => (
          <g key={i}>
            <circle
              cx={x(p.lon)}
              cy={y(p.lat)}
              r={i === 0 ? 7 : 4}
              fill={i === 0 ? "#f87171" : "#38bdf8"}
              stroke="#fff"
              strokeWidth="1"
            />
            {i === 0 && (
              <circle
                className="track-dot-current"
                cx={x(p.lon)}
                cy={y(p.lat)}
                r="12"
                fill="rgba(248,113,113,0.25)"
              />
            )}
            {i === 0 &&
              points[0].cone_km > 0 && (
                <circle
                  cx={x(p.lon)}
                  cy={y(p.lat)}
                  r={Math.min(30, Math.max(6, points[0].cone_km * 0.35))}
                  fill="rgba(248,113,113,0.12)"
                  stroke="rgba(248,113,113,0.35)"
                  strokeDasharray="3 3"
                />
              )}
            <text
              x={x(p.lon) + 8}
              y={y(p.lat) + 4}
              fill="#e0f2fe"
              fontSize="10"
              fontFamily="sans-serif"
            >
              {p.label}
            </text>
          </g>
        ))}
        <text x="10" y="15" fill="#7dd3fc" fontSize="10" fontFamily="sans-serif">
          Forecast Track ({points.length} points)
        </text>
      </svg>
    </div>
  );
}

function LivePanel({
  status,
  summary,
  dataset,
  scale,
  attributes,
  recent,
  intense,
  fetchedAt,
  stale,
}: {
  status: IBTrACSStatus | null;
  summary: IBTrACSSummary | null;
  dataset: IBTrACSDatasetResponse | null;
  scale: WindBand[];
  attributes: DatasetAttribute[];
  recent: IBTrACSStorm[];
  intense: IBTrACSStorm[];
  fetchedAt: Date | null;
  stale: boolean;
}) {
  const [search, setSearch] = useState("");
  const [sort, setSort] = useState("recent");
  const [basin, setBasin] = useState("all");
  const [minWind, setMinWind] = useState("0");
  const [showAll, setShowAll] = useState(true);

  const storms = dataset?.storms ?? [];
  const allAttributes = attributes.length > 0 ? attributes : FALLBACK_ATTRIBUTES;

  const basins = useMemo(() => {
    const found = new Set<string>();
    storms.forEach((s) => found.add(s.basin));
    return Array.from(found).sort();
  }, [storms]);

  /** Client-side filtering keeps the table instant and works offline. */
  const visible = useMemo(() => {
    const needle = search.trim().toLowerCase();
    const floor = Number(minWind) || 0;
    const rows = storms.filter((s) => {
      if (basin !== "all" && s.basin !== basin) return false;
      if (s.max_wind_knots < floor) return false;
      if (!needle) return true;
      return (
        s.name.toLowerCase().includes(needle) ||
        s.sid.toLowerCase().includes(needle) ||
        s.category.toLowerCase().includes(needle) ||
        s.basin.toLowerCase().includes(needle) ||
        String(s.year) === needle
      );
    });
    const sorted = [...rows];
    switch (sort) {
      case "oldest":
        sorted.sort((a, b) => a.year - b.year || b.max_wind_knots - a.max_wind_knots);
        break;
      case "intense":
        sorted.sort((a, b) => b.max_wind_knots - a.max_wind_knots || b.year - a.year);
        break;
      case "weakest":
        sorted.sort((a, b) => a.max_wind_knots - b.max_wind_knots || b.year - a.year);
        break;
      case "name":
        sorted.sort((a, b) => a.name.localeCompare(b.name) || b.year - a.year);
        break;
      case "pressure":
        sorted.sort(
          (a, b) =>
            (a.min_pressure_hpa ?? Number.MAX_SAFE_INTEGER) -
              (b.min_pressure_hpa ?? Number.MAX_SAFE_INTEGER) || b.year - a.year
        );
        break;
      default:
        sorted.sort((a, b) => b.year - a.year || b.max_wind_knots - a.max_wind_knots);
    }
    return sorted;
  }, [storms, search, sort, basin, minWind]);

  const rowsShown = showAll ? visible : visible.slice(0, 25);

  const sourceLabel =
    status?.source === "live"
      ? "Downloaded live from NOAA NCEI"
      : status?.source === "cache"
        ? "Loaded from the local cache"
        : status?.source === "bundled"
          ? "Bundled curated records (live archive unreachable)"
          : (status?.source ?? "Not reported");

  if (!status && !dataset) {
    return (
      <section className="card dataset-panel">
        <div className="panel-heading">
          <div><span className="eyebrow">NOAA / NCEI</span><h3>Live IBTrACS dataset</h3></div>
          <span className="timestamp"><Clock3 size={14} /> waiting for the backend…</span>
        </div>
        <div className="loading-state">
          <Loader2 size={18} className="spin-icon" /> Connecting to the dataset API… this
          panel fills in automatically as soon as the first response arrives.
        </div>
      </section>
    );
  }

  return (
    <>
      <section className="card dataset-panel">
        <div className="panel-heading">
          <div><span className="eyebrow">NOAA / NCEI</span><h3>Live IBTrACS dataset</h3></div>
          <span className="timestamp">
            <Clock3 size={14} /> fetched {formatTimestamp(fetchedAt)}
          </span>
        </div>
        <p className="reference-note">
          Real observed best-track data from the NOAA NCEI International Best
          Track Archive for Climate Stewardship (IBTrACS v04r01), North Indian
          Ocean basin. Public domain. Analysis results are calibrated against
          these observed storms.
        </p>
        <div className="stat-row">
          {SUMMARY_TILES.map((tile) => (
            <div className="stat" key={tile.key}>
              <div className="stat-value">
                {formatSummaryValue(tile.key, summary, status)}
              </div>
              <div className="stat-label">{tile.label}</div>
            </div>
          ))}
        </div>

        <div className="detail-grid">
          <div><dt>State</dt><dd>{datasetStateLabel(status?.state ?? dataset?.state)}</dd></div>
          <div><dt>Data source</dt><dd>{sourceLabel}</dd></div>
          <div><dt>Archive URL</dt><dd className="wrap-anywhere">{status?.url ?? "—"}</dd></div>
          <div><dt>Licence</dt><dd>{status?.license ?? "Public domain (NOAA/NCEI)"}</dd></div>
          <div><dt>Cache file</dt><dd className="wrap-anywhere">{status?.cache_file ?? "—"}</dd></div>
          <div><dt>Cache size</dt><dd>{status ? `${status.cache_size_mb} MB` : "—"}</dd></div>
          <div><dt>Cache age</dt><dd>{status?.cache_age_hours != null ? `${status.cache_age_hours} h` : "—"}</dd></div>
          <div><dt>Refresh TTL</dt><dd>{status ? `${status.cache_ttl_hours} h` : "—"}</dd></div>
          <div><dt>Downloaded</dt><dd>{formatTimestamp(status?.downloaded_at)}</dd></div>
          <div><dt>Records returned</dt><dd>{dataset ? `${dataset.returned} of ${dataset.total}` : "—"}</dd></div>
        </div>

        <div className="attribute-line">
          <b>Records by basin:</b>{" "}
          {formatCounts(summary?.records_by_basin, status?.records_by_basin) || "—"}
          <br />
          <b>Records by IMD category:</b>{" "}
          {formatCounts(summary?.records_by_category, status?.records_by_category) || "—"}
        </div>

        {status?.last_error && (
          <div className="dataset-alert info" style={{ marginBottom: 0 }}>
            <Info size={16} />
            <span>
              Last archive refresh reported: {status.last_error}. The records shown here
              come from the local best-track copy and stay fully usable.
            </span>
          </div>
        )}

        {stale && (
          <div className="dataset-alert info" style={{ marginBottom: 0 }}>
            <Info size={16} />
            <span>This panel is showing the last successfully loaded record set.</span>
          </div>
        )}

        <details className="dataset-docs" open>
          <summary>Dataset attributes ({allAttributes.length}) — every field in this dataset</summary>
          <div className="table-scroll">
            <table className="ref-table wide-table">
              <thead>
                <tr><th>Field</th><th>Label</th><th>Source column</th><th>Description</th></tr>
              </thead>
              <tbody>
                {allAttributes.map((attr) => (
                  <tr key={attr.field}>
                    <td className="mono">{attr.field}</td>
                    <td>{attr.label}</td>
                    <td>{attr.source}</td>
                    <td>{attr.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>

        {scale.length > 0 && (
          <details className="dataset-docs">
            <summary>IMD intensity scale used to label the record ({scale.length} bands)</summary>
            <div className="table-scroll">
              <table className="ref-table">
                <thead>
                  <tr><th>Class index</th><th>Minimum wind (kt)</th><th>Category</th></tr>
                </thead>
                <tbody>
                  {scale.map((band) => (
                    <tr key={band.class_index}>
                      <td>{band.class_index}</td>
                      <td>{band.min_wind_knots}</td>
                      <td>{band.category}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        )}
      </section>

      <section className="card mt-16 dataset-panel">
        <div className="panel-heading">
          <div>
            <span className="eyebrow">COMPLETE OBSERVED RECORD</span>
            <h3>{storms.length} storms, every attribute</h3>
          </div>
          <span className="timestamp">
            showing {rowsShown.length} of {visible.length} filtered ({storms.length} total)
          </span>
        </div>

        <div className="table-toolbar">
          <label className="filter-field">
            <Search size={14} />
            <input
              className="filter-input"
              type="search"
              placeholder="Search name, SID, basin, category or year"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
            />
          </label>
          <label className="filter-field">
            <span>Sort</span>
            <select className="filter-input" value={sort} onChange={(e) => setSort(e.target.value)}>
              {Object.entries(SORT_LABELS).map(([value, label]) => (
                <option key={value} value={value}>{label}</option>
              ))}
            </select>
          </label>
          <label className="filter-field">
            <span>Basin</span>
            <select className="filter-input" value={basin} onChange={(e) => setBasin(e.target.value)}>
              <option value="all">All basins</option>
              {basins.map((b) => <option key={b} value={b}>{b}</option>)}
            </select>
          </label>
          <label className="filter-field">
            <span>Min wind</span>
            <select className="filter-input" value={minWind} onChange={(e) => setMinWind(e.target.value)}>
              <option value="0">Any</option>
              <option value="34">34 kt+</option>
              <option value="48">48 kt+</option>
              <option value="64">64 kt+</option>
              <option value="90">90 kt+</option>
              <option value="120">120 kt+</option>
            </select>
          </label>
          <button
            type="button"
            className="pill-button"
            onClick={() => setShowAll((value) => !value)}
            disabled={visible.length <= 25}
          >
            {showAll ? "Show first 25" : `Show all ${visible.length}`}
          </button>
          <button
            type="button"
            className="pill-button"
            disabled={rowsShown.length === 0}
            onClick={() => downloadTextFile(
              "cyclo-vision-ibdtracs-storms.csv",
              stormsToCsv(rowsShown),
              "text/csv"
            )}
          >
            <Download size={14} /> CSV
          </button>
          <button
            type="button"
            className="pill-button"
            disabled={rowsShown.length === 0}
            onClick={() => downloadTextFile(
              "cyclo-vision-ibtracs-storms.json",
              JSON.stringify({ summary, scale, attributes: allAttributes, storms: rowsShown }, null, 2),
              "application/json"
            )}
          >
            <Download size={14} /> JSON
          </button>
        </div>

        {storms.length === 0 ? (
          <div className="loading-state">
            <Loader2 size={18} className="spin-icon" /> The observed record is being read —
            the table fills in automatically.
          </div>
        ) : (
          <div className="table-scroll">
            <table className="ref-table wide-table">
              <thead>
                <tr>
                  {allAttributes.map((attr) => (
                    <th key={attr.field} title={attr.description}>{attr.label}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rowsShown.map((storm) => (
                  <tr key={storm.sid}>
                    {allAttributes.map((attr) => (
                      <td
                        key={attr.field}
                        title={formatAttribute(attr.field, storm)}
                      >
                        {formatAttribute(attr.field, storm)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="card mt-16">
        <div className="panel-heading">
          <div><span className="eyebrow">LEADERBOARDS</span><h3>Most recent storms observed</h3></div>
          <span className="timestamp">{recent.length} shown</span>
        </div>
        <StormTable storms={recent} />
      </section>

      <section className="card mt-16">
        <div className="panel-heading">
          <div><span className="eyebrow">LEADERBOARDS</span><h3>Strongest storms on record (by peak sustained wind)</h3></div>
          <span className="timestamp">{intense.length} shown</span>
        </div>
        <StormTable storms={intense} />
      </section>
    </>
  );
}

function StormTable({ storms }: { storms: IBTrACSStorm[] }) {
  if (storms.length === 0) {
    return <p className="reference-note">No records returned yet.</p>;
  }
  return (
    <div className="table-scroll">
      <table className="ref-table wide-table">
        <thead>
          <tr>
            {FALLBACK_ATTRIBUTES.map((attr) => (
              <th key={attr.field} title={attr.description}>{attr.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {storms.map((storm) => (
            <tr key={storm.sid}>
              {FALLBACK_ATTRIBUTES.map((attr) => (
                <td key={attr.field} title={formatAttribute(attr.field, storm)}>
                  {formatAttribute(attr.field, storm)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ReferencePanel({ data, fetchedAt }: { data: ReferenceDatasetResponse | null; fetchedAt: Date | null }) {
  if (!data) {
    return <section className="card dataset-panel">
      <div className="panel-heading">
        <div><span className="eyebrow">BUNDLED CALIBRATION SET</span><h3>Reference cyclone records</h3></div>
        <span className="timestamp"><Clock3 size={14} /> waiting for the backend…</span>
      </div>
      <div className="loading-state">
        <Loader2 size={18} className="spin-icon" /> Loading the curated calibration set… it
        appears here automatically.
      </div>
    </section>;
  }

  const basins = formatCounts(data.summary.basins, undefined);
  return <section className="card dataset-panel">
    <div className="panel-heading">
      <div><span className="eyebrow">BUNDLED CALIBRATION SET</span><h3>Reference cyclone records</h3></div>
      <span className="timestamp"><Clock3 size={14} /> fetched {formatTimestamp(fetchedAt)}</span>
    </div>
    <p className="reference-note">Curated representative examples bundled with the app so it works offline. For the full observed record downloaded from NOAA, open the Live IBTrACS tab. Demo estimates are illustrative and must not be used for weather warnings.</p>
    <div className="stat-row">
      <div className="stat"><div className="stat-value">{data.summary.total_events}</div><div className="stat-label">Curated events</div></div>
      <div className="stat"><div className="stat-value">{data.summary.year_range}</div><div className="stat-label">Year range</div></div>
      <div className="stat"><div className="stat-value">{data.summary.strongest_wind_knots} kt</div><div className="stat-label">Strongest wind</div></div>
      <div className="stat"><div className="stat-value">{data.summary.lowest_pressure_hpa} hPa</div><div className="stat-label">Lowest pressure</div></div>
    </div>
    <div className="attribute-line">
      <b>Dataset attributes:</b> name, year, basin, category, peak wind, minimum pressure,
      landfall coordinates, track direction, forward speed, intensity index, and notes.
      <br /><b>Basins:</b> {basins || "—"}
      <br /><b>Categories:</b> {data.summary.categories.join(" · ") || "—"}
      <br /><b>Last updated:</b> {formatTimestamp(data.summary.updated)}
    </div>
    <div className="table-scroll"><table className="ref-table wide-table">
      <thead><tr><th>Name</th><th>Year</th><th>Basin</th><th>Category</th><th>Wind (knots)</th><th>Pressure (hPa)</th><th>Landfall</th><th>Track</th><th>Intensity</th><th>Notes</th></tr></thead>
      <tbody>{data.dataset.map((event) => <tr key={`${event.name}-${event.year}`}>
        <td title={event.notes}>{event.name}</td><td>{event.year}</td><td>{event.basin}</td>
        <td>{event.category}</td><td>{event.max_wind_knots}</td><td>{event.min_pressure_hpa}</td>
        <td>{event.landfall_lat.toFixed(1)}, {event.landfall_lon.toFixed(1)}</td>
        <td>{event.track_direction_deg.toFixed(0)}° / {event.forward_speed_knots.toFixed(1)} kt</td>
        <td>{event.intensity_index}</td><td title={event.notes}>{event.notes}</td>
      </tr>)}</tbody>
    </table></div>
  </section>;
}

function SourcesPanel({ sources, fetchedAt }: { sources: DataSource[]; fetchedAt: Date | null }) {
  return <section className="card dataset-panel"><div className="panel-heading"><div><span className="eyebrow">INGESTION CATALOG</span><h3>Satellite data sources ({sources.length})</h3></div><span className="timestamp"><Clock3 size={14} /> fetched {formatTimestamp(fetchedAt)}</span></div>
    <div className="attribute-line"><b>Dataset attributes:</b> identifier, provider, satellite type, description, status, availability, region, and source update time.</div>
    {sources.length === 0 ? (
      <div className="loading-state">
        <Loader2 size={18} className="spin-icon" /> Loading the ingestion catalog…
      </div>
    ) : (
      <div className="source-grid">{sources.map((source) => <article className="source-card" key={source.id}>
        <div className="source-card-head"><h4>{source.name}</h4><span className="source-chip">{source.status}</span></div>
        <p>{source.description}</p>
        <dl className="source-details"><div><dt>ID</dt><dd>{source.id}</dd></div><div><dt>Type</dt><dd>{source.satellite_type}</dd></div><div><dt>Region</dt><dd>{source.region}</dd></div><div><dt>Availability</dt><dd>{source.availability}</dd></div><div><dt>Updated</dt><dd>{formatTimestamp(source.last_updated)}</dd></div></dl>
      </article>)}</div>
    )}
  </section>;
}

function buildHeatmapDataUrl(b64: string | null | undefined): string {
  // P1-2: the API already returns an encoded PNG, so this is just the
  // data-URL wrapper the <img> tag expects.
  return b64 ? `data:image/png;base64,${b64}` : "";
}