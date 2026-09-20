  import { useCallback, useEffect, useRef, useState } from "react";
import {
  getHealth,
  getDataSources,
  getDemoSamples,
  getReferenceDataset,
  getIBTrACSStatus,
  getIBTrACSRecent,
  getIBTrACSIntense,
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
} from "lucide-react";
import type {
  AnalysisResult,
  DataSource,
  DemoSample,
  HealthStatus,
  IBTrACSStatus,
  IBTrACSStorm,
  ReferenceDatasetResponse,
} from "./types";

type TabKey = "analyze" | "live" | "reference" | "sources";

export default function App() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [samples, setSamples] = useState<DemoSample[]>([]);
  const [sources, setSources] = useState<DataSource[]>([]);
  const [refData, setRefData] = useState<ReferenceDatasetResponse | null>(null);
  const [ibtracsStatus, setIbtracsStatus] = useState<IBTrACSStatus | null>(null);
  const [ibtracsRecent, setIbtracsRecent] = useState<IBTrACSStorm[]>([]);
  const [ibtracsIntense, setIbtracsIntense] = useState<IBTrACSStorm[]>([]);
  const [selectedSample, setSelectedSample] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<TabKey>("analyze");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const loadHealth = useCallback(async () => {
    try {
      const h = await getHealth();
      setHealth(h);
    } catch {
      setHealth(null);
    }
  }, []);

  const loadData = useCallback(async () => {
    try {
      const [s, ds, rd] = await Promise.all([
        getDemoSamples(),
        getDataSources(),
        getReferenceDataset(),
      ]);
      setSamples(s.samples ?? []);
      setSources(ds.sources ?? []);
      setRefData(rd);
    } catch {
      // Graceful on error
    }
    // Live IBTrACS is loaded separately so a slow/failed download never
    // blocks the rest of the dashboard.
    try {
      const [st, rec, inten] = await Promise.all([
        getIBTrACSStatus(),
        getIBTrACSRecent(15),
        getIBTrACSIntense(15),
      ]);
      setIbtracsStatus(st);
      setIbtracsRecent(rec.storms ?? []);
      setIbtracsIntense(inten.storms ?? []);
    } catch {
      // Live dataset unavailable; the Live tab shows a notice.
    }
  }, []);

  useEffect(() => {
    loadHealth();
    loadData();
    const interval = setInterval(loadHealth, 30000);
    return () => clearInterval(interval);
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
          <h1>Cyclone Intelligence Dashboard</h1>
          <p>
            Upload a satellite image or pick a bundled storm sample — the
            pipeline segments the storm, classifies its intensity, calibrates
            wind and pressure against the observed NOAA record, and projects a
            forecast track with an uncertainty cone.
          </p>
        </header>
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

        {tab === "live" && (
          <LivePanel
            status={ibtracsStatus}
            recent={ibtracsRecent}
            intense={ibtracsIntense}
          />
        )}
        {tab === "reference" && refData && <ReferencePanel data={refData} />}
        {tab === "sources" && sources.length > 0 && (
          <SourcesPanel sources={sources} />
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
          <div className="value">{result.estimated_wind_speed_knots} kt</div>
          <div className="sub">
            ≈ {(result.estimated_wind_speed_knots * 1.852).toFixed(0)} km/h
          </div>
        </div>
        <div className="metric-box">
          <div className="label">Pressure</div>
          <div className="value">{result.estimated_pressure_hpa} hPa</div>
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
  recent,
  intense,
}: {
  status: IBTrACSStatus | null;
  recent: IBTrACSStorm[];
  intense: IBTrACSStorm[];
}) {
  if (!status) {
    return (
      <section className="card">
        <h3>Live IBTrACS dataset</h3>
        <div className="error-box">
          Live dataset unavailable. The backend could not reach the NOAA
          IBTrACS archive and no local cache was found. The bundled reference
          records on the previous tab remain available.
        </div>
      </section>
    );
  }

  const sourceLabel =
    status.source === "live"
      ? "Downloaded live from NOAA NCEI"
      : status.source === "cache"
      ? "Loaded from local cache"
      : status.source === "bundled"
      ? "Bundled fallback (download unavailable)"
      : status.source;

  return (
    <>
      <section className="card">
        <h3>Live IBTrACS dataset</h3>
        <p className="reference-note">
          Real observed best-track data from the NOAA NCEI International Best
          Track Archive for Climate Stewardship (IBTrACS v04r01), North Indian
          Ocean basin. Public domain. Analysis results are calibrated against
          these observed storms.
        </p>
        <div className="stat-row">
          <div className="stat">
            <div className="stat-value">{status.records}</div>
            <div className="stat-label">Observed storms</div>
          </div>
          <div className="stat">
            <div className="stat-value">{status.year_range}</div>
            <div className="stat-label">Year range</div>
          </div>
          <div className="stat">
            <div className="stat-value">{status.cache_size_mb} MB</div>
            <div className="stat-label">Local cache</div>
          </div>
          <div className="stat">
            <div className="stat-value">{sourceLabel}</div>
            <div className="stat-label">Data source</div>
          </div>
        </div>
        {status.last_error && (
          <div className="error-box" style={{ marginTop: 12 }}>
            Last download issue: {status.last_error}
          </div>
        )}
      </section>

      <section className="card mt-16">
        <h3>Most recent storms observed</h3>
        <StormTable storms={recent} />
      </section>

      <section className="card mt-16">
        <h3>Strongest storms on record (by peak sustained wind)</h3>
        <StormTable storms={intense} />
      </section>
    </>
  );
}

function StormTable({ storms }: { storms: IBTrACSStorm[] }) {
  if (storms.length === 0) {
    return <p className="reference-note">No records returned.</p>;
  }
  return (
    <div style={{ overflowX: "auto" }}>
      <table className="ref-table">
        <thead>
          <tr>
            <th>Name</th>
            <th>Year</th>
            <th>Category</th>
            <th>Wind (kt)</th>
            <th>Pressure (hPa)</th>
            <th>Genesis (lat, lon)</th>
          </tr>
        </thead>
        <tbody>
          {storms.map((s) => (
            <tr key={s.sid}>
              <td title={s.notes}>{s.name}</td>
              <td>{s.year}</td>
              <td>{s.category}</td>
              <td>{s.max_wind_knots}</td>
              <td>{s.min_pressure_hpa ?? "n/a"}</td>
              <td>
                {s.genesis_lat.toFixed(1)}, {s.genesis_lon.toFixed(1)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ReferencePanel({ data }: { data: ReferenceDatasetResponse }) {
  return <section className="card">
    <h3>Reference cyclone records</h3>
    <p className="reference-note">Curated representative examples bundled with the app so it works offline. For the full observed record downloaded from NOAA, open the Live IBTrACS tab. Demo estimates are illustrative and must not be used for weather warnings.</p>
    <p>{data.summary.total_events} events · {data.summary.year_range}</p>
    <div style={{ overflowX: "auto" }}><table className="ref-table">
      <thead><tr><th>Name</th><th>Year</th><th>Basin</th><th>Category</th><th>Wind (knots)</th><th>Pressure (hPa)</th></tr></thead>
      <tbody>{data.dataset.map((event) => <tr key={`${event.name}-${event.year}`}>
        <td title={event.notes}>{event.name}</td><td>{event.year}</td><td>{event.basin}</td>
        <td>{event.category}</td><td>{event.max_wind_knots}</td><td>{event.min_pressure_hpa}</td>
      </tr>)}</tbody>
    </table></div>
  </section>;
}

function SourcesPanel({ sources }: { sources: DataSource[] }) {
  return <section className="card"><h3>Satellite data sources</h3>
    {sources.map((source) => <article className="reference-note" key={source.id}>
      <h4>{source.name} <span className="source-chip">{source.status}</span></h4>
      <p>{source.description}</p><p>{source.region} · {source.availability}</p>
    </article>)}
  </section>;
}

function buildHeatmapDataUrl(b64: string | null | undefined): string {
  // P1-2: the API already returns an encoded PNG, so this is just the
  // data-URL wrapper the <img> tag expects.
  return b64 ? `data:image/png;base64,${b64}` : "";
}