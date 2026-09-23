/* =========================================================================
   CYCLO-VISION -- shared frontend logic (standalone pages, no build step)
   API base can be overridden with ?api=... or localStorage["cyclo_api"].
   ========================================================================= */
"use strict";

const API =
  new URLSearchParams(location.search).get("api") ||
  localStorage.getItem("cyclo_api") ||
  "http://127.0.0.1:8000";

const $ = (sel) => document.querySelector(sel);
const fmt = (v, d = 1) => (v == null || isNaN(v) ? "--" : Number(v).toFixed(d));

// Datasets must survive a backend that is still starting up (or a dropped
// packet), so idempotent GETs are retried a few times before the pane falls
// back to its error box. POSTs are never replayed.
const API_RETRIES = 3;
const API_RETRY_BACKOFF_MS = 700;

async function apiFetch(path, options) {
  const method = ((options && options.method) || "GET").toUpperCase();
  const attempts = method === "GET" ? API_RETRIES : 1;
  let lastError = new Error(`${path}: request failed`);

  for (let n = 1; n <= attempts; n += 1) {
    let retryable = true;
    try {
      const res = await fetch(`${API}/api${path}`, options);
      if (res.ok) return res.json();
      let msg = `HTTP ${res.status}`;
      try { msg = (await res.text()) || msg; } catch { /* ignore */ }
      lastError = new Error(msg);
      // 5xx is usually transient (backend restarting); 4xx will not improve.
      retryable = res.status >= 500;
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err));
    }
    if (!retryable || n === attempts) break;
    await new Promise((resolve) => setTimeout(resolve, API_RETRY_BACKOFF_MS * n));
  }
  throw lastError;
}

/* ---------------- shared: health pills ---------------- */
async function loadHealthPills() {
  const ml = $("#ml-pill"), db = $("#db-pill");
  try {
    const h = await apiFetch("/health");
    if (ml) {
      ml.innerHTML = `<span class="dot"></span> ML: ${h.ml_mode}` +
        (h.ml_mode === "model" && h.model_available ? " (ready)" : "");
      ml.classList.toggle("red", h.ml_mode === "model" && !h.model_available);
    }
    if (db) {
      const online = h.database === "connected";
      db.innerHTML = `<span class="dot"></span> DB: ${online ? "connected" : "offline"}`;
      db.classList.toggle("green", online);
      db.classList.toggle("red", !online);
    }
  } catch {
    if (ml) { ml.innerHTML = `<span class="dot"></span> API unreachable`; ml.classList.add("red"); }
    if (db) { db.innerHTML = `<span class="dot"></span> DB: ?`; db.classList.add("red"); }
  }
  try {
    const s = await apiFetch("/ibtracs/status");
    const ds = $("#ds-pill");
    if (ds) {
      ds.innerHTML = `<span class="dot"></span> IBTrACS: ${s.source} (${s.records})`;
      ds.title = `${s.year_range} - ${s.url}`;
    }
  } catch { /* pill stays as-is */ }
}

/* =========================================================================
   INPUT PAGE
   ========================================================================= */
function initInputPage() {
  loadHealthPills();

  let mode = null;        // {kind:"demo", id, name} | {kind:"upload", file}
  const samplesEl = $("#samples");
  const analyzeBtn = $("#analyze");
  const fileInput = $("#file");
  const drop = $("#drop");
  const preview = $("#preview");
  const previewWrap = $("#preview-wrap");
  const previewName = $("#preview-name");
  const statusEl = $("#status");
  const errorEl = $("#error");

  function setMode(next, el) {
    mode = next;
    document.querySelectorAll(".sample").forEach((s) => s.classList.remove("active"));
    if (el) el.classList.add("active");
    if (next && next.kind === "upload") {
      previewWrap.classList.remove("hidden");
      previewName.textContent = next.file.name;
    } else {
      previewWrap.classList.add("hidden");
    }
    updateSummary();
    analyzeBtn.disabled = !mode;
  }

  function updateSummary() {
    const box = $("#input-summary");
    if (!mode) { box.classList.add("hidden"); return; }
    const label = mode.kind === "demo" ? `Demo sample: ${mode.name}` : `Upload: ${mode.file.name}`;
    box.textContent = label;
    box.classList.remove("hidden");
  }

  /* ----- demo samples ----- */
  apiFetch("/analyze/samples")
    .then(({ samples }) => {
      samplesEl.innerHTML = "";
      samples.forEach((s) => {
        const div = document.createElement("div");
        div.className = "sample";
        div.innerHTML = `
          <img src="${API}/data/demo/${s.filename}" alt="${s.name}" loading="lazy" />
          <div class="cap"><b>${s.name}</b><span>${s.category}</span></div>`;
        div.addEventListener("click", () => setMode({ kind: "demo", id: s.id, name: s.name }, div));
        samplesEl.appendChild(div);
      });
    })
    .catch((e) => {
      samplesEl.innerHTML = `<div class="error-box">Could not load demo samples: ${e.message}<br>
        Is the backend running? Try <code>?api=http://127.0.0.1:8000</code> in the URL.</div>`;
    });

  /* ----- upload ----- */
  fileInput.addEventListener("change", () => {
    const f = fileInput.files[0];
    if (!f) return;
    preview.src = URL.createObjectURL(f);
    setMode({ kind: "upload", file: f }, null);
  });
  drop.addEventListener("click", () => fileInput.click());
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault(); drop.classList.remove("over");
    const f = e.dataTransfer.files[0];
    if (!f) return;
    fileInput.files = e.dataTransfer.files;
    preview.src = URL.createObjectURL(f);
    setMode({ kind: "upload", file: f }, null);
  });

  $("#region-clear").addEventListener("click", () => { $("#region").value = ""; });

  function parseRegion() {
    const raw = $("#region").value.trim();
    if (!raw) return null;
    const parts = raw.split(/[,\s]+/).filter(Boolean).map(Number);
    if (parts.length !== 2 || parts.some(isNaN) || Math.abs(parts[0]) > 90 || Math.abs(parts[1]) > 180) {
      throw new Error("Region must be two numbers: latitude, longitude (e.g. 17.5, 88.3).");
    }
    return parts;
  }

  function readAsDataUrl(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(new Error("Could not prepare the uploaded image for the results page."));
      reader.readAsDataURL(file);
    });
  }

  /* ----- run analysis -> results page ----- */
  analyzeBtn.addEventListener("click", async () => {
    errorEl.classList.add("hidden");
    if (!mode) return;
    let region;
    try { region = parseRegion(); }
    catch (e) { errorEl.textContent = e.message; errorEl.classList.remove("hidden"); return; }

    analyzeBtn.disabled = true;
    statusEl.classList.remove("hidden");
    statusEl.innerHTML = `<span class="spin"></span> Running analysis&hellip;`;

    try {
      let result;
      if (mode.kind === "demo") {
        const body = { source: "demo" };
        if (region) body.region = region;
        result = await apiFetch(`/analyze/demo/${mode.id}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
      } else {
        const fd = new FormData();
        fd.append("file", mode.file);
        fd.append("source", "upload");
        if (region) fd.append("region", JSON.stringify(region));
        result = await apiFetch("/analyze", { method: "POST", body: fd });
      }
      sessionStorage.setItem("cyclo_result", JSON.stringify(result));
      if (mode.kind === "demo") {
        const fn = (result.demo_sample && result.demo_sample.filename) || result.image_name || "";
        sessionStorage.setItem("cyclo_img", `${API}/data/demo/${fn}`);
      } else {
        sessionStorage.setItem("cyclo_img", await readAsDataUrl(mode.file));
      }
      location.href = "results.html";
    } catch (e) {
      statusEl.classList.add("hidden");
      errorEl.textContent = `Analysis failed: ${e.message}`;
      errorEl.classList.remove("hidden");
      analyzeBtn.disabled = false;
    }
  });
}

/* =========================================================================
   RESULTS PAGE
   ========================================================================= */
function initResultsPage() {
  loadHealthPills();

  const raw = sessionStorage.getItem("cyclo_result");
  if (!raw) {
    document.querySelector(".wrap").innerHTML =
      `<div class="card full"><h2>No result yet</h2>
       <p class="muted">Run an analysis on the input page first.</p>
       <p style="margin-top:14px"><a class="btn primary" href="index.html">Go to Input Page</a></p></div>`;
    return;
  }
  const r = JSON.parse(raw);

  /* ----- verdict ----- */
  $("#verdict-class").textContent = r.classification || "--";
  $("#conf-fill").style.width = `${Math.round((r.confidence || 0) * 100)}%`;
  $("#conf-val").textContent = `${Math.round((r.confidence || 0) * 100)}%`;
  $("#m-wind").textContent = fmt(r.estimated_wind_speed_knots);
  $("#m-pres").textContent = fmt(r.estimated_pressure_hpa, 0);
  $("#m-cat").textContent = r.intensity_category || "--";
  const riskEl = $("#m-risk");
  riskEl.textContent = r.risk_level || "--";
  riskEl.style.color =
    r.risk_level === "Extreme" || r.risk_level === "High" ? "var(--red)" :
    r.risk_level === "Moderate" ? "var(--amber)" : "var(--green)";

  /* ----- image + heatmap ----- */
  const img = $("#out-img");
  img.src = sessionStorage.getItem("cyclo_img") || "";
  const heat = r.explainability && r.explainability.heatmap_png_b64;
  const layer = $("#heat-layer");
  const toggle = $("#heat-toggle");
  const blend = $("#heat-blend");
  if (heat) {
    layer.style.backgroundImage = `url(data:image/png;base64,${heat})`;
    layer.style.opacity = blend.value / 100;
    toggle.addEventListener("click", () => {
      const nowHidden = layer.classList.toggle("hidden");
      toggle.textContent = nowHidden ? "Show Heatmap" : "Hide Heatmap";
    });
    blend.addEventListener("input", () => { layer.style.opacity = blend.value / 100; });
  } else {
    toggle.disabled = true;
    toggle.textContent = "No heatmap available";
  }
  $("#img-meta").textContent = [
    r.image_name ? `file: ${r.image_name}` : null,
    r.source ? `source: ${r.source}` : null,
    r.center ? `centre: ${fmt(r.center.lat, 2)}, ${fmt(r.center.lon, 2)}` : null,
    r.analysis_id != null ? `analysis #${r.analysis_id}` : null,
  ].filter(Boolean).join("  -  ");

  /* ----- mode / reference / warnings ----- */
  const banner = $("#mode-banner");
  if (r.inference_mode === "demo") {
    banner.classList.remove("hidden");
    banner.textContent =
      "Running in DEMO mode (no trained model weights found). Results come from a transparent " +
      "feature-based heuristic. Train CycloCNN and set ML_MODE=model for model-based inference.";
  }
  $("#mode-line").textContent = `inference mode: ${r.inference_mode}` +
    (r.calibration_source ? ` - calibrated via ${r.calibration_source}` : "");
  $("#ref-line").textContent = r.reference_cyclone
    ? `closest historical analog: ${r.reference_cyclone}${r.reference_year ? ` (${r.reference_year})` : ""}`
    : "";
  if (r.persist_warning) {
    const p = $("#persist-line");
    p.textContent = `note: ${r.persist_warning}`;
    p.classList.remove("hidden");
  }

  /* ----- explainability + risk factors ----- */
  $("#expl-note").textContent = (r.explainability && r.explainability.note) || "";
  const rf = $("#risk-factors");
  rf.innerHTML = "";
  (r.risk_factors || []).forEach((f) => {
    const d = document.createElement("div");
    d.className = "rf";
    d.innerHTML = `<b>${f.factor}</b>${f.impact}`;
    rf.appendChild(d);
  });

  /* ----- track map + table ----- */
  drawTrack(r.track || []);
  const tb = $("#track-tbl tbody");
  tb.innerHTML = "";
  (r.track || []).forEach((p) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${p.label}</td><td>${fmt(p.lat, 2)}</td><td>${fmt(p.lon, 2)}</td><td>${fmt(p.cone_km, 0)}</td>`;
    tb.appendChild(tr);
  });

  /* ----- accuracy panel ----- */
  $("#a-mode").textContent = r.inference_mode || "--";
  $("#a-conf").textContent = `${Math.round((r.confidence || 0) * 100)}%`;
  $("#a-band").textContent = "10-13 kt RMSE (published IR floor)";
  $("#a-cal").textContent = r.calibration_source || "class climatology";
}

/* ---------------- SVG track map ---------------- */
function drawTrack(points) {
  const g = document.getElementById("svg-track");
  const gl = document.getElementById("svg-lats");
  if (!g) return;
  g.innerHTML = ""; gl.innerHTML = "";
  if (!points.length) {
    g.innerHTML = `<text class="grid-lbl" x="4" y="8">No track available.</text>`;
    return;
  }
  const NS = "http://www.w3.org/2000/svg";
  const lats = points.map((p) => p.lat), lons = points.map((p) => p.lon);
  const latMin = Math.min(...lats) - 2, latMax = Math.max(...lats) + 2;
  const lonMin = Math.min(...lons) - 2, lonMax = Math.max(...lons) + 2;
  const W = 100, H = 70, pad = 8;
  const X = (lon) => pad + ((lon - lonMin) / (lonMax - lonMin || 1)) * (W - 2 * pad);
  const Y = (lat) => H - pad - ((lat - latMin) / (latMax - latMin || 1)) * (H - 2 * pad);

  /* lat/lon grid */
  for (let la = Math.ceil(latMin / 5) * 5; la <= latMax; la += 5) {
    const line = document.createElementNS(NS, "line");
    line.setAttribute("x1", 0); line.setAttribute("x2", W);
    line.setAttribute("y1", Y(la)); line.setAttribute("y2", Y(la));
    line.setAttribute("class", "grid-line");
    gl.appendChild(line);
    const t = document.createElementNS(NS, "text");
    t.setAttribute("x", 1); t.setAttribute("y", Y(la) - 0.6);
    t.setAttribute("class", "grid-lbl");
    t.textContent = `${la}N`;
    gl.appendChild(t);
  }
  for (let lo = Math.ceil(lonMin / 5) * 5; lo <= lonMax; lo += 5) {
    const line = document.createElementNS(NS, "line");
    line.setAttribute("y1", 0); line.setAttribute("y2", H);
    line.setAttribute("x1", X(lo)); line.setAttribute("x2", X(lo));
    line.setAttribute("class", "grid-line");
    gl.appendChild(line);
    const t = document.createElementNS(NS, "text");
    t.setAttribute("x", X(lo) + 0.6); t.setAttribute("y", H - 1);
    t.setAttribute("class", "grid-lbl");
    t.textContent = `${lo}E`;
    gl.appendChild(t);
  }

  /* uncertainty cone around current position */
  const now = points[0];
  const coneR = Math.max(2, (now.cone_km || 0) / 111);
  const circ = document.createElementNS(NS, "circle");
  circ.setAttribute("cx", X(now.lon)); circ.setAttribute("cy", Y(now.lat));
  circ.setAttribute("r", Math.max((coneR / (lonMax - lonMin || 1)) * (W - 2 * pad), 3));
  circ.setAttribute("class", "cone");
  g.appendChild(circ);

  /* track path + points */
  const path = document.createElementNS(NS, "polyline");
  path.setAttribute("points", points.map((p) => `${X(p.lon)},${Y(p.lat)}`).join(" "));
  path.setAttribute("class", "track-draw");
  g.appendChild(path);
  points.forEach((p, i) => {
    const c = document.createElementNS(NS, "circle");
    c.setAttribute("cx", X(p.lon)); c.setAttribute("cy", Y(p.lat));
    c.setAttribute("r", i === 0 ? 1.5 : 1);
    c.setAttribute("class", i === 0 ? "track-now" : "track-pt");
    const title = document.createElementNS(NS, "title");
    title.textContent = `${p.label}: ${p.lat.toFixed(2)}, ${p.lon.toFixed(2)} (cone ${p.cone_km} km)`;
    c.appendChild(title);
    g.appendChild(c);
  });
  const legend = document.querySelector(".map-legend");
  if (legend) legend.textContent =
    `white = current - blue = forecast (+6h..+48h) - dashed circle = ${now.cone_km} km uncertainty cone`;
}

/* =========================================================================
   DATASETS & SYSTEM TABS (input page)

   Every pane renders as soon as its data arrives and never blanks out: one
   slow dataset cannot hide the others, and the page keeps polling quietly
   while the backend is still reading the NOAA archive. There is deliberately
   no "Retry" prompt -- failures are reported inline as information.
   ========================================================================= */
const DATASET_POLL_MS = 3000;
const DATASET_MAX_POLLS = 40;

function esc(value) {
  return String(value == null ? "" : value).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function stateLabel(state) {
  if (state === "ready") return "loaded";
  if (state === "warming") return "loading archive...";
  if (state === "degraded") return "bundled records";
  return "starting";
}

function attrValue(storm, field) {
  const value = storm[field];
  if (value == null || value === "") return "n/a";
  if (typeof value === "number") {
    if (field.endsWith("_lat") || field.endsWith("_lon")) return value.toFixed(2);
    if (field.endsWith("_knots") || field.endsWith("_deg")) return value.toFixed(1);
    return String(value);
  }
  return String(value);
}

function countLine(counts) {
  if (!counts) return "-";
  return Object.entries(counts).map(([k, v]) => `${esc(k)}: ${v}`).join(" &middot; ");
}

/* -------- full record table with search / sort / export -------- */
function renderStormsTable(storms, attributes) {
  const cols = attributes && attributes.length ? attributes : DEFAULT_ATTRIBUTES;
  const head = cols.map((a) => `<th title="${esc(a.description)}">${esc(a.label)}</th>`).join("");
  const body = storms.map((s) => `<tr>${cols
    .map((a) => `<td title="${esc(attrValue(s, a.field))}">${esc(attrValue(s, a.field))}</td>`)
    .join("")}</tr>`).join("");
  return `<div class="tbl-wrap"><table class="tbl">
    <thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

const DEFAULT_ATTRIBUTES = [
  { field: "sid", label: "SID" },
  { field: "name", label: "Name" },
  { field: "year", label: "Year" },
  { field: "basin", label: "Basin" },
  { field: "category", label: "Category" },
  { field: "max_wind_knots", label: "Peak wind" },
  { field: "min_pressure_hpa", label: "Min pressure" },
  { field: "genesis_lat", label: "Genesis lat" },
  { field: "genesis_lon", label: "Genesis lon" },
  { field: "landfall_lat", label: "Final lat" },
  { field: "landfall_lon", label: "Final lon" },
  { field: "track_direction_deg", label: "Bearing" },
  { field: "forward_speed_knots", label: "Speed" },
  { field: "intensity_index", label: "Intensity" },
  { field: "notes", label: "Notes" },
];

function downloadText(filename, text, mime) {
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

function stormsToCsv(storms) {
  if (!storms.length) return "";
  const cols = Object.keys(storms[0]);
  const cell = (v) => {
    const text = v == null ? "" : String(v);
    return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  };
  return [cols.join(","), ...storms.map((s) => cols.map((c) => cell(s[c])).join(","))].join("\n");
}

function initDatasetsSection() {
  const tabsBox = document.querySelector(".tabs");
  if (!tabsBox) return; // not on the input page

  function activate(name) {
    document.querySelectorAll(".tab").forEach((t) =>
      t.classList.toggle("active", t.dataset.tab === name));
    document.querySelectorAll(".tabpane").forEach((p) =>
      p.classList.toggle("hidden", p.id !== `tab-${name}`));
  }
  tabsBox.addEventListener("click", (e) => {
    const btn = e.target.closest(".tab");
    if (btn) activate(btn.dataset.tab);
  });

  const statusBox = $("#datasets-status");
  const srcPane = $("#tab-sources");
  const livePane = $("#tab-live");
  const refPane = $("#tab-ref");

  let polls = 0;
  let liveState = "cold";
  let liveSearch = "";
  let liveSort = "recent";
  let liveMinWind = 0;
  let liveLimit = 0;          // 0 = every record
  let liveStorms = [];
  let liveAttributes = [];
  let liveSummary = null;
  let liveScale = [];
  let liveStatus = null;

  function setStatus(html, tone) {
    if (!statusBox) return;
    statusBox.className = `dataset-status${tone ? ` ${tone}` : ""}`;
    statusBox.innerHTML = html;
  }

  /* ----- renderers: every pane is self-contained and never blanks ----- */

  function renderSourcesPane(sources, health, errors) {
    if (!sources) {
      srcPane.innerHTML = `<div class="loading">Loading the ingestion catalog&hellip;</div>`;
      return;
    }
    const rows = sources.map((s) => `
      <tr>
        <td>${esc(s.name)}</td><td>${esc(s.satellite_type || "-")}</td>
        <td>${esc(s.region || "-")}</td>
        <td><span class="tag ${s.status === "operational" ? "ok" : ""}">${esc(s.status)}</span></td>
        <td>${esc(s.availability || "-")}</td>
        <td>${esc(s.last_updated || "-")}</td>
      </tr>`).join("");
    srcPane.innerHTML = `
      <div class="stat-row">
        <div class="metric"><div class="v">${sources.length}</div><div class="l">Catalogued sources</div></div>
        <div class="metric"><div class="v">${esc(health ? health.ml_mode : "-")}</div><div class="l">ML mode</div></div>
        <div class="metric"><div class="v">${health && health.model_available ? "yes" : "no"}</div><div class="l">Model weights</div></div>
        <div class="metric"><div class="v">${esc(health ? health.database : "-")}</div><div class="l">Database</div></div>
        <div class="metric"><div class="v">v${esc(health ? health.version : "-")}</div><div class="l">API version</div></div>
      </div>
      <p class="muted small"><strong>Dataset attributes:</strong> identifier, provider,
        satellite type, region, status, availability and source update time; the system
        block reports ML mode, model availability, database state and API version.</p>
      <div class="tbl-wrap"><table class="tbl">
        <thead><tr><th>Name</th><th>Satellite</th><th>Region</th><th>Status</th>
        <th>Availability</th><th>Updated</th></tr></thead>
        <tbody>${rows}</tbody>
      </table></div>
      ${errors && errors.sources ? `<div class="inline-note">Sources reported: ${esc(errors.sources)}</div>` : ""}`;
  }

  function filterStorms() {
    const needle = liveSearch.trim().toLowerCase();
    const rows = liveStorms.filter((s) => {
      if (s.max_wind_knots < liveMinWind) return false;
      if (!needle) return true;
      return (
        String(s.name).toLowerCase().includes(needle) ||
        String(s.sid).toLowerCase().includes(needle) ||
        String(s.category).toLowerCase().includes(needle) ||
        String(s.basin).toLowerCase().includes(needle) ||
        String(s.year) === needle
      );
    });
    switch (liveSort) {
      case "oldest":
        rows.sort((a, b) => a.year - b.year || b.max_wind_knots - a.max_wind_knots);
        break;
      case "intense":
        rows.sort((a, b) => b.max_wind_knots - a.max_wind_knots || b.year - a.year);
        break;
      case "weakest":
        rows.sort((a, b) => a.max_wind_knots - b.max_wind_knots || b.year - a.year);
        break;
      case "name":
        rows.sort((a, b) => String(a.name).localeCompare(String(b.name)) || b.year - a.year);
        break;
      case "pressure":
        rows.sort((a, b) =>
          (a.min_pressure_hpa || 9999) - (b.min_pressure_hpa || 9999) || b.year - a.year);
        break;
      default:
        rows.sort((a, b) => b.year - a.year || b.max_wind_knots - a.max_wind_knots);
    }
    return rows;
  }

  /** Summary block for the live pane: every statistic the API reports. */
  function liveSummaryHtml() {
    const status = liveStatus || {};
    const summary = liveSummary || {};
    return `
      <div class="stat-row">
        <div class="metric"><div class="v">${liveStorms.length}</div><div class="l">Observed storms</div></div>
        <div class="metric"><div class="v">${esc(summary.year_range || status.year_range || "-")}</div><div class="l">Year range</div></div>
        <div class="metric"><div class="v">${esc(stateLabel(status.state || liveState))}</div><div class="l">Dataset state</div></div>
        <div class="metric"><div class="v">${esc(status.source || "-")}</div><div class="l">Data source</div></div>
        <div class="metric"><div class="v">${esc(status.cache_size_mb != null ? status.cache_size_mb + " MB" : "-")}</div><div class="l">Local cache</div></div>
        <div class="metric"><div class="v">${esc(summary.strongest_wind_knots != null ? summary.strongest_wind_knots + " kt" : "-")}</div><div class="l">Strongest wind</div></div>
        <div class="metric"><div class="v">${esc(summary.lowest_pressure_hpa != null ? summary.lowest_pressure_hpa + " hPa" : "-")}</div><div class="l">Lowest pressure</div></div>
        <div class="metric"><div class="v">${esc(summary.named_records != null ? summary.named_records : "-")}</div><div class="l">Named storms</div></div>
      </div>
      <p class="muted small">Source: <strong>${esc(status.source || "-")}</strong>
        &middot; ${esc(status.records != null ? status.records : liveStorms.length)} records
        &middot; cache ${status.cache_present ? esc(status.cache_size_mb) + " MB" : "missing"}
        &middot; age ${status.cache_age_hours != null ? esc(status.cache_age_hours) + " h" : "n/a"}
        &middot; licence ${esc(status.license || "Public domain (NOAA/NCEI)")}.
        <a href="${esc(status.url || "https://www.ncei.noaa.gov/products/international-best-track-archive")}"
           target="_blank" rel="noreferrer">NOAA IBTrACS</a></p>
      <p class="muted small"><strong>Records by basin:</strong> ${countLine(summary.records_by_basin || status.records_by_basin)}<br>
        <strong>Records by IMD category:</strong> ${countLine(summary.records_by_category || status.records_by_category)}</p>
      ${status.last_error ? `<div class="inline-note">Last refresh reported: ${esc(status.last_error)}.
        The records below come from the local best-track copy.</div>` : ""}`;
  }

  /** Table + toolbar for the complete observed record. */
  function liveTableHtml(rows, limit) {
    return `
      <div class="subhead">Complete observed record &mdash; every attribute</div>
      <div class="toolbar">
        <input id="live-search" class="inp" type="search"
               placeholder="Search name, SID, basin, category or year" value="${esc(liveSearch)}" />
        <select id="live-sort" class="inp">
          ${[["recent", "Newest first"], ["oldest", "Oldest first"], ["intense", "Strongest wind"],
             ["weakest", "Weakest wind"], ["name", "Storm name (A-Z)"], ["pressure", "Lowest pressure"]]
            .map(([v, l]) => `<option value="${v}"${liveSort === v ? " selected" : ""}>${l}</option>`).join("")}
        </select>
        <select id="live-minwind" class="inp">
          ${[[0, "Any wind"], [34, "34 kt+"], [48, "48 kt+"], [64, "64 kt+"], [90, "90 kt+"], [120, "120 kt+"]]
            .map(([v, l]) => `<option value="${v}"${liveMinWind === v ? " selected" : ""}>${l}</option>`).join("")}
        </select>
        <button id="live-limit" class="btn ghost" type="button">
          ${liveLimit > 0 ? `Show all ${rows.length}` : "Show first 25"}
        </button>
        <button id="live-csv" class="btn ghost" type="button">Download CSV</button>
        <button id="live-json" class="btn ghost" type="button">Download JSON</button>
        <span class="muted small">showing ${limit.length} of ${rows.length} filtered
          (${liveStorms.length} total)</span>
      </div>
      ${renderStormsTable(limit, liveAttributes)}

      <div class="subhead">Dataset attributes (${(liveAttributes.length ? liveAttributes : DEFAULT_ATTRIBUTES).length})</div>
      <div class="tbl-wrap"><table class="tbl">
        <thead><tr><th>Field</th><th>Label</th><th>Source</th><th>Description</th></tr></thead>
        <tbody>${(liveAttributes.length ? liveAttributes : DEFAULT_ATTRIBUTES).map((a) => `
          <tr><td>${esc(a.field)}</td><td>${esc(a.label)}</td>
          <td>${esc(a.source || "-")}</td><td>${esc(a.description || "-")}</td></tr>`).join("")}
        </tbody></table></div>

      ${liveScale.length ? `<div class="subhead">IMD intensity scale (${liveScale.length} bands)</div>
        <div class="tbl-wrap"><table class="tbl">
          <thead><tr><th>Class index</th><th>Minimum wind (kt)</th><th>Category</th></tr></thead>
          <tbody>${liveScale.map((b) => `<tr><td>${b.class_index}</td>
            <td>${b.min_wind_knots}</td><td>${esc(b.category)}</td></tr>`).join("")}</tbody>
        </table></div>` : ""}
      <p class="muted small">Curated IMD-calibrated events are on the Reference Dataset tab;
        analysis results are calibrated against this observed record.</p>`;
  }

  function renderLivePane() {
    if (!liveStorms.length) {
      livePane.innerHTML = `<div class="loading">Reading the NOAA best-track archive&hellip;
        this pane fills in automatically.</div>`;
      return;
    }
    const rows = filterStorms();
    const limit = liveLimit > 0 ? rows.slice(0, liveLimit) : rows;
    livePane.innerHTML = liveSummaryHtml() + liveTableHtml(rows, limit);

    const search = $("#live-search");
    if (search) {
      search.addEventListener("input", (e) => { liveSearch = e.target.value; renderLivePane(); });
      search.focus();
      search.setSelectionRange(search.value.length, search.value.length);
    }
    const sort = $("#live-sort");
    if (sort) sort.addEventListener("change", (e) => { liveSort = e.target.value; renderLivePane(); });
    const minWind = $("#live-minwind");
    if (minWind) minWind.addEventListener("change", (e) => {
      liveMinWind = Number(e.target.value);
      renderLivePane();
    });
    const limitBtn = $("#live-limit");
    if (limitBtn) limitBtn.addEventListener("click", () => {
      liveLimit = liveLimit > 0 ? 0 : 25;
      renderLivePane();
    });
    const csv = $("#live-csv");
    if (csv) csv.addEventListener("click", () => {
      downloadText("cyclo-vision-ibtracs-storms.csv", stormsToCsv(limit), "text/csv");
    });
    const json = $("#live-json");
    if (json) json.addEventListener("click", () => {
      downloadText("cyclo-vision-ibtracs-storms.json", JSON.stringify({
        summary: liveSummary, scale: liveScale, attributes: liveAttributes, storms: limit,
      }, null, 2), "application/json");
    });
  }

  /* ----- tab 3: curated reference dataset ----- */

  function renderReferencePane(reference, errors) {
    if (!reference || !reference.dataset) {
      refPane.innerHTML = `<div class="loading">Loading the curated calibration set&hellip;</div>`;
      return;
    }
    const { dataset, summary } = reference;
    const rows = dataset.map((d) => `
      <tr>
        <td>${esc(d.name)}</td><td>${d.year}</td><td>${esc(d.basin)}</td>
        <td><span class="tag">${esc(d.category)}</span></td>
        <td>${Math.round(d.max_wind_knots)} kt</td>
        <td>${Math.round(d.min_pressure_hpa)} hPa</td>
        <td>${fmt(d.landfall_lat, 1)}, ${fmt(d.landfall_lon, 1)}</td>
        <td>${Math.round(d.track_direction_deg)}&deg; / ${fmt(d.forward_speed_knots, 1)} kt</td>
        <td>${d.intensity_index}</td>
        <td>${esc(d.notes)}</td>
      </tr>`).join("");
    refPane.innerHTML = `
      <div class="stat-row">
        <div class="metric"><div class="v">${summary.total_events}</div><div class="l">Curated events</div></div>
        <div class="metric"><div class="v">${esc(summary.year_range)}</div><div class="l">Years</div></div>
        <div class="metric"><div class="v">${summary.strongest_wind_knots} kt</div><div class="l">Strongest</div></div>
        <div class="metric"><div class="v">${summary.lowest_pressure_hpa} hPa</div><div class="l">Lowest pressure</div></div>
      </div>
      <p class="muted small"><strong>Dataset attributes:</strong> name, year, basin, category,
        peak wind, minimum pressure, landfall coordinates, track direction, forward speed,
        intensity index and notes.<br>
        <strong>Basins:</strong> ${countLine(summary.basins)}<br>
        <strong>Categories:</strong> ${(summary.categories || []).map(esc).join(" &middot; ") || "-"}<br>
        <strong>Updated:</strong> ${esc(summary.updated || "-")}</p>
      <div class="tbl-wrap"><table class="tbl">
        <thead><tr><th>Name</th><th>Year</th><th>Basin</th><th>Category</th><th>Peak wind</th>
        <th>Min pressure</th><th>Landfall</th><th>Track</th><th>Intensity</th><th>Notes</th></tr></thead>
        <tbody>${rows}</tbody>
      </table></div>
      <p class="muted small">Curated IMD-calibrated events used for intensity calibration and
        analog matching. The full observed record is on the Live IBTrACS tab.</p>
      ${errors && errors.reference ? `<div class="inline-note">Reference dataset reported:
        ${esc(errors.reference)} &mdash; showing the last known copy.</div>` : ""}`;
  }

  /* ----- one-shot loader with per-dataset degradation ----- */

  function applyOverview(data) {
    const errors = data.errors || {};
    renderSourcesPane(data.sources || [], null, errors);
    const ib = data.ibtracs || {};
    liveStatus = ib.status || null;
    liveSummary = ib.summary || null;
    liveScale = ib.scale || [];
    liveAttributes = ib.attributes || [];
    liveStorms = (ib.dataset && ib.dataset.storms) || [];
    liveState = (ib.status && ib.status.state) || (ib.dataset && ib.dataset.state) || "cold";
    renderLivePane();
    renderReferencePane(data.reference, errors);
    setStatus(
      `<strong>Datasets:</strong> ${liveStorms.length} IBTrACS records &middot; `
      + `${(data.reference && data.reference.dataset ? data.reference.dataset.length : 0)} reference events `
      + `&middot; ${(data.sources || []).length} sources &middot; ${(data.samples || []).length} demo samples `
      + `&middot; state: ${esc(stateLabel(liveState))}`,
      liveState === "warming" ? "busy" : liveState === "degraded" ? "warn" : "ok"
    );
  }

  async function loadDatasets() {
    try {
      const data = await apiFetch("/datasets/overview?recent_limit=5&intense_limit=5");
      applyOverview(data);
      // The backend answers instantly from its local copy; while it is still
      // warming the archive we poll quietly until the full record lands.
      if (liveState === "warming" && polls < DATASET_MAX_POLLS) {
        polls += 1;
        setTimeout(loadDatasets, DATASET_POLL_MS);
      }
    } catch (err) {
      // Older backend or unreachable API: fall back to the individual routes
      // so every pane still gets its own chance to render.
      const errors = {};
      const [sourcesRes, healthRes, liveRes, refRes] = await Promise.allSettled([
        apiFetch("/data-sources"),
        apiFetch("/health"),
        apiFetch("/ibtracs/dataset"),
        apiFetch("/reference-dataset"),
      ]);

      if (sourcesRes.status === "fulfilled") {
        renderSourcesPane(sourcesRes.value.sources || [],
          healthRes.status === "fulfilled" ? healthRes.value : null, errors);
      } else {
        errors.sources = String((sourcesRes.reason && sourcesRes.reason.message) || sourcesRes.reason);
        renderSourcesPane([], null, errors);
      }

      if (liveRes.status === "fulfilled") {
        const value = liveRes.value || {};
        liveStatus = value;
        liveSummary = value.summary || null;
        liveScale = value.scale || [];
        liveAttributes = value.attributes || [];
        liveStorms = value.storms || [];
        liveState = value.state || "cold";
        renderLivePane();
      } else {
        livePane.innerHTML = `<div class="loading">The live archive is unavailable right
          now &mdash; the backend keeps retrying automatically. The Reference Dataset tab
          stays fully available.</div>`;
      }

      if (refRes.status === "fulfilled") {
        renderReferencePane(refRes.value, errors);
      } else {
        errors.reference = String((refRes.reason && refRes.reason.message) || refRes.reason);
        renderReferencePane(null, errors);
      }

      setStatus(
        `<strong>Datasets:</strong> ${liveStorms.length} IBTrACS records &middot; `
        + `state: ${esc(stateLabel(liveState))} &middot; recovery runs automatically`,
        liveState === "warming" ? "busy" : liveState === "degraded" ? "warn" : "ok"
      );
      if (liveState === "warming" && polls < DATASET_MAX_POLLS) {
        polls += 1;
        setTimeout(loadDatasets, DATASET_POLL_MS);
      }
    }
  }

  renderSourcesPane(null, null, null);
  renderLivePane();
  renderReferencePane(null, null);
  setStatus("Reading the dataset catalog&hellip;", "busy");
  loadDatasets();
}

