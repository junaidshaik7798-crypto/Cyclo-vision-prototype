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

async function apiFetch(path, options) {
  const res = await fetch(`${API}/api${path}`, options);
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try { msg = (await res.text()) || msg; } catch { /* ignore */ }
    throw new Error(msg);
  }
  return res.json();
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
   ========================================================================= */
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

  /* ----- tab 1: data sources + system ----- */
  const srcPane = $("#tab-sources");
  Promise.all([apiFetch("/data-sources"), apiFetch("/health")])
    .then(([{ sources }, h]) => {
      const rows = sources.map((s) => `
        <tr>
          <td>${s.name}</td><td>${s.satellite_type || "-"}</td>
          <td>${s.region || "-"}</td>
          <td><span class="tag ${s.status === "operational" ? "ok" : ""}">${s.status}</span></td>
          <td>${s.availability || "-"}</td>
        </tr>`).join("");
      srcPane.innerHTML = `
        <table class="tbl">
          <thead><tr><th>Name</th><th>Satellite</th><th>Region</th><th>Status</th><th>Availability</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
        <div class="stat-row">
          <div class="metric"><div class="v">${h.ml_mode}</div><div class="l">ML mode</div></div>
          <div class="metric"><div class="v">${h.model_available ? "yes" : "no"}</div><div class="l">Model weights</div></div>
          <div class="metric"><div class="v">${h.database}</div><div class="l">Database</div></div>
          <div class="metric"><div class="v">v${h.version}</div><div class="l">API version</div></div>
        </div>`;
    })
    .catch((e) => { srcPane.innerHTML = `<div class="error-box">${e.message}</div>`; });

  /* ----- tab 2: live IBTrACS storms ----- */
  const livePane = $("#tab-live");
  Promise.all([
    apiFetch("/ibtracs/status"),
    apiFetch("/ibtracs/intense?limit=15"),
    apiFetch("/ibtracs/recent?limit=15"),
  ])
    .then(([st, intense, recent]) => {
      const row = (s) => `
        <tr>
          <td>${s.name}</td><td>${s.year}</td><td>${s.basin}</td>
          <td><span class="tag">${s.category}</span></td>
          <td>${Math.round(s.max_wind_knots)} kt</td>
          <td>${s.min_pressure_hpa == null ? "-" : Math.round(s.min_pressure_hpa) + " hPa"}</td>
          <td>${fmt(s.landfall_lat, 1)}, ${fmt(s.landfall_lon, 1)}</td>
        </tr>`;
      const head = `<thead><tr><th>Storm</th><th>Year</th><th>Basin</th><th>Category</th>
        <th>Peak wind</th><th>Min pressure</th><th>Landfall</th></tr></thead>`;
      livePane.innerHTML = `
        <p class="muted small">Source: <strong>${st.source}</strong> &middot; ${st.records} storms
          &middot; ${st.year_range} &middot; cache ${st.cache_present ? st.cache_size_mb + " MB" : "missing"}.
          License: ${st.license}. <a href="${st.url}" target="_blank" rel="noreferrer">NOAA IBTrACS</a></p>
        <div class="subhead">Most intense on record</div>
        <table class="tbl">${head}<tbody>${intense.storms.map(row).join("")}</tbody></table>
        <div class="subhead">Most recent seasons</div>
        <table class="tbl">${head}<tbody>${recent.storms.map(row).join("")}</tbody></table>`;
    })
    .catch((e) => { livePane.innerHTML = `<div class="error-box">${e.message}</div>`; });

  /* ----- tab 3: curated reference dataset ----- */
  const refPane = $("#tab-ref");
  apiFetch("/reference-dataset")
    .then(({ dataset, summary }) => {
      const rows = dataset.map((d) => `
        <tr>
          <td>${d.name}</td><td>${d.year}</td><td>${d.basin}</td>
          <td><span class="tag">${d.category}</span></td>
          <td>${Math.round(d.max_wind_knots)} kt</td>
          <td>${Math.round(d.min_pressure_hpa)} hPa</td>
          <td>${Math.round(d.forward_speed_knots)} kt</td>
          <td>${Math.round(d.track_direction_deg)}&deg;</td>
          <td>${d.notes}</td>
        </tr>`).join("");
      refPane.innerHTML = `
        <div class="stat-row">
          <div class="metric"><div class="v">${summary.total_events}</div><div class="l">Events</div></div>
          <div class="metric"><div class="v">${summary.year_range}</div><div class="l">Years</div></div>
          <div class="metric"><div class="v">${summary.strongest_wind_knots} kt</div><div class="l">Strongest</div></div>
          <div class="metric"><div class="v">${summary.lowest_pressure_hpa} hPa</div><div class="l">Lowest pressure</div></div>
        </div>
        <table class="tbl">
          <thead><tr><th>Name</th><th>Year</th><th>Basin</th><th>Category</th><th>Peak wind</th>
          <th>Min pressure</th><th>Fwd speed</th><th>Track dir</th><th>Notes</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
        <p class="muted small">Curated IMD-calibrated events used for intensity calibration and
          analog matching. Live data: tab 2.</p>`;
    })
    .catch((e) => { refPane.innerHTML = `<div class="error-box">${e.message}</div>`; });
}
