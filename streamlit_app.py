"""CYCLO-VISION Streamlit Cloud entrypoint.

This app calls the existing backend services directly, so Streamlit Cloud
doesn't need to start a separate FastAPI or Vite process.
"""

from __future__ import annotations

import base64
import sys
import threading
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
for candidate in (BACKEND, ROOT):
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from app.api.routes.data_sources import datasets_overview  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.services.analysis import analyze_image  # noqa: E402
from app.services.demo_data import get_demo_bytes  # noqa: E402
from app.services.ibtracs import dataset_snapshot, refresh  # noqa: E402


st.set_page_config(
    page_title="CYCLO-VISION | Cyclone Intelligence",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1440px; padding-top: 1.6rem; padding-bottom: 3rem;}
      [data-testid="stSidebar"] {background: #0c1724; border-right: 1px solid #1c3042;}
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {color:#a9bacb;}
      .cv-brand {font-size:.78rem; letter-spacing:.19em; font-weight:800; color:#36d6b2;}
      .cv-hero {padding:1.7rem 1.9rem; border:1px solid #1b3445; border-radius:22px;
        background:radial-gradient(ellipse at 85% 18%,rgba(36,163,154,.18),transparent 36%),
        linear-gradient(125deg,#102131 0%,#0c1724 58%,#0a1f29 100%); margin:.5rem 0 1.35rem;}
      .cv-hero h1 {font-size:clamp(2rem,4vw,3.2rem); line-height:1.04; margin:.65rem 0 .75rem;
        letter-spacing:-.045em; color:#f2f7fb;}
      .cv-hero p {max-width:800px; color:#b7c8d6; font-size:1.03rem; margin:0; line-height:1.65;}
      .cv-eyebrow {color:#53dfbd; font-size:.72rem; letter-spacing:.17em; font-weight:800;}
      .cv-chip {display:inline-block; border:1px solid #286c68; border-radius:999px; padding:.28rem .65rem;
        color:#7be3c8; background:#102b31; font-size:.72rem; font-weight:700; letter-spacing:.05em;}
      .cv-section {font-size:1.22rem; font-weight:750; color:#edf4fa; margin:.25rem 0 .2rem;}
      .cv-muted {color:#9cb0c1; font-size:.9rem;}
      [data-testid="stMetric"] {background:linear-gradient(145deg,#101f2d,#0d1925); border:1px solid #1c3445;
        border-radius:16px; padding:1rem 1.1rem;}
      [data-testid="stMetricLabel"] {color:#9bb0c1;}
      [data-testid="stMetricValue"] {color:#ecf5fb;}
      div.stButton > button[kind="primary"] {border:0; border-radius:11px; font-weight:750; min-height:2.8rem;}
      div.stButton > button {border-radius:11px;}
      [data-testid="stDataFrame"] {border:1px solid #1a3040; border-radius:14px; overflow:hidden;}
      div[data-testid="stExpander"] {border:1px solid #1d3445; border-radius:14px;}
      hr {border-color:#1b2e3e;}
      footer {visibility:hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=900, show_spinner=False)
def load_overview() -> dict[str, Any]:
    return datasets_overview(recent_limit=12, intense_limit=12)


@st.cache_resource(show_spinner=False)
def start_archive_refresh() -> str:
    """Refresh NOAA data in the background when only bundled records exist."""
    try:
        snapshot = dataset_snapshot()
        if snapshot.state in {"cold", "degraded"}:
            threading.Thread(target=refresh, daemon=True, name="ibtracs-refresh").start()
        return snapshot.state
    except Exception:
        return "degraded"


def header() -> None:
    st.markdown(
        """
        <div class="cv-hero">
          <div class="cv-brand">CYCLO-VISION <span style="color:#71899b">/</span> TROPICAL CYCLONE INTELLIGENCE</div>
          <h1>See the storm.<br>Understand what comes next.</h1>
          <p>Explore observed cyclone records, inspect satellite imagery, and create transparent prototype
          estimates for intensity, risk, and forecast track — in one focused workspace.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="cv-section">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="cv-muted">{subtitle}</div>', unsafe_allow_html=True)


def records_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    return frame


def render_analysis(result: dict[str, Any], image_bytes: bytes | None = None) -> None:
    st.divider()
    section_title("Analysis brief", "Prototype estimates depend on the selected image and are not operational warnings.")
    confidence = float(result.get("confidence", 0) or 0)
    confidence_pct = confidence * 100 if confidence <= 1 else confidence
    cols = st.columns(4)
    cols[0].metric("Classification", result.get("classification", "Unknown"))
    cols[1].metric("Estimated wind", f"{result.get('estimated_wind_speed_knots', '—')} kt")
    cols[2].metric("Estimated pressure", f"{result.get('estimated_pressure_hpa', '—')} hPa")
    cols[3].metric("Confidence", f"{confidence_pct:.0f}%")

    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.markdown(f"**Risk level:** `{result.get('risk_level', 'Not available')}`")
        st.markdown(f"**Inference mode:** `{result.get('inference_mode', 'demo')}`")
        if result.get("calibration_source"):
            st.markdown(f"**Calibration:** {result['calibration_source']}")
        factors = result.get("risk_factors") or []
        if factors:
            st.markdown("**Risk factors**")
            st.dataframe(pd.DataFrame(factors), hide_index=True, width="stretch")
        track = result.get("track") or []
        if track:
            st.markdown("**Forecast track**")
            track_df = pd.DataFrame(track)
            st.dataframe(track_df, hide_index=True, width="stretch")
            if {"lat", "lon"}.issubset(track_df.columns):
                st.map(track_df.rename(columns={"lat": "latitude", "lon": "longitude"})[["latitude", "longitude"]])
    with right:
        if image_bytes:
            st.image(image_bytes, caption="Analyzed satellite image", width="stretch")
        heat = (result.get("explainability") or {}).get("heatmap_png_b64")
        if heat:
            try:
                st.image(base64.b64decode(heat), caption="Prototype explainability heatmap", width="stretch")
            except Exception:
                st.caption("Heatmap preview is unavailable for this result.")
        factors = result.get("risk_factors") or []
        evacuation = result.get("evacuation")
        if evacuation:
            with st.expander("Evacuation-zone estimate"):
                st.write(evacuation)


def analyze_bytes(image_bytes: bytes, filename: str, source: str) -> dict[str, Any]:
    result = analyze_image(image_bytes, image_name=filename, source=source)
    return result


def page_overview(data: dict[str, Any]) -> None:
    ibtracs = data.get("ibtracs") or {}
    archive = ibtracs.get("status") or {}
    summary = ibtracs.get("summary") or {}
    reference = data.get("reference") or {}
    reference_summary = reference.get("summary") or {}
    samples = data.get("samples") or []
    sources = data.get("sources") or []

    st.markdown('<span class="cv-chip">● DATA PIPELINE CONNECTED</span>', unsafe_allow_html=True)
    st.write("")
    metrics = st.columns(4)
    metrics[0].metric("Observed storms", f"{summary.get('records', archive.get('records', 0)):,}")
    metrics[1].metric("Reference events", reference_summary.get("total_events", len(reference.get("dataset", []))))
    metrics[2].metric("Satellite sources", len(sources))
    metrics[3].metric("Demo scenes", len(samples))

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        section_title("Latest observed storms", "Recent records from the IBTrACS archive.")
        recent = ibtracs.get("recent") or []
        if recent:
            st.dataframe(pd.DataFrame(recent), hide_index=True, width="stretch", height=360)
        else:
            st.info("Storm records are warming up. The bundled reference set remains available.")
    with right:
        section_title("Archive status", "NOAA best-track source and local cache state.")
        state = str(archive.get("state", "unknown")).upper()
        st.markdown(f"### {state}")
        st.write(f"**Source:** {archive.get('source', 'Bundled reference data')}")
        st.write(f"**Record span:** {archive.get('year_range', summary.get('year_range', '—'))}")
        st.write(f"**Strongest wind:** {archive.get('strongest_wind_knots', summary.get('strongest_wind_knots', '—'))} kt")
        st.caption("Research prototype only. Estimates are not suitable for issuing weather warnings.")

    st.write("")
    section_title("Start with a scene", "Run a local prototype analysis on one of the bundled satellite samples.")
    sample_cols = st.columns(min(4, max(1, len(samples))))
    for index, sample in enumerate(samples[:4]):
        with sample_cols[index % len(sample_cols)]:
            st.markdown(f"**{sample.get('name', 'Demo scene')}**")
            st.caption(sample.get("description", "Bundled satellite sample"))
            if st.button("Analyze scene", key=f"overview_analyze_{sample['id']}", type="primary", width="stretch"):
                with st.spinner("Analyzing the selected satellite scene…"):
                    image_bytes, filename, _meta = get_demo_bytes(sample["id"])
                    st.session_state["cv_analysis"] = analyze_bytes(image_bytes, filename, "demo")
                    st.session_state["cv_image"] = image_bytes
    if st.session_state.get("cv_analysis"):
        render_analysis(st.session_state["cv_analysis"], st.session_state.get("cv_image"))


def page_analyze(data: dict[str, Any]) -> None:
    samples = data.get("samples") or []
    section_title("Satellite image analysis", "Choose a bundled scene or upload an image to create an analysis brief.")
    sample_col, upload_col = st.columns(2, gap="large")
    with sample_col:
        st.markdown("#### Bundled scenes")
        if samples:
            labels = {f"{s['name']} · {s['category']}": s["id"] for s in samples}
            selected_label = st.selectbox("Select a sample", list(labels), label_visibility="collapsed")
            if st.button("Analyze selected sample", type="primary", width="stretch"):
                with st.spinner("Analyzing scene and building the forecast brief…"):
                    image_bytes, filename, _meta = get_demo_bytes(labels[selected_label])
                    st.session_state["cv_analysis"] = analyze_bytes(image_bytes, filename, "demo")
                    st.session_state["cv_image"] = image_bytes
        else:
            st.info("No bundled satellite scenes were found.")
    with upload_col:
        st.markdown("#### Your image")
        upload = st.file_uploader("JPG, PNG, or TIFF · up to 10 MB", type=["jpg", "jpeg", "png", "tif", "tiff"])
        if upload and st.button("Analyze uploaded image", type="primary", width="stretch"):
            image_bytes = upload.getvalue()
            if len(image_bytes) > 10 * 1024 * 1024:
                st.error("Please upload an image smaller than 10 MB.")
            else:
                with st.spinner("Analyzing image and building the forecast brief…"):
                    st.session_state["cv_analysis"] = analyze_bytes(image_bytes, upload.name, "upload")
                    st.session_state["cv_image"] = image_bytes
    if st.session_state.get("cv_analysis"):
        render_analysis(st.session_state["cv_analysis"], st.session_state.get("cv_image"))
    else:
        st.info("Choose a demo scene or upload a satellite image to see classification, intensity estimates, risk factors, explainability, and forecast track here.")


def page_live(data: dict[str, Any]) -> None:
    ibtracs = data.get("ibtracs") or {}
    dataset = ibtracs.get("dataset") or {}
    storms = dataset.get("storms") or []
    summary = dataset.get("summary") or ibtracs.get("summary") or {}
    section_title("Live IBTrACS archive", "Explore observed tropical cyclone records and filter the complete local archive.")
    m = st.columns(4)
    m[0].metric("Records", f"{summary.get('records', len(storms)):,}")
    m[1].metric("Year range", summary.get("year_range", "—"))
    m[2].metric("Named storms", f"{summary.get('named_records', 0):,}")
    m[3].metric("Mean peak wind", f"{summary.get('mean_peak_wind_knots', '—')} kt")
    frame = records_frame(storms)
    if frame.empty:
        st.info("No observed records are available yet.")
        return
    filters = st.columns([1.4, 1, 1, 1])
    with filters[0]:
        search = st.text_input("Search records", placeholder="Storm name, SID, basin…")
    with filters[1]:
        basins = ["All basins"] + sorted(str(x) for x in frame.get("basin", pd.Series(dtype=str)).dropna().unique())
        basin = st.selectbox("Basin", basins)
    with filters[2]:
        years = pd.to_numeric(frame.get("year", pd.Series(dtype=float)), errors="coerce").dropna()
        year_floor, year_ceiling = (int(years.min()), int(years.max())) if not years.empty else (1900, 2026)
        year_range = st.slider("Season range", year_floor, year_ceiling, (year_floor, year_ceiling))
    with filters[3]:
        min_wind = st.selectbox("Minimum wind", [0, 34, 48, 64, 90, 120], format_func=lambda x: "Any" if x == 0 else f"{x}+ kt")
    filtered = frame.copy()
    if search:
        text_cols = [c for c in ["name", "sid", "basin", "category", "year"] if c in filtered.columns]
        filtered = filtered[filtered[text_cols].astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)]
    if basin != "All basins" and "basin" in filtered:
        filtered = filtered[filtered["basin"].astype(str) == basin]
    if "year" in filtered:
        year_values = pd.to_numeric(filtered["year"], errors="coerce")
        filtered = filtered[year_values.between(*year_range)]
    if "max_wind_knots" in filtered and min_wind:
        filtered = filtered[pd.to_numeric(filtered["max_wind_knots"], errors="coerce") >= min_wind]
    st.caption(f"Showing {len(filtered):,} of {len(frame):,} records")
    st.dataframe(filtered, hide_index=True, width="stretch", height=520)


def page_reference(data: dict[str, Any]) -> None:
    reference = data.get("reference") or {}
    events = reference.get("dataset") or []
    summary = reference.get("summary") or {}
    section_title("Reference library", "Curated historical cyclones used to calibrate prototype estimates.")
    cols = st.columns(4)
    cols[0].metric("Events", summary.get("total_events", len(events)))
    cols[1].metric("Year range", summary.get("year_range", "—"))
    cols[2].metric("Strongest wind", f"{summary.get('strongest_wind_knots', '—')} kt")
    cols[3].metric("Lowest pressure", f"{summary.get('lowest_pressure_hpa', '—')} hPa")
    frame = records_frame(events)
    if not frame.empty:
        st.dataframe(frame, hide_index=True, width="stretch", height=480)
        with st.expander("How to read these records"):
            st.write("These curated events provide reference context for the prototype calibration. They are not a live warning feed.")


def page_sources(data: dict[str, Any]) -> None:
    sources = data.get("sources") or []
    section_title("Data sources", "Configured ingestion catalog for satellite and cyclone datasets.")
    if not sources:
        st.info("The source catalog is unavailable.")
        return
    cols = st.columns(2)
    for index, source in enumerate(sources):
        with cols[index % 2]:
            with st.container(border=True):
                st.markdown(f"#### {source.get('name', source.get('id', 'Source'))}")
                st.caption(f"{source.get('satellite_type', 'Data source')} · {source.get('region', 'Global')}")
                st.write(source.get("description", ""))
                st.markdown(f"**Status:** {source.get('status', 'Unknown')}  ·  **Availability:** {source.get('availability', 'Unknown')}")


def main() -> None:
    start_archive_refresh()
    st.sidebar.markdown('<div class="cv-brand">CYCLO-VISION</div>', unsafe_allow_html=True)
    st.sidebar.caption("TROPICAL CYCLONE INTELLIGENCE")
    page = st.sidebar.radio(
        "Workspace",
        ["Overview", "Analyze imagery", "Live IBTrACS", "Reference library", "Data sources"],
        label_visibility="collapsed",
        key="cv_page",
    )
    st.sidebar.divider()
    st.sidebar.markdown("**RESEARCH PROTOTYPE**")
    st.sidebar.caption("Estimates are illustrative and must not be used as operational weather warnings.")

    header()
    with st.spinner("Loading cyclone records and source catalog…"):
        data = load_overview()
    if data.get("errors"):
        st.warning("Some data sources are temporarily unavailable. Available panels remain usable.")
    pages = {
        "Overview": page_overview,
        "Analyze imagery": page_analyze,
        "Live IBTrACS": page_live,
        "Reference library": page_reference,
        "Data sources": page_sources,
    }
    pages[page](data)
    st.divider()
    st.caption("CYCLO-VISION · Prototype intelligence workspace · Not a public warning service")


if __name__ == "__main__":
    main()
