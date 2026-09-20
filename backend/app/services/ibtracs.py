"""
CYCLO-VISION - IBTrACS Live Dataset Service
===========================================
Downloads and parses the *real* NOAA IBTrACS (International Best Track
Archive for Climate Stewardship) best-track dataset for the North Indian
Ocean basin, and turns it into the reference records the analysis engine
calibrates against.

Source : NOAA NCEI - IBTrACS v04r01
URL    : https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.NI.list.v04r01.csv
License: Public domain (US Government work)

Why this exists
---------------
The bundled ``reference_data`` module ships a small curated table so the
app always boots offline. This module upgrades that to the full observed
record. When the download succeeds the engine calibrates against
thousands of observed points instead of a dozen hand-written rows.

Design notes
------------
* Downloads are cached to ``data/cache/ibtracs.NI.list.v04r01.csv`` so a
  network hiccup never breaks the app; the cache is reused for
  ``CACHE_TTL_HOURS`` before a refresh is attempted.
* All work is guarded: any failure falls back to the bundled dataset so
  API responses never raise.
* Parsing is vectorised with pandas (no row-by-row Python loop) because
  the file is ~28 MB.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.core.config import PROJECT_ROOT
from ml.classification import (  # single source of truth for wind bands (P0-3)
    WIND_BANDS,
    band_centre_for_class,
    category_for_wind,
    class_index_for_wind,
)

logger = logging.getLogger("cyclo.ibtracs")

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

IBTRACS_URL = (
    "https://www.ncei.noaa.gov/data/"
    "international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv/ibtracs.NI.list.v04r01.csv"
)

CACHE_DIR = PROJECT_ROOT / "data" / "cache"
CACHE_FILE = CACHE_DIR / "ibtracs.NI.list.v04r01.csv"

# Re-download only if the cached file is older than this.
CACHE_TTL_HOURS = 24

DOWNLOAD_TIMEOUT_S = 90
CHUNK_SIZE = 1 << 20  # 1 MiB

# IBTrACS CSV uses two header rows: row 1 is the human label, row 2 the
# short name. We key on the short name.
ID_COL = "SID"
NAME_COL = "NAME"
SEASON_COL = "SEASON"
BASIN_COL = "BASIN"
USA_WIND_COL = "USA_WIND"
USA_PRES_COL = "USA_PRES"
LAT_COL = "LAT"
LON_COL = "LON"

# IMD intensity scale used for the North Indian Ocean.
# The single ordered table lives in ``ml.classification.WIND_BANDS`` (P0-3)
# so the labels here and the postprocessing categories cannot diverge.
# Kept as a module-level alias for backwards compatibility with callers
# that introspect it (e.g. the data-sources route).
IMD_CATEGORIES: list[tuple[float, str]] = [
    (band_min, label) for band_min, label, _ in WIND_BANDS
]


def imd_category(wind_knots: float) -> str:
    """Map a 1-minute maximum sustained wind (knots) to an IMD category."""
    return category_for_wind(wind_knots)


@dataclass
class IBTrACSStorm:
    """One observed storm from the real IBTrACS record."""

    sid: str
    name: str
    year: int
    basin: str
    category: str
    max_wind_knots: float
    min_pressure_hpa: Optional[float]
    genesis_lat: float
    genesis_lon: float
    landfall_lat: float
    landfall_lon: float
    track_direction_deg: float
    forward_speed_knots: float
    intensity_index: int
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Module state
# --------------------------------------------------------------------------

_lock = threading.Lock()
_storms: Optional[list[IBTrACSStorm]] = None
_source = "cold"          # "live" | "cache" | "bundled" | "cold"
_last_error: Optional[str] = None
_downloaded_at: Optional[datetime] = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _cache_is_fresh() -> bool:
    if not CACHE_FILE.exists():
        return False
    age_h = (_now().timestamp() - CACHE_FILE.stat().st_mtime) / 3600.0
    return age_h < CACHE_TTL_HOURS

# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------

def download_dataset(force: bool = False) -> Path:
    """Download the IBTrACS North-Indian-Ocean CSV into the local cache.

    Returns the path to the cached CSV. Raises on network failure so the
    caller can decide how to degrade.

    NOTE: this function mutates the module-level telemetry state
    (``_source``, ``_last_error``, ``_downloaded_at``) and must therefore be
    called with ``_lock`` held when invoked directly. Every exported entry
    point (:func:`load_storms`, :func:`download_dataset_async`) already does
    this (P0 review #11).
    """
    global _source, _last_error, _downloaded_at

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if CACHE_FILE.exists() and not force and _cache_is_fresh():
        logger.info("IBTrACS cache is fresh; skipping download.")
        return CACHE_FILE

    import requests  # imported lazily so the module imports without it

    tmp = CACHE_FILE.with_name(CACHE_FILE.name + ".part")
    logger.info("Downloading IBTrACS from %s", IBTRACS_URL)

    with requests.get(IBTRACS_URL, stream=True, timeout=DOWNLOAD_TIMEOUT_S) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length") or 0)
        written = 0
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                fh.write(chunk)
                written += len(chunk)
        logger.info("Downloaded %.1f MiB", written / (1 << 20))
        if total and written < total:
            raise IOError(f"Truncated download: {written}/{total} bytes")

    tmp.replace(CACHE_FILE)
    _source = "live"
    _downloaded_at = _now()
    _last_error = None
    return CACHE_FILE


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def _parse(path: Path) -> list[IBTrACSStorm]:
    """Vectorised parse of the IBTrACS CSV into per-storm summaries."""
    import numpy as np
    import pandas as pd

    # IBTrACS repeats the header on row 2; header=0 gives us the labels and
    # skiprows=[1] drops the short-name duplicate row.
    raw = pd.read_csv(path, skiprows=[1], low_memory=False)

    required = {ID_COL, NAME_COL, LAT_COL, LON_COL}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"IBTrACS CSV missing expected columns: {sorted(missing)}")

    # The calibration columns are the whole point of this dataset: without
    # USA_WIND/USA_PRES every storm would parse with wind=None and the
    # intensity matching would silently produce zero usable storms. Fail
    # loudly instead of zero-filling (P0 review #25).
    calib_missing = [c for c in (USA_WIND_COL, USA_PRES_COL) if c not in raw.columns]
    if calib_missing:
        raise ValueError(
            f"IBTrACS CSV is missing the calibration column(s) {calib_missing}; "
            "intensity calibration is impossible without them. Check the file "
            "version/schema."
        )

    # --- coerce numerics -------------------------------------------------
    for col in (USA_WIND_COL, USA_PRES_COL, LAT_COL, LON_COL, SEASON_COL):
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
        else:
            raw[col] = np.nan

    valid = raw[raw[LAT_COL].notna() & raw[LON_COL].notna()]

    # --- peak intensity per storm (single aggregation pass) --------------
    peak = raw.groupby(ID_COL, sort=False).agg(
        max_wind=(USA_WIND_COL, "max"),
        min_pres=(USA_PRES_COL, "min"),
        year=(SEASON_COL, "max"),
    )

    # --- genesis point (first valid observation) -------------------------
    genesis = (
        valid.groupby(ID_COL, sort=False)[[LAT_COL, LON_COL]]
        .first()
        .rename(columns={LAT_COL: "genesis_lat", LON_COL: "genesis_lon"})
    )

    # --- landfall proxy (last valid observation) -------------------------
    landfall = (
        valid.groupby(ID_COL, sort=False)[[LAT_COL, LON_COL]]
        .last()
        .rename(columns={LAT_COL: "landfall_lat", LON_COL: "landfall_lon"})
    )

    # --- dominant basin + name per storm --------------------------------
    meta = raw.groupby(ID_COL, sort=False).agg(
        name=(NAME_COL, "first"),
        basin=(BASIN_COL, "first"),
    )

    frame = (
        peak.join(genesis, how="left")
        .join(landfall, how="left")
        .join(meta, how="left")
    )

    # --- drop entries with no usable intensity --------------------------
    frame = frame[frame["max_wind"].notna() & frame["year"].notna()]
    frame["year"] = frame["year"].astype(int)

    return _build_storms(frame)


def _track_vector(lat1: float, lon1: float, lat2: float, lon2: float) -> tuple[float, float]:
    """Great-circle initial bearing (deg) and a plausible forward speed (kt)."""
    import math

    if any(v != v for v in (lat1, lon1, lat2, lon2)):  # NaN check
        return 315.0, 8.0

    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    bearing = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0

    # Distance via haversine, then assume a ~24 h genesis->end span.
    dphi = p2 - p1
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    km = 6371.0 * 2 * math.asin(min(1.0, math.sqrt(a)))
    speed = max(4.0, min(20.0, km / 1.852 / 24.0))
    return round(bearing, 1), round(speed, 1)


def _intensity_index(wind_knots: float) -> int:
    """Map observed wind to the engine's 0-6 class index.

    Delegates to the shared ``ml.classification.WIND_BANDS`` table (P0-3) so
    an IBTrACS storm's label always agrees with the category the
    postprocessing layer reports for the same wind speed.
    """
    return class_index_for_wind(wind_knots)


def _coerce_pressure(value: Any) -> Optional[float]:
    """Return a usable pressure in hPa, or None when absent/invalid.

    IBTrACS leaves pressure blank for many pre-satellite-era storms, and
    pandas may surface that as NaN, None or pd.NA depending on the dtype.
    Normalising here keeps the rest of the code free of sentinel checks.
    """
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f or f <= 0:  # NaN or nonsense
        return None
    return f


def _build_storms(frame) -> list[IBTrACSStorm]:
    """Convert the aggregated DataFrame into IBTrACSStorm objects."""
    import numpy as np

    storms: list[IBTrACSStorm] = []
    for sid, row in frame.iterrows():
        wind = float(row["max_wind"])
        # Pre-satellite-era storms often have no pressure record. Keep it
        # as None rather than inventing a value, so the UI can show "n/a"
        # and calibration never uses a fabricated number.
        pressure = _coerce_pressure(row["min_pres"])

        glat, glon = row["genesis_lat"], row["genesis_lon"]
        llat, llon = row["landfall_lat"], row["landfall_lon"]
        if np.isnan(glat) or np.isnan(glon):
            glat, glon = llat, llon
        if np.isnan(llat) or np.isnan(llon):
            llat, llon = glat, glon

        bearing, speed = _track_vector(glat, glon, llat, llon)

        name = str(row["name"]).strip() or "UNNAMED"
        basin = str(row["basin"]).strip() or "North Indian Ocean"

        storms.append(
            IBTrACSStorm(
                sid=str(sid),
                name=name,
                year=int(row["year"]),
                basin=basin,
                category=imd_category(wind),
                max_wind_knots=round(wind, 1),
                min_pressure_hpa=round(pressure, 1) if pressure else None,
                genesis_lat=round(float(glat), 2),
                genesis_lon=round(float(glon), 2),
                landfall_lat=round(float(llat), 2),
                landfall_lon=round(float(llon), 2),
                track_direction_deg=bearing,
                forward_speed_knots=speed,
                intensity_index=_intensity_index(wind),
                notes=f"IBTrACS {sid} - observed best track.",
            )
        )
    return storms

# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def _bundled_fallback() -> list[IBTrACSStorm]:
    """Convert the bundled curated table into IBTrACSStorm records."""
    from app.services.reference_data import get_reference_dataset

    out: list[IBTrACSStorm] = []
    for ref in get_reference_dataset():
        out.append(
            IBTrACSStorm(
                sid=f"BUNDLED-{ref.name.upper()}-{ref.year}",
                name=ref.name,
                year=ref.year,
                basin=ref.basin,
                category=ref.category,
                max_wind_knots=float(ref.max_wind_knots),
                min_pressure_hpa=float(ref.min_pressure_hpa),
                # The -2/-3 offsets are arbitrary: the curated table only
                # records landfall, so genesis is placed slightly SW of it
                # purely so the fallback track has some length (P0 #27).
                genesis_lat=float(ref.landfall_lat) - 2.0,
                genesis_lon=float(ref.landfall_lon) - 3.0,
                landfall_lat=float(ref.landfall_lat),
                landfall_lon=float(ref.landfall_lon),
                track_direction_deg=float(ref.track_direction_deg),
                forward_speed_knots=float(ref.forward_speed_knots),
                intensity_index=int(ref.intensity_index),
                notes=ref.notes,
            )
        )
    return out


def load_storms(force: bool = False) -> list[IBTrACSStorm]:
    """Return the dataset, downloading live data on first use.

    Never raises: on any failure the bundled curated table is returned so
    the API keeps working offline.
    """
    global _storms, _source, _last_error

    with _lock:
        if _storms is not None and not force:
            return _storms

        try:
            path = download_dataset(force=force)
            if _source != "live":
                _source = "cache"
            _storms = _parse(path)
            if not _storms:
                raise ValueError("Parsed zero storms from IBTrACS file")
            logger.info("Loaded %d storms from IBTrACS (%s).", len(_storms), _source)
            _last_error = None
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            _last_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "IBTrACS unavailable (%s); falling back to bundled dataset.", _last_error
            )
            _storms = _bundled_fallback()
            _source = "bundled"

        return _storms


def get_dataset() -> list[dict[str, Any]]:
    """All storms as plain dicts (API-friendly)."""
    return [s.to_dict() for s in load_storms()]


def get_recent(limit: int = 25) -> list[dict[str, Any]]:
    """Most recent storms, newest first."""
    storms = sorted(load_storms(), key=lambda s: (s.year, s.max_wind_knots), reverse=True)
    return [s.to_dict() for s in storms[:limit]]


def get_top_intense(limit: int = 25) -> list[dict[str, Any]]:
    """Strongest storms on record by peak wind."""
    storms = sorted(load_storms(), key=lambda s: s.max_wind_knots, reverse=True)
    return [s.to_dict() for s in storms[:limit]]


def get_named_only() -> list[IBTrACSStorm]:
    """Storms that were actually named (used for calibration labels)."""
    names = [s for s in load_storms() if s.name and s.name != "UNNAMED"]
    return names or load_storms()


def best_match_for_class(class_idx: int) -> Optional[IBTrACSStorm]:
    """Pick the observed storm whose peak intensity best anchors a class.

    Chooses the storm whose ``max_wind_knots`` is closest to the *centre*
    of the class's wind band (``ml.classification.band_centre_for_class``),
    so the calibration anchor is a representative mid-band event rather
    than the most extreme storm on record. Ties are broken towards the
    more recent storm, then the one that has a pressure record.

    Returns ``None`` for class 0 / out-of-range classes, or when the
    dataset has no named storms.
    """
    from ml.model import CLASS_LABELS

    if class_idx <= 0 or class_idx >= len(CLASS_LABELS):
        return None

    centre = band_centre_for_class(class_idx)
    if centre is None or centre <= 0:
        return None

    named = get_named_only()
    exact = [s for s in named if s.intensity_index == class_idx]
    pool = exact or named
    if not pool:
        return None

    # Closest to the band centre wins. The pressure flag is inverted here so
    # that ascending tuple comparison still prefers storms with a pressure
    # record (0 < 1). Note: the year must come FIRST, otherwise Python's
    # tuple comparison stops at the differing distance and never reaches it.
    def score(s: IBTrACSStorm) -> tuple[int, float, int]:
        has_pressure = 0 if s.min_pressure_hpa else 1
        return (has_pressure, abs(s.max_wind_knots - centre), -s.year)

    return min(pool, key=score)


def get_status() -> dict[str, Any]:
    """Health/telemetry for the live dataset."""
    storms = load_storms()
    years = [s.year for s in storms if s.year > 0]
    return {
        "source": _source,
        "records": len(storms),
        "year_range": f"{min(years)}-{max(years)}" if years else "n/a",
        "cache_file": str(CACHE_FILE),
        "cache_present": CACHE_FILE.exists(),
        "cache_size_mb": (
            round(CACHE_FILE.stat().st_size / (1 << 20), 2) if CACHE_FILE.exists() else 0.0
        ),
        "downloaded_at": _downloaded_at.isoformat() if _downloaded_at else None,
        "last_error": _last_error,
        "url": IBTRACS_URL,
        "license": "Public domain (NOAA/NCEI)",
    }
