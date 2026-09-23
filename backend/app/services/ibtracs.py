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

import json
import logging
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.core.config import PROJECT_ROOT
from app.core.net import ca_bundle  # fixes stale CURL_CA_BUNDLE on Windows
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

# A single dropped connection to NOAA must not degrade the app to the tiny
# bundled table, so a failed attempt is retried a few times with a short
# backoff before ``load_storms`` gives up. Kept small so a *cold* start on a
# blocked network still serves a response quickly.
DOWNLOAD_ATTEMPTS = 3
RETRY_BACKOFF_S = 2.0

# The real NI list is ~27 MB; anything smaller is an error page or a partial
# transfer and must never be promoted to the cache.
MIN_VALID_BYTES = 100_000

# The parsed record is mirrored to a small JSON sidecar so a restart can serve
# the full observed record in milliseconds instead of re-parsing 28 MB. The
# version guards against a schema change being read by an older build.
SIDECAR_VERSION = 1

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
# Dataset documentation (exposed to the UI so every field can be explained)
# --------------------------------------------------------------------------

# Every attribute of ``IBTrACSStorm`` with the column it comes from. The
# dashboard renders this table verbatim, so a reader can see exactly what the
# dataset contains instead of guessing from the JSON keys.
DATASET_ATTRIBUTES: list[dict[str, str]] = [
    {
        "field": "sid",
        "label": "Storm ID (SID)",
        "source": "IBTrACS SID",
        "description": "Unique storm identifier: basin prefix, season and sequence number.",
    },
    {
        "field": "name",
        "label": "Storm name",
        "source": "IBTrACS NAME",
        "description": "Official name, or UNNAMED for systems from before the naming era.",
    },
    {
        "field": "year",
        "label": "Season (year)",
        "source": "IBTrACS SEASON",
        "description": "Season the storm belongs to; the archive spans 1842 to the present.",
    },
    {
        "field": "basin",
        "label": "Basin",
        "source": "IBTrACS BASIN",
        "description": "Reporting basin for the track (North Indian Ocean sub-basins).",
    },
    {
        "field": "category",
        "label": "IMD category",
        "source": "derived from USA_WIND",
        "description": "IMD intensity band of the peak 1-minute sustained wind.",
    },
    {
        "field": "max_wind_knots",
        "label": "Peak wind",
        "source": "IBTrACS USA_WIND (max)",
        "description": "Peak 1-minute sustained wind at 10 m, in knots.",
    },
    {
        "field": "min_pressure_hpa",
        "label": "Minimum pressure",
        "source": "IBTrACS USA_PRES (min)",
        "description": "Lowest central pressure in hPa; null where the archive has no record.",
    },
    {
        "field": "genesis_lat",
        "label": "Genesis latitude",
        "source": "IBTrACS LAT (first fix)",
        "description": "Latitude of the first valid track point, in degrees north.",
    },
    {
        "field": "genesis_lon",
        "label": "Genesis longitude",
        "source": "IBTrACS LON (first fix)",
        "description": "Longitude of the first valid track point, in degrees east.",
    },
    {
        "field": "landfall_lat",
        "label": "Final latitude",
        "source": "IBTrACS LAT (last fix)",
        "description": "Latitude of the last valid track point (landfall/dissipation proxy).",
    },
    {
        "field": "landfall_lon",
        "label": "Final longitude",
        "source": "IBTrACS LON (last fix)",
        "description": "Longitude of the last valid track point (landfall/dissipation proxy).",
    },
    {
        "field": "track_direction_deg",
        "label": "Track bearing",
        "source": "derived (great circle)",
        "description": "Initial great-circle bearing from genesis to the final fix, in degrees.",
    },
    {
        "field": "forward_speed_knots",
        "label": "Forward speed",
        "source": "derived (great circle)",
        "description": "Typical translation speed implied by the track length, in knots.",
    },
    {
        "field": "intensity_index",
        "label": "Intensity index",
        "source": "derived from USA_WIND",
        "description": "Class index 0-6 used by the analysis engine's calibration step.",
    },
    {
        "field": "notes",
        "label": "Notes",
        "source": "generated",
        "description": "Short provenance note (SID reference) for the record.",
    },
]


def get_attributes() -> list[dict[str, str]]:
    """Field-level documentation for the live dataset payload."""
    return [dict(item) for item in DATASET_ATTRIBUTES]


# --------------------------------------------------------------------------
# Module state
# --------------------------------------------------------------------------

# Two locks, deliberately separate:
#   * ``_state_lock`` guards the in-memory record + telemetry and is only ever
#     held for microseconds,
#   * ``_refresh_lock`` serialises the expensive download/parse pipeline,
#   * ``_parse_lock`` makes sure only one thread parses the 28 MB CSV at a time.
#
# They used to be a single lock held across the whole NOAA download, so every
# dataset request -- /ibtracs/status, /recent, /intense, /api/health and the
# analysis calibration -- blocked for ~20 s (up to ~5 min on a dead network)
# whenever the cache was stale. That is longer than the dashboard's request
# timeout, which is why the UI kept reporting "could not fetch the datasets"
# and asked for a retry while nothing was actually broken.
_state_lock = threading.Lock()
_refresh_lock = threading.Lock()
_parse_lock = threading.Lock()

_storms: Optional[list[IBTrACSStorm]] = None
_source = "cold"          # "live" | "cache" | "bundled" | "cold"
_last_error: Optional[str] = None
_downloaded_at: Optional[datetime] = None
_loading = False          # a download/parse cycle is running right now


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _cache_is_fresh() -> bool:
    if not CACHE_FILE.exists():
        return False
    age_h = (_now().timestamp() - CACHE_FILE.stat().st_mtime) / 3600.0
    return age_h < CACHE_TTL_HOURS


def _cache_age_hours() -> Optional[float]:
    """Age of the local CSV in hours, or ``None`` when there is no cache."""
    try:
        return round((_now().timestamp() - CACHE_FILE.stat().st_mtime) / 3600.0, 2)
    except OSError:
        return None


def _cache_size_mb() -> float:
    """Size of the local CSV in MiB (0.0 when absent)."""
    try:
        return round(CACHE_FILE.stat().st_size / (1 << 20), 2)
    except OSError:
        return 0.0

# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------

def _looks_like_ibtracs(path: Path) -> bool:
    """Cheap sanity check that ``path`` is the IBTrACS CSV, not an error page.

    Guards against a captive-portal/HTML response or an aborted transfer being
    promoted to the cache and poisoning every later cold start.
    """
    try:
        if path.stat().st_size < MIN_VALID_BYTES:
            return False
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            header = fh.readline()
    except OSError:
        return False
    return ID_COL in header and NAME_COL in header


def download_dataset(force: bool = False) -> Path:
    """Download the IBTrACS North-Indian-Ocean CSV into the local cache.

    Returns the path to the cached CSV. Raises on network failure so the
    caller can decide how to degrade.

    NOTE: this function mutates the module-level telemetry state
    (``_source``, ``_last_error``, ``_downloaded_at``) and is always called
    through :func:`refresh`, which holds ``_refresh_lock`` while the download
    runs. It is the *only* place that performs network I/O, and it is never
    executed while a dataset read is in flight, so a slow NOAA response can no
    longer stall the API (P0 review #11).
    """
    global _source, _last_error, _downloaded_at

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if CACHE_FILE.exists() and not force and _cache_is_fresh():
        logger.info("IBTrACS cache is fresh; skipping download.")
        return CACHE_FILE

    import requests  # imported lazily so the module imports without it

    tmp = CACHE_FILE.with_name(CACHE_FILE.name + ".part")
    last_exc: Exception | None = None

    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            logger.info(
                "Downloading IBTrACS (attempt %d/%d) from %s",
                attempt,
                DOWNLOAD_ATTEMPTS,
                IBTRACS_URL,
            )
            with requests.get(
                IBTRACS_URL,
                stream=True,
                timeout=DOWNLOAD_TIMEOUT_S,
                # Explicit verify target: a stale machine-wide
                # CURL_CA_BUNDLE/REQUESTS_CA_BUNDLE makes plain requests.get()
                # raise "Could not find a suitable TLS CA certificate bundle".
                verify=ca_bundle(),
            ) as resp:
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

            if not _looks_like_ibtracs(tmp):
                raise IOError(
                    "Downloaded file is not a usable IBTrACS CSV "
                    f"({tmp.stat().st_size if tmp.exists() else 0} bytes)"
                )

            tmp.replace(CACHE_FILE)
            _source = "live"
            _downloaded_at = _now()
            _last_error = None
            return CACHE_FILE
        except Exception as exc:  # noqa: BLE001 - retried below
            last_exc = exc
            logger.warning(
                "IBTrACS download attempt %d/%d failed: %s: %s",
                attempt,
                DOWNLOAD_ATTEMPTS,
                type(exc).__name__,
                exc,
            )
            # Never leave a partial file behind: it would otherwise be
            # mistaken for a cache by later runs.
            tmp.unlink(missing_ok=True)
            if attempt < DOWNLOAD_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_S * attempt)

    assert last_exc is not None  # loop always sets it on failure
    raise last_exc


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


# --------------------------------------------------------------------------
# Fast, non-blocking dataset access
# --------------------------------------------------------------------------


def _install(storms: list[IBTrACSStorm], source: str) -> None:
    """Publish a resolved dataset under the short-lived state lock."""
    global _storms, _source
    with _state_lock:
        _storms = storms
        _source = source


def _sidecar_file() -> Path:
    """Path of the JSON sidecar that mirrors the parsed CSV.

    Derived from ``CACHE_FILE`` (rather than a module constant) so tests and
    operators can relocate the cache without stranding the sidecar.
    """
    stem = CACHE_FILE.name[:-4] if CACHE_FILE.name.endswith(".csv") else CACHE_FILE.name
    return CACHE_FILE.with_name(f"{stem}.storms.json")


def _csv_fingerprint() -> dict[str, Any]:
    """Identity of the CSV a sidecar was derived from (size + mtime)."""
    try:
        stat = CACHE_FILE.stat()
    except OSError:
        return {}
    return {"csv_size": stat.st_size, "csv_mtime": round(stat.st_mtime, 3)}


def _storm_rows(storms: list[IBTrACSStorm]) -> list[dict[str, Any]]:
    """Serialisable rows for the sidecar (tolerates test doubles)."""
    rows: list[dict[str, Any]] = []
    for storm in storms:
        if isinstance(storm, IBTrACSStorm):
            rows.append(asdict(storm))
        elif hasattr(storm, "to_dict"):
            rows.append(storm.to_dict())
    return rows


def write_sidecar(storms: list[IBTrACSStorm]) -> None:
    """Persist the parsed record next to the CSV (best effort).

    Parsing the 28 MB archive costs ~2 s; replaying this small JSON on the
    next start makes the dataset readable in milliseconds instead. Written
    atomically, so a crash can never leave a half-written sidecar behind.
    """
    rows = _storm_rows(storms)
    if not rows:
        return
    path = _sidecar_file()
    tmp = path.with_name(path.name + ".part")
    payload = {
        "version": SIDECAR_VERSION,
        "generated_at": _now().isoformat(),
        "source_url": IBTRACS_URL,
        "records": len(rows),
        "attributes": [attr["field"] for attr in DATASET_ATTRIBUTES],
        **_csv_fingerprint(),
        "storms": rows,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:  # read-only data dir, disk full, ...
        logger.debug("Could not write IBTrACS sidecar: %s", exc)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _read_sidecar() -> Optional[list[IBTrACSStorm]]:
    """Load the sidecar when it still matches the CSV on disk."""
    path = _sidecar_file()
    if not path.exists():
        return None
    fingerprint = _csv_fingerprint()
    if not fingerprint:
        return None  # no CSV to authenticate the sidecar against
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != SIDECAR_VERSION:
            return None
        if any(payload.get(key) != value for key, value in fingerprint.items()):
            return None  # CSV was replaced/re-downloaded -> re-parse it
        fields = set(IBTrACSStorm.__dataclass_fields__)
        storms = [
            IBTrACSStorm(**{k: v for k, v in row.items() if k in fields})
            for row in payload.get("storms", [])
            if isinstance(row, dict)
        ]
    except Exception as exc:  # noqa: BLE001 - a bad sidecar is not fatal
        logger.debug("Ignoring unreadable IBTrACS sidecar: %s", exc)
        return None
    return storms or None


def _parse_local_cache() -> Optional[list[IBTrACSStorm]]:
    """Sidecar first, then the downloaded CSV. Never touches the network."""
    with _parse_lock:
        with _state_lock:
            if _storms:
                return _storms

        storms = _read_sidecar()
        if storms is None and CACHE_FILE.exists():
            try:
                storms = _parse(CACHE_FILE)
            except Exception as exc:  # noqa: BLE001 - corrupt cache is not fatal
                logger.warning(
                    "IBTrACS cache unreadable (%s: %s).", type(exc).__name__, exc
                )
                storms = None
            if storms:
                write_sidecar(storms)

        return storms or None


@dataclass(frozen=True)
class DatasetSnapshot:
    """Result of a non-blocking dataset lookup.

    ``state`` is:

    * ``"ready"``    -- the real observed record (live download or local copy),
    * ``"warming"``  -- a refresh is running, the payload is provisional,
    * ``"degraded"`` -- no local copy and the last attempt failed, so the
      bundled curated table is served until the network recovers.
    """

    storms: list[IBTrACSStorm]
    source: str
    state: str
    loading: bool

    @property
    def ready(self) -> bool:
        return self.state == "ready"


def _state_for(source: str, storms: list[IBTrACSStorm], loading: bool) -> str:
    """Classify a snapshot without ever reporting a false 'ready'."""
    if storms and source in ("live", "cache"):
        return "ready"
    if loading:
        return "warming"
    return "degraded"


def dataset_snapshot() -> DatasetSnapshot:
    """Return the best available dataset *without* waiting on the network.

    Resolution order (local I/O only, so it answers in milliseconds once the
    archive has been parsed once):

    1. the in-memory record,
    2. the JSON sidecar written by an earlier parse,
    3. the downloaded CSV itself (~2 s for 28 MB),
    4. the bundled curated table.

    Whenever the local copy is missing or older than the TTL a background
    refresh is started, but the caller never pays the download cost. That is
    what keeps the dataset panels loading instantly instead of hitting the
    request timeout and asking the user to retry.
    """
    with _state_lock:
        storms, source, loading = _storms, _source, _loading

    if storms:
        if not _cache_is_fresh():
            warm_async()
        return DatasetSnapshot(
            storms, source, _state_for(source, storms, loading), loading
        )

    local = _parse_local_cache()
    if local:
        _install(local, "cache")
        if not _cache_is_fresh():
            warm_async()
        return DatasetSnapshot(local, "cache", "ready", loading)

    # Nothing usable on disk: serve the curated anchors and fetch in the
    # background so the very next request can answer with the full record.
    warm_async()
    with _state_lock:
        storms, source, loading = _storms, _source, _loading
    if not storms:
        fallback = _bundled_fallback()
        _install(fallback, "bundled")
        return DatasetSnapshot(fallback, "bundled", "degraded", loading)
    return DatasetSnapshot(storms, source, _state_for(source, storms, loading), loading)


def warm_async(force: bool = False) -> None:
    """Kick a background download/parse cycle; safe to call repeatedly."""
    with _state_lock:
        if _loading:
            return
    threading.Thread(
        target=refresh,
        kwargs={"force": force},
        name="ibtracs-refresh",
        daemon=True,
    ).start()


def refresh(force: bool = False) -> list[IBTrACSStorm]:
    """Download (if needed) and (re)parse the dataset. Never raises.

    Serialised by ``_refresh_lock``, so concurrent callers cannot trigger
    duplicate downloads: a caller that arrives while a refresh is running
    simply receives that refresh's result.
    """
    global _storms, _source, _last_error, _loading

    with _refresh_lock:
        # Another worker may have completed the fetch while we waited.
        if not force:
            with _state_lock:
                ready = _storms is not None and _source in ("live", "cache")
                live = _source == "live"
            if ready and (live or _cache_is_fresh()):
                with _state_lock:
                    return _storms  # type: ignore[return-value]

        with _state_lock:
            _loading = True
            # ``download_dataset`` flips this back to "live" when it actually
            # fetches fresh data; staying on "cache" means the local copy won.
            _source = "cache"
        try:
            try:
                path = download_dataset(force=force)
                storms = _parse(path)
                if not storms:
                    raise ValueError("Parsed zero storms from IBTrACS file")
                write_sidecar(storms)
                with _state_lock:
                    _storms = storms
                    _last_error = None
                logger.info("Loaded %d storms from IBTrACS (%s).", len(storms), _source)
                return storms
            except Exception as exc:  # noqa: BLE001 - degrade gracefully
                with _state_lock:
                    _last_error = f"{type(exc).__name__}: {exc}"
                rescued = _rescue_cache()
                if rescued is not None:
                    with _state_lock:
                        _storms = rescued
                        _source = "cache"
                    return rescued
                fallback = _bundled_fallback()
                with _state_lock:
                    _storms = fallback
                    _source = "bundled"
                return fallback
        finally:
            with _state_lock:
                _loading = False


def load_storms(force: bool = False, wait: bool = True) -> list[IBTrACSStorm]:
    """Return the dataset, downloading live data on first use.

    Resolution order (never raises):

    1. a fresh download (or a fresh cache),
    2. an older but readable cache -- a stale 342-storm file is far more
       accurate than the tiny bundled table, so it is preferred whenever the
       download fails,
    3. the bundled curated table, only when no usable cache exists at all.

    ``wait=True`` (the default, kept for existing callers) resolves the
    dataset synchronously. ``wait=False`` is the request-path variant: it
    returns the observed record the moment it can be resolved locally and
    otherwise hands back the bundled table while the download continues in
    the background.
    """
    if not force:
        snapshot = dataset_snapshot()
        if snapshot.ready or not wait:
            return snapshot.storms
    return refresh(force=force)


def _rescue_cache() -> Optional[list[IBTrACSStorm]]:
    """Parse the on-disk cache after a failed download; ``None`` if unusable.

    ``_last_error`` is intentionally left set so ``/ibtracs/status`` still
    reports why the refresh failed while the UI keeps showing real storms.
    """
    if not CACHE_FILE.exists():
        return None
    try:
        storms = _parse(CACHE_FILE)
    except Exception as exc:  # noqa: BLE001 - corrupt cache is not fatal
        logger.warning(
            "Cached IBTrACS file unreadable (%s: %s); falling back to bundled data.",
            type(exc).__name__,
            exc,
        )
        return None
    if not storms:
        return None

    cached_at = datetime.fromtimestamp(
        CACHE_FILE.stat().st_mtime, tz=timezone.utc
    ).isoformat()
    logger.warning(
        "IBTrACS refresh failed (%s); serving %d cached storms from %s.",
        _last_error,
        len(storms),
        cached_at,
    )
    return storms


def get_dataset() -> list[dict[str, Any]]:
    """All storms as plain dicts (API-friendly)."""
    return [s.to_dict() for s in dataset_snapshot().storms]


# Sort orders offered by the dataset view. Each key is a stable callable so
# the ordering is identical for every client (and easy to test).
SORT_ORDERS: dict[str, Any] = {
    "recent": lambda s: (-s.year, -s.max_wind_knots),
    "oldest": lambda s: (s.year, -s.max_wind_knots),
    "intense": lambda s: (-s.max_wind_knots, -s.year),
    "weakest": lambda s: (s.max_wind_knots, -s.year),
    "name": lambda s: (s.name.lower(), -s.year),
    "pressure": lambda s: (
        s.min_pressure_hpa if s.min_pressure_hpa else 9_999.0,
        -s.year,
    ),
}


def query_dataset(
    limit: Optional[int] = None,
    offset: int = 0,
    sort: str = "recent",
    basin: Optional[str] = None,
    category: Optional[str] = None,
    min_wind: Optional[float] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    search: Optional[str] = None,
) -> tuple[list[IBTrACSStorm], int]:
    """Filter and sort the full observed record for the dataset view.

    Returns ``(page, matched)`` where ``matched`` counts every storm that
    satisfied the filters (before ``limit``/``offset``), so the UI can report
    "showing 342 of 342 storms" truthfully.
    """
    storms = dataset_snapshot().storms

    needle = (search or "").strip().lower()
    if basin:
        basin_lower = basin.strip().lower()
        storms = [s for s in storms if s.basin.lower() == basin_lower]
    if category:
        category_lower = category.strip().lower()
        storms = [s for s in storms if s.category.lower() == category_lower]
    if min_wind is not None:
        storms = [s for s in storms if s.max_wind_knots >= float(min_wind)]
    if year_min is not None:
        storms = [s for s in storms if s.year >= int(year_min)]
    if year_max is not None:
        storms = [s for s in storms if s.year <= int(year_max)]
    if needle:
        storms = [
            s
            for s in storms
            if needle in s.name.lower()
            or needle in s.sid.lower()
            or needle in s.category.lower()
            or needle in s.basin.lower()
            or needle == str(s.year)
        ]

    storms = sorted(storms, key=SORT_ORDERS.get(sort, SORT_ORDERS["recent"]))
    matched = len(storms)

    start = max(0, int(offset or 0))
    if limit is None or int(limit) <= 0:
        page = storms[start:]
    else:
        page = storms[start : start + int(limit)]
    return page, matched


def get_recent(limit: int = 25) -> list[dict[str, Any]]:
    """Most recent storms, newest first."""
    page, _ = query_dataset(limit=limit, sort="recent")
    return [s.to_dict() for s in page]


def get_top_intense(limit: int = 25) -> list[dict[str, Any]]:
    """Strongest storms on record by peak wind."""
    page, _ = query_dataset(limit=limit, sort="intense")
    return [s.to_dict() for s in page]


def get_named_only() -> list[IBTrACSStorm]:
    """Storms that were actually named (used for calibration labels).

    Uses the non-blocking snapshot: calibration must never wait on a NOAA
    download, it can always fall back to the curated anchors instead.
    """
    storms = dataset_snapshot().storms
    names = [s for s in storms if s.name and s.name != "UNNAMED"]
    return names or storms


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


def _counts(storms: list[IBTrACSStorm], attr: str) -> dict[str, int]:
    """Tally storms by a string attribute, largest group first."""
    counts: dict[str, int] = {}
    for storm in storms:
        key = str(getattr(storm, attr, "") or "Unknown")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def get_scale() -> list[dict[str, Any]]:
    """The IMD intensity scale used to label the record (from WIND_BANDS)."""
    return [
        {"class_index": index, "min_wind_knots": band_min, "category": label}
        for band_min, label, index in WIND_BANDS
    ]


def dataset_summary() -> dict[str, Any]:
    """Aggregate statistics describing the *whole* observed record.

    Everything a reader needs to judge the dataset at a glance: coverage,
    strongest/lowest values, how the storms distribute across basins and
    IMD categories, and where the numbers came from.
    """
    snapshot = dataset_snapshot()
    storms = snapshot.storms
    years = [s.year for s in storms if s.year > 0]
    winds = [s.max_wind_knots for s in storms if s.max_wind_knots]
    pressures = [s.min_pressure_hpa for s in storms if s.min_pressure_hpa]
    return {
        "state": snapshot.state,
        "loading": snapshot.loading,
        "source": snapshot.source,
        "records": len(storms),
        "named_records": sum(1 for s in storms if s.name and s.name != "UNNAMED"),
        "records_with_pressure": len(pressures),
        "year_range": f"{min(years)}-{max(years)}" if years else "n/a",
        "first_year": min(years) if years else None,
        "last_year": max(years) if years else None,
        "strongest_wind_knots": max(winds) if winds else None,
        "lowest_pressure_hpa": min(pressures) if pressures else None,
        "mean_peak_wind_knots": round(sum(winds) / len(winds), 1) if winds else None,
        "basins": _counts(storms, "basin"),
        "categories": _counts(storms, "category"),
        # Aliases used by /ibtracs/status so both payloads expose the same
        # aggregate keys and the UI can render either without special cases.
        "records_by_basin": _counts(storms, "basin"),
        "records_by_category": _counts(storms, "category"),
        "attribute_count": len(DATASET_ATTRIBUTES),
        "generated_at": _now().isoformat(),
    }


def get_status() -> dict[str, Any]:
    """Instant health/telemetry for the live dataset.

    Never blocks on the network: it reports whatever is already resolved
    (memory, sidecar or the downloaded CSV) and says so through ``state``.
    """
    snapshot = dataset_snapshot()
    storms = snapshot.storms
    years = [s.year for s in storms if s.year > 0]
    winds = [s.max_wind_knots for s in storms if s.max_wind_knots]
    pressures = [s.min_pressure_hpa for s in storms if s.min_pressure_hpa]
    return {
        "source": snapshot.source,
        "state": snapshot.state,
        "loading": snapshot.loading,
        "records": len(storms),
        "year_range": f"{min(years)}-{max(years)}" if years else "n/a",
        "cache_file": str(CACHE_FILE),
        "cache_present": CACHE_FILE.exists(),
        "cache_size_mb": _cache_size_mb(),
        "cache_age_hours": _cache_age_hours(),
        "cache_fresh": _cache_is_fresh(),
        "cache_ttl_hours": CACHE_TTL_HOURS,
        "sidecar_file": str(_sidecar_file()),
        "sidecar_present": _sidecar_file().exists(),
        "downloaded_at": _downloaded_at.isoformat() if _downloaded_at else None,
        "last_error": _last_error,
        "url": IBTRACS_URL,
        "license": "Public domain (NOAA/NCEI)",
        "records_by_basin": _counts(storms, "basin"),
        "records_by_category": _counts(storms, "category"),
        "strongest_wind_knots": max(winds) if winds else None,
        "lowest_pressure_hpa": min(pressures) if pressures else None,
        "attribute_count": len(DATASET_ATTRIBUTES),
    }


def get_full_dataset(
    limit: Optional[int] = None,
    offset: int = 0,
    sort: str = "recent",
    basin: Optional[str] = None,
    category: Optional[str] = None,
    min_wind: Optional[float] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    search: Optional[str] = None,
) -> dict[str, Any]:
    """Complete, self-documenting dataset payload.

    Returns every matching storm with every attribute, plus the summary
    statistics, the IMD scale and the field-level documentation, so one
    request is enough for the dashboard to render the whole dataset view.
    """
    page, matched = query_dataset(
        limit=limit,
        offset=offset,
        sort=sort,
        basin=basin,
        category=category,
        min_wind=min_wind,
        year_min=year_min,
        year_max=year_max,
        search=search,
    )
    snapshot = dataset_snapshot()
    return {
        "state": snapshot.state,
        "loading": snapshot.loading,
        "source": snapshot.source,
        "total": len(snapshot.storms),
        "matched": matched,
        "returned": len(page),
        "offset": max(0, int(offset or 0)),
        "sort": sort if sort in SORT_ORDERS else "recent",
        "sort_orders": sorted(SORT_ORDERS),
        "storms": [s.to_dict() for s in page],
        "summary": dataset_summary(),
        "scale": get_scale(),
        "attributes": get_attributes(),
    }
