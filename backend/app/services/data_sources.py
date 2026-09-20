"""CYCLO-VISION -- Multi-source satellite data service.

Defines the pluggable data-source registry. Real integrations (INSAT,
GOES, NASA Earthdata) can be added later; for the prototype, sources are
reported as demo/unavailable with a clear status so the UI is honest.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.core.config import settings

# Registry of satellite sources in the architecture.
_SOURCE_CATALOG = [
    {
        "name": "INSAT-3DR",
        "satellite_type": "Geo-stationary (India)",
        "description": "Indian National Satellite System -- Himawari/INSAT series IR & visible.",
    },
    {
        "name": "NOAA GOES-16/17",
        "satellite_type": "Geo-stationary (US)",
        "description": "GOES / ABI multi-band imagery for West Atlantic & Pacific basins.",
    },
    {
        "name": "NASA Terra/Aqua MODIS",
        "satellite_type": "Low-Earth Orbit",
        "description": "NASA Earth Observation (Worldview) -- MODIS true-color & thermal.",
    },
    {
        "name": "EUMETSAT Meteosat",
        "satellite_type": "Geo-stationary (Europe)",
        "description": "Meteosat SEVIFI imagery -- Indian Ocean & Atlantic coverage.",
    },
    {
        "name": "Himawari-8/9",
        "satellite_type": "Geo-stationary (Japan)",
        "description": "JMA Himawari AHI -- complementary Western Pacific coverage.",
    },
]


def get_data_sources() -> list[dict[str, Any]]:
    """Return source availability. In demo mode all sources are labeled
    'demo' (available only as bundled sample data)."""
    sources = []
    for src in _SOURCE_CATALOG:
        sources.append({
            "id": src["name"].lower().replace(" ", "-").replace("/", "-"),
            "name": src["name"],
            "satellite_type": src["satellite_type"],
            "description": src["description"],
            "status": "demo",
            "availability": "Bundled demo data only",
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "region": "Bay of Bengal / North Indian Ocean (demo)",
        })
    return sources