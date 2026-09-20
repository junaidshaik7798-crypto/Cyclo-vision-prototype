"""
CYCLO-VISION -- Reference Cyclone Dataset Service
=================================================
Provides real-world reference data for tropical cyclones in the North
Indian Ocean / Bay of Bengal region (from IBTrACS / IMD historical
records). This dataset serves as a knowledge base that the analysis
engine uses to produce calibrated, realistic intensity estimates and
risk assessments, instead of purely synthetic values.

Data includes:
  * Historical cyclone events (name, year, category, winds, pressure)
  * Basin statistics
  * Reference intensity curves
  * Typical track direction / speed for the Bay of Bengal
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Historical reference cyclones -- Bay of Bengal / North Indian Ocean
# (Source: IMD/IBTrACS consolidated records, representative values)
# ---------------------------------------------------------------------------

CYCLONE_REFERENCE_DATASET: list[dict[str, Any]] = [
    # --- Super / Extremely Severe ---
    {
        "name": "Fani",
        "year": 2019,
        "basin": "Bay of Bengal",
        "category": "Extremely Severe Cyclonic Storm",
        "max_wind_knots": 130,
        "min_pressure_hpa": 915,
        "landfall_lat": 20.2,
        "landfall_lon": 86.2,
        "track_direction_deg": 300,   # WNW
        "forward_speed_knots": 12.0,
        "intensity_index": 6,         # matches class_index
        "notes": "Strongest Odisha cyclone since 1999 Super Cyclone.",
    },
    {
        "name": "Amphan",
        "year": 2020,
        "basin": "Bay of Bengal",
        "category": "Super Cyclonic Storm",
        "max_wind_knots": 160,
        "min_pressure_hpa": 920,
        "landfall_lat": 21.6,
        "landfall_lon": 88.3,
        "track_direction_deg": 0,     # N
        "forward_speed_knots": 14.0,
        "intensity_index": 6,
        "notes": "Strongest cyclone on record in the Bay of Bengal.",
    },
    {
        "name": "Mocha",
        "year": 2023,
        "basin": "Bay of Bengal",
        "category": "Extremely Severe Cyclonic Storm",
        "max_wind_knots": 130,
        "min_pressure_hpa": 938,
        "landfall_lat": 21.8,
        "landfall_lon": 92.3,
        "track_direction_deg": 15,    # NNE
        "forward_speed_knots": 10.0,
        "intensity_index": 6,
        "notes": "Crossed Myanmar/Bangladesh border region.",
    },
    {
        "name": "Hudhud",
        "year": 2014,
        "basin": "Bay of Bengal",
        "category": "Very Severe Cyclonic Storm",
        "max_wind_knots": 110,
        "min_pressure_hpa": 960,
        "landfall_lat": 17.6,
        "landfall_lon": 83.3,
        "track_direction_deg": 330,   # NNW
        "forward_speed_knots": 9.0,
        "intensity_index": 5,
        "notes": "Major damage at Visakhapatnam.",
    },
    {
        "name": "Bulbul",
        "year": 2019,
        "basin": "Bay of Bengal",
        "category": "Very Severe Cyclonic Storm",
        "max_wind_knots": 100,
        "min_pressure_hpa": 976,
        "landfall_lat": 21.2,
        "landfall_lon": 88.0,
        "track_direction_deg": 355,   # N
        "forward_speed_knots": 8.0,
        "intensity_index": 5,
        "notes": "Crossed West Bengal / Bangladesh coast.",
    },
    {
        "name": "Vardah",
        "year": 2016,
        "basin": "Bay of Bengal",
        "category": "Very Severe Cyclonic Storm",
        "max_wind_knots": 85,
        "min_pressure_hpa": 975,
        "landfall_lat": 13.2,
        "landfall_lon": 80.4,
        "track_direction_deg": 280,   # WNW
        "forward_speed_knots": 7.0,
        "intensity_index": 5,
        "notes": "Crossed Chennai coast.",
    },
    {
        "name": "Gaja",
        "year": 2018,
        "basin": "Bay of Bengal",
        "category": "Severe Cyclonic Storm",
        "max_wind_knots": 75,
        "min_pressure_hpa": 978,
        "landfall_lat": 10.3,
        "landfall_lon": 79.5,
        "track_direction_deg": 270,   # W
        "forward_speed_knots": 7.0,
        "intensity_index": 4,
        "notes": "Crossed Tamil Nadu coast near Vedaranyam.",
    },
    {
        "name": "Nivar",
        "year": 2020,
        "basin": "Bay of Bengal",
        "category": "Severe Cyclonic Storm",
        "max_wind_knots": 75,
        "min_pressure_hpa": 978,
        "landfall_lat": 12.4,
        "landfall_lon": 80.0,
        "track_direction_deg": 295,   # WNW
        "forward_speed_knots": 8.0,
        "intensity_index": 4,
        "notes": "Crossed near Puducherry / Tamil Nadu.",
    },
    {
        "name": "Titli",
        "year": 2018,
        "basin": "Bay of Bengal",
        "category": "Very Severe Cyclonic Storm",
        "max_wind_knots": 90,
        "min_pressure_hpa": 972,
        "landfall_lat": 18.8,
        "landfall_lon": 84.5,
        "track_direction_deg": 340,   # NNW
        "forward_speed_knots": 8.0,
        "intensity_index": 5,
        "notes": "Crossed Odisha / Andhra Pradesh border.",
    },
    {
        "name": "Phailin",
        "year": 2013,
        "basin": "Bay of Bengal",
        "category": "Very Severe Cyclonic Storm",
        "max_wind_knots": 140,
        "min_pressure_hpa": 940,
        "landfall_lat": 19.0,
        "landfall_lon": 84.9,
        "track_direction_deg": 315,   # NW
        "forward_speed_knots": 10.0,
        "intensity_index": 6,
        "notes": "One of the strongest to hit Odisha.",
    },
    {
        "name": "Tauktae",
        "year": 2021,
        "basin": "Arabian Sea",
        "category": "Extremely Severe Cyclonic Storm",
        "max_wind_knots": 120,
        "min_pressure_hpa": 950,
        "landfall_lat": 21.4,
        "landfall_lon": 72.7,
        "track_direction_deg": 350,   # N
        "forward_speed_knots": 12.0,
        "intensity_index": 6,
        "notes": "Major cyclone on the Gujarat / Saurashtra coast.",
    },
    {
        "name": "Yaas",
        "year": 2021,
        "basin": "Bay of Bengal",
        "category": "Very Severe Cyclonic Storm",
        "max_wind_knots": 85,
        "min_pressure_hpa": 970,
        "landfall_lat": 21.0,
        "landfall_lon": 87.0,
        "track_direction_deg": 0,     # N
        "forward_speed_knots": 9.0,
        "intensity_index": 5,
        "notes": "Crossed Odisha / West Bengal coast.",
    },
    {
        "name": "Michaung",
        "year": 2023,
        "basin": "Bay of Bengal",
        "category": "Severe Cyclonic Storm",
        "max_wind_knots": 65,
        "min_pressure_hpa": 984,
        "landfall_lat": 14.0,
        "landfall_lon": 80.5,
        "track_direction_deg": 300,   # WNW
        "forward_speed_knots": 6.0,
        "intensity_index": 4,
        "notes": "Crossed near Nellore, Andhra Pradesh.",
    },
    {
        "name": "Remal",
        "year": 2024,
        "basin": "Bay of Bengal",
        "category": "Severe Cyclonic Storm",
        "max_wind_knots": 60,
        "min_pressure_hpa": 985,
        "landfall_lat": 21.5,
        "landfall_lon": 89.0,
        "track_direction_deg": 10,    # NNE
        "forward_speed_knots": 8.0,
        "intensity_index": 3,
        "notes": "Crossed Bangladesh / West Bengal coast.",
    },
]
# Class-index -> typical track direction & speed (for Bay of Bengal)
TRACK_PARAMETERS: dict[int, dict[str, float]] = {
    0: {"direction_deg": 0.0, "speed_knots": 0.0},      # No cyclone
    1: {"direction_deg": 290.0, "speed_knots": 5.0},    # Depression
    2: {"direction_deg": 295.0, "speed_knots": 6.0},    # Deep Depression
    3: {"direction_deg": 300.0, "speed_knots": 7.0},    # Cyclonic Storm
    4: {"direction_deg": 310.0, "speed_knots": 8.0},    # Severe Cyclonic Storm
    5: {"direction_deg": 320.0, "speed_knots": 9.0},    # Very Severe Cyclonic Storm
    6: {"direction_deg": 325.0, "speed_knots": 11.0},   # Extremely Severe Cyclonic Storm
}

# Reference wind/pressure intensity curves (knots & hPa) by class index
INTENSITY_CURVE: dict[int, dict[str, float]] = {
    0: {"wind": 15.0, "pressure": 1008.0},
    1: {"wind": 25.0, "pressure": 1002.0},
    2: {"wind": 32.0, "pressure": 996.0},
    3: {"wind": 40.0, "pressure": 990.0},
    4: {"wind": 60.0, "pressure": 972.0},
    5: {"wind": 85.0, "pressure": 958.0},
    6: {"wind": 115.0, "pressure": 938.0},
}

# Default center coordinates for each class (Bay of Bengal / NIO)
CLASS_CENTERS: dict[int, tuple[float, float]] = {
    0: (8.0, 90.0),
    1: (12.5, 89.0),
    2: (13.0, 90.5),
    3: (13.5, 90.0),
    4: (14.0, 90.5),
    5: (15.0, 89.5),
    6: (16.0, 89.0),
}


@dataclass
class ReferenceCyclone:
    """A reference cyclone event from the dataset.

    Field notes
    -----------
    The curated table in this module is hand-written and therefore carries
    only the fields needed for calibration; it is **not** a full IBTrACS
    record. In particular there are no ``genesis_lat`` / ``genesis_lon``
    fields here -- ``app.services.ibtracs._bundled_fallback`` synthesises
    them from the landfall position (genesis ~ landfall offset by 2deg S and
    3deg W) purely so the offline fallback still produces a plausible track
    bearing. Treat those synthesised values as placeholders, not data.
    """

    name: str
    year: int
    basin: str
    category: str
    max_wind_knots: float
    min_pressure_hpa: float
    landfall_lat: float
    landfall_lon: float
    track_direction_deg: float
    forward_speed_knots: float
    intensity_index: int
    notes: str = ""


def get_reference_dataset() -> list[ReferenceCyclone]:
    """Return the full reference dataset as ReferenceCyclone objects."""
    return [ReferenceCyclone(**item) for item in CYCLONE_REFERENCE_DATASET]


def find_reference_cyclone(class_idx: int) -> Optional[ReferenceCyclone]:
    """Find the best-matching reference cyclone for a given class index."""
    matches = [c for c in get_reference_dataset() if c.intensity_index == class_idx]
    if not matches:
        return None
    # Pick the most recent one (best relevance)
    return max(matches, key=lambda c: c.year)


def get_center_for_class(class_idx: int) -> tuple[float, float]:
    """Return a plausible storm center for a cyclone class."""
    return CLASS_CENTERS.get(class_idx, (10.0, 88.0))


def get_track_params_for_class(class_idx: int) -> dict[str, float]:
    """Return track direction & forward speed for a cyclone class."""
    return TRACK_PARAMETERS.get(class_idx, {"direction_deg": 300.0, "speed_knots": 8.0})


def get_intensity_for_class(class_idx: int) -> dict[str, float]:
    """Return reference wind speed & pressure for a cyclone class."""
    return INTENSITY_CURVE.get(class_idx, {"wind": 40.0, "pressure": 990.0})


def list_recent_cyclones(limit: int = 10) -> list[dict[str, Any]]:
    """Return recent cyclone events sorted by year (newest first)."""
    events = sorted(
        CYCLONE_REFERENCE_DATASET, key=lambda c: c["year"], reverse=True
    )
    return events[:limit]


def get_dataset_summary() -> dict[str, Any]:
    """Return summary statistics about the reference dataset."""
    events = CYCLONE_REFERENCE_DATASET
    by_basin: dict[str, int] = {}
    for ev in events:
        basin = ev["basin"]
        by_basin[basin] = by_basin.get(basin, 0) + 1

    categories = sorted(set(ev["category"] for ev in events))
    return {
        "total_events": len(events),
        "year_range": f"{min(e['year'] for e in events)}-{max(e['year'] for e in events)}",
        "basins": by_basin,
        "categories": categories,
        "strongest_wind_knots": max(e["max_wind_knots"] for e in events),
        "lowest_pressure_hpa": min(e["min_pressure_hpa"] for e in events),
        "updated": datetime.now(timezone.utc).isoformat(),
    }