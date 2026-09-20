"""
CYCLO-VISION -- Postprocessing / Output Mapping
===============================================
Turns raw model probabilities (or demo feature scores) into a structured,
scientifically-framed result:

    * class label
    * confidence
    * estimated max wind speed (knots)
    * estimated central pressure (hPa)
    * risk level + factors
    * forecast track points (prototype)

All document-prototype mappings are defined here so they can be revised
as real training data and validated models are introduced.
"""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np

# Re-export labels for convenience
from ml.model import CLASS_LABELS, CLASS_WIND_KNOTS, CLASS_PRESSURE_HPA
from ml.classification import category_for_wind

# ---------------------------------------------------------------------------
# Calibration (documented prototype mapping)
# ---------------------------------------------------------------------------

# Confidence/wind thresholds for risk tiers. Wind is the primary driver
# (see assess_risk); the second element is the minimum confidence percent
# required to *keep* the tier when wind alone would grant it.
RISK_THRESHOLDS = {
    "EXTREME": (110, 35),  # (min_wind_knots, min_confidence_pct)
    "HIGH": (70, 35),
    "MODERATE": (40, 20),
    "LOW": (0, 0),
}


def intensity_category(wind_knots: float) -> str:
    """Maps wind speed to a category name using the shared IMD wind bands.

    Delegates to :mod:`ml.classification` so the reported category and the
    calibration class index can never disagree (see P0-3 / WIND_BANDS).
    """
    return category_for_wind(wind_knots)


def estimate_intensity(class_idx: int, confidence: float) -> tuple[float, float]:
    """Prototype intensity estimate.

    Derived from the representative wind/pressure of the class, scaled
    slightly by confidence so distinct images yield distinct numbers.
    """
    base_wind = float(CLASS_WIND_KNOTS[class_idx])
    base_pressure = float(CLASS_PRESSURE_HPA[class_idx])
    jitter = (confidence - 0.5) * 4  # small, deterministic-ish
    wind = round(base_wind + jitter, 1)
    pressure = round(base_pressure - jitter, 1)
    return wind, pressure


def compute_confidence(probs: np.ndarray, class_idx: int) -> float:
    """Confidence from softmax probability with mild sharpening."""
    p = float(probs[class_idx])
    sharpened = p ** 0.8
    return round(min(0.999, max(0.0, sharpened)), 4)


def assess_risk(
    class_idx: int, wind: float, pressure: float, confidence: float
) -> tuple[str, list[dict[str, str]]]:
    """Return (risk_level, factor_list) per documented prototype rule."""
    factors: list[dict[str, str]] = []
    if class_idx >= 1:
        factors.append(
            {"factor": "Cyclone detected", "impact": str(CLASS_LABELS[class_idx])}
        )
    factors.append({"factor": "Wind speed", "impact": f"{wind:g} knots"})
    factors.append({"factor": "Pressure", "impact": f"{pressure:g} hPa"})

    # Primary rule: wind drives risk (thresholds from RISK_THRESHOLDS).
    conf_pct = confidence * 100
    level = "LOW"
    for tier in ("EXTREME", "HIGH", "MODERATE"):
        min_wind, min_conf = RISK_THRESHOLDS[tier]
        if wind >= min_wind and conf_pct >= min_conf:
            level = tier
            break

    # Downscale risk if confidence is very low.
    if conf_pct < 35 and level not in ("LOW",):
        level = "MODERATE" if level == "HIGH" else "LOW"

    factors.append(
        {
            "factor": "Model confidence",
            "impact": f"{conf_pct:.1f}% {'(supports risk)' if conf_pct > 40 else '(limited)'}",
        }
    )
    if class_idx >= 4:
        factors.append({"factor": "Intensity trend", "impact": "Rapid intensification"})
    return level, factors


# ---------------------------------------------------------------------------
# Deterministic pseudo-random track generator (prototype)
# ---------------------------------------------------------------------------


def _pseudo_rng(seed: int) -> random.Random:
    return random.Random(seed ^ 0x5EED)


def generate_track(
    center_lat: float,
    center_lon: float,
    storm_strength: float = 1.0,
    seed: int = 42,
    direction_deg: float | None = None,
    forward_speed_knots: float | None = None,
) -> list[dict[str, Any]]:
    """Generate prototype forecast track points.

    Produces [current, +6h, +12h, +24h, +48h] with a cone of uncertainty.
    Labeled as prototype/simulated -- not operational forecast.
    Uses reference cyclone track parameters when provided.
    """
    # Typical cyclone drift in the demo region (Bay of Bengal, NW movement).
    base_dir_deg = direction_deg if direction_deg is not None else 300.0  # ~ WNW
    drift_knots = forward_speed_knots if forward_speed_knots is not None else (
        8.0 + 4.0 * storm_strength
    )
    # Physical scaling: 1 degree of latitude ~= 60 nautical miles, so a
    # storm moving at N knots covers N/60 degrees per hour. The previous
    # constant (0.33 per 6h) moved a 12-kt storm ~240 nmi in 6 hours --
    # a jet, not a cyclone (P0 review #3).
    deg_per_nmi = 1.0 / 60.0

    rng = _pseudo_rng(seed)

    points = [
        {"label": "Current", "hours": 0, "lat": center_lat, "lon": center_lon}
    ]
    lat, lon = center_lat, center_lon
    for hours, label in [(6, "+6h"), (12, "+12h"), (24, "+24h"), (48, "+48h")]:
        # Compass-bearing convention: bearing is measured clockwise from
        # north, so the northward component is cos(bearing) and the
        # eastward component is sin(bearing). The previous -sin/cos form
        # treated the bearing as a maths angle and moved 300-degree (WNW)
        # storms to the NE, roughly 90 degrees off course.
        rad = math.radians(base_dir_deg + rng.uniform(-12, 12))
        # Knots are nautical miles per hour, so displacement in degrees is
        # speed * hours / 60. Each point is the *cumulative* position from
        # the storm centre (not an increment on the previous point -- the
        # earlier version compounded the drift 1x/2x/4x/8x per leg).
        # Longitude shrinks with cos(lat) but the prototype ignores that
        # ~5% Bay of Bengal effect.
        dlat = math.cos(rad) * drift_knots * hours * deg_per_nmi
        dlon = math.sin(rad) * drift_knots * hours * deg_per_nmi
        lat = center_lat + dlat + rng.uniform(-0.2, 0.2)
        lon = center_lon + dlon + rng.uniform(-0.2, 0.2)
        points.append({"label": label, "hours": hours, "lat": lat, "lon": lon})

    # Uncertainty cone radii (km) per forecast hour -- prototype values.
    # Exactly one entry per point (current + 4 forecast); the old 6-entry
    # list silently truncated via zip() and dropped the +48h radius.
    cone = [0, 40, 60, 90, 150]
    if len(cone) != len(points):  # pragma: no cover - guard against drift
        raise RuntimeError("cone radii must match track point count")
    for p, r in zip(points, cone):
        p["cone_km"] = r
    return points


def default_region_for_class(class_idx: int) -> tuple[float, float]:
    """Return a plausible (lat, lon) center in the Bay of Bengal."""
    base = [
        (10.0, 88.0),   # No cyclone -- random point
        (12.5, 90.0),
        (13.0, 91.0),
        (13.5, 90.5),
        (14.0, 90.0),
        (15.0, 89.5),
        (16.0, 89.0),
    ]
    lat, lon = base[max(0, min(class_idx, len(base) - 1))]
    return lat, lon