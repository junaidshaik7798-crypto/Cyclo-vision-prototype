"""
CYCLO-VISION -- Intensity classification (single source of truth)
===============================================================

The IMD/RSMC New Delhi wind-speed bands in **one** table, used by every
consumer:

    * ``ml.postprocessing.intensity_category``  (reported category string)
    * ``app.services.ibtracs``                  (observed-storm labels and
                                                 calibration class index)

Why this module exists
----------------------
Previously `_intensity_index` (IBTrACS) and `intensity_category`
(postprocessing) each hard-coded slightly different thresholds, so the same
wind speed could be labelled "Cyclonic Storm" by one and "Severe Cyclonic
Storm" by the other -- e.g. a storm classified as Cyclonic Storm was
reported as Severe Cyclonic Storm once calibrated against the reference
dataset. Keeping a single ordered table removes that class of bug by
construction.

The bands are the official IMD thresholds in knots:
    >= 120 kt  Super Cyclonic Storm            (>= 222 km/h)
    90-119 kt  Extremely Severe Cyclonic Storm (167-221 km/h)
    64-89  kt  Very Severe Cyclonic Storm      (119-166 km/h)
    48-63  kt  Severe Cyclonic Storm           (89-117 km/h)
    34-47  kt  Cyclonic Storm                  (63-88 km/h)
    28-33  kt  Deep Depression                 (52-61 km/h)
    17-27  kt  Depression                      (31-49 km/h)
    < 17   kt  No Cyclone / Depression
"""

from __future__ import annotations

# (minimum wind in knots, IMD category label, model class index)
# Ordered strongest-first so the first `kt >= band_min` match wins.
# Class indices match `ml.model.CLASS_LABELS` (0 = no cyclone .. 6 = strongest).
WIND_BANDS: list[tuple[float, str, int]] = [
    (120.0, "Super Cyclonic Storm", 6),
    (90.0, "Extremely Severe Cyclonic Storm", 6),
    (64.0, "Very Severe Cyclonic Storm", 5),
    (48.0, "Severe Cyclonic Storm", 4),
    (34.0, "Cyclonic Storm", 3),
    (28.0, "Deep Depression", 2),
    (17.0, "Depression", 1),
    (0.0, "No Cyclone / Depression", 0),
]


def band_for_wind(kt: float) -> tuple[float, str, int]:
    """Return the ``(band_min, category, class_index)`` row for a wind speed.

    This is the only place the wind bands are evaluated; both public
    helpers below derive from it so they can never disagree.
    """
    for band_min, category, class_index in WIND_BANDS:
        if kt >= band_min:
            return band_min, category, class_index
    # Unreachable in practice: the last band starts at 0.0, and negative
    # inputs (never produced by inference) are clamped to it for safety.
    return WIND_BANDS[-1]


def category_for_wind(kt: float) -> str:
    """Map a 1-minute maximum sustained wind (knots) to an IMD category."""
    return band_for_wind(kt)[1]


def class_index_for_wind(kt: float) -> int:
    """Map a 1-minute maximum sustained wind (knots) to a model class index."""
    return band_for_wind(kt)[2]


def band_centre_for_class(class_index: int) -> float | None:
    """Mid-point wind speed (knots) that *best* represents a class index.

    A class index can span more than one band (6 covers both the Super
    Cyclonic and Extremely Severe Cyclonic bands), so the widest span is
    used. This is what ``ibtracs.best_match_for_class`` anchors calibration
    on, so the reference storm is a mid-band event rather than the most
    extreme storm on record. Returns ``None`` for an unknown class.
    """
    mins = [b[0] for b in WIND_BANDS if b[2] == class_index]
    if not mins:
        return None
    highest, lowest = max(mins), min(mins)
    # Open-ended top band (Super Cyclonic, >= 120 kt): use a documented
    # nominal top of 150 kt so the centre stays a realistic peak intensity.
    top = 150.0 if highest == WIND_BANDS[0][0] else highest
    # Bottom band (< 17 kt) is reported as "No Cyclone"; there is no storm
    # to anchor on, so the centre is the band start.
    if lowest == 0.0:
        return 0.0
    return (lowest + top) / 2.0