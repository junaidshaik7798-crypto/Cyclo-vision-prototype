"""P0-3 regression: wind speed -> class index / category consistency.

``postprocessing.intensity_category`` and ``ibtracs._intensity_index`` used
to hard-code different thresholds for the same wind bands, so a storm's
reported category could contradict its classification class. Both now derive
from the single ``ml.classification.WIND_BANDS`` table; these tests fail if
that ever diverges again.
"""

from __future__ import annotations

from ml.classification import (
    WIND_BANDS,
    category_for_wind,
    class_index_for_wind,
)
from ml import postprocessing as post
from app.services.ibtracs import _intensity_index, imd_category


def test_class_index_and_category_agree_on_detection():
    """For 0..179 kt both views must agree that a cyclone is (or isn't) present."""
    for kt in range(0, 180):
        class_idx = class_index_for_wind(kt)
        category = category_for_wind(kt)
        detected_by_class = class_idx > 0
        detected_by_category = category != "No Cyclone / Depression"
        assert detected_by_class == detected_by_category, (
            f"disagreement at {kt} kt: class={class_idx}, category={category!r}"
        )


def test_single_source_of_truth_is_used_everywhere():
    """All four public entry points must return the same values."""
    for kt in range(0, 180):
        expected_idx = class_index_for_wind(kt)
        expected_cat = category_for_wind(kt)
        assert post.intensity_category(kt) == expected_cat
        assert imd_category(kt) == expected_cat
        assert _intensity_index(kt) == expected_idx


def test_band_table_is_ordered_and_monotonic():
    """WIND_BANDS must descend and cover 0 kt (first match wins)."""
    thresholds = [b[0] for b in WIND_BANDS]
    assert thresholds == sorted(thresholds, reverse=True)
    assert thresholds[-1] == 0.0

    indices = [b[2] for b in WIND_BANDS]
    assert indices == sorted(indices, reverse=True)


def test_band_boundaries_map_to_their_own_row():
    """A wind exactly on a band minimum belongs to that band."""
    for band_min, category, class_index in WIND_BANDS:
        assert category_for_wind(band_min) == category
        assert class_index_for_wind(band_min) == class_index
        # Just below a (non-zero) threshold it must drop to the next band.
        if band_min > 0:
            assert category_for_wind(band_min - 0.1) != category
