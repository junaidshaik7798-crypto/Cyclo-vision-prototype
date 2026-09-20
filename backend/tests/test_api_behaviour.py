"""Verification-checklist tests for the code-review fixes.

These cover the end-to-end acceptance checks that are easy to run from a
terminal but deserve a permanent regression guard:

  * ``/api/health`` exposes ``ibtracs.source`` (P2-6)
  * ``/api/analyze/demo`` honours an explicit ``region`` (P0-2)
  * the heatmap payload is a base64 string, not a list (P1-2)
  * class index + category come from the *same* WIND_BANDS row (P0-3)
  * demo sample listing is cached / does not rescan (P1-3)
"""

from __future__ import annotations

import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ml.classification import WIND_BANDS, category_for_wind, class_index_for_wind

CHECK_WINDS = (10, 25, 40, 55, 75, 100, 130)


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (48, 48), (30, 30, 30)).save(buf, format="PNG")
    return buf.getvalue()


def test_health_reports_ibtracs(client: TestClient):
    resp = client.get("/api/health")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "ibtracs" in body
    assert body["ibtracs"]["source"] in {
        "live",
        "cache",
        "bundled",
        "cold",
        "unavailable",
    }


def test_analyze_demo_respects_region(client: TestClient):
    """P0-2: /analyze/demo must use the supplied region as the storm centre."""
    region = [12.5, 88.25]
    resp = client.post("/api/analyze/demo", json={"region": region})
    assert resp.status_code == 200, resp.text
    center = resp.json()["center"]
    assert center["lat"] == pytest.approx(region[0])
    assert center["lon"] == pytest.approx(region[1])


def test_analyze_demo_sample_respects_region(client: TestClient):
    samples = client.get("/api/analyze/samples").json()["samples"]
    assert samples, "no bundled demo samples available"
    region = [-5.0, 175.0]
    resp = client.post(
        f"/api/analyze/demo/{samples[0]['id']}", json={"region": region}
    )
    assert resp.status_code == 200, resp.text
    center = resp.json()["center"]
    assert center["lat"] == pytest.approx(region[0])
    assert center["lon"] == pytest.approx(region[1])


def test_heatmap_payload_is_small_base64_string(client: TestClient):
    """P1-2: base64 PNG must be far smaller than the old RGB array."""
    resp = client.post(
        "/api/analyze", files={"file": ("shot.png", _png(), "image/png")}
    )
    assert resp.status_code == 200, resp.text
    heat = resp.json()["explainability"]["heatmap_png_b64"]
    assert isinstance(heat, str)
    assert len(heat) < 500_000, "heatmap payload is suspiciously large"
    assert base64.b64decode(heat).startswith(b"\x89PNG")


def test_class_and_category_share_a_band_row():
    """For each checklist wind, the class index and category come from one row."""
    for kt in CHECK_WINDS:
        class_idx = class_index_for_wind(kt)
        category = category_for_wind(kt)
        rows = [(b, c, i) for b, c, i in WIND_BANDS if c == category]
        assert rows, f"{category!r} is not a WIND_BANDS label"
        assert any(i == class_idx for _, _, i in rows), (
            f"{kt} kt -> class {class_idx} but category {category!r} "
            "belongs to a different band row"
        )


def test_demo_samples_are_cached_across_calls():
    """P1-3: repeated listings must reuse the cached scan."""
    from app.services import demo_data

    first = demo_data.list_demo_samples()
    second = demo_data.list_demo_samples()
    assert first is second, "list_demo_samples rescanned instead of using cache"

    # get_demo_bytes must resolve via the cached listing, not a fresh scan.
    calls = {"n": 0}
    original_scan = demo_data._scan

    def counting_scan():
        calls["n"] += 1
        return original_scan()

    demo_data._scan = counting_scan
    try:
        if first:
            data, filename, meta = demo_data.get_demo_bytes(first[0]["id"])
            assert data and filename == first[0]["filename"]
            assert meta["id"] == first[0]["id"]
    finally:
        demo_data._scan = original_scan
    assert calls["n"] == 0, "get_demo_bytes triggered a directory rescan"


def test_reference_without_pressure_record_does_not_crash():
    """IBTrACS storms may have min_pressure_hpa = None (pre-satellite era).

    Calibrating against such a storm must fall back to the documented class
    curve instead of raising TypeError on float(None). The reference is
    forced for *any* class so the test does not depend on which demo image
    happens to be listed first (sample_clear.png is class 0 / no cyclone).
    """
    import ml.inference as inference

    class _NoPressureRef:
        name = "OLD-UNNAMED"
        year = 1935
        max_wind_knots = 95.0
        min_pressure_hpa = None

    original = inference._live_reference
    # `run_inference` resolves `_live_reference` from its own module globals,
    # so patching the module attribute is what takes effect.
    inference._live_reference = lambda class_idx: _NoPressureRef()
    try:
        with open(_demo_png_path(), "rb") as fh:
            result = inference.run_inference(fh.read())
    finally:
        inference._live_reference = original

    assert result["reference_cyclone"] == "OLD-UNNAMED"
    assert 870.0 <= result["estimated_pressure_hpa"] <= 1010.0
    assert result["estimated_wind_speed_knots"] > 0


def test_best_match_prefers_storm_with_pressure_record():
    """The band-centre anchor must still avoid pressure-less storms."""
    from app.services.ibtracs import IBTrACSStorm, best_match_for_class

    def storm(name, wind, pressure):
        return IBTrACSStorm(
            sid=name, name=name, year=2000, basin="North Indian Ocean",
            category="", max_wind_knots=wind, min_pressure_hpa=pressure,
            genesis_lat=10.0, genesis_lon=88.0, landfall_lat=12.0,
            landfall_lon=86.0, track_direction_deg=300.0,
            forward_speed_knots=8.0, intensity_index=5, notes="",
        )

    import app.services.ibtracs as ib

    # Class 5 band centre is 64 kt; both storms are 2 kt away, but only one
    # has a pressure record, so that one must win.
    pool = [storm("NO-PRES", 62.0, None), storm("HAS-PRES", 66.0, 970.0)]
    original_named, original_load = ib.get_named_only, ib.load_storms
    ib.get_named_only = lambda: pool
    try:
        chosen = best_match_for_class(5)
    finally:
        ib.get_named_only, ib.load_storms = original_named, original_load

    assert chosen is not None
    assert chosen.name == "HAS-PRES"
    assert chosen.min_pressure_hpa is not None


def _demo_png_path() -> str:
    """Absolute path to a bundled demo PNG (skips if absent)."""
    from app.core.config import settings
    from pathlib import Path

    root = Path(settings.DEMO_DATA_DIR)
    candidates = sorted(root.glob("*.png"))
    if not candidates:
        pytest.skip("no bundled demo PNG available")
    return str(candidates[0])


def test_safe_filename_lowercases_extension():
    """P2-3: uppercase/mixed-case extensions survive validate_upload."""
    from app.api.routes.analyze import _safe_filename

    assert _safe_filename("IMG.PNG") == "IMG.png"
    assert _safe_filename("Pic.Jpg") == "Pic.jpg"
    assert _safe_filename(r"C:\tmp\storm.TIFF") == "storm.tiff"
    assert _safe_filename("no_extension") == "no_extension"