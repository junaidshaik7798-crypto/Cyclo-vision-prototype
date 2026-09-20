"""P0-1 regression: the analysis response contract.

`AnalysisResult` used to lack ``analysis_id``/``source``/``image_name``/
``demo_sample``, so Pydantic v2 silently dropped them and the frontend could
not follow up with ``/analyses/{id}`` or ``/cyclone-track/{id}``. These tests
lock the response shape down.

The database is intentionally unavailable in CI (no PostgreSQL), which is a
supported degraded mode: ``analysis_id`` is then ``None`` but the *keys* must
still be present so the client contract does not change.
"""

from __future__ import annotations

import base64
import io

from fastapi.testclient import TestClient
from PIL import Image


def _png(size: int = 32) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), (10, 90, 160)).save(buf, format="PNG")
    return buf.getvalue()


def test_analyze_returns_analysis_id(client: TestClient):
    """POST a small PNG to /api/analyze and check the contract fields."""
    resp = client.post(
        "/api/analyze",
        files={"file": ("upload.PNG", _png(), "image/png")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # P0-1: the keys must exist even when persistence is unavailable.
    assert "analysis_id" in body, body.keys()
    assert body["analysis_id"] is None or isinstance(body["analysis_id"], int)
    assert body["source"] == "upload"
    assert body["image_name"] == "upload.png"  # P2-3 lowercases the extension

    # P1-2: the heatmap is a base64 PNG string, never a nested RGB list.
    heat = body["explainability"]["heatmap_png_b64"]
    assert isinstance(heat, str) and heat
    decoded = base64.b64decode(heat)
    assert decoded.startswith(b"\x89PNG\r\n\x1a\n")
    with Image.open(io.BytesIO(decoded)) as im:
        assert im.mode == "RGB"

    # Core analysis fields still present.
    for key in ("cyclone_detected", "classification", "class_index",
                "estimated_wind_speed_knots", "intensity_category",
                "risk_level", "track", "center"):
        assert key in body, key


def test_analyze_demo_returns_demo_sample(client: TestClient):
    """The demo endpoint must also echo the sample metadata (P0-1)."""
    resp = client.post("/api/analyze/demo", json={})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["demo_sample"] is not None
    assert body["source"] == "demo"
    assert body["image_name"]


def test_analyze_demo_region_is_validated(client: TestClient):
    """P1-4: region must be one in-range [lat, lon] pair."""
    bad = client.post("/api/analyze/demo", json={"region": [12.0, 20.0, 30.0]})
    assert bad.status_code == 422, bad.text

    out_of_range = client.post("/api/analyze/demo", json={"region": [200.0, 90.0]})
    assert out_of_range.status_code == 422, out_of_range.text
