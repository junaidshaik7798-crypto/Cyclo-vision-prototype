"""Final end-to-end verification of the CYCLO-VISION code-review fixes.

Run from the repo root with the backend on sys.path:
    cd backend && python ../scripts/verify_fixes.py
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import urllib.request
from pathlib import Path

# Allow "python scripts/verify_fixes.py" from backend/ (backend on sys.path).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

BASE = os.environ.get("CYCLO_BASE_URL", "http://127.0.0.1:8000")
PNG_SIG = b"\x89PNG\r\n\x1a\n"

results: list[tuple[str, bool, str]] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    results.append((label, bool(condition), detail))


def get_json(path: str) -> dict:
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        return json.loads(r.read())


def post_json(path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


def post_multipart(path: str, filename: str, content: bytes) -> dict:
    boundary = "----cyclovisionboundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode() + content + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        BASE + path, data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


def main() -> int:
    from app.core.config import settings
    from ml.classification import category_for_wind, class_index_for_wind
    from pathlib import Path

    # ---- /api/health : P2-6 -------------------------------------------------
    health = get_json("/api/health")
    check(
        "P2-6 /api/health exposes ibtracs.source",
        health.get("ibtracs", {}).get("source")
        in {"live", "cache", "bundled", "cold", "unavailable"},
        str(health.get("ibtracs")),
    )

    # ---- upload PNG : P0-1 + P1-2 ------------------------------------------
    png = sorted(Path(settings.DEMO_DATA_DIR).glob("*.png"))[0].read_bytes()
    body = post_multipart("/api/analyze", "UPLOAD.PNG", png)
    check("P0-1 analysis_id key present", "analysis_id" in body, repr(body.get("analysis_id")))
    check("P0-1 source present", body.get("source") == "upload", repr(body.get("source")))
    check("P0-1 image_name present", body.get("image_name") == "UPLOAD.png", repr(body.get("image_name")))
    # P2-3 lowercases the *extension* only; the stem is preserved as sent.
    check("P2-3 extension lowercased", (body.get("image_name") or "").endswith(".png"), repr(body.get("image_name")))

    heat = body.get("explainability", {}).get("heatmap_png_b64")
    check("P1-2 heatmap is a string", isinstance(heat, str), type(heat).__name__)
    decoded = base64.b64decode(heat or "")
    check("P1-2 heatmap decodes to PNG", decoded[:8] == PNG_SIG, repr(decoded[:8]))
    check(
        "P1-2 no raw heatmap_rgb key",
        "heatmap_rgb" not in body.get("explainability", {}),
        str(list(body.get("explainability", {}).keys())),
    )
    with Image.open(io.BytesIO(decoded)) as im:
        check("P1-2 PNG opens as RGB image", im.mode == "RGB", f"{im.mode} {im.size}")

    # ---- class/category coherence on the live response : P0-3 --------------
    wind = body["estimated_wind_speed_knots"]
    check(
        "P0-3 live response category matches shared table",
        body["intensity_category"] == category_for_wind(wind),
        f"{wind} kt -> {body['intensity_category']!r}",
    )
    check(
        "P0-3 live response class matches shared table",
        body["class_index"] == class_index_for_wind(wind),
        f"{wind} kt -> class {body['class_index']}",
    )

    # ---- demo with region : P0-2 -------------------------------------------
    demo = post_json("/api/analyze/demo", {"region": [12.5, 88.25]})
    check("P0-2 /demo honours region", demo["center"] == {"lat": 12.5, "lon": 88.25}, str(demo["center"]))
    check("P0-1 /demo returns demo_sample", bool(demo.get("demo_sample")), str((demo.get("demo_sample") or {}).get("id")))

    samples = get_json("/api/analyze/samples")["samples"]
    demo2 = post_json(f"/api/analyze/demo/{samples[0]['id']}", {"region": [-5.0, 175.0]})
    check("P0-2 /demo/{id} honours region", demo2["center"] == {"lat": -5.0, "lon": 175.0}, str(demo2["center"]))

    # ---- checklist wind values : P0-3 --------------------------------------
    from ml.classification import WIND_BANDS

    ok = True
    detail = []
    for kt in (10, 25, 40, 55, 75, 100, 130):
        idx, cat = class_index_for_wind(kt), category_for_wind(kt)
        row = next(b for b in WIND_BANDS if b[1] == cat)
        if row[2] != idx:
            ok = False
        detail.append(f"{kt}->({idx},{cat})")
    check("P0-3 checklist winds share one WIND_BANDS row", ok, "; ".join(detail))

    # ---- report -------------------------------------------------------------
    width = max(len(label) for label, _, _ in results)
    failures = 0
    for label, passed, detail in results:
        if not passed:
            failures += 1
        print(f"{'PASS' if passed else 'FAIL'}  {label:<{width}}  {detail}")
    print(f"\n{len(results) - failures}/{len(results)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
