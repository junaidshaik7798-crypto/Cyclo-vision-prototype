"""
CYCLO-VISION - End-to-End Demo Verification
===========================================
Exercises every user-facing path against a running server and prints a
single consolidated report. Safe to re-run; it never mutates state.

Usage
-----
    # terminal 1
    cd backend && python -m uvicorn app.main:app --port 8000
    # terminal 2
    python scripts/verify_full_stack.py

Pass ``--base http://localhost:5173`` to point at the Vite dev server
(which proxies /api and /data to the backend) instead of the backend.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import requests

TIMEOUT = 120


def _hr(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _get(base: str, path: str, **params: Any) -> Any:
    r = requests.get(f"{base}{path}", params=params or None, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def _post(base: str, path: str) -> Any:
    r = requests.post(f"{base}{path}", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    args = ap.parse_args()
    base = args.base.rstrip("/")

    failures: list[str] = []

    def check(label: str, fn) -> Any:
        try:
            val = fn()
            print(f"  [OK]   {label}")
            return val
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {label}: {type(exc).__name__}: {exc}")
            failures.append(label)
            return None

    # ------------------------------------------------------------------
    _hr("1. SERVICE HEALTH")
    health = check("GET /api/health", lambda: _get(base, "/api/health"))
    if health:
        for k in ("status", "app", "version", "ml_mode", "model_available", "database"):
            print(f"         {k:<18} = {health.get(k)}")

    # ------------------------------------------------------------------
    _hr("2. LIVE IBTrACS DATASET (real NOAA best-track archive)")
    status = check("GET /api/ibtracs/status", lambda: _get(base, "/api/ibtracs/status"))
    if status:
        for k in ("source", "records", "year_range", "cache_size_mb", "license"):
            print(f"         {k:<18} = {status.get(k)}")

    for label, ep in (("RECENT", "/api/ibtracs/recent"), ("STRONGEST", "/api/ibtracs/intense")):
        data = check(f"GET {ep}", lambda e=ep: _get(base, e, limit=10))
        if not data:
            continue
        print()
        print(f"         {label}")
        print(f"         {'#':>3}  {'NAME':<13} {'YEAR':<5} {'WIND':>8} {'PRES':>8}  CATEGORY")
        for i, s in enumerate(data.get("storms", [])[:10], 1):
            print(
                f"         {i:>3}  {s['name']:<13} {s['year']:<5} "
                f"{s['max_wind_knots']:>6.1f}kt {s['min_pressure_hpa']:>6.0f}hPa  {s['category']}"
            )

    # ------------------------------------------------------------------
    _hr("3. BUNDLED REFERENCE DATASET")
    ref = check("GET /api/reference-dataset", lambda: _get(base, "/api/reference-dataset"))
    if ref:
        print(f"         summary = {json.dumps(ref.get('summary'), default=str)[:240]}")

    # ------------------------------------------------------------------
    _hr("4. DATA SOURCES")
    src = check("GET /api/data-sources", lambda: _get(base, "/api/data-sources"))
    if src:
        for s in src.get("sources", []):
            print(f"         - {str(s.get('id', '?')):<22} {s.get('name', '')}")

    # ------------------------------------------------------------------
    _hr("5. DEMO SAMPLES + ANALYSIS (calibrated vs real IBTrACS storms)")
    samples = check("GET /api/analyze/samples", lambda: _get(base, "/api/analyze/samples"))
    ids: list[str] = []
    if samples:
        for s in samples.get("samples", []):
            ids.append(s["id"])
            print(f"         - {s['id']:<18} {s.get('name', '')}")

    if ids:
        print()
        print(
            f"         {'SAMPLE':<18} {'CLASSIFICATION':<32} {'CONF':>6} "
            f"{'WIND':>8} {'PRES':>8} {'RISK':<9} {'CALIB':<15} REFERENCE"
        )
        print("         " + "-" * 120)
        rows: list[tuple[str, Any]] = []
        for sid in ids:
            try:
                r = _post(base, f"/api/analyze/demo/{sid}")
            except Exception as exc:  # noqa: BLE001
                print(f"         {sid:<18} FAILED: {exc}")
                failures.append(f"analyze {sid}")
                continue
            ref_txt = (
                f"{r.get('reference_cyclone')} ({r.get('reference_year')})"
                if r.get("reference_cyclone")
                else "-"
            )
            print(
                f"         {sid:<18} {r['classification']:<32} {r['confidence']:>6.3f} "
                f"{r['estimated_wind_speed_knots']:>6.1f}kt {r['estimated_pressure_hpa']:>6.0f}hPa "
                f"{r['risk_level']:<9} {str(r.get('calibration_source') or '-'):<15} {ref_txt}"
            )
            rows.append((sid, r))

        if rows:
            winds = [r["estimated_wind_speed_knots"] for _, r in rows]
            classes = len({r["classification"] for _, r in rows})
            print()
            print(f"         distinct classes : {classes} of {len(rows)}")
            print(f"         wind monotonic   : {all(b >= a for a, b in zip(winds, winds[1:]))}")
            print(f"         wind range       : {min(winds):.1f} - {max(winds):.1f} kt")

            _, strong = rows[-1]
            for k, ok in {
                "risk_factors": bool(strong.get("risk_factors")),
                "heatmap_rgb": bool(strong.get("explainability", {}).get("heatmap_rgb")),
                "track": bool(strong.get("track")),
                "center": bool(strong.get("center")),
                "calibration_source": bool(strong.get("calibration_source")),
            }.items():
                print(f"         payload {k:<20} {'present' if ok else 'MISSING'}")
                if not ok:
                    failures.append(f"payload {k}")

    # ------------------------------------------------------------------
    _hr("6. DEMO IMAGE SERVING")
    for name in ("sample_clear.png", "sample_dev.png", "sample_severe.png", "sample_vsevere.png"):
        try:
            r = requests.get(f"{base}/data/demo/{name}", timeout=TIMEOUT)
            r.raise_for_status()
            print(
                f"  [OK]   /data/demo/{name:<20} {r.status_code}  "
                f"{len(r.content):>7} bytes  {r.headers.get('Content-Type')}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] /data/demo/{name}: {exc}")
            failures.append(f"image {name}")

    # ------------------------------------------------------------------
    _hr("RESULT")
    if failures:
        print(f"  {len(failures)} CHECK(S) FAILED:")
        for f in failures:
            print(f"    - {f}")
        return 1
    print("  ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())