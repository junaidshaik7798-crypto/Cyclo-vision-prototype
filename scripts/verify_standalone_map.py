"""Browser-level check: the standalone results page labels track endpoints.

Drives the no-build standalone flow (input page -> results page) against the
Live Server on :5500 with a backend on :8000, then asserts:

  * the permanent "Start — cyclone now" / "End — +48h forecast" pills render,
  * the Start pill is centred on the storm marker,
  * the legend sits outside the map frame (so it can never clip the pills).

Usage (backend on :8000, ``frontend/standalone`` served on :5500):

    python scripts/verify_standalone_map.py [base_url]

Screenshot: logs/standalone_endpoints.png
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parents[1]
LOGS = REPO / "logs"

failures: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
    if not ok:
        failures.append(label)


def run(base_url: str) -> None:
    with sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(channel="chrome", headless=True)
        except Exception:
            browser = pw.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.goto(base_url, wait_until="load", timeout=30_000)

        # Run an analysis pinned to a known storm location.
        page.wait_for_selector(".sample", timeout=20_000)
        page.fill("#region", "17.5, 88.3")
        page.locator(".sample").first.click()
        page.click("#analyze")
        page.wait_for_url("**/results.html", timeout=30_000)
        page.wait_for_selector(".cyclone-div-icon", timeout=20_000)
        page.wait_for_timeout(1200)

        snap = page.evaluate(
            """
            () => {
              const rect = (sel) => {
                const el = document.querySelector(sel);
                if (!el) return null;
                const b = el.getBoundingClientRect();
                return { x: b.x, y: b.y, w: b.width, h: b.height,
                         text: (el.textContent || '').trim() };
              };
              const c = (r) => (r ? { cx: r.x + r.w / 2, cy: r.y + r.h / 2 } : null);
              const map = rect('#cyclone-map');
              const legend = rect('.map-legend');
              const start = rect('.track-endpoint-start .track-endpoint-text');
              const marker = rect('.cyclone-div-icon');
              let overlap = null;
              if (map && legend) {
                overlap = !(
                  legend.x + legend.w <= map.x || map.x + map.w <= legend.x ||
                  legend.y + legend.h <= map.y || map.y + map.h <= legend.y
                );
              }
              const sc = c(start), mc = c(marker);
              return {
                map, legend, overlap, start,
                end: rect('.track-endpoint-end .track-endpoint-text'),
                dx: sc && mc ? sc.cx - mc.cx : null,
                dy: sc && mc ? sc.cy - mc.cy : null,
              };
            }
            """
        )

        start = snap.get("start")
        end = snap.get("end")
        check(
            bool(start) and "Start" in start["text"] and "cyclone now" in start["text"],
            "standalone: start label present",
            start["text"] if start else "missing",
        )
        check(
            bool(end) and "End" in end["text"] and "+48h" in end["text"],
            "standalone: end label present",
            end["text"] if end else "missing",
        )
        if snap["dx"] is None:
            check(False, "standalone: start label aligned with marker", "no boxes")
        else:
            ok = abs(snap["dx"]) <= 4 and 0 <= snap["dy"] <= 48
            check(
                ok,
                "standalone: start label aligned with marker",
                f"dx={snap['dx']:.1f}px dy={snap['dy']:.1f}px",
            )
        check(
            snap["overlap"] is False,
            "standalone: legend outside the map (never clips labels)",
            (
                f"legend {snap['legend']['w']:.0f}x{snap['legend']['h']:.0f} vs "
                f"map {snap['map']['w']:.0f}x{snap['map']['h']:.0f}"
                if snap["legend"] and snap["map"]
                else "missing box"
            ),
        )
        page.screenshot(path=str(LOGS / "standalone_endpoints.png"), full_page=True)
        browser.close()


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5500/index.html"
    LOGS.mkdir(exist_ok=True)
    try:
        run(url)
    except Exception as exc:  # noqa: BLE001 - surface driver/timeout issues
        failures.append(f"exception: {exc}")
        print(f"[FAIL] standalone check crashed: {exc}")
    if failures:
        print(f"\n{len(failures)} check(s) failed.")
        sys.exit(1)
    print("\nAll standalone map checks passed.")
