"""Browser-level check: the analysis map must be visible and accurate.

Runs the built dashboard (``vite preview`` on :4173 by default) in headless
Chrome via Playwright and asserts the properties users actually see:

  1. VISIBILITY -- analysing a cyclone image renders the real Leaflet map
     (centre marker, track, evacuation circle, basemap tiles) and never the
     schematic SVG fallback.
  2. ACCURACY -- an upload with a supplied storm location pins the marker
     tooltip and the risk-card readout to exactly those coordinates.
  3. ENDPOINTS -- the forecast track carries permanent "Start" and "End"
     labels, and the Start label is centred on the storm marker (the track
     begins where the cyclone is starting).

Usage (from the repo root, with the preview server and backend running):

    python scripts/verify_map_render.py [base_url]

Requires ``pip install playwright`` and a system Chrome or Edge. Screenshots
are written to logs/.
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parents[1]
LOGS = REPO / "logs"
UPLOAD_IMAGE = REPO / "data" / "demo" / "sample_severe.png"

failures: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {label}" + (f" -- {detail}" if detail else ""))
    if not ok:
        failures.append(label)


def launch(pw, browser_name: str):
    return pw.chromium.launch(channel=browser_name, headless=True)


def wait_for_map(page) -> None:
    page.wait_for_selector(".cyclone-map.leaflet-container", timeout=20_000)
    page.wait_for_selector(".cyclone-div-icon .cyclone-pulse", timeout=10_000)


def marker_inside_map(page) -> bool:
    """The centre marker must actually be positioned inside the map frame."""
    box_map = page.locator(".cyclone-map").bounding_box()
    box_icon = page.locator(".cyclone-div-icon").first.bounding_box()
    if not box_map or not box_icon:
        return False
    cx = box_icon["x"] + box_icon["width"] / 2
    cy = box_icon["y"] + box_icon["height"] / 2
    return (
        box_map["x"] <= cx <= box_map["x"] + box_map["width"]
        and box_map["y"] <= cy <= box_map["y"] + box_map["height"]
    )


def endpoint_boxes(page) -> dict:
    """Snapshot map / centre-marker / Start / End label boxes in ONE JS tick.

    Measuring everything in a single evaluation keeps the relative positions
    valid even if the map is still settling after fitBounds().
    """
    return page.evaluate(
        """
        () => {
          const box = (el) => {
            if (!el) return null;
            const r = el.getBoundingClientRect();
            return {
              cx: r.x + r.width / 2,
              cy: r.y + r.height / 2,
              text: (el.textContent || '').trim(),
            };
          };
          const m = document.querySelector('.cyclone-map');
          const r = m ? m.getBoundingClientRect() : null;
          const rect = (el) => {
            if (!el) return null;
            const b = el.getBoundingClientRect();
            return { x: b.x, y: b.y, w: b.width, h: b.height };
          };
          return {
            map: r ? { x: r.x, y: r.y, w: r.width, h: r.height } : null,
            legend: rect(document.querySelector('.map-legend')),
            marker: box(document.querySelector('.cyclone-div-icon')),
            start: box(document.querySelector('.track-endpoint-start .track-endpoint-text')),
            end: box(document.querySelector('.track-endpoint-end .track-endpoint-text')),
          };
        }
        """
    )


def legend_off_the_map(ep: dict) -> tuple[bool, str]:
    """The legend must not overlap the map frame at all.

    Leaflet paints every map layer inside .leaflet-map-pane (z-index 400) as
    one block, so a legend drawn *on* the map can only ever be painted over
    (clipping the permanent Start/End track pills and the zoom control) or
    behind (hidden by the tiles). Keeping it outside the map frame is the
    one layout where the pills, the tiles and the legend are all visible.
    """
    lg, mp = ep.get("legend"), ep.get("map")
    if not lg or not mp:
        return False, "missing legend or map"
    overlap = not (
        lg["x"] + lg["w"] <= mp["x"]
        or mp["x"] + mp["w"] <= lg["x"]
        or lg["y"] + lg["h"] <= mp["y"]
        or mp["y"] + mp["h"] <= lg["y"]
    )
    return (
        not overlap,
        f"legend {lg['w']:.0f}x{lg['h']:.0f} at ({lg['x']:.0f},{lg['y']:.0f}) vs "
        f"map {mp['w']:.0f}x{mp['h']:.0f} at ({mp['x']:.0f},{mp['y']:.0f})",
    )


def start_labels(ep: dict) -> tuple[bool, str]:
    """Start label must exist and read 'Start ... cyclone now'."""
    s = ep.get("start")
    if not s:
        return False, "missing"
    ok = "Start" in s["text"] and "cyclone now" in s["text"]
    return ok, s["text"]


def end_labels(ep: dict) -> tuple[bool, str]:
    """End label must exist and name the final forecast hour (+48h)."""
    e = ep.get("end")
    if not e:
        return False, "missing"
    ok = "End" in e["text"] and "+48h" in e["text"]
    return ok, e["text"]


def start_on_marker(ep: dict) -> tuple[bool, str]:
    """The track's Start pill sits exactly on the storm marker (same point)."""
    s, m = ep.get("start"), ep.get("marker")
    if not s or not m:
        return False, "missing label or marker"
    dx = s["cx"] - m["cx"]
    dy = s["cy"] - m["cy"]
    # Horizontally centred on the marker; vertically the pill hangs below it
    # (the permanent centre tooltip owns the space above).
    ok = abs(dx) <= 4 and 0 <= dy <= 48
    return ok, f"dx={dx:.1f}px dy={dy:.1f}px"


def end_inside_map(ep: dict) -> tuple[bool, str]:
    """The track's End pill must be visible inside the map frame."""
    e, mp = ep.get("end"), ep.get("map")
    if not e or not mp:
        return False, "missing label or map"
    ok = (
        mp["x"] <= e["cx"] <= mp["x"] + mp["w"]
        and mp["y"] <= e["cy"] <= mp["y"] + mp["h"]
    )
    return ok, f"at ({e['cx']:.0f},{e['cy']:.0f}) of {mp['w']:.0f}x{mp['h']:.0f}"


def run(base_url: str) -> None:
    with sync_playwright() as pw:
        try:
            browser = launch(pw, "chrome")
        except Exception:
            browser = launch(pw, "msedge")
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.goto(base_url, wait_until="load", timeout=30_000)

        # ---- 1. Demo analysis -> the real map must render -------------------
        page.wait_for_selector(".demo-sample", timeout=20_000)
        page.locator(".demo-sample").first.click()
        wait_for_map(page)

        check(
            page.locator(".track-map").count() == 0,
            "demo: real Leaflet map shown (SVG fallback not used)",
        )
        box = page.locator(".cyclone-map").bounding_box()
        check(
            bool(box and box["height"] >= 300),
            "demo: map frame visible",
            f"height={box['height'] if box else 0}",
        )
        check(marker_inside_map(page), "demo: centre marker positioned in map")
        check(
            page.locator(".cyclone-map-wrap path.leaflet-interactive").count() >= 1,
            "demo: evacuation zone / track vectors drawn",
        )
        # Basemap: either tiles load (online) or the offline notice shows.
        try:
            page.wait_for_function(
                "() => [...document.querySelectorAll('img.leaflet-tile')]"
                ".filter(i => i.naturalWidth > 0).length >= 3",
                timeout=15_000,
            )
            tiles_ok = True
        except Exception:
            tiles_ok = False
        offline_notice = page.locator(".tile-warn").count() > 0
        check(
            tiles_ok or offline_notice,
            "demo: basemap visible (tiles) or offline notice shown",
            f"tiles_loaded={tiles_ok} offline_notice={offline_notice}",
        )
        check(page.locator(".map-legend").count() == 1, "demo: map legend visible")
        tooltip = page.locator(".cyclone-map .leaflet-tooltip").first.inner_text()
        check(
            "Cyclone centre" in tooltip,
            "demo: permanent centre tooltip",
            tooltip.replace("\n", " "),
        )
        ep = endpoint_boxes(page)
        ok, detail = start_labels(ep)
        check(ok, "demo: track start label ('Start — cyclone now')", detail)
        ok, detail = end_labels(ep)
        check(ok, "demo: track end label ('End — +48h forecast')", detail)
        ok, detail = start_on_marker(ep)
        check(ok, "demo: start label centred on the storm marker", detail)
        ok, detail = end_inside_map(ep)
        check(ok, "demo: end label visible inside the map", detail)
        ok, detail = legend_off_the_map(ep)
        check(ok, "demo: legend sits outside the map (never clips labels)", detail)
        page.screenshot(path=str(LOGS / "map_render_demo.png"), full_page=True)

        # ---- 2. Upload + supplied storm location -> exact coordinates -------
        page.locator("#storm-region-input").fill("17.5, 88.3")
        page.set_input_files("input[type=file]", str(UPLOAD_IMAGE))
        wait_for_map(page)
        page.wait_for_function(
            "() => ((document.querySelector('.leaflet-tooltip') || {}).textContent || '')"
            ".includes('17.50')",
            timeout=15_000,
        )
        tooltip = page.locator(".cyclone-map .leaflet-tooltip").first.inner_text()
        check(
            "17.50°N" in tooltip and "88.30°E" in tooltip,
            "upload: marker tooltip pinned to 17.50°N 88.30°E",
            tooltip.replace("\n", " "),
        )
        risk = page.locator(".risk-card").inner_text()
        check(
            "17.50°N, 88.30°E" in risk,
            "upload: risk-card readout matches",
            risk.replace("\n", " | "),
        )
        check(
            "from your storm location" in risk,
            "upload: readout labelled as user-provided",
        )
        check(marker_inside_map(page), "upload: centre marker positioned in map")
        check(
            page.locator(".track-map").count() == 0,
            "upload: real Leaflet map shown (SVG fallback not used)",
        )
        ep = endpoint_boxes(page)
        ok, detail = start_labels(ep)
        check(ok, "upload: track start label present", detail)
        ok, detail = end_labels(ep)
        check(ok, "upload: track end label present", detail)
        ok, detail = start_on_marker(ep)
        check(ok, "upload: start label sits on the supplied location", detail)
        ok, detail = end_inside_map(ep)
        check(ok, "upload: end label visible inside the map", detail)
        page.screenshot(path=str(LOGS / "map_render_upload_region.png"), full_page=True)

        browser.close()


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:4173/"
    LOGS.mkdir(exist_ok=True)
    try:
        run(url)
    except Exception as exc:  # noqa: BLE001 - surface any driver/timeout issue
        failures.append(f"exception: {exc}")
        print(f"[FAIL] render check crashed: {exc}")
    if failures:
        print(f"\n{len(failures)} check(s) failed.")
        sys.exit(1)
    print("\nAll map render checks passed.")
