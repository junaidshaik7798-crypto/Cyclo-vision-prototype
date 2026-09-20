"""
CYCLO-VISION -- Build a synthetic 4-panel figure for testing the preparer.
=========================================================================

Purpose: regression-test ``prepare_reference_dataset`` without needing the
real reference images. It renders a figure in the same layout the project's
reference products use (Raw IR / BT Contours / BT Calibrated / Cloud Height)
and asserts that the preparer picks out the top-left IR panel.

Run:
    python scripts/selftest_prepare.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_reference_dataset import (  # noqa: E402
    auto_crop_quadrant,
    classify_panel,
    prepare,
)


def _raw_ir_panel(h: int, w: int, seed: int, storm: bool) -> np.ndarray:
    """Grayscale-ish IR panel: dark ocean, optional bright cold cloud top."""
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 40.0 + rng.normal(0, 6, (h, w))
    if storm:
        cy, cx = h * 0.45, w * 0.5
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        # cold (bright) cloud shield with a warm eye
        shield = np.exp(-((d / (min(h, w) * 0.42)) ** 2))
        eye = np.exp(-((d / (min(h, w) * 0.07)) ** 2))
        base = base + shield * 190.0 - eye * 150.0
    base = np.clip(base, 0, 255)
    # Raw IR is displayed greyscale, so replicate across channels.
    return np.repeat(base[..., None], 3, axis=2).astype(np.uint8)


def _colour_panel(h: int, w: int, seed: int) -> np.ndarray:
    """Vivid blue/red panel standing in for the BT-calibrated product."""
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2)
    cold = np.exp(-((d / (min(h, w) * 0.30)) ** 2))
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[..., 0] = 220 * (1 - cold) + 20 * cold       # red background
    rgb[..., 1] = 60 * (1 - cold) + 40 * cold
    rgb[..., 2] = 30 * (1 - cold) + 230 * cold       # blue core
    rgb += rng.normal(0, 4, rgb.shape)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _contour_panel(h: int, w: int, seed: int) -> np.ndarray:
    """Greyscale with thin dark contour lines."""
    gray = _raw_ir_panel(h, w, seed, storm=True).mean(axis=2)
    lines = (np.sin(gray / 8.0) > 0.985).astype(np.uint8) * 90
    out = np.clip(gray * 0.9 + lines, 0, 255)
    return np.repeat(out[..., None], 3, axis=2).astype(np.uint8)


def _cloud_height_panel(h: int, w: int, seed: int) -> np.ndarray:
    """Greyscale 3-D surface view -- dark base, tall grey spire, axis lines.

    Modelled on the real "Cloud Height (km)" panel: most of the frame is a
    near-black ground plane, with a bright surface rising in the middle and
    thin axis/grid lines down each edge.
    """
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # Ground plane: very dark, gently sloping (the 3-D floor).
    out = 8.0 + 12.0 * (yy / h) + rng.normal(0, 3, (h, w))

    # A spire of elevated cloud tops, brightest at the peak.
    cx, cy = w * 0.5, h * 0.55
    d = np.sqrt(((xx - cx) / (w * 0.16)) ** 2 + ((yy - cy) / (h * 0.28)) ** 2)
    spire = np.clip(1.0 - d, 0.0, 1.0) ** 1.6
    out = out + spire * 235.0

    # Axis / grid lines: bright hairlines along the frame edges.
    for x in range(0, w, max(8, w // 12)):
        out[:, x] += 55.0
    for y in range(0, h, max(8, h // 12)):
        out[y, :] += 35.0

    out = np.clip(out, 0, 255)
    return np.repeat(out[..., None], 3, axis=2).astype(np.uint8)


def build_figure(px: int = 320, seed: int = 0) -> np.ndarray:
    """Assemble a 4-panel figure on a white background."""
    gap, margin = 18, 16
    total_w = margin * 2 + px * 2 + gap
    total_h = margin * 2 + px * 2 + gap
    fig = np.full((total_h, total_w, 3), 255, dtype=np.uint8)

    panels = [
        (0, 0, _raw_ir_panel(px, px, seed, storm=True)),
        (0, 1, _contour_panel(px, px, seed + 10)),
        (1, 0, _colour_panel(px, px, seed + 20)),
        (1, 1, _cloud_height_panel(px, px, seed + 30)),
    ]
    for row, col, panel in panels:
        y0 = margin + row * (px + gap)
        x0 = margin + col * (px + gap)
        fig[y0 : y0 + px, x0 : x0 + px] = panel[:px, :px]
    return fig


def main() -> int:
    failures: list[str] = []

    print("=" * 78)
    print("SELFTEST -- prepare_reference_dataset")
    print("=" * 78)

    # --- 1. auto crop must isolate the grayscale IR panel ----------------
    print("\n[1] auto_crop_quadrant on a synthetic 4-panel figure")
    for seed in (0, 1, 2):
        fig = build_figure(px=320, seed=seed)
        panel = auto_crop_quadrant(fig)
        if panel is None:
            failures.append(f"seed {seed}: auto crop returned None")
            print(f"  seed {seed}: FAIL -- no panel located")
            continue
        ok, reason = classify_panel(panel)
        if not ok:
            failures.append(f"seed {seed}: {reason}")
        print(f"  seed {seed}: {'PASS' if ok else 'FAIL'} -- shape={panel.shape} {reason}")

    # --- 2. the preparer must REJECT a whole figure ---------------------
    print("\n[2] reject a full figure passed as if it were a tile")
    fig = build_figure(px=320, seed=0)
    ok, reason = classify_panel(fig)
    if ok:
        failures.append("whole figure was accepted as an IR panel")
    print(f"  {'PASS' if not ok else 'FAIL'} -- {reason}")

    # --- 3. the colour / contour panels must be rejected ----------------
    print("\n[3] reject non-IR panels")
    for name, panel in (
        ("BT calibrated (vivid colour)", _colour_panel(256, 256, 7)),
        ("cloud-height surface", _cloud_height_panel(256, 256, 7)),
    ):
        ok, reason = classify_panel(panel)
        if ok:
            failures.append(f"{name} was accepted")
        print(f"  {name:<30} {'PASS' if not ok else 'FAIL'} -- {reason}")

    # --- 4. end-to-end prepare() on a temp tree -------------------------
    print("\n[4] end-to-end prepare() with folder-based labels")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "raw"
        out = Path(tmp) / "labeled"
        for label, count in (("Cyclonic Storm", 3), ("Severe Cyclonic Storm", 3)):
            d = src / label
            d.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                seed = (abs(hash((label, i))) % 9999)
                Image.fromarray(build_figure(px=200, seed=seed)).save(d / f"fig_{i}.png")

        manifest = prepare(src, out, crop=None, dry_run=False, keep_panels=None)
        total = sum(manifest["per_class"].values())
        expected = 6
        if total != expected:
            failures.append(f"prepare wrote {total} tiles, expected {expected}")
        print(f"  {'PASS' if total == expected else 'FAIL'} -- wrote {total}/{expected} tiles")
        for label, count in sorted(manifest["per_class"].items()):
            on_disk = len(list((out / label).glob("*.png")))
            if on_disk != count:
                failures.append(f"{label}: manifest={count} files={on_disk}")
            print(f"      {label:<28} {count}")

        # every written tile must still pass verification and be 256x256
        written = sorted(out.rglob("*.png"))
        for png in written:
            with Image.open(png) as im:
                arr = np.asarray(im.convert("RGB"))
            if arr.shape[:2] != (256, 256):
                failures.append(f"{png.name}: wrong size {arr.shape[:2]}")
            ok, reason = classify_panel(arr)
            if not ok:
                failures.append(f"{png.name}: written tile failed verify ({reason})")
        print(f"  verification sweep over {len(written)} written tiles done")

    print("\n" + "=" * 78)
    if failures:
        print(f"SELFTEST FAILED -- {len(failures)} problem(s):")
        for f in failures:
            print(f"  - {f}")
        print("=" * 78)
        return 1
    print("SELFTEST PASSED -- crop + verification + export all behave.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())