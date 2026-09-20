"""
CYCLO-VISION -- Prepare a trainable dataset from reference analysis figures.
==========================================================================

The reference figures supplied for this project are 4-panel *analysis
products*, laid out as::

    +----------------------+----------------------+
    | Raw Image            | Brightness Temp      |
    | (the TIR channel)    | Contours             |
    +----------------------+----------------------+
    | BT Calibrated Image  | Cloud Height (km)    |
    | (colour-mapped)      | (3-D surface)        |
    +----------------------+----------------------+
    LAT: 10.65 N  LON: 80.51 E     12DEC20130400

Only the top-left "Raw Image" panel is usable as CNN training data. The
other three are derived renderings of the same field, and every panel is
surrounded by matplotlib furniture (colourbars, contour labels, the
"LAT: .. LON: .." caption). Training on an uncropped figure teaches the
model to read the caption text instead of the storm -- textbook leakage.

This script therefore:

    1. finds the figure's plot area,
    2. crops the top-left quadrant,
    3. **verifies the crop is monochrome IR data** (the raw panel is
       effectively grayscale; the bottom-left panel is a vivid
       blue/red colour map, and the colourbars are saturated strips),
    4. resizes to the model input size and writes labelled PNGs.

Anything that fails verification is reported and skipped -- the script
never guesses. Use ``--crop`` to override the automatic quadrant split
for figures with an unusual layout.

Usage
-----
    # inspect without writing anything
    python scripts/prepare_reference_dataset.py --source data/reference/raw --dry-run

    # real run
    python scripts/prepare_reference_dataset.py --source data/reference/raw

    # manual crop for odd layouts (left, top, right, bottom as 0-1 fractions)
    python scripts/prepare_reference_dataset.py --crop 0.06 0.02 0.55 0.50
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent

DEFAULT_SOURCE = PROJECT_ROOT / "data" / "reference" / "raw"
DEFAULT_OUT = PROJECT_ROOT / "data" / "reference" / "labeled"
MANIFEST = PROJECT_ROOT / "data" / "reference" / "manifest.json"

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# Model input size -- must match settings.INPUT_SIZE so training and
# inference see the same resolution.
INPUT_SIZE = 256


# ---------------------------------------------------------------------------
# Colour analysis helpers
# ---------------------------------------------------------------------------


def _saturation(rgb: np.ndarray) -> float:
    """Mean HSV saturation (0-1) of an RGB uint8 array."""
    f = rgb.astype(np.float32) / 255.0
    mx = f.max(axis=2)
    mn = f.min(axis=2)
    denom = np.where(mx > 1e-6, mx, 1.0)
    return float(np.mean((mx - mn) / denom))


def _colour_energy(rgb: np.ndarray) -> float:
    """Fraction of "vivid" pixels (high saturation, non-black).

    The BT-calibrated panel is dominated by strong red/blue, the contour
    panel has coloured contours, and every colourbar is a saturated strip.
    The raw IR panel is near-grayscale, so this fraction stays low.
    """
    f = rgb.astype(np.float32) / 255.0
    mx = f.max(axis=2)
    mn = f.min(axis=2)
    denom = np.where(mx > 1e-6, mx, 1.0)
    sat = (mx - mn) / denom
    vivid = (sat > 0.45) & (mx > 0.15)
    return float(np.mean(vivid))


def classify_panel(rgb: np.ndarray) -> tuple[bool, str]:
    """Judge whether a crop looks like the raw-IR panel.

    Returns ``(ok, reason)``. Deliberately conservative: a crop that
    merely *might* be the right panel is rejected, because a silently
    mislabelled training set is worse than a missing one.
    """
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return False, "not an RGB image"

    h, w = rgb.shape[:2]
    if min(h, w) < 64:
        return False, f"crop too small ({w}x{h})"

    sat = _saturation(rgb)
    energy = _colour_energy(rgb)
    gray = rgb.astype(np.float32).mean(axis=2)
    contrast = float(gray.std())

    # The raw panel is essentially grayscale: channel spread is tiny.
    channel_spread = float(
        np.mean(np.abs(rgb[..., 0].astype(np.float32) - rgb[..., 2].astype(np.float32)))
    )

    if energy > 0.08:
        return False, f"too colourful (vivid={energy:.3f}) -- likely BT/colourbar"
    if sat > 0.22:
        return False, f"saturation {sat:.3f} too high -- not raw IR"
    if channel_spread > 12.0:
        return False, f"R-B spread {channel_spread:.1f} -- coloured panel"
    if contrast < 8.0:
        return False, f"contrast {contrast:.1f} too flat -- blank/background"

    # Greyscale-only guard. A "Cloud Height" 3-D surface panel is also
    # greyscale, so colour tests cannot catch it. The discriminator is
    # texture: a 3-D surface render is dominated by long straight axis
    # lines and large flat black/dark regions, whereas IR imagery has
    # smooth low-frequency cloud structure across the whole frame.
    hist = np.histogram(gray, bins=32, range=(0, 255))[0].astype(np.float64)
    hist /= hist.sum() + 1e-9
    # Fraction of pixels in the darkest bin (3-D panels have a big black base)
    dark_frac = float(hist[0] + hist[1])
    # Vertical line energy: axis grids/lines produce strong 1-px columns
    col_energy = float(np.mean(np.abs(np.diff(gray, axis=1))))
    row_energy = float(np.mean(np.abs(np.diff(gray, axis=0))))
    # A 3-D surface plot is axis-aligned: one direction much smoother.
    axis_bias = abs(col_energy - row_energy) / (col_energy + row_energy + 1e-6)

    if dark_frac > 0.42:
        return False, f"dark-bin fraction {dark_frac:.3f} -- 3-D surface/base region"
    if axis_bias > 0.55:
        return False, f"axis bias {axis_bias:.3f} -- axis-aligned plot furniture"

    return True, f"gray IR ok (sat={sat:.3f}, vivid={energy:.3f}, contrast={contrast:.1f})"


# ---------------------------------------------------------------------------
# Crop detection
# ---------------------------------------------------------------------------


def detect_plot_area(rgb: np.ndarray) -> tuple[int, int, int, int] | None:
    """Locate the figure's white-margin-bounded content box.

    Returns ``(left, top, right, bottom)`` in pixels, or ``None`` when the
    image has no clear white margin (i.e. it is already a cropped tile --
    the caller then treats it as such).
    """
    gray = rgb.astype(np.float32).mean(axis=2)
    h, w = gray.shape

    # A margin row/column is near-white (matplotlib figures default white).
    def is_white(values: np.ndarray) -> bool:
        return float(np.mean(values > 235)) > 0.80

    top = 0
    while top < h - 1 and is_white(gray[top]):
        top += 1
    bottom = h - 1
    while bottom > top and is_white(gray[bottom]):
        bottom -= 1
    left = 0
    while left < w - 1 and is_white(gray[:, left]):
        left += 1
    right = w - 1
    while right > left and is_white(gray[:, right]):
        right -= 1

    # No meaningful margin => the file is already a tight tile.
    if top < 4 and bottom > h - 5 and left < 4 and right > w - 5:
        return None
    if right - left < 32 or bottom - top < 32:
        return None
    return left, top, right, bottom


def _content_bands(active: np.ndarray, min_len: int = 24) -> list[tuple[int, int]]:
    """Group a 1-D boolean 'content present' mask into runs of length >= min_len.

    Short runs are hairlines, axis ticks or antialiased edges and are
    discarded; the long runs are the actual image panels.
    """
    bands: list[tuple[int, int]] = []
    start: int | None = None
    for i, on in enumerate(active):
        if on and start is None:
            start = i
        elif not on and start is not None:
            if i - start >= min_len:
                bands.append((start, i))
            start = None
    if start is not None and len(active) - start >= min_len:
        bands.append((start, len(active)))
    return bands


def _find_content_columns(rgb: np.ndarray, top: int, bottom: int) -> list[tuple[int, int]]:
    """Return horizontal content bands (columns that are not all-white).

    Two panels side by side are separated by a white gutter; this finds
    that gutter rather than assuming an exact 50/50 split.
    """
    gray = rgb[top:bottom].astype(np.float32).mean(axis=2)
    non_white = (gray < 235).mean(axis=0) > 0.05
    return _content_bands(non_white)


def _find_content_rows(
    rgb: np.ndarray, left: int, right: int, top: int, bottom: int
) -> list[tuple[int, int]]:
    """Return vertical content bands within the given column range."""
    gray = rgb[top:bottom, left:right].astype(np.float32).mean(axis=2)
    non_white = (gray < 235).mean(axis=1) > 0.05
    return _content_bands(non_white)


def auto_crop_quadrant(rgb: np.ndarray) -> np.ndarray | None:
    """Extract the top-left image panel from a 4-panel figure.

    Locates the two column bands and two row bands, then takes the
    intersection of the first of each. Returns ``None`` when the grid
    structure cannot be found confidently, so the caller never silently
    trains on a whole figure.
    """
    box = detect_plot_area(rgb)
    if box is None:
        return None
    left, top, right, bottom = box

    col_bands = _find_content_columns(rgb, top, bottom)
    if len(col_bands) < 2:
        return None

    first_left, first_right = col_bands[0]
    row_bands = _find_content_rows(rgb, first_left, first_right, top, bottom)
    if len(row_bands) < 2:
        return None

    first_top, first_bottom = row_bands[0]
    crop = rgb[
        top + first_top : top + first_bottom,
        left + first_left : left + first_right,
    ]
    if crop.size == 0:
        return None
    return crop


def manual_crop(rgb: np.ndarray, fractions: tuple[float, float, float, float]) -> np.ndarray:
    """Crop using ``(left, top, right, bottom)`` fractions of width/height."""
    h, w = rgb.shape[:2]
    fl, ft, fr, fb = fractions
    x0, x1 = int(round(fl * w)), int(round(fr * w))
    y0, y1 = int(round(ft * h)), int(round(fb * h))
    x0, x1 = max(0, min(x0, x1)), min(w, max(x0, x1))
    y0, y1 = max(0, min(y0, y1)), min(h, max(y0, y1))
    return rgb[y0:y1, x0:x1]


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------


def _resize_square(rgb: np.ndarray, size: int) -> np.ndarray:
    """Center-crop to a square then resize -- mirrors ml.preprocessing."""
    pil = Image.fromarray(rgb.astype(np.uint8))
    w, h = pil.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    pil = pil.crop((left, top, left + side, top + side))
    return np.asarray(pil.resize((size, size), Image.LANCZOS))


def _label_for(path: Path, source: Path) -> str:
    """Class label: an explicit ``__<class>`` suffix, else the folder name.

    The suffix form lets a flat folder of figures carry labels without the
    user having to create one directory per class.
    """
    if "__" in path.stem:
        return path.stem.rsplit("__", 1)[1]
    rel = path.relative_to(source)
    return rel.parts[0] if len(rel.parts) > 1 else "unlabeled"


def prepare(
    source: Path,
    out_dir: Path,
    *,
    crop: tuple[float, float, float, float] | None = None,
    dry_run: bool = False,
    keep_panels: Path | None = None,
) -> dict:
    """Crop, verify and export every figure found under ``source``."""
    if not source.exists():
        raise SystemExit(
            f"Source folder not found: {source}\n"
            "Create it and drop the reference figures inside, either as\n"
            f"  {source}\\<Class Name>\\<figure>.png\n"
            f"  {source}\\<figure>__<Class Name>.png"
        )

    files = sorted(
        p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not files:
        raise SystemExit(f"No images found under {source}")

    # Files sitting loose in the root with no "__<class>" suffix cannot be
    # labelled at all -- warn before doing any work so the user is not left
    # wondering why everything came out as "unlabeled".
    loose = [
        p
        for p in files
        if len(p.relative_to(source).parts) == 1 and "__" not in p.stem
    ]
    if loose:
        print("!" * 78)
        print(f"WARNING -- {len(loose)} file(s) in the root have no class label.")
        print("They will be filed under 'unlabeled' and cannot be trained on.")
        print("Rename them as  <figure>__<Class Name>.png  or move them into")
        print("a sub-folder named after the class. First few:")
        for p in loose[:5]:
            print(f"  {p.name}")
        print("!" * 78)

    print("=" * 78)
    print("CYCLO-VISION -- reference figure -> training tile extraction")
    print("=" * 78)
    print(f"source      : {source}")
    print(f"destination : {out_dir}")
    print(f"figures     : {len(files)}")
    print(f"crop mode   : {'manual ' + str(crop) if crop else 'auto (top-left quadrant)'}")
    print(f"input size  : {INPUT_SIZE}x{INPUT_SIZE}")
    print("-" * 78)

    if out_dir.exists() and not dry_run:
        shutil.rmtree(out_dir)

    per_class: dict[str, int] = {}
    rejected: list[dict] = []
    accepted: list[dict] = []

    for path in files:
        rel = path.relative_to(source)
        label = _label_for(path, source)

        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB"))

        panel = manual_crop(rgb, crop) if crop is not None else auto_crop_quadrant(rgb)
        if panel is None or panel.size == 0:
            rejected.append({"file": str(rel), "label": label, "reason": "no panel located"})
            print(f"  REJECT  {str(rel):<50} no panel located")
            continue

        ok, reason = classify_panel(panel)
        if not ok:
            rejected.append({"file": str(rel), "label": label, "reason": reason})
            print(f"  REJECT  {str(rel):<50} {reason}")
            continue

        tile = _resize_square(panel, INPUT_SIZE)
        per_class[label] = per_class.get(label, 0) + 1
        accepted.append({"file": str(rel), "label": label, "panel_shape": list(panel.shape)})
        print(f"  OK      {str(rel):<50} {reason}")

        if dry_run:
            continue

        class_dir = out_dir / label
        class_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(tile).save(class_dir / f"{path.stem}.png", format="PNG", optimize=True)

        if keep_panels is not None:
            keep_panels.mkdir(parents=True, exist_ok=True)
            Image.fromarray(panel.astype(np.uint8)).save(keep_panels / f"{path.stem}_panel.png")

    print("-" * 78)
    print("tiles accepted per class:")
    if per_class:
        for label, count in sorted(per_class.items()):
            flag = "" if count >= 3 else "   <-- TOO FEW (need >= 3)"
            print(f"  {label:<34} {count:>3}{flag}")
    else:
        print("  (none)")
    if rejected:
        print(f"rejected: {len(rejected)} figure(s) -- see manifest for reasons")
    print("=" * 78)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(source),
        "output": str(out_dir),
        "input_size": INPUT_SIZE,
        "dry_run": dry_run,
        "per_class": per_class,
        "accepted": accepted,
        "rejected": rejected,
    }
    if not dry_run:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"manifest written -> {MANIFEST}")

    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract raw-IR panels from 4-panel reference figures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"folder of reference figures (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"output root for labelled tiles (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--crop",
        type=float,
        nargs=4,
        metavar=("L", "T", "R", "B"),
        default=None,
        help="manual crop as left top right bottom fractions (0-1)",
    )
    parser.add_argument("--dry-run", action="store_true", help="analyse without writing")
    parser.add_argument(
        "--keep-panels",
        type=Path,
        default=None,
        help="also save the un-resized panels here (for visual spot-checks)",
    )
    args = parser.parse_args(argv)

    crop = tuple(args.crop) if args.crop else None
    if crop is not None and not all(0.0 <= v <= 1.0 for v in crop):
        raise SystemExit("--crop values must be fractions between 0 and 1")

    prepare(
        args.source,
        args.out,
        crop=crop,
        dry_run=args.dry_run,
        keep_panels=args.keep_panels,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
