"""CYCLO-VISION -- Demo dataset service.

Enumerates bundled sample satellite images in data/demo/ and returns their
metadata. Demo sample metadata can be supplied via sidecar .json files, or
derived from the filename/PNG metadata.

The directory is scanned at most once per change: the listing is cached
against ``os.path.getmtime(_DEMO_ROOT)`` (P1-3), so repeated calls from
``/analyze/demo`` and ``/analyze/samples`` do not re-stat every image.
"""

from __future__ import annotations

import os
from pathlib import Path

from app.core.config import settings

_DEMO_ROOT = Path(settings.DEMO_DATA_DIR)

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

# Fallback catalog (name, description, category) keyed by filename stem
_FALLBACK = {
    "sample_clear": ("Clear Sky -- No Cyclone", "Calm ocean, no organized convection.", "No Cyclone"),
    "sample_dev": ("Developing Low", "Early convective organization, disorganized bands.", "Depression"),
    "sample_severe": ("Severe Cyclone", "Well-formed eyewall with intense convection.", "Severe Cyclonic Storm"),
    "sample_vsevere": ("Very Severe Cyclone", "Symmetrical storm with visible eye.", "Very Severe Cyclonic Storm"),
}

# (directory mtime, samples) -- invalidated when the demo folder changes.
_CACHE: tuple[float, list[dict]] | None = None


def _dir_mtime() -> float:
    """Directory mtime, or -1.0 when the demo folder does not exist."""
    try:
        return os.path.getmtime(_DEMO_ROOT)
    except OSError:
        return -1.0


def _scan() -> list[dict]:
    """Read the demo directory once and build the sample metadata list."""
    samples: list[dict] = []
    if not _DEMO_ROOT.exists():
        return samples
    for fname in sorted(os.listdir(_DEMO_ROOT)):
        if not fname.lower().endswith(_IMAGE_SUFFIXES):
            continue
        if fname.startswith("."):
            continue
        stem = Path(fname).stem
        # A single stat() is enough for file_size_kb. The previous
        # Image.open(...).verify() call invalidated the handle and the
        # pixel data was never used anyway (P1-3).
        try:
            size_kb = round((_DEMO_ROOT / fname).stat().st_size / 1024)
        except OSError:
            size_kb = 0
        name, desc, cat = _FALLBACK.get(stem, (stem.replace("_", " ").title(), "Bundled demo satellite image.", "Demo"))
        samples.append({
            "id": stem,
            "name": name,
            "description": desc,
            "category": cat,
            "filename": fname,
            "file_size_kb": size_kb,
        })
    return samples


def list_demo_samples() -> list[dict]:
    """Return metadata for bundled demo images present on disk.

    Cached on the demo directory's mtime, so a directory scan happens only
    when images are added/removed (or their timestamps change).
    """
    global _CACHE
    mtime = _dir_mtime()
    if _CACHE is not None and _CACHE[0] == mtime:
        return _CACHE[1]
    samples = _scan()
    _CACHE = (mtime, samples)
    return samples


def get_demo_bytes(sample_id: str) -> tuple[bytes, str, dict]:
    """Read a demo image by id. Returns (bytes, filename, meta)."""
    if not sample_id:
        raise ValueError("sample_id is required")
    # Single pass over the cached listing builds the id -> sample map, so
    # the lookup itself is O(1) and never rescans the directory (P1-3).
    by_id = {s["id"]: s for s in list_demo_samples()}
    sample = by_id.get(sample_id)
    if sample is None:
        raise FileNotFoundError(f"Demo sample '{sample_id}' not found.")
    path = _DEMO_ROOT / sample["filename"]
    with open(path, "rb") as f:
        return f.read(), sample["filename"], sample