"""
CYCLO-VISION -- extract PNG windows from HURSAT-B1 NetCDF files
==============================================================

One-off ingestion: open every ``.nc`` in ``--src`` (HURSAT-B1 IR windows,
4 km geostationary, centred on each IBTrACS fix), grab the infrared channel,
normalise to 8-bit and save one PNG per file into ``--dst``. PNGs keep the
NetCDF stem (``b1.<SID>.IR.<YYYYMMDDHH>...png``); ``build_manifest.py``
indexes both that and the normalized ``season_SID_date`` convention, so no
renaming is required here.

Keep it dumb: one file per time step, no labelling logic here.

Requires the ``netCDF4`` package (``pip install netCDF4``). Without it the
script exits with an install hint instead of failing mid-way.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Candidate variable names for the IR channel across HURSAT-B1 revisions.
_IR_CANDIDATES = ["infrared", "irwin_cdr", "irwin", "temp", "brightness_temp"]


def extract_nc(nc_path: Path, dst: Path) -> str:
    """Convert one HURSAT-B1 NetCDF to a PNG. Returns 'saved' | 'skipped'."""
    from netCDF4 import Dataset as NC

    with NC(nc_path, "r") as nc:
        ir_name = next((v for v in _IR_CANDIDATES if v in nc.variables), None)
        if ir_name is None:
            return "skipped"
        data = np.asarray(nc.variables[ir_name][:], dtype=np.float32)
        if data.ndim == 3:  # (time, y, x) -> single frame
            data = data[0]

    valid = data[np.isfinite(data)]
    if valid.size == 0:
        return "skipped"
    # Robust 1-99 percentile stretch -> uint8 (cloud tops cold/bright).
    lo, hi = np.percentile(valid, [1, 99])
    span = max(1e-6, float(hi - lo))
    img = np.clip((data - lo) / span, 0, 1)
    img = np.nan_to_num(img, nan=0.0)
    arr = (img * 255).astype(np.uint8)

    out = dst / (nc_path.stem + ".png")
    Image_.fromarray(arr).convert("RGB").save(out)
    return "saved"


# Imported late so the argparse header stays readable; PIL is a hard dep
# of the rest of the project anyway.
from PIL import Image as Image_  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="HURSAT-B1 NetCDF -> PNG extractor")
    p.add_argument("--src", default="../data/hursat_nc", help="folder of .nc files")
    p.add_argument("--dst", default="../data/hursat_png", help="PNG output folder")
    args = p.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if not src.is_dir():
        raise SystemExit(f"source folder not found: {src}")
    try:
        import netCDF4  # noqa: F401
    except ImportError:
        raise SystemExit(
            "netCDF4 is required: pip install netCDF4"
        )
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob("*.nc"))
    if not files:
        raise SystemExit(f"no .nc files in {src}")
    saved = skipped = 0
    for i, f in enumerate(files, 1):
        try:
            if extract_nc(f, dst) == "saved":
                saved += 1
            else:
                skipped += 1
        except Exception as exc:  # keep going; report at the end
            print(f"  ! {f.name}: {exc}")
            skipped += 1
        if i % 200 == 0:
            print(f"  {i}/{len(files)} (saved {saved}, skipped {skipped})")
    print(f"done: {saved} saved, {skipped} skipped -> {dst}")


if __name__ == "__main__":
    main()
