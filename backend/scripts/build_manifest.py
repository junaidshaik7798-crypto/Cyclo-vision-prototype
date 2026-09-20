"""
CYCLO-VISION -- build manifest.csv joining HURSAT frames to IBTrACS labels
=========================================================================

Joins extracted HURSAT PNGs (see ``extract_hursat.py``) against the local
IBTrACS-NI cache by SID + ISO_TIME and writes the manifest that
``ml.dataset.CycloneImageDataset`` and ``scripts.train_cnn`` consume:

    filename,usa_wind,usa_pres,sid,year,lat,lon

Two filename conventions on disk are recognised (indexed once per run):

    raw HURSAT stem:  b1.2019121N05089.IR.2019050100.png
    normalized:       2019_2019121N05089_2019-05-01_00.png

Only rows whose image actually exists on disk are kept.

Usage:
    cd backend
    python -m scripts.build_manifest --img-dir ../data/hursat_png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

IBT = Path("../data/cache/ibtracs.NI.list.v04r01.csv")
OUT = Path("../data/manifest.csv")

# normalized: <SEASON>_<SID>_<YYYY-MM-DD_HH...>.png
_PAT_NORM = re.compile(r"^\d{4}_(?P<sid>[^_]+)_(?P<iso>\d{4}-\d{2}-\d{2}_\d{2})")
# raw HURSAT-b1: b1.<SID>.IR.<YYYYMMDDHH>...
_PAT_RAW = re.compile(r"^b1\.(?P<sid>[^.]+)\.IR\.(?P<stamp>\d{10})")


def _index_images(img_dir: Path) -> dict[tuple[str, str], str]:
    """Map (SID, YYYYMMDDHH00) -> png filename for both conventions."""
    index: dict[tuple[str, str], str] = {}
    for f in sorted(img_dir.glob("*.png")):
        m = _PAT_NORM.match(f.name)
        if m:
            stamp = m.group("iso").replace("-", "").replace("_", "") + "00"
            index.setdefault((m.group("sid"), stamp), f.name)
            continue
        m = _PAT_RAW.match(f.name)
        if m:
            index.setdefault((m.group("sid"), m.group("stamp") + "00"), f.name)
    return index


def build_manifest(img_dir: Path, ibt: Path, out: Path) -> int:
    # IBTrACS has a duplicate units row right after the header -- skip it
    # (same trick as app/services/ibtracs.py).
    df = pd.read_csv(ibt, skiprows=[1], low_memory=False)
    df = df[["SID", "SEASON", "ISO_TIME", "USA_WIND", "USA_PRES", "LAT", "LON"]].copy()
    df["usa_wind"] = pd.to_numeric(df["USA_WIND"], errors="coerce")
    df["usa_pres"] = pd.to_numeric(df["USA_PRES"], errors="coerce")
    df = df.dropna(subset=["usa_wind", "LAT", "LON"])

    # ISO_TIME ('2019-05-01 00:00:00') -> lookup key (SID, YYYYMMDDHH00).
    stamp = pd.to_datetime(df["ISO_TIME"], errors="coerce").dt.strftime("%Y%m%d%H")
    df["_key"] = list(zip(df["SID"].astype(str), stamp.astype(str) + "00"))

    index = _index_images(img_dir)
    df["_img"] = df["_key"].map(index)  # NaN when no image on disk
    df = df.dropna(subset=["_img"])
    if df.empty:
        return 0

    df = df.rename(
        columns={
            "SEASON": "year",
            "LAT": "lat",
            "LON": "lon",
            "SID": "sid",
            "_img": "filename",
        }
    )
    cols = ["filename", "usa_wind", "usa_pres", "sid", "year", "lat", "lon"]
    df[cols].to_csv(out, index=False)
    return len(df)


def main() -> None:
    p = argparse.ArgumentParser(description="Build training manifest CSV")
    p.add_argument("--img-dir", default="../data/hursat_png")
    p.add_argument("--ibtracs", default=str(IBT))
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args()

    img_dir = Path(args.img_dir)
    if not img_dir.is_dir():
        raise SystemExit(
            f"image folder not found: {img_dir} -- run "
            "scripts/extract_hursat.py first"
        )
    n = build_manifest(img_dir, Path(args.ibtracs), Path(args.out))
    if n == 0:
        raise SystemExit(
            "manifest is empty: no IBTrACS rows matched extracted filenames. "
            "Check the filename convention (season_SID_YYYY-MM-DD_HHMM.png)."
        )
    print(f"wrote {n} rows -> {args.out}")


if __name__ == "__main__":
    main()
