"""
CYCLO-VISION -- end-to-end sanity check for the real-data training stack
========================================================================

Verifies, without any HURSAT download:

    1. ml.dataset reuses the serve-path preprocessing (denoise + CLAHE /255)
    2. wind_to_class matches ml.classification bands exactly
    3. a hand-labelled demo manifest loads (one fake SID per frame so the
       split is storm-level)
    4. split_by_storm partitions by SID with no overlap
    5. the weighted sampler balances classes
    6. CycloCNN trains a few epochs on CPU (exercises the whole loop)
    7. the checkpoint loads through ml.model.build_model and serves a
       prediction via ml.inference.run_inference

Run:
    cd backend
    python scripts/selftest_train_stack.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.classification import class_index_for_wind  # noqa: E402
from ml.dataset import CycloneImageDataset, wind_to_class  # noqa: E402
from ml.inference import run_inference  # noqa: E402
from ml.model import CLASS_LABELS, CycloCNN, build_model  # noqa: E402
from ml.preprocessing import preprocess_image  # noqa: E402

DEMO = Path(__file__).resolve().parents[2] / "data" / "demo"

# Hand labels for generate_demo_data.py output (kt). Trivially separable.
SAMPLES = [
    ("sample_clear.png", 12.0),  # No Cyclone
    ("sample_dev.png", 24.0),  # Depression
    ("sample_severe.png", 72.0),  # Very Severe
    ("sample_vsevere.png", 120.0),  # Extremely Severe (>= 120 -> class 6)
]

_checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _checks.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="cyclo_train_selftest_"))
    try:
        # --- 1. preprocessing parity ----------------------------------------
        raw = (DEMO / SAMPLES[0][0]).read_bytes()
        pre = preprocess_image(raw)
        check(
            "preprocess_image runs (denoise+CLAHE+/255)",
            pre.success
            and pre.image is not None
            and float(pre.image.tensor.max()).__le__(1.0),
        )

        # --- 2. wind_to_class parity ----------------------------------------
        probes = [12.0, 17.0, 28.0, 34.0, 48.0, 64.0, 90.0, 120.0, 137.0]
        got = [wind_to_class(w) for w in probes]
        expect = [class_index_for_wind(kt=w) for w in probes]
        check("wind_to_class == ml.classification bands", got == expect)
        check("wind band spread", len(set(got)) >= 5, f"classes hit: {set(got)}")

        # --- 3. manifest -------------------------------------------------------
        img_dir = tmp / "imgs"
        img_dir.mkdir()
        rows = []
        for i, (fname, wind) in enumerate(SAMPLES):
            shutil.copy(DEMO / fname, img_dir / fname)
            rows.append(
                {
                    "filename": fname,
                    "usa_wind": wind,
                    "usa_pres": 1004.0,
                    "sid": f"TEST{i:04d}",  # one fake storm per frame
                    "year": 2019,
                    "lat": 15.0,
                    "lon": 87.0,
                }
            )
        manifest = tmp / "manifest.csv"
        pd.DataFrame(rows).to_csv(manifest, index=False)
        check("demo manifest written", manifest.exists())

        # --- 4. storm-level split ----------------------------------------------
        from scripts.train_cnn import build_transforms, make_sampler, split_by_storm

        tr, va = split_by_storm(manifest, val_frac=0.25)
        overlap = set(tr["sid"]) & set(va["sid"])
        check(
            "split_by_storm: no SID overlap",
            not overlap,
            f"{len(tr)} train / {len(va)} val frames",
        )

        # --- 5. sampler ---------------------------------------------------------
        labels = np.array([wind_to_class(w) for w in tr["usa_wind"]])
        sampler = make_sampler(labels)
        idx = list(sampler)[:200]
        sampled = [labels[i] for i in idx]
        c = np.bincount(np.array(sampled), minlength=7)
        present = c[labels]
        check(
            "sampler balances classes",
            (present.max() - present.min()) <= 40,
            f"counts over 200 draws: {c.tolist()}",
        )

        # --- 6. short training run ----------------------------------------------
        from scripts.train_cnn import train as train_fn

        args = argparse.Namespace(
            manifest=str(manifest),
            img_dir=str(img_dir),
            out=str(tmp / "cyclo_cnn.pt"),
            size=64,
            batch=4,
            epochs=3,
            lr=3e-4,
            val_frac=0.25,
            workers=0,
            no_preprocess=False,
        )
        train_fn(args)
        ckpt = tmp / "cyclo_cnn.pt"
        sidecar = Path(str(ckpt) + ".json")
        check("checkpoint + sidecar written", ckpt.exists() and sidecar.exists())

        # --- 7. serve through the real stack -------------------------------------
        model, _ = build_model(str(ckpt))
        sd = torch.load(str(ckpt), map_location="cpu")
        same = all(
            torch.allclose(model.state_dict()[k].cpu(), sd[k])
            for k in list(sd)[:3]
        )
        check(
            "build_model loads bare state dict",
            isinstance(model, CycloCNN) and same,
        )
        raw = (DEMO / "sample_vsevere.png").read_bytes()
        from app.core.config import settings

        old_mode, old_path = settings.ML_MODE, settings.MODEL_PATH
        settings.ML_MODE, settings.MODEL_PATH = "model", str(ckpt)
        try:
            result = run_inference(raw)
        finally:
            settings.ML_MODE, settings.MODEL_PATH = old_mode, old_path
        check(
            "run_inference uses the trained model",
            result.get("inference_mode") == "model",
            f"class={result.get('classification')}",
        )
        # The class itself is meaningless for a 4-image / 3-epoch toy -- what
        # matters is that the API result provably comes from THIS checkpoint
        # (not the demo heuristic, not a stale cached model).
        model, _ = build_model(str(ckpt))
        x = torch.from_numpy(
            np.ascontiguousarray(preprocess_image(raw).image.tensor)
        ).unsqueeze(0)
        with torch.no_grad():
            direct = int(model(x).argmax(1))
        check(
            "serving class == direct checkpoint prediction",
            result.get("class_index") == direct,
            f"{CLASS_LABELS[direct]}",
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    failed = [n for n, ok, _ in _checks if not ok]
    print(f"\n{len(_checks) - len(failed)}/{len(_checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

