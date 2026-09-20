"""
CYCLO-VISION -- end-to-end rehearsal for the training pipeline.
==============================================================

Builds synthetic 4-panel analysis figures (same layout as the real
reference images), runs prepare -> train -> evaluate -> explain against
them, and asserts every artifact is produced. This proves the plumbing
works before real figures are supplied.

Run:
    python scripts/selftest_pipeline.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
for extra in (BACKEND_DIR, BACKEND_DIR / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import train_and_report as pipeline  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    (PASSED if ok else FAILED).append(label)
    line = f"  [{'PASS' if ok else 'FAIL'}] {label}"
    if detail:
        line += f" -- {detail}"
    print(line)


def make_figure(path: Path, *, strong: bool, seed: int, n: int = 200) -> None:
    """Write a 4-panel figure: raw IR (TL), colour BT (BL), grey (TR/BR)."""
    rng = np.random.RandomState(seed)

    # Top-left: monochrome raw IR with a bright cold cloud mass.
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    r = np.sqrt((xx - n / 2) ** 2 + (yy - n / 2) ** 2) / (n / 2)
    ir = np.full((n, n), 60.0, dtype=np.float32)
    if strong:
        # Tight, very cold, organised core -- an eye-like structure.
        ir += np.exp(-(r ** 2) / 0.035) * 190.0
        ir += np.exp(-((r - 0.18) ** 2) / 0.006) * 150.0
    else:
        # Diffuse, warm, disorganised cloud shield.
        ir += np.exp(-(r ** 2) / 0.16) * 85.0
    ir += rng.normal(0, 4.0, (n, n))
    ir = np.clip(ir, 0, 255).astype(np.uint8)
    raw_panel = np.stack([ir, ir, ir], axis=2)

    # Bottom-left: vivid colour-mapped BT panel (must fail the IR check).
    bt = np.zeros((n, n, 3), dtype=np.uint8)
    bt[..., 0] = np.clip(ir.astype(np.int16) + 60, 0, 255)
    bt[..., 1] = 40
    bt[..., 2] = np.clip(255 - ir.astype(np.int16), 0, 255)

    tr = np.clip(128 + rng.normal(0, 30, (n, n)), 0, 255).astype(np.uint8)
    br = np.clip(100 + rng.normal(0, 45, (n, n)), 0, 255).astype(np.uint8)

    canvas = np.full((2 * n + 40, 2 * n + 20, 3), 255, dtype=np.uint8)
    canvas[10:10 + n, 10:10 + n] = raw_panel
    canvas[10:10 + n, 20 + n:20 + 2 * n] = np.stack([tr] * 3, axis=2)
    canvas[30 + n:30 + 2 * n, 10:10 + n] = bt
    canvas[30 + n:30 + 2 * n, 20 + n:20 + 2 * n] = np.stack([br] * 3, axis=2)

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(path)


def _loads_as_cyclocnn(ckpt_path: Path, tiles: Path) -> bool:
    """Prove the checkpoint works through ml.model.build_model's code path."""
    import torch

    from ml.model import build_model

    model, _ = build_model(str(ckpt_path), "cpu")

    from app.core.config import settings
    from ml.preprocessing import preprocess_image

    tile = next(iter(sorted(tiles.rglob("*.png"))))
    pre = preprocess_image(tile.read_bytes(), size=settings.INPUT_SIZE)
    if not pre.success or pre.image is None:
        return False
    with torch.no_grad():
        logits = model(torch.from_numpy(pre.image.tensor).unsqueeze(0))
    return tuple(logits.shape) == (1, 7)


def main() -> int:
    print("=" * 78)
    print("CYCLO-VISION -- training-pipeline rehearsal (synthetic fixtures)")
    print("=" * 78)

    tmp = Path(tempfile.mkdtemp(prefix="cyclo_selftest_"))
    raw = tmp / "raw"
    tiles = tmp / "labeled"
    ckpt = tmp / "models" / "cyclo_cnn.pt"
    heatmaps = tmp / "heatmaps"

    try:
        # Two classes, 5 figures each: clears --min-per-class 3 and still
        # leaves tiles in both the train and the val split.
        spec = [("Cyclonic Storm", False), ("Severe Cyclonic Storm", True)]
        for label, strong in spec:
            for i in range(5):
                make_figure(
                    raw / label / f"fig_{i}.png",
                    strong=strong,
                    seed=100 + i + (0 if not strong else 50),
                )
        print(f"\nsynthetic figures: 10 across {len(spec)} classes")
        print(f"  {raw}")

        # --- STEP 1: prepare ---------------------------------------------
        print("\n" + "-" * 78)
        print("STEP 1 -- prepare (crop + verify)")
        print("-" * 78)
        manifest = pipeline.prepare(raw, tiles)
        per_class = manifest["per_class"]
        check("prepare produced tiles", sum(per_class.values()) > 0,
              f"per_class={per_class}")
        check("both classes survived verification", len(per_class) == 2,
              str(sorted(per_class)))
        check("nothing rejected", not manifest["rejected"],
              f"{len(manifest['rejected'])} rejected")
        check("tiles are 256x256",
              all(Image.open(p).size == (256, 256) for p in tiles.rglob("*.png")))

        # --- STEP 2: train ------------------------------------------------
        print("\n" + "-" * 78)
        print("STEP 2 -- train")
        print("-" * 78)
        targs = argparse.Namespace(
            data=tiles, out=ckpt, backbone="cnn", pretrained=False,
            epochs=3, batch_size=4, lr=1e-3, weight_decay=1e-4,
            val_split=0.3, size=256, device="cpu", seed=42, min_per_class=3,
        )
        report = pipeline.train(targs)
        check("checkpoint written", ckpt.exists(),
              f"{ckpt.stat().st_size if ckpt.exists() else 0} bytes")
        check("report written", pipeline.REPORT_PATH.exists())
        check("one history row per epoch", len(report["history"]) == 3,
              f"{len(report['history'])} rows")
        check("val split non-empty", report["val_tiles"] > 0,
              f"val={report['val_tiles']} train={report['train_tiles']}")
        check("metrics in [0,1]",
              0.0 <= report["best"]["val_accuracy"] <= 1.0
              and 0.0 <= report["best"]["val_macro_f1"] <= 1.0,
              f"acc={report['best']['val_accuracy']} "
              f"f1={report['best']['val_macro_f1']}")

        import torch

        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        meta_path = ckpt.with_suffix(".meta.json")
        check("weights saved as a bare state dict (build_model-compatible)",
              isinstance(state, dict) and "features.0.weight" in state,
              f"{len(state)} tensors")
        check("sidecar metadata written", meta_path.exists(),
              str(meta_path.name))
        if meta_path.exists():
            import json as _json

            meta = _json.loads(meta_path.read_text(encoding="utf-8"))
            check("metadata intact",
                  meta.get("input_size") == 256
                  and meta.get("backbone") == "cnn"
                  and len(meta.get("class_labels", [])) == 7,
                  f"backbone={meta.get('backbone')} "
                  f"size={meta.get('input_size')}")
        check("loads into bare CycloCNN (ml.inference path)",
              _loads_as_cyclocnn(ckpt, tiles))

        # --- STEP 4: heatmaps --------------------------------------------
        print("\n" + "-" * 78)
        print("STEP 4 -- explain")
        print("-" * 78)
        written = pipeline.make_heatmaps(ckpt, tiles, heatmaps, limit=3)
        check("heatmaps rendered", written > 0, f"{written} written")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 78)
    print(f"REHEARSAL RESULT -- {len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        for name in FAILED:
            print(f"  FAILED: {name}")
        print("=" * 78)
        return 1
    print("ALL CHECKS PASSED")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())