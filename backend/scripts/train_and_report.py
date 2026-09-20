"""
CYCLO-VISION -- One-command end-to-end pipeline.
===============================================

Runs the whole thing in a single command and prints every step:

    1. **prepare** -- crop the raw-IR panel out of each 4-panel analysis
       figure under ``data/reference/raw/``, verify it really is
       monochrome IR, resize to 256x256 and write labelled tiles.
    2. **train**   -- fine-tune the classifier, selecting on val macro-F1
       and saving the best checkpoint to ``models/cyclo_cnn.pt``.
    3. **evaluate**-- per-class precision/recall/F1 and a confusion matrix
       on the held-out validation split.
    4. **explain** -- render saliency heatmap samples to
       ``data/reference/heatmaps/`` so the model's attention can be
       eyeballed (this is the same code path the API's heatmap uses).

Nothing is written if the data is too thin: the pipeline aborts with an
explicit per-class count instead of producing an untrustworthy model.

Usage
-----
    python scripts/train_and_report.py
    python scripts/train_and_report.py --epochs 40 --backbone cnn
    python scripts/train_and_report.py --skip-prepare     # tiles already exist
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(BACKEND_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR / "scripts"))

from prepare_reference_dataset import DEFAULT_OUT as DEFAULT_TILES  # noqa: E402
from prepare_reference_dataset import DEFAULT_SOURCE as DEFAULT_RAW  # noqa: E402
from prepare_reference_dataset import MANIFEST as MANIFEST_PATH  # noqa: E402
from prepare_reference_dataset import prepare  # noqa: E402
from train_model import DEFAULT_OUT as DEFAULT_CKPT  # noqa: E402
from train_model import REPORT_PATH, build_backbone, train  # noqa: E402

HEATMAP_DIR = PROJECT_ROOT / "data" / "reference" / "heatmaps"


def banner(title: str, step: int, total: int) -> None:
    print()
    print("#" * 78)
    print(f"# STEP {step}/{total} -- {title}")
    print("#" * 78)


def make_heatmaps(ckpt_path: Path, data_dir: Path, out_dir: Path, limit: int = 6) -> int:
    """Render saliency heatmaps for a few tiles so attention is checkable.

    Uses the repo's own preprocessing path (``ml.preprocessing`` via
    ``ml.inference``) rather than a reimplementation, so the tensor being
    explained is exactly what ``/api/analyze`` would feed the model.
    """
    if not ckpt_path.exists():
        print(f"  skipped -- no checkpoint at {ckpt_path}")
        return 0

    import io

    import ml.inference as inference
    import numpy as np
    from PIL import Image
    import torch
    from ml.model import CLASS_LABELS

    # Weights are stored as a bare state dict so ml.model.build_model can
    # load them; backbone/size come from the sidecar written by train_model.
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta_path = ckpt_path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    backbone_name = str(meta.get("backbone", "cnn"))
    input_size = int(meta.get("input_size", 256))

    model, _, _ = build_backbone(backbone_name, False, len(CLASS_LABELS), torch)
    model.load_state_dict(state)
    model.eval()

    tiles = sorted(p for p in data_dir.rglob("*.png") if p.is_file())[:limit]
    if not tiles:
        print("  no tiles to render")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for path in tiles:
        # Go through the project's own pipeline so the tensor we explain is
        # byte-for-byte the one /api/analyze would feed the model.
        im = inference.preprocess_image(path.read_bytes(), size=input_size)
        if not im.success or im.image is None:
            continue
        x = torch.from_numpy(im.image.tensor).unsqueeze(0)

        # Input-gradient saliency for the predicted class: cheap, and it
        # needs no assumptions about the backbone's internals.
        x.requires_grad_(True)
        logits = model(x)
        cls = int(logits.argmax(dim=1).item())
        model.zero_grad(set_to_none=True)
        logits[0, cls].backward()
        arr = x.grad.detach().abs().numpy()[0].mean(axis=0)

        arr = arr - arr.min()
        span = arr.max()
        arr = arr / span if span > 0 else arr
        grey = (arr * 255).astype(np.uint8)

        # Side-by-side: input panel next to its saliency, so the overlay is
        # readable without opening two files.
        panel = np.asarray(im.display_rgb if hasattr(im, "display_rgb")
                           else Image.open(io.BytesIO(path.read_bytes())).convert("RGB"))
        panel = np.asarray(Image.fromarray(panel).resize((input_size, input_size)))
        side = np.concatenate([panel, np.stack([grey] * 3, axis=2)], axis=1)

        name = f"{path.parent.name}_{path.stem}_heat.png".replace(" ", "_")
        Image.fromarray(side).save(out_dir / name)
        written += 1

    print(f"  {written} heatmap(s) -> {out_dir}")
    return written


def main(argv: list[str] | None = None) -> int:
    total = 4
    parser = argparse.ArgumentParser(
        description="Prepare -> train -> evaluate -> explain, in one command.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW,
                        help=f"folder of reference figures (default: {DEFAULT_RAW})")
    parser.add_argument("--data", type=Path, default=DEFAULT_TILES,
                        help=f"labelled tile root (default: {DEFAULT_TILES})")
    parser.add_argument("--out", type=Path, default=DEFAULT_CKPT,
                        help=f"checkpoint path (default: {DEFAULT_CKPT})")
    parser.add_argument("--backbone", choices=("resnet18", "cnn"), default="resnet18")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.25)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-per-class", type=int, default=3)
    parser.add_argument("--skip-prepare", action="store_true",
                        help="tiles are already prepared; go straight to training")
    parser.add_argument("--skip-heatmaps", action="store_true")
    parser.add_argument("--crop", type=float, nargs=4, metavar=("L", "T", "R", "B"),
                        default=None, help="manual crop fractions for odd layouts")
    parser.set_defaults(pretrained=True)
    args = parser.parse_args(argv)

    print("=" * 78)
    print("CYCLO-VISION -- reference-image pipeline (prepare -> train -> evaluate)")
    print("=" * 78)
    print(f"raw figures : {args.raw}")
    print(f"tile root   : {args.data}")
    print(f"checkpoint  : {args.out}")
    print("=" * 78)

    # --- STEP 1: prepare -------------------------------------------------
    if args.skip_prepare:
        banner("PREPARE (skipped)", 1, total)
    else:
        banner("PREPARE - crop + verify raw-IR panels", 1, total)
        manifest = prepare(
            args.raw,
            args.data,
            crop=tuple(args.crop) if args.crop else None,
        )

        # Fail fast with a diagnosis instead of letting STEP 2 crash on a
        # tile root that was never created.
        if sum(manifest["per_class"].values()) == 0:
            print()
            print("X" * 78)
            print("PIPELINE ABORTED - no raw-IR panel could be extracted")
            print("from any figure under:", args.raw)
            if manifest["rejected"]:
                reasons: dict[str, int] = {}
                for entry in manifest["rejected"]:
                    reasons[entry["reason"]] = reasons.get(entry["reason"], 0) + 1
                print("  rejection reasons:")
                for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
                    print(f"    {count:>4}  {reason}")
                print(f"  (full list: {MANIFEST_PATH})")
            print()
            print("  The prepare step expects 4-panel IMD analysis sheets whose")
            print("  TOP-LEFT panel is the monochrome raw-IR image. Single-panel")
            print("  figures (e.g. the data/demo samples) are rejected. Either")
            print("    * supply the real 4-panel figures, or")
            print(f"    * drop ready-made 256x256 tiles directly in {args.data}\\<Class Name>\\,")
            print("      then rerun with --skip-prepare, or")
            print("    * pass --crop L T R B for an unusual figure layout.")
            print("X" * 78)
            raise SystemExit(2)

    # --- STEP 2 + 3: train (prints its own eval table) -------------------
    train_args = argparse.Namespace(
        data=args.data,
        out=args.out,
        backbone=args.backbone,
        pretrained=args.pretrained,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_split=args.val_split,
        size=args.size,
        device=args.device,
        seed=args.seed,
        min_per_class=args.min_per_class,
    )
    banner("TRAIN -- fine-tune and select best checkpoint", 2, total)
    report = train(train_args)

    # A second, explicit read-out of the held-out metrics: the training
    # log scrolls, so the headline numbers are repeated here.
    banner("EVALUATE -- held-out metrics", 3, total)
    best = report["best"]
    print(f"  validation tiles : {report['val_tiles']}")
    print(f"  best epoch       : {best['epoch']} of {args.epochs}")
    print(f"  accuracy         : {best['val_accuracy']:.4f}")
    print(f"  macro F1         : {best['val_macro_f1']:.4f}")
    print()
    print(f"  {'class':<34} {'support':>8} {'prec':>7} {'recall':>7} {'f1':>7}")
    for label in best["classes_present"]:
        m = best["per_class"][label]
        print(f"  {label:<34} {m['support']:>8} {m['precision']:>7.3f} "
              f"{m['recall']:>7.3f} {m['f1']:>7.3f}")

    # --- STEP 4: heatmaps ------------------------------------------------
    if args.skip_heatmaps:
        banner("EXPLAIN (skipped)", 4, total)
    else:
        banner("EXPLAIN -- saliency heatmap samples", 4, total)
        make_heatmaps(Path(args.out), Path(args.data), HEATMAP_DIR)

    # --- summary ---------------------------------------------------------
    print()
    print("=" * 78)
    print("PIPELINE COMPLETE")
    print("=" * 78)
    print(f"  checkpoint : {args.out}")
    print(f"  report     : {REPORT_PATH}")
    if not args.skip_heatmaps:
        print(f"  heatmaps   : {HEATMAP_DIR}")
    print()
    print("  Next: set ML_MODE=model in your .env and restart the backend so")
    print("  /api/analyze uses these trained weights instead of demo heuristics.")
    print()
    print("  Honest limits worth reading in the report:")
    print(f"    * {report['train_tiles']} train + {report['val_tiles']} val tiles only.")
    print("    * Validation accuracy on this few tiles is a smoke test, not a")
    print("      generalisation estimate. Collect more figures per class before")
    print("      quoting any accuracy figure.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())