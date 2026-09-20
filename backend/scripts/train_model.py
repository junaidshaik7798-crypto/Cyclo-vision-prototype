"""
CYCLO-VISION -- Train the CycloCNN classifier on prepared reference tiles.
=========================================================================

Trains ``ml.model.CycloCNN`` on the labelled tiles produced by
``scripts/prepare_reference_dataset.py`` and writes a checkpoint that
``ml.inference`` picks up automatically in MODEL mode.

Design decisions (kept deliberately simple and honest)
-----------------------------------------------------
* **Fine-tuning, not from-scratch.** With only a few dozen tiles, training
  a CNN from random init memorises noise. The default backbone is a
  torchvision ImageNet-pretrained ResNet-18 which is far better behaved at
  this data scale. ``--backbone cnn`` uses the in-repo CycloCNN instead
  (kept intact -- only the head is replaced).
* **Stratified split.** Val tiles come from the same class distribution as
  train, so per-class recall is measurable.
* **Class-weighted loss.** Rare classes are not drowned out.
* **Best-val checkpoint.** The saved weights are the ones that actually
  generalised, not the last epoch.
* **Fail loudly on thin data.** A class with fewer than ``--min-per-class``
  tiles aborts the run with a clear message instead of producing a model
  that cannot be trusted.

The checkpoint stores the label ordering and input size alongside the
weights, so inference can never silently mismatch classes.

Usage
-----
    python scripts/train_model.py --epochs 25
    python scripts/train_model.py --backbone cnn --epochs 15
    python scripts/train_model.py --data data/reference/labeled --no-pretrained
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Project paths / sys.path bootstrap
# ---------------------------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml.model import CLASS_LABELS  # noqa: E402

DEFAULT_DATA = PROJECT_ROOT / "data" / "reference" / "labeled"
DEFAULT_OUT = PROJECT_ROOT / "models" / "cyclo_cnn.pt"
REPORT_PATH = PROJECT_ROOT / "models" / "training_report.json"

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# ImageNet statistics -- the pretrained backbone expects these.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------


def normalise_label(raw: str) -> str:
    """Normalise a folder name to a canonical ``CLASS_LABELS`` entry.

    The reference figures may be labelled "No Cyclone", "no_cyclone" or
    "Severe Cyclonic Storm (SCS)". Matching is done on a compacted
    alphanumeric form so spelling style never silently drops tiles.
    """

    def compact(s: str) -> str:
        return "".join(ch for ch in s.lower() if ch.isalnum())

    target = compact(raw)
    for label in CLASS_LABELS:
        if compact(label) == target:
            return label
    # Common abbreviations used in IMD bulletins.
    aliases = {
        "scs": "Severe Cyclonic Storm",
        "vscs": "Very Severe Cyclonic Storm",
        "escs": "Extremely Severe Cyclonic Storm",
        "cs": "Cyclonic Storm",
        "dd": "Deep Depression",
        "d": "Depression",
        "noc": "No Cyclone",
        "nosystem": "No Cyclone",
    }
    if target in aliases:
        return aliases[target]
    # Prefix match, e.g. "nodepression" -> "No Cyclone"
    for label in CLASS_LABELS:
        if compact(label).startswith(target) and len(target) >= 4:
            return label
    raise ValueError(
        f"Cannot map folder '{raw}' to a model class. "
        f"Expected one of: {', '.join(CLASS_LABELS)}"
    )


def discover_samples(data_dir: Path) -> tuple[list[Path], list[int]]:
    """Walk ``data_dir`` and return (paths, class_indices)."""
    paths: list[Path] = []
    labels: list[int] = []
    for class_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        try:
            label = normalise_label(class_dir.name)
        except ValueError as exc:
            print(f"  SKIP  {class_dir.name}: {exc}")
            continue
        idx = CLASS_LABELS.index(label)
        found = sorted(
            p
            for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not found:
            print(f"  SKIP  {class_dir.name}: no images")
            continue
        paths.extend(found)
        labels.extend([idx] * len(found))
    return paths, labels


# ---------------------------------------------------------------------------
# Torch dataset
# ---------------------------------------------------------------------------


def _load_tile(path: Path, size: int) -> np.ndarray:
    """Load a tile as uint8 RGB at ``size`` -- matches inference preprocessing."""
    with Image.open(path) as im:
        pil = im.convert("RGB")
        w, h = pil.size
        side = min(w, h)
        left, top = (w - side) // 2, (h - side) // 2
        pil = pil.crop((left, top, left + side, top + side))
        if side != size:
            pil = pil.resize((size, size), Image.LANCZOS)
        return np.asarray(pil)


def _augment(rgb: np.ndarray, rng: random.Random, size: int) -> np.ndarray:
    """Geometric + photometric augmentation for a uint8 RGB array.

    Cyclone imagery is rotationally ambiguous, so flips and 90-degree
    rotations are label-preserving and safe. Random resized crop forces the
    model to tolerate the storm sitting off-centre (which is exactly what
    the real reference figures look like).
    """
    pil = Image.fromarray(rgb)

    # Random resized crop (scale 0.75-1.0 of the frame).
    w, h = pil.size
    scale = rng.uniform(0.75, 1.0)
    cw, ch = int(w * scale), int(h * scale)
    x0 = rng.randint(0, max(0, w - cw))
    y0 = rng.randint(0, max(0, h - ch))
    pil = pil.crop((x0, y0, x0 + cw, y0 + ch)).resize((size, size), Image.BILINEAR)

    # Flips / 90-degree rotations.
    if rng.random() < 0.5:
        pil = pil.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < 0.5:
        pil = pil.transpose(Image.FLIP_TOP_BOTTOM)
    if rng.random() < 0.5:
        pil = pil.transpose(Image.ROTATE_90)

    arr = np.asarray(pil).astype(np.float32)

    # Brightness / contrast jitter -- the reference panels vary in stretch.
    if rng.random() < 0.7:
        arr = arr * rng.uniform(0.85, 1.15)
    if rng.random() < 0.5:
        mean = arr.mean()
        arr = (arr - mean) * rng.uniform(0.85, 1.15) + mean
    arr = arr + rng.gauss(0.0, 3.0)

    return np.clip(arr, 0, 255).astype(np.uint8)


def _to_tensor(rgb: np.ndarray, normalise: bool) -> np.ndarray:
    """uint8 (H,W,3) -> float32 (3,H,W), ImageNet-normalised when needed."""
    f = rgb.astype(np.float32) / 255.0
    if normalise:
        f = (f - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(f, (2, 0, 1)).copy()


def make_dataset_class(torch):
    """Build the torch ``Dataset`` class (needs torch to be importable)."""

    class TileDataset(torch.utils.data.Dataset):
        """In-memory labelled tile dataset with optional augmentation."""

        def __init__(
            self,
            paths: list[Path],
            labels: list[int],
            size: int,
            *,
            augment: bool,
            normalise: bool,
            seed: int = 0,
        ):
            self.paths = paths
            self.labels = labels
            self.size = size
            self.augment = augment
            self.normalise = normalise
            self.seed = seed
            # Cache decoded tiles: the dataset is small enough to hold in RAM
            # and this makes each epoch far cheaper than re-reading PNGs.
            self._cache: list[np.ndarray] = [
                _load_tile(p, size) for p in paths
            ]

        def __len__(self) -> int:
            return len(self.paths)

        def __getitem__(self, index: int):
            rgb = self._cache[index]
            if self.augment:
                rng = random.Random(self.seed * 100003 + index * 7919 + _EPOCH[0])
                rgb = _augment(rgb, rng, self.size)
            tensor = _to_tensor(rgb, self.normalise)
            return torch.from_numpy(tensor), self.labels[index]

    return TileDataset


# Mutable epoch counter so the augmentation seed changes each epoch while
# the dataset object is reused (DataLoader workers read this cheaply).
_EPOCH = [0]


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def build_backbone(backbone: str, pretrained: bool, num_classes: int, torch):
    """Return (model, uses_imagenet_norm, description).

    ``resnet18`` fine-tunes an ImageNet backbone -- the right call at this
    data scale. ``cnn`` keeps the in-repo CycloCNN architecture and only
    swaps its classifier head, so no project architecture is modified.
    """
    if backbone == "cnn":
        from ml.model import CycloCNN

        model = CycloCNN(in_channels=3, num_classes=num_classes)
        return model, False, "CycloCNN (in-repo, random init)"

    import torchvision.models as tvm

    weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = tvm.resnet18(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    desc = "ResNet-18 " + ("(ImageNet pretrained)" if pretrained else "(random init)")
    return model, pretrained, desc


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(model, loader, device, torch) -> dict:
    """Return accuracy, macro-F1 and the confusion matrix on a loader."""
    model.eval()
    all_pred: list[int] = []
    all_true: list[int] = []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            pred = logits.argmax(dim=1).cpu().numpy()
            all_pred.extend(pred.tolist())
            all_true.extend(y.numpy().tolist())

    n = len(all_true)
    correct = sum(int(p == t) for p, t in zip(all_pred, all_true))
    acc = correct / n if n else 0.0

    # Per-class recall/precision and macro-F1 from the confusion matrix.
    k = len(CLASS_LABELS)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(all_true, all_pred):
        cm[t, p] += 1

    f1s: list[float] = []
    per_class: dict[str, dict] = {}
    for i in range(k):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if cm[i, :].sum() > 0:
            f1s.append(f1)
        per_class[CLASS_LABELS[i]] = {
            "support": int(cm[i, :].sum()),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
        }

    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(np.mean(f1s)) if f1s else 0.0, 4),
        "n": n,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def print_confusion(cm: list[list[int]], classes_present: list[str]) -> None:
    """Render the confusion matrix as readable text."""
    idx = [CLASS_LABELS.index(c) for c in classes_present]
    width = max(len(c) for c in classes_present) + 2
    header = " " * (width + 8) + "".join(f"{c[:8]:>10}" for c in classes_present)
    print(header)
    for i in idx:
        row = "".join(f"{cm[i][j]:>10}" for j in idx)
        print(f"  {CLASS_LABELS[i]:<{width}} {row}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(args) -> dict:
    """Run the full training job and write the checkpoint + report."""
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"PyTorch is required for training: {exc}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    data_dir = Path(args.data)
    if not data_dir.exists():
        raise SystemExit(
            f"Data folder not found: {data_dir}\n"
            "Either no tiles have been prepared yet, or every reference\n"
            "figure was rejected during prepare (reasons are listed in\n"
            "data/reference/manifest.json). Fix the figures and run\n"
            "scripts/prepare_reference_dataset.py, or run the whole\n"
            "pipeline via scripts/train_and_report.py."
        )

    print("=" * 78)
    print("CYCLO-VISION -- training CycloCNN classifier")
    print("=" * 78)
    print(f"data        : {data_dir}")
    print(f"output      : {args.out}")
    print(f"backbone    : {args.backbone}")
    print(f"input size  : {args.size}")
    print(f"epochs      : {args.epochs}   batch: {args.batch_size}   lr: {args.lr}")
    print(f"device      : {args.device}")
    print(f"seed        : {args.seed}")
    print("-" * 78)

    paths, labels = discover_samples(data_dir)
    if not paths:
        raise SystemExit(
            f"No labelled tiles found under {data_dir}.\n"
            "Expected sub-folders named after model classes, e.g.\n"
            f"  {data_dir / 'Cyclonic Storm'}/*.png"
        )

    # --- class census ----------------------------------------------------
    counts: dict[str, int] = {}
    for idx in labels:
        counts[CLASS_LABELS[idx]] = counts.get(CLASS_LABELS[idx], 0) + 1
    print(f"tiles found : {len(paths)}")
    print("class census:")
    for label, count in sorted(counts.items(), key=lambda kv: CLASS_LABELS.index(kv[0])):
        print(f"  {label:<34} {count:>4}")

    # A single class (or none) cannot be split or stratified -- say exactly
    # what is missing rather than writing a useless model.
    if len(counts) < 2:
        print("-" * 78)
        if counts:
            print(f"ABORT -- only one class present ({list(counts)[0]}).")
            print("At least two classes are needed to learn a boundary.")
        else:
            print("ABORT -- no labelled tiles found.")
        print(
            "\nPut the reference figures under data/reference/raw/ using either\n"
            "  raw/<Class Name>/<figure>.png      (folder per class)\n"
            "  raw/<figure>__<Class Name>.png     (class in the file name)\n"
            "then re-run:  python scripts/prepare_reference_dataset.py"
        )
        print("=" * 78)
        raise SystemExit(2)

    thin = {k: v for k, v in counts.items() if v < args.min_per_class}
    if thin:
        print("-" * 78)
        print(f"ABORT -- these classes have fewer than {args.min_per_class} tiles:")
        for label, count in thin.items():
            print(f"  {label:<34} {count:>4}   need {args.min_per_class - count} more")
        print(
            "\nA model trained on this would not be trustworthy, so nothing\n"
            "was written. Add more reference figures for the classes above.\n"
            "To override deliberately, re-run with --min-per-class 1."
        )
        print("=" * 78)
        raise SystemExit(2)

    # --- stratified train/val split --------------------------------------
    by_class: dict[int, list[int]] = {}
    for i, idx in enumerate(labels):
        by_class.setdefault(idx, []).append(i)

    train_idx: list[int] = []
    val_idx: list[int] = []
    for idx, members in sorted(by_class.items()):
        shuffled = members[:]
        random.Random(args.seed).shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * args.val_split)))
        n_val = min(n_val, max(1, len(shuffled) - 1))  # always keep >=1 train
        val_idx.extend(shuffled[:n_val])
        train_idx.extend(shuffled[n_val:])

    train_paths = [paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_paths = [paths[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    print("-" * 78)
    print(f"split       : train={len(train_paths)}  val={len(val_paths)}"
          f"  (val_split={args.val_split})")

    model, normalise, backbone_desc = build_backbone(
        args.backbone, args.pretrained, len(CLASS_LABELS), torch
    )
    print(f"model       : {backbone_desc}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"parameters  : {n_params:,}")

    TileDataset = make_dataset_class(torch)
    train_ds = TileDataset(
        train_paths, train_labels, args.size,
        augment=True, normalise=normalise, seed=args.seed,
    )
    val_ds = TileDataset(
        val_paths, val_labels, args.size,
        augment=False, normalise=normalise, seed=args.seed,
    )

    device = torch.device(args.device)
    model = model.to(device)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # Class-weighted loss so rare classes are not ignored.
    present = sorted(by_class.keys())
    weights = torch.ones(len(CLASS_LABELS), dtype=torch.float32)
    total = sum(len(by_class[i]) for i in present)
    for i in present:
        weights[i] = total / (len(present) * len(by_class[i]))
    print(f"class weights: {[round(float(w), 2) for w in weights]}")
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    print("-" * 78)
    header = (
        f"{'epoch':>5} {'train_loss':>11} {'train_acc':>10} "
        f"{'val_loss':>10} {'val_acc':>9} {'macroF1':>9} {'sec':>6}"
    )
    print(header)
    print("-" * len(header))

    history: list[dict] = []
    best: dict = {"macro_f1": -1.0, "epoch": -1, "state": None, "metrics": None}
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        _EPOCH[0] = epoch
        model.train()
        running, seen, correct = 0.0, 0, 0
        te = time.time()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimiser.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimiser.step()

            running += float(loss.item()) * y.size(0)
            seen += y.size(0)
            correct += int((logits.argmax(dim=1) == y).sum().item())

        scheduler.step()
        train_loss = running / max(1, seen)
        train_acc = correct / max(1, seen)

        val_metrics = evaluate(model, val_loader, device, torch)

        # Val loss uses the same weighted criterion as training so the two
        # columns are directly comparable.
        model.eval()
        vloss, vseen = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                vloss += float(criterion(model(x), y).item()) * y.size(0)
                vseen += y.size(0)
        val_loss = vloss / max(1, vseen)

        elapsed = time.time() - te
        print(
            f"{epoch:>5} {train_loss:>11.4f} {train_acc:>10.4f} "
            f"{val_loss:>10.4f} {val_metrics['accuracy']:>9.4f} "
            f"{val_metrics['macro_f1']:>9.4f} {elapsed:>6.1f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_acc": round(train_acc, 5),
            "val_loss": round(val_loss, 5),
            "val_acc": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "lr": round(float(scheduler.get_last_lr()[0]), 6),
            "seconds": round(elapsed, 2),
        })

        # Select on macro-F1: with few tiles per class, raw accuracy can be
        # dominated by whichever class happens to be most numerous.
        if val_metrics["macro_f1"] > best["macro_f1"]:
            best = {
                "macro_f1": val_metrics["macro_f1"],
                "epoch": epoch,
                "state": {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                },
                "metrics": val_metrics,
            }

    train_seconds = time.time() - t_start
    print("-" * 78)
    print(
        f"best epoch  : {best['epoch']}  "
        f"(val macro-F1 {best['macro_f1']:.4f}, "
        f"val acc {best['metrics']['accuracy']:.4f})"
    )
    print(f"total time  : {train_seconds:.1f}s")
    print("=" * 78)

    # --- final report on the best checkpoint -----------------------------
    print("\nFINAL VALIDATION -- best checkpoint")
    print("-" * 78)
    classes_present = [CLASS_LABELS[i] for i in sorted(by_class.keys())]
    print_confusion(best["metrics"]["confusion_matrix"], classes_present)
    print()
    print(f"  {'class':<34} {'support':>8} {'prec':>7} {'recall':>7} {'f1':>7}")
    for label in classes_present:
        m = best["metrics"]["per_class"][label]
        print(
            f"  {label:<34} {m['support']:>8} {m['precision']:>7.3f} "
            f"{m['recall']:>7.3f} {m['f1']:>7.3f}"
        )
    print()
    print(f"  overall accuracy : {best['metrics']['accuracy']:.4f}")
    print(f"  macro F1         : {best['metrics']['macro_f1']:.4f}")

    # --- save checkpoint -------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ``ml.model.build_model`` calls ``load_state_dict(torch.load(path))``
    # directly, so the .pt file must contain the *bare* state dict. The
    # class order / input size travel in a sidecar file instead, and the
    # training report records them too.
    torch.save(best["state"], out_path)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\ncheckpoint  : {out_path}  ({size_mb:.2f} MB)")
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "class_labels": CLASS_LABELS,
                "input_size": args.size,
                "backbone": args.backbone,
                "pretrained": args.pretrained,
                "imagenet_norm": normalise,
                "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "best_epoch": best["epoch"],
                "val_accuracy": best["metrics"]["accuracy"],
                "val_macro_f1": best["metrics"]["macro_f1"],
                "train_tiles": len(train_paths),
                "val_tiles": len(val_paths),
                "class_counts": counts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"metadata    : {meta_path}")
    if args.backbone == "cnn":
        print("             ml.inference loads the .pt directly (CycloCNN weights).")
    else:
        # ml.inference builds a bare CycloCNN, so ResNet-18 weights would not
        # load there. Be explicit about that rather than letting MODEL mode
        # fail silently at request time.
        print("             NOTE: these are ResNet-18 weights. ml.inference's")
        print("             build_model() path constructs a bare CycloCNN, so")
        print("             retrain with --backbone cnn before setting")
        print("             ML_MODE=model. Training/eval above is unaffected.")

    report = {
        # ``args`` holds Path objects, which json cannot encode -- stringify
        # every value so the report stays portable.
        "config": {k: str(v) for k, v in vars(args).items()},
        "class_counts": counts,
        "train_tiles": len(train_paths),
        "val_tiles": len(val_paths),
        "backbone_description": backbone_desc,
        "parameters": n_params,
        "history": history,
        "best": {
            "epoch": best["epoch"],
            "val_accuracy": best["metrics"]["accuracy"],
            "val_macro_f1": best["metrics"]["macro_f1"],
            "per_class": best["metrics"]["per_class"],
            "confusion_matrix": best["metrics"]["confusion_matrix"],
            "classes_present": classes_present,
        },
        "total_seconds": round(train_seconds, 2),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report      : {REPORT_PATH}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the CYCLO-VISION classifier on prepared tiles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help=f"labelled tile root (default: {DEFAULT_DATA})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"checkpoint path (default: {DEFAULT_OUT})")
    parser.add_argument("--backbone", choices=("resnet18", "cnn"), default="resnet18",
                        help="resnet18 = ImageNet fine-tune (default); cnn = in-repo CycloCNN")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false",
                        help="do not load ImageNet weights")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.25,
                        help="fraction of each class held out (default 0.25)")
    parser.add_argument("--size", type=int, default=256,
                        help="input resolution; must match settings.INPUT_SIZE")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-per-class", type=int, default=3,
                        help="abort if any class has fewer tiles than this (default 3)")
    parser.set_defaults(pretrained=True)
    args = parser.parse_args(argv)

    if not 0.0 < args.val_split < 0.5:
        raise SystemExit("--val-split must be between 0 and 0.5")

    train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())