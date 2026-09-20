"""
CYCLO-VISION -- train CycloCNN on labeled TC imagery
====================================================

Usage
-----
    cd backend
    python -m scripts.train_cnn --manifest ../data/manifest.csv \
        --img-dir ../data/hursat_png

Outputs a **bare state dict** at ``--out`` plus a sidecar
``<out>.json`` (classes, input size, val accuracy) -- the exact format
``ml.model.build_model`` and ``ml.inference._load_model`` consume. After
training, set ``ML_MODE=model`` and restart the API.

The two things that make or break accuracy live here:

1. **Storm-level split** (``split_by_storm``): GroupShuffleSplit on SID, never
   on rows. Random row splits leak 20+ frames of the same storm into both
   splits and val accuracy reads ~20 points better than reality.
2. **Class-balanced sampling** (``make_sampler``): IBTrACS-NI is heavily
   skewed (a single Depression vs hundreds of Cyclonic Storms). Without
   inverse-frequency sampling the model predicts the majority class for
   everything.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from ml.dataset import CycloneImageDataset, wind_to_class
from ml.model import CLASS_LABELS, CycloCNN

NUM_CLASSES = len(CLASS_LABELS)


def build_transforms(size: int):
    """Train/val transforms.

    Random flips + 180-degree rotation are legitimate here: IR cloud fields
    have no canonical "up", so these add diversity without label noise.
    """
    train_tf = transforms.Compose(
        [
            transforms.Resize(int(size * 1.15)),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),  # /255 -- matches serving _normalize
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
        ]
    )
    return train_tf, val_tf


def make_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """Inverse-frequency sampling so rare classes aren't drowned out."""
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float32)
    counts = np.where(counts == 0, 1.0, counts)
    weights = (1.0 / counts)[labels]
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights).double(),
        num_samples=len(labels),
        replacement=True,
    )


def split_by_storm(manifest: str | Path, val_frac: float = 0.15, seed: int = 42):
    """Split on SID -- NOT on filename (see module docstring)."""
    df = pd.read_csv(manifest)
    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr_idx, va_idx = next(gss.split(df, groups=df["sid"]))
    tr = df.iloc[tr_idx].reset_index(drop=True)
    va = df.iloc[va_idx].reset_index(drop=True)
    tr_path = Path(str(manifest).replace(".csv", "_train.csv"))
    va_path = Path(str(manifest).replace(".csv", "_val.csv"))
    tr.to_csv(tr_path, index=False)
    va.to_csv(va_path, index=False)
    return tr, va

def train(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    tr_df, va_df = split_by_storm(args.manifest, args.val_frac)
    print(
        f"storm-level split: {tr_df['sid'].nunique()} train storms / "
        f"{va_df['sid'].nunique()} val storms, {len(tr_df)}/{len(va_df)} frames"
    )
    size = args.size
    train_tf, val_tf = build_transforms(size)

    train_ds = CycloneImageDataset(
        Path(str(args.manifest).replace(".csv", "_train.csv")),
        args.img_dir,
        transform=train_tf,
        preprocess=not args.no_preprocess,
    )
    val_ds = CycloneImageDataset(
        Path(str(args.manifest).replace(".csv", "_val.csv")),
        args.img_dir,
        transform=val_tf,
        preprocess=not args.no_preprocess,
    )

    train_labels = np.array([wind_to_class(w) for w in tr_df["usa_wind"]])
    sampler = make_sampler(train_labels)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=device == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device == "cuda",
    )

    model = CycloCNN(in_channels=3, num_classes=NUM_CLASSES).to(device)

    # Label smoothing: 64 vs 70 kt is not a meaningful physical difference.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    best_acc = 0.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        tl, tn = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device).long()
            optim.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optim.step()
            tl += loss.item() * x.size(0)
            tn += x.size(0)
        sched.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device).long()
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        vacc = correct / max(1, total)
        print(f"epoch {epoch:03d}  train_loss {tl / tn:.4f}  val_acc {vacc:.4f}")

        if vacc >= best_acc:
            best_acc = vacc
            _save(out_path, model, size, vacc, epoch + 1, not args.no_preprocess)
            print(f"  -> saved {out_path} (val_acc {vacc:.4f})")

    print(f"best val acc: {best_acc:.4f}")


def _save(
    out_path: Path,
    model: CycloCNN,
    size: int,
    vacc: float,
    epochs_done: int,
    preprocess_at_serve: bool,
) -> None:
    """Bare state dict + sidecar JSON -- what the serving stack expects."""
    torch.save(model.state_dict(), out_path)
    Path(str(out_path) + ".json").write_text(
        json.dumps(
            {
                "classes": CLASS_LABELS,
                "size": size,
                "val_acc": vacc,
                "epochs": epochs_done,
                "preprocess_at_serve": preprocess_at_serve,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train CycloCNN on labeled imagery")
    p.add_argument("--manifest", default="../data/manifest.csv")
    p.add_argument("--img-dir", default="../data/hursat_png")
    p.add_argument("--out", default="../models/cyclo_cnn.pt")
    p.add_argument("--size", type=int, default=256, help="match app INPUT_SIZE")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--workers", type=int, default=0, help="0 is safest on Windows")
    p.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Skip denoise/CLAHE in the dataset (only if images on disk are "
        "already preprocessed exactly like the serving path).",
    )
    train(p.parse_args())

