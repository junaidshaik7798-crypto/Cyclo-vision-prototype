"""
CYCLO-VISION -- PyTorch dataset for labeled TC imagery
======================================================

Reads a manifest CSV (``filename,usa_wind,usa_pres,sid,year,lat,lon``) plus an
image directory and yields tensors for training CycloCNN.

Consistency contract
--------------------
The dataset applies the *same* preprocessing as the serving path
(``ml.preprocessing.preprocess_image``: bilateral denoise + CLAHE) so the
train and serve distributions match exactly. Images are normalized with a
plain ``/255`` -- identical to ``_normalize`` at inference -- because the
from-scratch CycloCNN was trained that way. If you later swap in an
ImageNet-pretrained backbone, add the ImageNet mean/std to BOTH this dataset
and ``ml.preprocessing._normalize`` in the same commit.

Class labels are *not* re-derived here: ``wind_to_class`` delegates to
``ml.classification.class_index_for_wind`` (the single source of truth for
IMD bands), so the training labels can never drift from what the app reports.

Augmentation is the caller's job (see ``scripts/train_cnn.py``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from ml.classification import class_index_for_wind
from ml.preprocessing import denoise, enhance_contrast


def wind_to_class(w: float) -> int:
    """Map a 1-minute sustained wind (knots) to a CycloCNN class index."""
    return class_index_for_wind(float(w))


class CycloneImageDataset(Dataset):
    """Reads ``manifest.csv`` + image directory.

    Parameters
    ----------
    manifest:
        CSV with at least ``filename`` and ``usa_wind`` columns.
    img_dir:
        Directory containing the manifest's image files.
    transform:
        Optional torchvision transform applied *after* preprocessing
        (denoise + CLAHE). Receives and returns a PIL image.
    preprocess:
        Apply the serve-path denoise + CLAHE. Keep True unless the images on
        disk were already preprocessed the same way.
    target:
        ``"class"`` -> class index; ``"wind"`` -> knots; ``"both"`` -> dict.
    """

    def __init__(
        self,
        manifest: str | Path,
        img_dir: str | Path,
        transform=None,
        preprocess: bool = True,
        target: str = "class",  # "class" | "wind" | "both"
    ):
        self.df = pd.read_csv(manifest)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.preprocess = preprocess
        self.target = target

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(self.img_dir / row["filename"]).convert("RGB")
        arr = np.asarray(img)

        if self.preprocess:
            arr = denoise(arr)
            arr = enhance_contrast(arr)

        img = Image.fromarray(arr)
        if self.transform is not None:
            img = self.transform(img)

        wind = float(row["usa_wind"])
        if self.target == "class":
            y: object = wind_to_class(wind)
        elif self.target == "wind":
            y = wind
        else:
            y = {"class": wind_to_class(wind), "wind": wind}
        return img, y
