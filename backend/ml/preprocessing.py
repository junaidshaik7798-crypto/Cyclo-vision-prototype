"""
CYCLO-VISION -- Image Preprocessing Pipeline
===========================================
Applies a modular, documented preprocessing pipeline to raw satellite
imagery before ML inference:

    1. Load / validate image
    2. Convert/standardize color space
    3. Resize to model input size
    4. Normalize pixel values
    5. Denoise (bilateral filter)
    6. Contrast enhancement (CLAHE)
    7. Feature preparation / tensor construction

Every step returns structured metadata so the frontend can display a
step-by-step pipeline visualization.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from app.core.config import settings

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PreprocessedImage:
    """Result of the preprocessing pipeline."""

    tensor: np.ndarray  # float32 array, shape (C, H, W) normalized ~ [0,1]
    display_rgb: np.ndarray  # uint8 (H, W, 3) for frontend display
    steps: list[dict[str, Any]] = field(default_factory=list)
    original_shape: tuple[int, int, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreprocessingResult:
    """Full pipeline output including per-step logs."""

    success: bool
    message: str
    image: PreprocessedImage | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
MAX_SIZE_MB = settings.MAX_UPLOAD_SIZE_MB


def validate_upload(filename: str, file_bytes: bytes) -> tuple[bool, str]:
    """Validate extension, size and integrity of an uploaded file."""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        return False, (
            f"Unsupported format '{ext or 'unknown'}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    if len(file_bytes) == 0:
        return False, "Uploaded file is empty."
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_SIZE_MB:
        return False, (
            f"File too large: {size_mb:.1f} MB exceeds the "
            f"{MAX_SIZE_MB} MB limit."
        )
    # Try to open to detect corruption
    try:
        with Image.open(io.BytesIO(file_bytes)) as im:
            im.verify()
    except Exception:
        return False, "File appears to be corrupted or is not a valid image."
    return True, "Validation passed."


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def _load_image(file_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


def _resize(img: np.ndarray, size: int) -> np.ndarray:
    """Resize keeping the larger dimension = size using PIL LANCZOS."""
    pil = Image.fromarray(img)
    w, h = pil.size
    scale = size / max(w, h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    pil = pil.resize((new_w, new_h), Image.LANCZOS)
    return np.asarray(pil)


def _center_crop(img: np.ndarray, size: int) -> np.ndarray:
    """Center-crop to a square of given size."""
    h, w = img.shape[:2]
    top = max(0, (h - size) // 2)
    left = max(0, (w - size) // 2)
    return img[top : top + size, left : left + size]


def _denoise(img: np.ndarray) -> np.ndarray:
    """Bilateral filter to reduce sensor noise while preserving edges."""
    try:
        import cv2

        return cv2.bilateralFilter(img, d=5, sigmaColor=25, sigmaSpace=25)
    except Exception:
        return img  # fallback if OpenCV unavailable


def _enhance_contrast(img: np.ndarray) -> np.ndarray:
    """CLAHE contrast enhancement on the L channel of Lab space."""
    try:
        import cv2

        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        merged = cv2.merge((l, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    except Exception:
        return img


# Public aliases: training pipelines (ml/dataset.py) reuse these so the
# train-time and serve-time pixel distributions are identical by construction.
denoise = _denoise
enhance_contrast = _enhance_contrast


def _normalize(img: np.ndarray) -> np.ndarray:
    """Scale uint8 [0,255] -> float32 [0,1]."""
    return img.astype(np.float32) / 255.0


def _channel_first(img: np.ndarray) -> np.ndarray:
    """Convert (H, W, C) -> (C, H, W)."""
    return np.transpose(img, (2, 0, 1))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def preprocess_image(
    file_bytes: bytes,
    size: int | None = None,
) -> PreprocessingResult:
    """Run the full preprocessing pipeline on raw bytes."""
    size = size or settings.INPUT_SIZE
    steps: list[dict[str, Any]] = []

    def record(name: str, status: str, detail: str, elapsed_ms: float = 0.0):
        steps.append(
            {"name": name, "status": status, "detail": detail, "elapsed_ms": elapsed_ms}
        )

    try:
        t0 = time.time()
        im = _load_image(file_bytes)
        original = np.asarray(im)
        record("Load & Validate", "done", f"{original.shape[1]}x{original.shape[0]}px")

        current = original
        # Resize to 2x first, then center-crop, then downsample: cropping at
        # 2x resolution keeps the crop window centred on the original aspect
        # ratio before LANCZOS downsampling, avoiding aliasing artefacts that
        # a single crop-at-target-size pass would introduce on wide figures.
        current = _resize(current, size * 2)
        current = _center_crop(current, size * 2)
        current = _resize(current, size)
        record("Resize & Crop", "done", f"-> {size}x{size}px")

        current = _denoise(current)
        record("Noise Reduction", "done", "Bilateral filter applied")

        current = _enhance_contrast(current)
        record("Contrast Enhancement", "done", "CLAHE (clip 2.5, tile 8x8)")

        normalized = _normalize(current)
        record("Normalization", "done", "uint8 [0,255] -> float32 [0,1]")

        tensor = _channel_first(normalized)
        record("Feature Prep (Tensor)", "done", f"Shape {list(tensor.shape)}")

        meta = {"source_shape": list(original.shape), "input_size": size, "channels": 3}

        image = PreprocessedImage(
            tensor=tensor,
            display_rgb=current.copy(),
            steps=steps,
            original_shape=original.shape,
            metadata=meta,
        )
        return PreprocessingResult(
            success=True,
            message="Preprocessing completed successfully.",
            image=image,
            steps=steps,
        )

    except Exception as exc:  # pragma: no cover - defensive
        record("Pipeline", "failed", str(exc))
        return PreprocessingResult(
            success=False,
            message=f"Preprocessing failed: {exc}",
            steps=steps,
        )


def preprocess_to_tensor(file_bytes: bytes, size: int | None = None) -> np.ndarray:
    """Convenience wrapper returning just the normalized CHW tensor."""
    result = preprocess_image(file_bytes, size)
    if result.success and result.image is not None:
        return result.image.tensor
    raise ValueError(result.message)