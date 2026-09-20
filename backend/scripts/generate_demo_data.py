"""
CYCLO-VISION -- Generate bundled demo satellite images.
=======================================================
Creates synthetic, satellite-style imagery under data/demo/ representing
four cyclone scenarios:

    sample_clear.png    -- No cyclone / calm ocean
    sample_dev.png      -- Developing low / disorganized convection
    sample_severe.png   -- Severe cyclone with visible eyewall
    sample_vsevere.png  -- Very severe cyclone, tight eye

These are NOT real satellite images -- they are computationally generated
textures used only to exercise the full pipeline for the SIH demo.
"""

from __future__ import annotations

import os
import numpy as np
from PIL import Image, ImageFilter

OUT_DIR = os.path.join("data", "demo")
SIZE = 512


def ocean_background(rng, h, w, base_bg=(25, 45, 90)):
    """Pseudo-ocean: smooth gradient + subtle noise."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    grad = (xx / w) * 8.0 + (yy / h) * 6.0
    noise = rng.normal(0, 4.0, (h, w, 3))
    arr = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        arr[..., c] = base_bg[c] + grad * 0.5 + noise[..., c]
    return np.clip(arr, 0, 255).astype(np.uint8)


def draw_cloud(rgb, cx, cy, r, intensity, rng, cloud_rgb=(230, 235, 245)):
    """Add a fuzzy cloud blob."""
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = np.exp(-((d / r) ** 2))
    # mottle
    nz = rng.uniform(-0.3, 0.3, size=(h, w))
    mask = np.clip(mask + nz * mask, 0, 1) * intensity
    # Cast rgb to float32 for safe arithmetic, then clip back to uint8
    rgb_f = rgb.astype(np.float32)
    for c in range(3):
        rgb_f[..., c] += mask * (cloud_rgb[c] - rgb_f[..., c])
    np.clip(rgb_f, 0, 255, out=rgb_f)
    return rgb_f.astype(np.uint8)


def draw_eye(rgb, cx, cy, r_inner, r_outer, intensity, rng, eye_rgb=(40, 55, 110)):
    """Draw a cyclone eye + surrounding ring."""
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    # eye hole (calm center)
    eye = np.exp(-((d / r_inner) ** 2)) * 0.95
    # bright eyewall ring
    ring = np.exp(-(((d - (r_inner + r_outer * 0.5)) / (r_outer * 0.4)) ** 2))
    ring = ring * intensity
    warm = (245, 240, 230)
    rgb_f = rgb.astype(np.float32)
    for c in range(3):
        rgb_f[..., c] += eye * (eye_rgb[c] - rgb_f[..., c])
        rgb_f[..., c] += ring * (warm[c] - rgb_f[..., c])
    np.clip(rgb_f, 0, 255, out=rgb_f)
    return rgb_f.astype(np.uint8)


def draw_bands(rgb, cx, cy, n, intensity, rng):
    """Draw spiral rain bands rotating around the center."""
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    angle = np.arctan2(yy - cy, xx - cx)
    radius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (w * 0.55)
    bands = np.zeros((h, w), dtype=np.float32)
    for i in range(n):
        offset = rng.uniform(-np.pi, np.pi)
        bands += 0.5 + 0.5 * np.cos(angle * (2 + i * 0.7) - radius * (3 + i * 1.8) + offset)
    bands = (bands / n) * intensity
    bands = np.clip(bands, 0, 1)
    col = (235, 238, 248)
    rgb_f = rgb.astype(np.float32)
    for c in range(3):
        rgb_f[..., c] += bands * (col[c] - rgb_f[..., c])
    np.clip(rgb_f, 0, 255, out=rgb_f)
    return rgb_f.astype(np.uint8), float(bands.max())


def save(rgb, path):
    img = Image.fromarray(rgb).convert("RGB")
    img = img.filter(ImageFilter.GaussianBlur(0.4))
    img.save(path)
    print(f"Wrote {path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.RandomState(1234)
    H = W = SIZE

    # 1) Clear / no cyclone
    rgb = ocean_background(rng, H, W)
    rgb = draw_cloud(rgb, W * 0.62, H * 0.35, 55, 0.5, rng)
    rgb = draw_cloud(rgb, W * 0.25, H * 0.7, 40, 0.35, rng)
    save(rgb, os.path.join(OUT_DIR, "sample_clear.png"))

    # 2) Developing low
    rng = np.random.RandomState(5678)
    rgb = ocean_background(rng, H, W)
    rgb = draw_cloud(rgb, W * 0.5, H * 0.5, 95, 0.7, rng)
    rgb, _ = draw_bands(rgb, W * 0.5, H * 0.5, 2, 0.4, rng)
    rgb = draw_eye(rgb, W * 0.5, H * 0.5, 12, 40, 0.55, rng, eye_rgb=(55, 70, 115))
    save(rgb, os.path.join(OUT_DIR, "sample_dev.png"))

    # 3) Severe cyclone
    rng = np.random.RandomState(9101)
    rgb = ocean_background(rng, H, W)
    rgb = draw_cloud(rgb, W * 0.5, H * 0.5, 120, 0.9, rng)
    rgb, _ = draw_bands(rgb, W * 0.5, H * 0.5, 3, 0.7, rng)
    rgb = draw_eye(rgb, W * 0.5, H * 0.5, 16, 60, 0.9, rng, eye_rgb=(45, 60, 105))
    save(rgb, os.path.join(OUT_DIR, "sample_severe.png"))

    # 4) Very severe cyclone
    rng = np.random.RandomState(11213)
    rgb = ocean_background(rng, H, W)
    rgb, _ = draw_bands(rgb, W * 0.5, H * 0.5, 4, 0.9, rng)
    rgb = draw_cloud(rgb, W * 0.5, H * 0.5, 100, 1.0, rng)
    rgb = draw_eye(rgb, W * 0.5, H * 0.5, 22, 70, 1.0, rng, eye_rgb=(35, 48, 92))
    save(rgb, os.path.join(OUT_DIR, "sample_vsevere.png"))


if __name__ == "__main__":
    main()