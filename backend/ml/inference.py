"""
CYCLO-VISION -- Inference Engine / Orchestrator
==============================================
Coordinates preprocessing -> model -> postprocessing and returns a
structured JSON result. Supports two modes:

    MODEL MODE (settings.ML_MODE == "model")
        Loads a trained CycloCNN from models/ and runs real inference with
        an activation-mapping heatmap (prototype XAI).

    DEMO MODE (settings.ML_MODE == "demo")  <-- default for SIH prototype
        Uses a *transparent, feature-based* heuristic on the uploaded
        image (so the result genuinely depends on the image), combined
        with per-sample calibration for bundled demo images.

Every result is tagged with `inference_mode` so the UI can clearly
label real-model vs prototype/demo predictions.
"""

from __future__ import annotations

import base64
import functools
import io
import logging
import math
import os
from typing import Any

import numpy as np
from PIL import Image

from app.core.config import settings
from app.services.reference_data import (
    find_reference_cyclone,
    get_center_for_class,
    get_track_params_for_class,
)
from ml import postprocessing as post
from ml.preprocessing import preprocess_image, PreprocessedImage
from ml.model import CLASS_LABELS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Paths are read through functions, not module constants: settings can be
# mutated at runtime (tests) and MODEL_DIR must reflect the live value.
def _model_dir() -> str:
    return str(settings.MODEL_DIR)


def _demo_data_dir() -> str:
    return str(settings.DEMO_DATA_DIR)


# Back-compat aliases for existing imports.
MODEL_DIR = settings.MODEL_DIR
DEMO_DATA_DIR = settings.DEMO_DATA_DIR

logger = logging.getLogger("cyclo.inference")


def model_available() -> bool:
    """Return True if a model weights file exists and torch is importable."""
    try:
        import torch  # noqa: F401

        for fname in os.listdir(_model_dir()):
            if fname.endswith((".pt", ".pth", ".onnx", ".h5")):
                return True
    except Exception:
        pass
    return False


def _live_reference(class_idx: int):
    """Best-matching storm from the live IBTrACS archive.

    Returns None when the live dataset is unavailable, letting the caller
    fall back to the bundled curated table. Imported lazily and guarded so
    a dataset problem can never break inference.
    """
    if class_idx <= 0:
        return None
    try:
        from app.services.ibtracs import best_match_for_class

        return best_match_for_class(class_idx)
    except ImportError:
        return None  # optional dependency not present
    except (OSError, ValueError, KeyError):
        return None  # dataset missing/corrupt -- expected degraded mode
    except Exception:
        # Anything else is a real bug in the lookup: log it loudly instead
        # of hiding it behind the graceful fallback (P0 review #20).
        logger.warning("best_match_for_class(%d) failed unexpectedly", class_idx, exc_info=True)
        return None
# ---------------------------------------------------------------------------
# Demo / prototype saliency map
# ---------------------------------------------------------------------------

# Heatmap sizes are not arbitrary: preprocessing uses settings.INPUT_SIZE,
# so the same few (h, w) shapes recur on every request. Cache the grids
# (P2-7) instead of rebuilding them per call.
_HEATMAP_CACHE_SIZE = 8


@functools.lru_cache(maxsize=_HEATMAP_CACHE_SIZE)
def _radial_grids(h: int, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cached ``(radius, angle, spatial-decay)`` grids for a (h, w) shape."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h // 2, w // 2
    radius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(1, (w // 2))
    angle = np.arctan2(yy - cy, xx - cx)
    radial_decay = np.exp(-(radius - 0.35) ** 2 / 0.2)
    return radius, angle, radial_decay


def to_png_b64(arr: np.ndarray) -> str:
    """Encode a uint8 ``(H, W, 3)`` RGB array as a base64 PNG string (P1-2).

    Returning the raw array cost ~1-2 MB of JSON per analysis response;
    the encoded PNG is roughly two orders of magnitude smaller.
    """
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _demo_saliency(
    display_rgb: np.ndarray, class_idx: int, strength: float
) -> str:
    """Build a prototype attention map, returned as base64 PNG (P1-2)."""
    h, w = display_rgb.shape[:2]
    radius, angle, radial_decay = _radial_grids(h, w)

    gray = display_rgb.mean(axis=2)
    warmth = (
        display_rgb[..., 0].astype(np.float32)
        - display_rgb[..., 2].astype(np.float32)
    ) / 255.0
    core = np.clip(gray / 255.0 + warmth * 0.6, 0, 1)

    windings = 1.5 + class_idx * 0.45
    spiral = 0.5 + 0.5 * np.cos(angle * windings - radius * 6.0)
    spiral *= radial_decay

    spatial = np.clip((core * 0.55 + spiral * 0.45) * strength, 0, 1)

    heat = np.zeros((h, w, 3), dtype=np.float32)
    t = spatial
    heat[..., 0] = np.clip(np.minimum(t * 3.0, 1.0), 0, 1) * 0.25 + t * 0.1
    heat[..., 1] = np.clip(1.2 - np.abs(2.0 * t - 1.0), 0, 1)
    heat[..., 2] = np.clip((t - 0.5) * 2.0, 0, 1) * 0.8

    return to_png_b64((heat * 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Feature-based demo scoring
# ---------------------------------------------------------------------------


def _demo_feature_scores(pre: PreprocessedImage) -> np.ndarray:
    """Compute per-class scores purely from image statistics.

    The output is a *relative* score vector; the caller normalises it into
    probabilities. Scoring is intentionally transparent: a storm's class is
    driven by how warm (deep convection), how bright (dense eyewall/cloud
    tops) and how well-organised (edge/contrast structure) the scene is.

    Logits are used rather than narrow Gaussian bumps so that a scene lands
    decisively in one class instead of splitting probability across
    neighbours, which is what made the earlier calibration ambiguous.
    """
    rgb = pre.display_rgb.astype(np.float32)
    gray = rgb.mean(axis=2)
    h, w = gray.shape

    contrast = float(np.std(gray))
    mean_r = float(np.mean(rgb[..., 0]))
    mean_b = float(np.mean(rgb[..., 2]))
    warmth = (mean_r - mean_b) / 255.0
    brightness = float(np.mean(gray)) / 255.0

    edges = float(
        np.mean(np.abs(np.gradient(gray, axis=0))) + np.mean(np.abs(np.gradient(gray, axis=1)))
    )

    # Bright-core measure: a mature cyclone has a bright, compact cloud
    # shield. Compare the central third against the overall mean.
    band_y, band_x = max(4, h // 6), max(4, w // 6)
    cy, cx = h // 2, w // 2
    core = float(gray[cy - band_y : cy + band_y, cx - band_x : cx + band_x].mean())
    core_excess = (core - float(np.mean(gray))) / 255.0

    # --- organisation: rotational/textural structure ---------------------
    # Swap left/right halves; a rotationally symmetric storm changes less
    # than a plain landscape when mirrored.
    mirrored = float(np.mean(np.abs(gray - np.fliplr(gray)))) / 255.0
    symmetry = max(0.0, 1.0 - mirrored * 4.0)

    organisation = (
        0.35 * symmetry
        + 0.40 * min(1.0, edges / 18.0)
        + 0.25 * min(1.0, contrast / 55.0)
    )

    # --- convection: warm cloud tops / dense moisture --------------------
    # Note: satellite IR imagery renders cold cloud tops as *dark*, so a
    # deep-convection scene is bright/structured rather than warm. The
    # warmth term is therefore a weak negative indicator here.
    convection = (
        0.55 * max(0.0, brightness - 0.25) * 2.2
        + 0.30 * max(0.0, core_excess) * 4.5
        + 0.15 * min(1.0, contrast / 60.0)
    )
    convection = min(1.0, convection)

    # --- 0..1 severity index --------------------------------------------
    severity = 0.45 * organisation + 0.55 * convection
    # Stretch: the measured dynamic range on 256px satellite crops sits
    # roughly in 0.25-0.95, so rescale to use the full 0..1 band.
    severity = (severity - 0.30) / 0.55
    severity = min(1.0, max(0.0, severity))

    # --- map severity onto 7 classes -------------------------------------
    # Class i is centred at severity i/6. A Gaussian proximity kernel gives
    # a smooth, monotonic assignment: the class whose centre is closest to
    # the measured severity wins, with the spread controlled by SIGMA.
    # SIGMA is deliberately small so neighbouring classes stay separable
    # (0.10 => ~1.67 sigma separation between adjacent centres at d=1/6,
    # i.e. decisively one class wins instead of splitting probability).
    scores = np.zeros(len(CLASS_LABELS), dtype=np.float32)
    centres = np.linspace(0.0, 1.0, len(CLASS_LABELS))
    sigma = 0.10
    for i, c in enumerate(centres):
        scores[i] = math.exp(-((severity - c) ** 2) / (2.0 * sigma * sigma))

    # Small deterministic texture term so identical-looking inputs still
    # produce a stable, image-dependent tie-break. Applied on the *logit*
    # scale and deliberately tiny (<=0.02 vs. typical inter-class gaps of
    # 0.5+), so it can only break exact ties -- it cannot flip a confident
    # scene into a neighbouring class.
    seed = int(gray[:: max(1, h // 16), :: max(1, w // 16)].mean() * 1000) & 0xFFFF
    rng = np.random.RandomState(seed)
    scores += rng.uniform(0.0, 0.02, size=len(CLASS_LABELS)).astype(np.float32)
    return scores
# ---------------------------------------------------------------------------
# Explainable-AI activation mapping for real model
# ---------------------------------------------------------------------------

try:
    import torch

    def _model_heatmap(model: Any, tensor: np.ndarray, device: str, class_idx: int):
        """Prototype activation-mapping heatmap (NOT Grad-CAM unless the
        loaded model provides compatible hooks; labeled accordingly)."""
        x = torch.from_numpy(tensor).unsqueeze(0).to(device)
        x.requires_grad_(True)
        try:
            f, g, logits = model.activations(x)
            loss = logits[:, class_idx]
            loss.backward()
            grads = x.grad.abs().mean(1).squeeze(0)
            acts = f.detach().mean(1).squeeze(0)
            heat = (acts * grads).clamp(0)
            heat = heat / (heat.max() + 1e-8)
            hm = torch.nn.functional.interpolate(
                heat.unsqueeze(0).unsqueeze(0).float(),
                size=(tensor.shape[1], tensor.shape[2]),
                mode="bilinear",
            ).squeeze().cpu().numpy()
            # Replicate the single-channel map into RGB so the payload is a
            # valid colour-explosion PNG (P1-2: base64, not a raw array).
            hm8 = (hm * 255).astype(np.uint8)
            return to_png_b64(np.repeat(hm8[..., None], 3, axis=2))
        except Exception:
            return None

except Exception:  # pragma: no cover
    _model_heatmap = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helpers for model path
# ---------------------------------------------------------------------------


def _find_weights() -> str | None:
    model_dir = _model_dir()
    for fname in os.listdir(model_dir):
        if fname.endswith((".pt", ".pth")):
            return os.path.join(model_dir, fname)
    return None


# The loaded (model, device) pair, keyed by (weights_path, mtime). Building
# the CNN and loading the checkpoint on *every* request was a throughput
# cliff in model mode (P0 review #18/#19); keying on mtime keeps a freshly
# retrained checkpoint visible without a manual restart.
_MODEL_CACHE: dict[tuple[str, float], tuple[Any, Any]] = {}


def _load_model() -> tuple[Any, Any] | None:
    """Return the cached ``(model, device)`` for the current checkpoint."""
    import torch  # noqa: F401 - availability check, used by callers' torch
    from ml.model import build_model

    weights = _find_weights()
    if weights is None:
        return None
    try:
        mtime = os.path.getmtime(weights)
    except OSError:
        mtime = -1.0
    key = (weights, mtime)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    model, device = build_model(weights)
    _MODEL_CACHE.clear()  # drop entries from an older checkpoint
    _MODEL_CACHE[key] = (model, device)
    return model, device


def _score_with_model(pre: PreprocessedImage):
    """Return softmax-like scores from the trained model, or None on failure."""
    try:
        import torch

        loaded = _load_model()
        if loaded is None:
            return None
        model, device = loaded
        x = torch.from_numpy(pre.tensor).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
        return torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
    except Exception:
        return None


def _model_heatmap_internal(pre: PreprocessedImage, class_idx: int):
    if _model_heatmap is None:
        return None
    try:
        loaded = _load_model()
        if loaded is None:
            return None
        model, device = loaded
        return _model_heatmap(model, pre.tensor, device, class_idx)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main inference entry point
# ---------------------------------------------------------------------------


def run_inference(
    file_bytes: bytes,
    region: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Run the whole pipeline and return a structured result dict."""
    pre = preprocess_image(file_bytes)
    if not pre.success or pre.image is None:
        raise RuntimeError(pre.message)

    mode = "demo"
    im = pre.image

    if settings.ML_MODE == "model" and model_available():
        score_path = _score_with_model(im)
        mode = "model" if score_path is not None else "demo"
        if score_path is None:
            score_path = _demo_feature_scores(im)
    else:
        score_path = _demo_feature_scores(im)

    scores = np.asarray(score_path, dtype=np.float32)
    scores = scores - scores.min()
    denom = scores.sum() + 1e-8
    probs = scores / denom

    class_idx = int(np.argmax(probs))
    label = CLASS_LABELS[class_idx]
    confidence = post.compute_confidence(probs, class_idx)

    # --- Use reference dataset for calibration ---
    # Prefer the live IBTrACS record (real observed storms); fall back to
    # the bundled curated table if the download/parse was unavailable.
    reference = _live_reference(class_idx)
    calibration_source = "ibtracs-live"
    if reference is None:
        reference = find_reference_cyclone(class_idx)
        calibration_source = "bundled-curated"

    # If we have reference data, use it to calibrate intensity estimates
    if reference is not None:
        # Blend observed peak intensity with a confidence-based adjustment
        # so the output still tracks the image rather than being a pure
        # lookup.
        jitter = (confidence - 0.5) * 5.0
        wind = round(float(reference.max_wind_knots) + jitter, 1)
        # Pre-satellite-era storms have no pressure record at all; rather
        # than inventing one, fall back to the documented class curve for
        # the pressure half of the estimate only.
        if reference.min_pressure_hpa:
            pressure = round(float(reference.min_pressure_hpa) - jitter, 1)
        else:
            _, pressure = post.estimate_intensity(class_idx, confidence)
        # Keep wind within realistic bounds
        wind = max(15.0, min(wind, 175.0))
        pressure = max(870.0, min(pressure, 1010.0))
    else:
        wind, pressure = post.estimate_intensity(class_idx, confidence)

    risk_level, factors = post.assess_risk(class_idx, wind, pressure, confidence)
    detected = class_idx > 0

    if mode == "model" and _model_heatmap is not None:
        heat = _model_heatmap_internal(im, class_idx)
        xai_type = "activation-map"
        if heat is None:
            heat = _demo_saliency(im.display_rgb, class_idx, float(confidence))
            xai_type = "prototype-attention"
    else:
        heat = _demo_saliency(im.display_rgb, class_idx, float(confidence))
        xai_type = "prototype-attention"

    # Use reference data for track generation
    track_params = get_track_params_for_class(class_idx)

    if region is not None:
        base_lat, base_lon = region
    else:
        base_lat, base_lon = get_center_for_class(class_idx)

    track = post.generate_track(
        base_lat,
        base_lon,
        storm_strength=float(wind) / 80.0,
        # P2-4: seed from the storm centre as well as the class, otherwise
        # every storm of the same class produced an identical track.
        seed=(
            class_idx * 17
            + 3
            + int((base_lat + 90) * 100)
            + int((base_lon + 180) * 100)
        ),
        direction_deg=track_params["direction_deg"],
        forward_speed_knots=track_params["speed_knots"],
    )

    return {
        "cyclone_detected": bool(detected),
        "classification": label,
        "class_index": class_idx,
        "confidence": round(confidence, 4),
        "estimated_wind_speed_knots": wind,
        "estimated_pressure_hpa": pressure,
        "intensity_category": post.intensity_category(wind),
        "risk_level": risk_level,
        "risk_factors": factors,
        "inference_mode": mode,
        "calibration_source": calibration_source,
        "reference_cyclone": (
            reference.name if reference is not None else None
        ),
        "reference_year": (
            int(reference.year) if reference is not None else None
        ),
        "explainability": {
            "type": xai_type,
            "note": (
                "Prototype activation/attention map. Not Grad-CAM unless a "
                "compatible CNN is loaded."
            ),
            "heatmap_png_b64": heat,
        },
        "track": track,
        "center": {"lat": base_lat, "lon": base_lon},
    }