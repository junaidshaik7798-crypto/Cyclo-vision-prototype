"""CYCLO-VISION -- Analysis API endpoints."""

from __future__ import annotations

import logging
import json
import math
import re

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import UnidentifiedImageError

from app.schemas import AnalyzeRequest, AnalysisResult
from app.services.analysis import analyze_image
from app.services.demo_data import get_demo_bytes, list_demo_samples
from ml import postprocessing as post
from ml.preprocessing import validate_upload

logger = logging.getLogger("cyclo.api.analyze")
router = APIRouter(prefix="/analyze", tags=["analysis"])


def _safe_filename(name: str) -> str:
    """Sanitize a filename -- strip path and keep only safe characters."""
    base = name.replace("\\", "/").split("/")[-1]
    base = re.sub(r"[^A-Za-z0-9_.-]", "_", base)
    base = base[:120] or "upload"
    # P2-3: normalise the extension to lower case so "IMG.PNG"/"Pic.Jpg"
    # survive validate_upload's extension check unchanged.
    stem, dot, ext = base.rpartition(".")
    if not dot or not stem:
        return base
    return f"{stem}.{ext.lower()}"


def _to_response(
    result: dict,
    default_source: str,
    default_name: str,
    demo_sample: dict | None = None,
) -> AnalysisResult:
    """Build the API response, keeping the P0-1 contract fields explicit.

    The pass-through keys are filtered out of ``**result`` because
    ``analyze_image`` already sets them; passing both would be a duplicate
    keyword argument.
    """
    payload = {
        k: v
        for k, v in result.items()
        if k not in ("analysis_id", "source", "image_name")
    }

    # Keep the public response usable even when an older inference worker
    # returns a partial result. Recompute only missing/non-finite estimates;
    # valid calibrated values from inference are left unchanged.
    wind = payload.get("estimated_wind_speed_knots")
    pressure = payload.get("estimated_pressure_hpa")
    if not (
        isinstance(wind, (int, float))
        and math.isfinite(float(wind))
        and isinstance(pressure, (int, float))
        and math.isfinite(float(pressure))
    ):
        fallback_wind, fallback_pressure = post.estimate_intensity(
            int(payload["class_index"]), float(payload["confidence"])
        )
        payload["estimated_wind_speed_knots"] = fallback_wind
        payload["estimated_pressure_hpa"] = fallback_pressure

    return AnalysisResult(
        **payload,
        analysis_id=result.get("analysis_id"),
        source=result.get("source") or default_source,
        image_name=result.get("image_name") or default_name,
        demo_sample=demo_sample,
    )


@router.post("", response_model=AnalysisResult)
async def upload_and_analyze(
    file: UploadFile = File(...),
    source: str = "upload",
    region: str | None = Form(default=None),
):
    """Upload a satellite image and run the analysis pipeline."""
    parsed_region = None
    if region:
        try:
            parsed_region = AnalyzeRequest(region=json.loads(region)).region
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"Invalid region: {exc}")

    try:
        file_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read file: {exc}")

    filename = _safe_filename(file.filename or "upload.png")

    # Validate the upload
    ok, message = validate_upload(filename, file_bytes)
    if not ok:
        raise HTTPException(status_code=400, detail=message)

    try:
        result = analyze_image(
            file_bytes,
            image_name=filename,
            source=source,
            region=tuple(parsed_region) if parsed_region else None,
        )
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Not a readable image file.")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:  # pragma: no cover
        logger.exception("Internal analysis error")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")

    # P0-1: contract fields passed explicitly rather than relying on
    # **result alone (Pydantic v2 drops unknown keys silently).
    return _to_response(result, default_source=source, default_name=filename)


@router.post("/demo", response_model=AnalysisResult)
async def analyze_demo(payload: AnalyzeRequest):
    """Run the sales demo: pick a bundled sample and analyze it."""
    body = payload if payload else AnalyzeRequest()
    # If no explicit sample_id provided, pick the first available.
    samples = list_demo_samples()
    if not samples:
        raise HTTPException(status_code=503, detail="No demo samples available.")
    sample = samples[0]
    try:
        file_bytes, filename, meta = get_demo_bytes(sample["id"])
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    # P0-2: honour an explicit region here too, mirroring /demo/{sample_id}
    # (list -> tuple conversion happens at the call site, per P1-4).
    region = tuple(body.region) if body.region else None

    result = analyze_image(
        file_bytes,
        image_name=filename,
        source=body.source or "demo",
        region=region,
    )
    # Demo endpoints always report the sample they analysed (P0-1).
    return _to_response(
        result, default_source="demo", default_name=filename, demo_sample=meta
    )


@router.post("/demo/{sample_id}", response_model=AnalysisResult)
async def analyze_demo_sample(sample_id: str, payload: AnalyzeRequest | None = None):
    """Analyze a specific bundled demo sample by id."""
    try:
        file_bytes, filename, meta = get_demo_bytes(sample_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    region = None
    if payload and payload.region:
        region = tuple(payload.region)

    source = payload.source if payload else "demo"

    result = analyze_image(
        file_bytes,
        image_name=filename,
        source=source,
        region=region,
    )
    # Demo endpoints always report the sample they analysed (P0-1).
    return _to_response(
        result, default_source="demo", default_name=filename, demo_sample=meta
    )


@router.get("/samples")
async def demo_samples():
    """List bundled demo satellite images."""
    return {"samples": list_demo_samples()}