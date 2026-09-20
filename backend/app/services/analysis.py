"""CYCLO-VISION -- Analysis service layer.

Wraps the ML inference engine and persists results to PostgreSQL when
available. If the DB is down, results are still returned (with a warning),
so the demo never fails on DB unavailability.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from app.core.config import settings
from ml.inference import run_inference
from app.database.session import SessionLocal, db_available
from app.database.models import Analysis, CycloneTrack

logger = logging.getLogger("cyclo.analysis")


def _persist(
    result: dict[str, Any], image_name: str, source: str
) -> tuple[Optional[int], bool, Optional[str]]:
    """Persist an analysis.

    Returns ``(analysis_id, db_was_up, error_message)``. ``analysis_id`` is
    None when the row could not be written; ``db_was_up`` distinguishes
    "DB not configured/offline" (a normal degraded mode) from "DB reachable
    but the write failed" (which the caller surfaces to the user, P1-5).
    """
    if not db_available():
        logger.warning("Skipping persist: DB unavailable.")
        return None, False, None
    try:
        db = SessionLocal()
    except Exception as exc:  # pragma: no cover - raced offline between checks
        logger.warning("Skipping persist: DB connection failed: %s", exc)
        return None, False, None
    try:
        try:
            center = result.get("center", {})
            # P1-2: the heatmap is now a base64 PNG string; the column is
            # still TEXT (just no longer a JSON array).
            heat = result.get("explainability", {}).get("heatmap_png_b64")
            import json

            record = Analysis(
                image_name=image_name[:255],
                source=source[:80],
                latitude=center.get("lat"),
                longitude=center.get("lon"),
                cyclone_detected=result["cyclone_detected"],
                classification=result["classification"][:80],
                class_index=result.get("class_index"),
                confidence=result["confidence"],
                wind_speed=result["estimated_wind_speed_knots"],
                pressure=result["estimated_pressure_hpa"],
                intensity_category=result["intensity_category"][:80],
                risk_level=result["risk_level"][:20],
                inference_mode=result["inference_mode"],
                # Keep the column name (heatmap_json); it is just a TEXT blob.
                heatmap_json=json.dumps(heat) if heat else None,
            )
            db.add(record)
            db.flush()
            for pt in result.get("track", []):
                db.add(
                    CycloneTrack(
                        analysis_id=record.id,
                        latitude=pt["lat"],
                        longitude=pt["lon"],
                        forecast_hours=pt.get("hours", 0),
                        forecast_type="current" if pt.get("hours", 0) == 0 else "forecast",
                        label=pt.get("label"),
                        cone_km=pt.get("cone_km"),
                    )
                )
            db.commit()
            db.refresh(record)
            return record.id, True, None
        finally:
            db.close()
    except Exception as exc:  # pragma: no cover
        logger.error("Persist failed: %s", exc)
        return None, True, f"{type(exc).__name__}: {exc}"


def analyze_image(
    file_bytes: bytes,
    image_name: str = "upload",
    source: str = "upload",
    region: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Run full inference + persist. Returns result dict."""
    result = run_inference(file_bytes, region=region)
    analysis_id, db_was_up, persist_error = _persist(result, image_name, source)
    result["analysis_id"] = analysis_id
    result["source"] = source
    result["image_name"] = image_name
    # P1-5: a failed write while the DB *was* reachable is a real problem
    # (bad migration, permissions, constraint) and must not look like a
    # successful save. DB simply being offline stays silent by design.
    if analysis_id is None and db_was_up:
        logger.error("Analysis was not persisted: %s", persist_error)
        result["persist_warning"] = "Analysis not saved to history."
    return result