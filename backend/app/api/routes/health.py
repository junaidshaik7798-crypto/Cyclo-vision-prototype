"""CYCLO-VISION -- Health / system status endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from app.core.config import settings
from app.database.session import db_available
from ml.inference import model_available

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    """Report system health: DB availability, ML mode, model presence."""
    # P2-6: expose the IBTrACS dataset status so /api/health answers
    # "is calibration using live NOAA data or the bundled fallback?".
    # Never fatal: any failure degrades to {"source": "unavailable"}.
    try:
        from app.services.ibtracs import get_status

        status = get_status()
        ibtracs = {k: status[k] for k in ("source", "records", "year_range")}
    except Exception:
        ibtracs = {"source": "unavailable"}

    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "version": "1.0.0",
        "ml_mode": settings.ML_MODE,
        "model_available": model_available(),
        "database": "connected" if db_available() else "unavailable",
        "message": "CYCLO-VISION API is running. Demo mode active." if settings.ML_MODE == "demo"
        else "CYCLO-VISION API is running. Model mode active.",
        "ibtracs": ibtracs,
    }