"""CYCLO-VISION - Data-sources API endpoint."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.services.data_sources import get_data_sources
from app.services.reference_data import (
    list_recent_cyclones,
    get_dataset_summary,
    get_reference_dataset,
)
from app.services.ibtracs import (
    get_recent as get_ibtracs_recent,
    get_top_intense as get_ibtracs_intense,
    get_status as get_ibtracs_status,
)

router = APIRouter(tags=["data-sources"])


@router.get("/data-sources")
def list_sources():
    """Return multi-source satellite data architecture availability."""
    try:
        return {"sources": get_data_sources()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to list sources: {exc}")


@router.get("/reference-dataset")
def reference_dataset():
    """Return the reference cyclone dataset."""
    try:
        events = [vars(c) for c in get_reference_dataset()]
        return {
            "dataset": events,
            "summary": get_dataset_summary(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load reference data: {exc}")


@router.get("/reference-dataset/recent")
def recent_cyclones(limit: int = 10):
    """Return recent cyclone events from the reference dataset."""
    try:
        return {"cyclones": list_recent_cyclones(limit=limit)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load reference data: {exc}")
# --------------------------------------------------------------------------
# Live IBTrACS (real NOAA best-track archive)
# --------------------------------------------------------------------------

@router.get("/ibtracs/status")
def ibtracs_status():
    """Report live-dataset telemetry (source, record count, cache state)."""
    try:
        return get_ibtracs_status()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"IBTrACS status failed: {exc}")


@router.get("/ibtracs/recent")
def ibtracs_recent(limit: int = 25):
    """Most recent real storms in the North Indian Ocean, newest first."""
    try:
        return {"source": get_ibtracs_status()["source"], "storms": get_ibtracs_recent(limit)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"IBTrACS recent failed: {exc}")


@router.get("/ibtracs/intense")
def ibtracs_intense(limit: int = 25):
    """Strongest real storms on record by peak 1-minute sustained wind."""
    try:
        return {"source": get_ibtracs_status()["source"], "storms": get_ibtracs_intense(limit)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"IBTrACS intense failed: {exc}")
