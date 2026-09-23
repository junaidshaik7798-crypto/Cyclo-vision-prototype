"""CYCLO-VISION -- Dataset API endpoints.

Everything the dashboard's dataset views need. Design rule for this module:

* no endpoint may block on a NOAA download -- each one answers from whatever
  is already local (see ``app.services.ibtracs.dataset_snapshot``) and says so
  through its ``state`` field, and
* a failure in one dataset must never empty another panel, so
  ``/datasets/overview`` collects each dataset independently and reports
  per-key errors instead of failing the whole request.

That is what removes the "could not fetch the datasets -- retry" loop the UI
used to show while the archive was still being downloaded in the background.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from fastapi import APIRouter, HTTPException, Query

from app.services.data_sources import get_data_sources
from app.services.demo_data import list_demo_samples
from app.services.reference_data import (
    list_recent_cyclones,
    get_dataset_summary,
    get_reference_dataset,
)
from app.services.ibtracs import (
    dataset_summary as get_ibtracs_summary,
    get_attributes as get_ibtracs_attributes,
    get_full_dataset as get_ibtracs_dataset,
    get_recent as get_ibtracs_recent,
    get_scale as get_ibtracs_scale,
    get_status as get_ibtracs_status,
    get_top_intense as get_ibtracs_intense,
)

logger = logging.getLogger("cyclo.datasets")

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


@router.get("/reference-dataset/summary")
def reference_dataset_summary():
    """Summary-only view of the bundled reference dataset.

    Compatibility alias: older frontend bundles (still cached in browsers)
    requested this path and received a 404, which the UI reported as
    "could not fetch the reference dataset". Keeping the route alive means a
    stale client can no longer produce that failure.
    """
    try:
        return get_dataset_summary()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load reference summary: {exc}")


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
    """Report live-dataset telemetry (source, record count, cache state).

    Answered from memory / the local cache: it never waits for the NOAA
    download, so the dashboard can always render the panel.
    """
    try:
        return get_ibtracs_status()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"IBTrACS status failed: {exc}")


@router.get("/ibtracs/dataset")
def ibtracs_full_dataset(
    limit: Optional[int] = Query(
        None, ge=0, description="Maximum rows to return; 0 or omitted = every storm."
    ),
    offset: int = Query(0, ge=0),
    sort: str = Query("recent", description="recent | oldest | intense | weakest | name | pressure"),
    basin: Optional[str] = None,
    category: Optional[str] = None,
    min_wind: Optional[float] = Query(None, description="Minimum peak wind in knots."),
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    search: Optional[str] = Query(None, description="Free text over name, SID, basin, category, year."),
):
    """The complete observed record: every storm, every attribute, every filter.

    Returns the full payload (summary statistics, IMD scale and field-level
    documentation) so the dataset view needs exactly one request.
    """
    try:
        return get_ibtracs_dataset(
            limit=limit,
            offset=offset,
            sort=sort,
            basin=basin,
            category=category,
            min_wind=min_wind,
            year_min=year_min,
            year_max=year_max,
            search=search,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"IBTrACS dataset failed: {exc}")


@router.get("/ibtracs/recent")
def ibtracs_recent(limit: int = 25):
    """Most recent real storms in the North Indian Ocean, newest first."""
    try:
        status = get_ibtracs_status()
        return {
            "source": status["source"],
            "state": status["state"],
            "records": status["records"],
            "storms": get_ibtracs_recent(limit),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"IBTrACS recent failed: {exc}")


@router.get("/ibtracs/intense")
def ibtracs_intense(limit: int = 25):
    """Strongest real storms on record by peak 1-minute sustained wind."""
    try:
        status = get_ibtracs_status()
        return {
            "source": status["source"],
            "state": status["state"],
            "records": status["records"],
            "storms": get_ibtracs_intense(limit),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"IBTrACS intense failed: {exc}")


# --------------------------------------------------------------------------
# One-shot dashboard payload
# --------------------------------------------------------------------------


def _collect(errors: dict[str, str], key: str, loader: Callable[[], Any]) -> Any:
    """Run one dataset loader, recording (not raising) a per-key failure."""
    try:
        return loader()
    except Exception as exc:  # noqa: BLE001 - one dataset must not sink the rest
        logger.warning("Dataset '%s' failed: %s", key, exc, exc_info=True)
        errors[key] = f"{type(exc).__name__}: {exc}"
        return None


@router.get("/datasets/overview")
def datasets_overview(
    recent_limit: int = Query(10, ge=0, le=500),
    intense_limit: int = Query(10, ge=0, le=500),
):
    """Every dataset the dashboard shows, in a single request.

    The dashboard previously fired seven separate requests; one slow endpoint
    was enough to blank a panel and trigger the retry banner. This route
    gathers all of them, degrades per dataset, and always answers 200 with
    whatever is available.
    """
    errors: dict[str, str] = {}

    samples = _collect(errors, "samples", lambda: list_demo_samples() or [])
    sources = _collect(errors, "sources", lambda: get_data_sources() or [])
    reference = _collect(
        errors,
        "reference",
        lambda: {
            "dataset": [vars(c) for c in get_reference_dataset()],
            "summary": get_dataset_summary(),
        },
    )
    status = _collect(errors, "ibtracs_status", get_ibtracs_status)
    full = _collect(
        errors,
        "ibtracs_dataset",
        lambda: get_ibtracs_dataset(limit=None, sort="recent"),
    )
    summary = _collect(errors, "ibtracs_summary", get_ibtracs_summary)
    scale = _collect(errors, "ibtracs_scale", get_ibtracs_scale)
    attributes = _collect(errors, "ibtracs_attributes", get_ibtracs_attributes)

    ibtracs: dict[str, Any] = {
        "status": status,
        "summary": summary,
        "scale": scale or [],
        "attributes": attributes or [],
        "dataset": full,
    }
    if recent_limit:
        ibtracs["recent"] = _collect(
            errors, "ibtracs_recent", lambda: get_ibtracs_recent(recent_limit)
        ) or []
    if intense_limit:
        ibtracs["intense"] = _collect(
            errors, "ibtracs_intense", lambda: get_ibtracs_intense(intense_limit)
        ) or []

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "errors": errors,
        "samples": samples or [],
        "sources": sources or [],
        "reference": reference or {"dataset": [], "summary": {}},
        "ibtracs": ibtracs,
    }
