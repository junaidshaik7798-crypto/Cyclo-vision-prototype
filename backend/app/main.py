"""CYCLO-VISION -- FastAPI application entry point.

Run with:
    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import PROJECT_ROOT, settings
from app.database.session import init_db
from app.api.routes import analyze, history, data_sources, health

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cyclo.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown hook. Attempts DB init (non-fatal if missing)."""
    ok = init_db()
    if ok:
        logger.info("Database initialized (tables ensured).")
    else:
        logger.warning(
            "Database unavailable at startup -- running with in-memory/error "
            "fallbacks. Run PostgreSQL and set DATABASE_URL to enable history."
        )
    # Warm the live IBTrACS dataset in a background thread so the first
    # request needing calibration does not pay the download cost. The
    # wrapper logs the outcome -- previously the thread finished silently,
    # so operators could not tell a completed warm-up from a hung one (#26).
    def _warm_ibtracs() -> None:
        try:
            from app.services.ibtracs import get_status, load_storms

            storms = load_storms()
            status = get_status()
            logger.info(
                "IBTrACS warm-up done: %d storms (source=%s).",
                len(storms),
                status.get("source", "unknown"),
            )
        except Exception:
            logger.warning("IBTrACS warm-up failed.", exc_info=True)

    try:
        threading.Thread(target=_warm_ibtracs, daemon=True).start()
        logger.info("IBTrACS live dataset warm-up started.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not start IBTrACS warm-up: %s", exc)
    yield


app = FastAPI(
    title="CYCLO-VISION API",
    description=(
        "AI-powered tropical cyclone intelligence platform. Classifies, "
        "assesses risk, and provides prototype forecasts from satellite imagery."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS -- allow configured frontend origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health.router, prefix="/api")
app.include_router(data_sources.router, prefix="/api")
app.include_router(history.router, prefix="/api")
app.include_router(analyze.router, prefix="/api")

# Static: bundled demo imagery, served at /data/demo/<file> so the frontend
# can render sample thumbnails directly.
_DEMO_DIR = PROJECT_ROOT / "data" / "demo"
if _DEMO_DIR.exists():
    app.mount(
        "/data/demo",
        StaticFiles(directory=str(_DEMO_DIR)),
        name="demo-data",
    )


# Root
@app.get("/")
def root():
    return {
        "app": "CYCLO-VISION",
        "message": "AI-Powered Tropical Cyclone Intelligence Platform",
        "docs": "/docs",
        "health": "/api/health",
    }