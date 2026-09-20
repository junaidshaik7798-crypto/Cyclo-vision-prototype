"""CYCLO-VISION -- Pydantic schemas for API request/response."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Common
# ---------------------------------------------------------------------------


class PreprocessStep(BaseModel):
    name: str
    status: str = "done"
    detail: str = ""
    elapsed_ms: float = 0.0


class Explainability(BaseModel):
    type: str = "prototype-attention"
    note: str = ""
    # Base64-encoded PNG of the attention/activation overlay. Shipping the
    # raw (H, W, 3) RGB array cost ~1-2 MB of JSON per request, so the
    # heatmap is encoded once server-side instead.
    heatmap_png_b64: Optional[str] = None


class TrackPoint(BaseModel):
    label: str
    hours: int
    lat: float
    lon: float
    cone_km: float = 0.0


class RiskFactor(BaseModel):
    factor: str
    impact: str


# ---------------------------------------------------------------------------
# Analysis response
# ---------------------------------------------------------------------------


class AnalysisResult(BaseModel):
    cyclone_detected: bool
    classification: str
    class_index: int
    confidence: float
    estimated_wind_speed_knots: float
    estimated_pressure_hpa: float
    intensity_category: str
    risk_level: str
    risk_factors: list[RiskFactor] = []
    inference_mode: str
    calibration_source: Optional[str] = None
    reference_cyclone: Optional[str] = None
    reference_year: Optional[int] = None
    explainability: Explainability
    track: list[TrackPoint] = []
    center: dict[str, float]

    # --- P0-1: pass-through fields the frontend needs ---------------------
    # Without these, Pydantic v2 silently dropped them from the response and
    # the UI could not call /analyses/{id} or /cyclone-track/{id}.
    analysis_id: Optional[int] = None
    source: str = "upload"
    image_name: Optional[str] = None
    demo_sample: Optional[dict] = None

    # --- P1-5: surfaced persistence failure ------------------------------
    persist_warning: Optional[str] = None


# ---------------------------------------------------------------------------
# Demo / query inputs
# ---------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    """Body for /api/analyze/demo -- overridable source label."""

    source: str = "demo"
    region: Optional[list[float]] = None

    @field_validator("region")
    @classmethod
    def _check_region(cls, v):
        """A region is exactly one [lat, lon] pair with in-range values."""
        if v is None:
            return v
        if len(v) != 2:
            raise ValueError("region must be [lat, lon]")
        lat, lon = v
        if not (-90 <= lat <= 90):
            raise ValueError("lat out of range")
        if not (-180 <= lon <= 180):
            raise ValueError("lon out of range")
        return v


class TrackResponse(BaseModel):
    analysis_id: int
    points: list[TrackPoint] = []


# ---------------------------------------------------------------------------
# History records (DB-backed)
# ---------------------------------------------------------------------------


class AnalysisRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    image_name: Optional[str] = None
    source: str = "upload"
    timestamp: datetime
    cyclone_detected: bool
    classification: Optional[str] = None
    confidence: Optional[float] = None
    wind_speed: Optional[float] = None
    pressure: Optional[float] = None
    risk_level: Optional[str] = None
    inference_mode: Optional[str] = "demo"
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class AnalysisDetail(AnalysisRecord):
    track: list[TrackPoint] = []
    intensity_category: Optional[str] = None
    risk_factors: list[RiskFactor] = []


class DataSourceRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    satellite_type: Optional[str] = None
    status: str = "unavailable"
    last_updated: Optional[datetime] = None
    description: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    ml_mode: str
    database: str
    model_available: bool
    version: str = "1.0.0"
    message: str = ""


class DemoSample(BaseModel):
    id: str
    name: str
    description: str
    category: str
    filename: str
    file_size_kb: Optional[int] = None