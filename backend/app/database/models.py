"""CYCLO-VISION -- SQLAlchemy ORM models (PostgreSQL tables)."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.database.base import Base


def _utcnow() -> datetime:
    """Timezone-aware UTC now (datetime.utcnow is deprecated in 3.12+)."""
    return datetime.now(timezone.utc)


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    image_name = Column(String(255), nullable=True)
    source = Column(String(80), default="upload")
    timestamp = Column(DateTime, default=_utcnow)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    cyclone_detected = Column(Boolean, default=False)
    classification = Column(String(80), nullable=True)
    class_index = Column(Integer, nullable=True)
    confidence = Column(Float, nullable=True)
    wind_speed = Column(Float, nullable=True)  # knots
    pressure = Column(Float, nullable=True)  # hPa
    intensity_category = Column(String(80), nullable=True)
    risk_level = Column(String(20), nullable=True)
    inference_mode = Column(String(20), default="demo")
    heatmap_json = Column(Text, nullable=True)  # optional base64/list
    created_at = Column(DateTime, default=_utcnow)

    # order_by guarantees the UI always renders points in forecast order;
    # without it Postgres returns them in arbitrary physical order.
    # ("all, delete-orphan" already implies save-update/merge/delete.)
    track = relationship(
        "CycloneTrack",
        back_populates="analysis",
        cascade="all, delete-orphan",
        order_by="CycloneTrack.forecast_hours",
    )


class CycloneTrack(Base):
    __tablename__ = "cyclone_tracks"

    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("analyses.id", ondelete="CASCADE"), index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    forecast_hours = Column(Integer, default=0)
    forecast_type = Column(String(20), default="current")  # current | forecast
    label = Column(String(20), nullable=True)
    cone_km = Column(Float, nullable=True)

    analysis = relationship("Analysis", back_populates="track")


class DataSource(Base):
    __tablename__ = "data_sources"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(120), unique=True, index=True)
    satellite_type = Column(String(80), nullable=True)
    status = Column(String(20), default="unavailable")
    last_updated = Column(DateTime, nullable=True)
    description = Column(Text, nullable=True)