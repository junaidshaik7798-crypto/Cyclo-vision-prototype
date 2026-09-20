"""CYCLO-VISION -- History (analyses) API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.database.models import Analysis, CycloneTrack
from app.database.session import get_db
from app.schemas import AnalysisDetail, AnalysisRecord, TrackPoint, TrackResponse

router = APIRouter(tags=["history"])


@router.get("/analyses", response_model=list[AnalysisRecord])
def list_analyses(
    db: Session = Depends(get_db),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Return most recent analyses from PostgreSQL.

    The session is provided by ``Depends(get_db)``; only genuine database
    failures (``SQLAlchemyError``) are translated to 503. Any other
    exception propagates as a 500 so real bugs are not misreported as
    "database unavailable".
    """
    try:
        return (
            db.query(Analysis)
            .order_by(Analysis.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}")


@router.get("/analyses/{analysis_id}", response_model=AnalysisDetail)
def get_analysis(analysis_id: int, db: Session = Depends(get_db)):
    """Return a single analysis including its forecast track."""
    try:
        row = db.query(Analysis).filter(Analysis.id == analysis_id).first()
        if not row:
            raise HTTPException(status_code=404, detail="Analysis not found")
        track = [
            TrackPoint(
                label=t.label or ("Current" if t.forecast_type == "current" else f"+{t.forecast_hours}h"),
                hours=t.forecast_hours or 0,
                lat=t.latitude,
                lon=t.longitude,
                cone_km=t.cone_km or 0,
            )
            for t in row.track
        ]
        return AnalysisDetail(
            **{
                "id": row.id,
                "image_name": row.image_name,
                "source": row.source,
                "timestamp": row.timestamp,
                "cyclone_detected": row.cyclone_detected,
                "classification": row.classification,
                "confidence": row.confidence,
                "wind_speed": row.wind_speed,
                "pressure": row.pressure,
                "risk_level": row.risk_level,
                "inference_mode": row.inference_mode,
                "latitude": row.latitude,
                "longitude": row.longitude,
                "intensity_category": row.intensity_category,
                "track": track,
            }
        )
    except HTTPException:
        raise
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}")


@router.get("/cyclone-track/{analysis_id}", response_model=TrackResponse)
def get_track(analysis_id: int, db: Session = Depends(get_db)):
    """Return the cyclone forecast track for an analysis."""
    try:
        row = db.query(Analysis).filter(Analysis.id == analysis_id).first()
        if not row:
            raise HTTPException(status_code=404, detail="Analysis not found")
        track = [
            TrackPoint(
                label=t.label or "+" + str(t.forecast_hours or 0) + "h",
                hours=t.forecast_hours or 0,
                lat=t.latitude,
                lon=t.longitude,
                cone_km=t.cone_km or 0,
            )
            for t in row.track
        ]
        return TrackResponse(analysis_id=analysis_id, points=track)
    except HTTPException:
        raise
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}")


@router.get("/analyses", response_model=list[AnalysisRecord])
def list_analyses(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    """Return most recent analyses from PostgreSQL."""
    try:
        db = SessionLocal()
        try:
            rows = (
                db.query(Analysis)
                .order_by(Analysis.created_at.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )
            return rows
        finally:
            db.close()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}")


@router.get("/analyses/{analysis_id}", response_model=AnalysisDetail)
def get_analysis(analysis_id: int):
    """Return a single analysis including its forecast track."""
    try:
        db = SessionLocal()
        try:
            row = db.query(Analysis).filter(Analysis.id == analysis_id).first()
            if not row:
                raise HTTPException(status_code=404, detail="Analysis not found")
            track = [
                TrackPoint(
                    label=t.label or ("Current" if t.forecast_type == "current" else f"+{t.forecast_hours}h"),
                    hours=t.forecast_hours or 0,
                    lat=t.latitude,
                    lon=t.longitude,
                    cone_km=t.cone_km or 0,
                )
                for t in row.track
            ]
            detail = AnalysisDetail(
                **{
                    "id": row.id,
                    "image_name": row.image_name,
                    "source": row.source,
                    "timestamp": row.timestamp,
                    "cyclone_detected": row.cyclone_detected,
                    "classification": row.classification,
                    "confidence": row.confidence,
                    "wind_speed": row.wind_speed,
                    "pressure": row.pressure,
                    "risk_level": row.risk_level,
                    "inference_mode": row.inference_mode,
                    "latitude": row.latitude,
                    "longitude": row.longitude,
                    "intensity_category": row.intensity_category,
                    "track": track,
                }
            )
            return detail
        finally:
            db.close()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}")


@router.get("/cyclone-track/{analysis_id}", response_model=TrackResponse)
def get_track(analysis_id: int):
    """Return the cyclone forecast track for an analysis."""
    try:
        db = SessionLocal()
        try:
            row = db.query(Analysis).filter(Analysis.id == analysis_id).first()
            if not row:
                raise HTTPException(status_code=404, detail="Analysis not found")
            track = [
                TrackPoint(
                    label=t.label or "+" + str(t.forecast_hours or 0) + "h",
                    hours=t.forecast_hours or 0,
                    lat=t.latitude,
                    lon=t.longitude,
                    cone_km=t.cone_km or 0,
                )
                for t in row.track
            ]
            return TrackResponse(analysis_id=analysis_id, points=track)
        finally:
            db.close()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}")