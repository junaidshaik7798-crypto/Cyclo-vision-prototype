"""CYCLO-VISION -- Database base & engine/session management.

Gracefully handles the case where PostgreSQL is unavailable so API calls
return meaningful errors instead of crashing the whole app.
"""

from __future__ import annotations

import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import settings
from app.database.base import Base

logger = logging.getLogger("cyclo.database")

try:
    engine = create_engine(
        settings.DATABASE_URL,
        pool_pre_ping=True,
        pool_recycle=3600,
        connect_args={"connect_timeout": 4} if settings.DATABASE_URL.startswith("postgresql") else {},
    )
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
except Exception as exc:  # pragma: no cover
    logger.warning("Database engine init failed: %s", exc)
    engine = None
    SessionLocal = None


def get_db():
    """FastAPI dependency yielding a DB session."""
    if SessionLocal is None:
        raise RuntimeError("Database is not configured/available.")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> bool:
    """Create tables if possible. Returns True on success, False otherwise."""
    if engine is None:
        logger.warning("init_db skipped: no engine.")
        return False
    try:
        # Import models so they register on the metadata.
        import app.database.models  # noqa: F401

        Base.metadata.create_all(bind=engine)
        return True
    except SQLAlchemyError as exc:
        logger.error("Database init failed: %s", exc)
        return False


def db_available() -> bool:
    """Quick check whether the DB is reachable."""
    if engine is None or SessionLocal is None:
        return False
    try:
        from sqlalchemy import text

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False