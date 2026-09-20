-- ============================================================
-- CYCLO-VISION — PostgreSQL schema
-- ============================================================
-- Run against an existing PostgreSQL server, e.g.:
--   psql -U cyclo -d cyclo_vision -f database/schema.sql
-- The FastAPI app can also create these tables automatically via
--   scripts/init_db.py  (uses SQLAlchemy metadata)
-- ============================================================

BEGIN;

-- ------------------------------------------------------------
-- analyses
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS analyses (
    id              SERIAL PRIMARY KEY,
    image_name      VARCHAR(255),
    source          VARCHAR(80)  DEFAULT 'upload',
    timestamp       TIMESTAMP    DEFAULT now(),
    latitude        DOUBLE PRECISION,
    longitude       DOUBLE PRECISION,
    cyclone_detected  BOOLEAN    DEFAULT FALSE,
    classification   VARCHAR(80),
    class_index      INTEGER,
    confidence       DOUBLE PRECISION,
    wind_speed       DOUBLE PRECISION,   -- knots
    pressure         DOUBLE PRECISION,   -- hPa
    intensity_category VARCHAR(80),
    risk_level       VARCHAR(20),
    inference_mode   VARCHAR(20)  DEFAULT 'demo',
    heatmap_json     TEXT,
    created_at       TIMESTAMP    DEFAULT now()
);

-- ------------------------------------------------------------
-- cyclone_tracks
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS cyclone_tracks (
    id           SERIAL PRIMARY KEY,
    analysis_id  INTEGER NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    latitude     DOUBLE PRECISION NOT NULL,
    longitude    DOUBLE PRECISION NOT NULL,
    forecast_hours INTEGER DEFAULT 0,
    forecast_type  VARCHAR(20) DEFAULT 'current',  -- current | forecast
    label        VARCHAR(20),
    cone_km      DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_tracks_analysis ON cyclone_tracks(analysis_id);

-- ------------------------------------------------------------
-- data_sources
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS data_sources (
    id            SERIAL PRIMARY KEY,
    name          VARCHAR(120) UNIQUE NOT NULL,
    satellite_type VARCHAR(80),
    status        VARCHAR(20) DEFAULT 'unavailable',
    last_updated  TIMESTAMP,
    description   TEXT
);

COMMIT;