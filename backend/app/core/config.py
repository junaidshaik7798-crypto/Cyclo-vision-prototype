"""
CYCLO-VISION -- Application Configuration
=========================================
Central settings loaded from environment variables (see .env.example).
Uses pydantic-settings for typed access with safe defaults.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = backend/
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server
    APP_NAME: str = "CYCLO-VISION"
    BACKEND_HOST: str = "127.0.0.1"
    BACKEND_PORT: int = 8000
    APP_ENV: str = "development"

    # CORS
    # Vite dev (5173), ``vite preview`` (4173), VS Code Live Server (5500) and
    # the ``null`` origin used when the standalone pages are opened straight
    # from disk. Any of these missing means the browser blocks the dataset
    # fetch and the UI reports it as a backend failure.
    ALLOWED_ORIGINS: str = (
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:4173,http://127.0.0.1:4173,"
        "http://localhost:5500,http://127.0.0.1:5500,"
        "null"
    )

    # Database
    DATABASE_URL: str = (
        "postgresql://cyclo:cyclo_pass@localhost:5432/cyclo_vision"
    )

    # ML
    ML_MODE: str = "demo"  # "model" | "demo"
    MODEL_PATH: str = str(PROJECT_ROOT / "models" / "cyclo_cnn.pt")
    INPUT_SIZE: int = 256

    # Data
    DEMO_DATA_DIR: str = str(PROJECT_ROOT / "data" / "demo")

    # Upload
    MAX_UPLOAD_SIZE_MB: int = 10
    ALLOWED_EXTENSIONS: str = ".jpg,.jpeg,.png,.tif,.tiff"

    # Security
    SECRET_KEY: str = "change-me"

    # Optional external
    INSAT_API_URL: str = ""
    GOES_API_URL: str = ""
    NASA_EARTHDATA_TOKEN: str = ""

    # ------------------------------------------------------------------
    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]

    @property
    def allowed_extensions_set(self) -> set[str]:
        return {e.strip().lower() for e in self.ALLOWED_EXTENSIONS.split(",") if e}

    @property
    def model_dir(self) -> Path:
        p = Path(self.MODEL_PATH).parent
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def MODEL_DIR(self) -> Path:
        return self.model_dir


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()