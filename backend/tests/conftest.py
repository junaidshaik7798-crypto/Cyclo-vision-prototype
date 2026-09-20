"""Shared pytest fixtures for the CYCLO-VISION backend tests.

The FastAPI ``TestClient`` is used in-process (no live server required), so
the tests run anywhere without PostgreSQL or network access.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image


def _png_bytes(size: int = 64, colour: tuple[int, int, int] = (40, 120, 200)) -> bytes:
    """Build a small, valid PNG entirely in memory."""
    buf = io.BytesIO()
    Image.new("RGB", (size, size), colour).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def client() -> TestClient:
    """In-process FastAPI client (lifespan is not triggered by default)."""
    from app.main import app

    return TestClient(app)


@pytest.fixture(scope="session")
def png_bytes() -> bytes:
    """A small valid PNG used for upload tests."""
    return _png_bytes()
