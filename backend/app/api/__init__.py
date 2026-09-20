"""CYCLO-VISION -- API package.

Kept as a real package because ``app.main`` imports the routes through it
(``from app.api.routes import analyze, history, data_sources, health``).
This module intentionally holds no code.
"""

