"""CYCLO-VISION — Initialize PostgreSQL tables.

Run from project root:
    python scripts/init_db.py
Requires DATABASE_URL in backend env /.env. Non-fatal if DB is absent.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.database.session import init_db  # noqa: E402


def main():
    ok = init_db()
    if ok:
        print("Database initialized successfully.")
    else:
        print(
            "Database init failed. Is PostgreSQL running and is DATABASE_URL "
            "correct (see .env.example)?"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()