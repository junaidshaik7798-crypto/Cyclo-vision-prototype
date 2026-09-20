"""CYCLO-VISION — Generate bundled demo satellite images.

Run from project root:
    python scripts/generate_demo_data.py
Creates data/demo/*.png files.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

DEFAULT_OUT = os.path.join("data", "demo")


def run():
    from PIL import Image  # noqa: F401  (ensure available)
    import numpy as np

    import importlib.util

    # Load the generator from backend/scripts
    gen_path = os.path.join("backend", "scripts", "generate_demo_data.py")
    spec = importlib.util.spec_from_file_location("gendemo", gen_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.OUT_DIR = DEFAULT_OUT
    mod.main()
    print("Demo data generated in", DEFAULT_OUT)


if __name__ == "__main__":
    run()