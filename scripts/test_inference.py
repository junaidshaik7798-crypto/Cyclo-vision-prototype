"""Test all demo samples through the inference pipeline."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from ml.inference import run_inference
from app.services.demo_data import get_demo_bytes, list_demo_samples


def main():
    samples = list_demo_samples()
    for s in samples:
        file_bytes, filename, _ = get_demo_bytes(s["id"])
        result = run_inference(file_bytes)
        print(
            f"{s['id']}: {result['classification']} "
            f"(conf={result['confidence']:.2f}, "
            f"wind={result['estimated_wind_speed_knots']}kts, "
            f"risk={result['risk_level']})"
        )


if __name__ == "__main__":
    main()