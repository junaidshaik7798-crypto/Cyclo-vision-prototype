"""Test file upload endpoint."""
import httpx
import os

BASE = "http://127.0.0.1:8000"


def main():
    # Test upload
    with open(os.path.join("data", "demo", "sample_vsevere.png"), "rb") as f:
        r = httpx.post(
            f"{BASE}/api/analyze",
            files={"file": ("sample_vsevere.png", f, "image/png")},
        )
        d = r.json()
        print(f"Upload: status={r.status_code}, class={d.get('classification')}, "
              f"conf={d.get('confidence')}, risk={d.get('risk_level')}")

    # Test unsupported file type
    r = httpx.post(
        f"{BASE}/api/analyze",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    print(f"Bad upload: status={r.status_code} (expected 400)")


if __name__ == "__main__":
    main()