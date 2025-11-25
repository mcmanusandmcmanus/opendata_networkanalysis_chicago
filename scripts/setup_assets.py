"""
Download pinned frontend assets for offline use.

Run:
    python scripts/setup_assets.py
"""
import hashlib
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "static" / "vendor"
TARGET.mkdir(parents=True, exist_ok=True)

ASSETS = {
    "leaflet.css": "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css",
    "leaflet.js": "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js",
    "chart.js": "https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.js",
}


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_assets():
    print(f"Downloading assets to {TARGET}")
    for name, url in ASSETS.items():
        dest = TARGET / name
        if dest.exists():
            print(f"✔ {name} already present ({dest.stat().st_size} bytes)")
            continue
        try:
            print(f"→ Fetching {name} from {url}")
            urllib.request.urlretrieve(url, dest)
            print(f"  saved {dest} (sha256={sha256sum(dest)})")
        except Exception as exc:
            print(f"✖ Failed to download {name}: {exc}")
            if dest.exists():
                dest.unlink(missing_ok=True)
            raise


if __name__ == "__main__":
    download_assets()
