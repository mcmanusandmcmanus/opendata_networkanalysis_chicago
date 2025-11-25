import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Environment variables
DATA_PATH = os.getenv(
    "CRIME_CSV_PATH",
    str(BASE_DIR / "data" / "chicago_crime_odp_2022_202511.csv"),
)

# Analysis defaults
SPATIAL_RADIUS_MILES = float(os.getenv("SPATIAL_RADIUS_MILES", 0.5))
TEMPORAL_WINDOW_DAYS = int(os.getenv("TEMPORAL_WINDOW_DAYS", 3))
DBSCAN_EPS_MILES = float(os.getenv("DBSCAN_EPS_MILES", 0.2))
DBSCAN_MIN_SAMPLES = int(os.getenv("DBSCAN_MIN_SAMPLES", 5))
DBSCAN_SAMPLE_LIMIT = int(os.getenv("DBSCAN_SAMPLE_LIMIT", 500_000))
NETWORK_NODE_LIMIT = int(os.getenv("NETWORK_NODE_LIMIT", 20_000))

# Render settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))

# Paths for static assets
STATIC_DIR = BASE_DIR / "static"
VENDOR_DIR = STATIC_DIR / "vendor"
