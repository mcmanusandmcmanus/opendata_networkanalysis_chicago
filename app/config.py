import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment from common locations
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / "Secure.env")

# Environment variables
DATA_PATH = os.getenv(
    "CRIME_CSV_PATH",
    str(BASE_DIR / "data" / "chicago_crime_odp_2022_202511.csv"),
)

# Data source selection: "local" (CSV) or "api" (default)
DATA_SOURCE = os.getenv("DATA_SOURCE", "api").lower()

# API configuration (for Socrata/City of Chicago crimes endpoint)
API_URL = os.getenv(
    "API_URL",
    "https://data.cityofchicago.org/resource/ijzp-q8t2.json",  # Crimes - 2001 to Present
)
API_LIMIT = int(os.getenv("API_LIMIT", 200_000))  # max rows to pull per refresh (safe for ~2GB RAM; adjust down if needed)
API_YEARS_BACK = int(os.getenv("API_YEARS_BACK", 4))  # limit by recent years for coverage (e.g., up to current year)
APP_TOKEN = os.getenv("app_token") or os.getenv("APP_TOKEN")
SECRET_KEY = os.getenv("secret_key") or os.getenv("SECRET_KEY")

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
BOUNDARY_DIR = STATIC_DIR / "boundaries"
DISTRICT_GEOJSON = BOUNDARY_DIR / "police_districts.geojson"
NEIGHBORHOOD_GEOJSON = BOUNDARY_DIR / "neighborhoods.geojson"
TRACT_GEOJSON = BOUNDARY_DIR / "tracts.geojson"
