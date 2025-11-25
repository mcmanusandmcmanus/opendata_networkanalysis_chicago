# Open Data Network Analysis — Chicago Crime

Offline-first micro SaaS for Chicago crime intelligence: FastAPI backend + static dashboard (Leaflet + Chart.js) operating on a local CSV (`chicago_crime_odp_2022_202511.csv`) or, optionally, pulling from the Chicago crimes API (Socrata) with your `app_token`/`secret_key`. CJIS-safe posture remains if you stay on local data and vendor assets.

## Quickstart (local)
1) Place the CSV at `data/chicago_crime_odp_2022_202511.csv` or set `CRIME_CSV_PATH` to your path. For API mode, set `DATA_SOURCE=api` and provide `API_URL` (default: crimes endpoint), `app_token`, and `secret_key` in `.env` or `Secure.env`.
2) Install deps: `python -m venv .venv && .venv\\Scripts\\activate && pip install -r requirements.txt`
3) Fetch frontend assets (Leaflet + Chart.js): `python scripts/setup_assets.py`
4) Run the API/UI: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
5) Open `http://localhost:8000` for the dashboard. APIs: `/api/summary`, `/api/hotspots?crime_type=...`, `/api/network?crime_type=...`

## Deploy to Render
- Update `CRIME_CSV_PATH` in `render.yaml` to the mounted dataset path on Render.
- Render will run `pip install -r requirements.txt` and start `uvicorn app.main:app --host 0.0.0.0 --port 10000`.
- Ensure `static/vendor` assets are present in the repo or fetched during build (network allowed at build time).

## Repo Layout
- `app/` — FastAPI app (`main.py`) and analytics engine (`analysis.py`).
- `static/` — HTML dashboard and vendor assets (Leaflet, Chart.js). Run `scripts/setup_assets.py` to populate `static/vendor/`.
- `static/boundaries/` — optional GeoJSON boundaries (e.g., police districts, neighborhoods, tracts) for map overlays; keep files local.
- `docs/` — data card, prompt recipes, workflow checklist for LLM collaboration.
- `render.yaml` — Render web service config.
- `requirements.txt` — pinned Python dependencies.

## Notes
- Analysis defaults: haversine DBSCAN eps=0.2 miles, min_samples=5; network links within 0.5 miles & 3 days; node cap 20k; DBSCAN sample cap 500k.
- Data source: `DATA_SOURCE=local` (default) reads the CSV. `DATA_SOURCE=api` fetches from `API_URL` (default Socrata crimes dataset), limited by `API_LIMIT` (default 100k rows) and `API_YEARS_BACK` (default 3 years), using `app_token`/`secret_key` from env.
- Boundaries: If you want overlays, convert shapefiles to GeoJSON (WGS84) and place under `static/boundaries/` (e.g., `police_districts.geojson`, `neighborhoods.geojson`, `tracts.geojson`). The UI boundary toggle will pick them up; no external services are called.
- Outputs are cached per crime type (network) and refreshed via `/api/refresh` or app restart.
