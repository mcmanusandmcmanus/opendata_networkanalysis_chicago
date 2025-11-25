# Open Data Network Analysis — Chicago Crime

Offline-first micro SaaS for Chicago crime intelligence: FastAPI backend + static dashboard (Leaflet + Chart.js) operating on a local CSV (`chicago_crime_odp_2022_202511.csv`). CJIS-safe posture: no external calls at runtime; vendor assets locally.

## Quickstart (local)
1) Place the CSV at `data/chicago_crime_odp_2022_202511.csv` or set `CRIME_CSV_PATH` to your path.
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
- `docs/` — data card, prompt recipes, workflow checklist for LLM collaboration.
- `render.yaml` — Render web service config.
- `requirements.txt` — pinned Python dependencies.

## Notes
- Analysis defaults: haversine DBSCAN eps=0.2 miles, min_samples=5; network links within 0.5 miles & 3 days; node cap 20k; DBSCAN sample cap 500k.
- Outputs are cached per crime type (network) and refreshed via `/api/refresh` or app restart.
