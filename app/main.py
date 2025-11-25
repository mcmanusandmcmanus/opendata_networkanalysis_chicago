import logging
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import config
from app.analysis import analyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("app")

app = FastAPI(title="Chicago Crime Intel", version="1.0.0")

# Static assets
static_dir = config.STATIC_DIR
static_dir.mkdir(parents=True, exist_ok=True)
vendor_dir = config.VENDOR_DIR
vendor_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


class SummaryStats(BaseModel):
    total_incidents: int
    date_range: Dict[str, str]
    top_types: Dict[str, int]
    arrest_rate: float
    hourly_counts: Dict[int, int]
    monthly_counts: Dict[str, int]
    pareto_concentration: Dict[str, Any]


class Hotspot(BaseModel):
    cluster_id: int
    count: int
    center_lat: float
    center_lon: float
    crime_type: str
    radius_approx_ft: float


class NetworkData(BaseModel):
    summary: Dict[str, int]
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, str]]


@app.on_event("startup")
def startup_event():
    if analyzer.df is None:
        logger.info("Preloading dataset (source=%s)...", analyzer.data_source)
        analyzer.load_data()


@app.get("/health")
def health_check():
    loaded = analyzer.df is not None and analyzer.df is not None and not analyzer.df.empty
    return {
        "status": "online",
        "data_loaded": loaded,
        "records": len(analyzer.df) if loaded else 0,
        "data_path": str(analyzer.csv_path),
        "data_source": analyzer.data_source,
        "last_load_status": analyzer.last_load_status,
        "last_load_error": analyzer.last_load_error,
        "last_loaded_at": analyzer.last_loaded_at,
    }


@app.get("/", response_class=HTMLResponse)
def root():
    index_path = static_dir / "index.html"
    if not index_path.exists():
        return HTMLResponse("System offline. static/index.html missing.", status_code=503)
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/api/summary", response_model=SummaryStats)
def get_summary():
    stats = analyzer.get_summary_stats()
    if not stats:
        raise HTTPException(status_code=503, detail="Dataset not loaded or empty.")
    return stats


@app.get("/api/hotspots", response_model=List[Hotspot])
def get_hotspots(crime_type: str = Query("ALL", description="Primary Type")):
    results = analyzer.compute_hotspots(crime_type)
    if results is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded or empty.")
    return results


@app.get("/api/network", response_model=NetworkData)
def get_network(crime_type: str = Query("ROBBERY", description="Primary Type")):
    data = analyzer.build_network(crime_type)
    if data is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded or empty.")
    return data


@app.post("/api/refresh")
def refresh_data():
    ok = analyzer.load_data()
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to reload dataset.")
    return {"status": "refreshed", "count": len(analyzer.df) if analyzer.df is not None else 0}


def create_app():
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=config.HOST, port=config.PORT, reload=False)
