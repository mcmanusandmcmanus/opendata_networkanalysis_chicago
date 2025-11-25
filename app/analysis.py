import logging
import os
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd
import requests
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree

from app import config

logger = logging.getLogger("CrimeAnalyzer")


class CrimeAnalyzer:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df: Optional[pd.DataFrame] = None
        self.cache: Dict[str, Any] = {}
        self._lock = Lock()
        self.data_source = config.DATA_SOURCE
        self.last_load_status: str = "not_loaded"
        self.last_load_error: Optional[str] = None
        self.last_loaded_at: Optional[str] = None

        # Tunable defaults
        self.spatial_radius_miles = config.SPATIAL_RADIUS_MILES
        self.temporal_window_days = config.TEMPORAL_WINDOW_DAYS
        self.dbscan_eps_miles = config.DBSCAN_EPS_MILES
        self.dbscan_min_samples = config.DBSCAN_MIN_SAMPLES
        self.dbscan_sample_limit = config.DBSCAN_SAMPLE_LIMIT
        self.network_node_limit = config.NETWORK_NODE_LIMIT
        self.earth_radius_miles = 3958.8

    def _load_chunks(self) -> List[pd.DataFrame]:
        cols = [
            "ID",
            "Case Number",
            "Date",
            "Block",
            "IUCR",
            "Primary Type",
            "Description",
            "Location Description",
            "Arrest",
            "Domestic",
            "Beat",
            "District",
            "Ward",
            "Community Area",
            "FBI Code",
            "X Coordinate",
            "Y Coordinate",
            "Year",
            "Updated On",
            "Latitude",
            "Longitude",
            "Location",
        ]
        dtypes = {c: "string" for c in cols}
        dtypes.update(
            {
                "Latitude": "float32",
                "Longitude": "float32",
                "X Coordinate": "float32",
                "Y Coordinate": "float32",
                "Year": "Int64",
                "Beat": "string",
                "District": "string",
                "Ward": "string",
                "Community Area": "string",
            }
        )

        parse_dates = ["Date", "Updated On"]
        chunk_size = 200_000
        chunks: List[pd.DataFrame] = []

        for chunk in pd.read_csv(
            self.csv_path,
            usecols=cols,
            dtype=dtypes,
            parse_dates=parse_dates,
            infer_datetime_format=True,
            chunksize=chunk_size,
            on_bad_lines="skip",
        ):
            chunk = chunk.dropna(subset=["Date", "Latitude", "Longitude"])

            # Normalize booleans
            for col in ["Arrest", "Domestic"]:
                chunk[col] = (
                    chunk[col]
                    .astype("string")
                    .str.lower()
                    .map({"true": True, "false": False})
                )

            chunk["hour"] = chunk["Date"].dt.hour.astype("int8")
            chunk["lat_rad"] = np.radians(chunk["Latitude"]).astype("float32")
            chunk["lon_rad"] = np.radians(chunk["Longitude"]).astype("float32")
            chunks.append(chunk)

        return chunks

    def _load_from_csv(self) -> bool:
        if not os.path.exists(self.csv_path):
            logger.error("CSV not found at %s", self.csv_path)
            return False
        logger.info("Loading dataset from %s", self.csv_path)
        chunks = self._load_chunks()
        if not chunks:
            logger.error("No data loaded; chunks are empty.")
            return False
        self.df = pd.concat(chunks, ignore_index=True)
        logger.info("Data loaded: %s records", f"{len(self.df):,}")
        return True

    def _load_from_api(self) -> bool:
        logger.info("Loading dataset from API: %s", config.API_URL)
        headers = {}
        if config.APP_TOKEN:
            headers["X-App-Token"] = config.APP_TOKEN
        if config.SECRET_KEY:
            headers["Authorization"] = f"Bearer {config.SECRET_KEY}"
        if config.DATA_SOURCE == "api" and not headers:
            logger.warning("DATA_SOURCE=api but no app_token/secret_key provided; API may rate-limit or fail.")

        # Limit to recent years for performance
        now_year = datetime.utcnow().year
        start_year = now_year - config.API_YEARS_BACK + 1
        where_clause = f"year >= {start_year}"

        limit = min(config.API_LIMIT, 1_000_000)
        batch_size = 50_000
        frames: List[pd.DataFrame] = []
        offset = 0

        field_map = {
            "id": "ID",
            "case_number": "Case Number",
            "date": "Date",
            "block": "Block",
            "iucr": "IUCR",
            "primary_type": "Primary Type",
            "description": "Description",
            "location_description": "Location Description",
            "arrest": "Arrest",
            "domestic": "Domestic",
            "beat": "Beat",
            "district": "District",
            "ward": "Ward",
            "community_area": "Community Area",
            "fbi_code": "FBI Code",
            "x_coordinate": "X Coordinate",
            "y_coordinate": "Y Coordinate",
            "year": "Year",
            "updated_on": "Updated On",
            "latitude": "Latitude",
            "longitude": "Longitude",
            "location": "Location",
        }

        while offset < limit:
            params = {
                "$limit": batch_size,
                "$offset": offset,
                "$where": where_clause,
            }
            try:
                resp = requests.get(config.API_URL, headers=headers, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                logger.exception("API fetch failed at offset %s: %s", offset, exc)
                break

            if not data:
                break

            df = pd.DataFrame(data)
            if df.empty:
                break

            # Rename to internal schema
            df = df.rename(columns=field_map)
            # Keep only known columns
            df = df[[v for v in field_map.values() if v in df.columns]]
            # Parse datetimes
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            if "Updated On" in df.columns:
                df["Updated On"] = pd.to_datetime(df["Updated On"], errors="coerce")
            # Coerce numerics
            for col in ["Latitude", "Longitude", "X Coordinate", "Y Coordinate"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            # Booleans
            for col in ["Arrest", "Domestic"]:
                if col in df.columns:
                    df[col] = df[col].astype("string").str.lower().map({"true": True, "false": False})

            df = df.dropna(subset=["Date", "Latitude", "Longitude"])
            if not df.empty:
                frames.append(df)

            offset += batch_size
            if offset >= limit:
                break

        if not frames:
            logger.error("API returned no usable data")
            return False

        self.df = pd.concat(frames, ignore_index=True)
        logger.info("API data loaded: %s records", f"{len(self.df):,}")
        return True

    def load_data(self) -> bool:
        with self._lock:
            self.cache = {}
            if self.data_source == "api":
                ok = self._load_from_api()
                if ok:
                    self._post_load_vector_ops()
                    self.last_load_status = "ok"
                    self.last_load_error = None
                    self.last_loaded_at = datetime.utcnow().isoformat()
                    return True
                logger.warning("API load failed; falling back to CSV")
            ok = self._load_from_csv()
            if ok:
                self._post_load_vector_ops()
                self.last_load_status = "ok"
                self.last_load_error = None
                self.last_loaded_at = datetime.utcnow().isoformat()
                return True
            self.last_load_status = "error"
            self.last_load_error = f"Failed to load from {self.data_source}"
            return False

    def _post_load_vector_ops(self):
        if self.df is None or self.df.empty:
            return
        self.df["hour"] = self.df["Date"].dt.hour.astype("int8")
        self.df["lat_rad"] = np.radians(self.df["Latitude"]).astype("float32")
        self.df["lon_rad"] = np.radians(self.df["Longitude"]).astype("float32")

    # --- Stats ---
    def _require_data(self) -> bool:
        return self.df is not None and not self.df.empty

    def compute_pareto_stats(self) -> Dict[str, Any]:
        if not self._require_data() or "Beat" not in self.df.columns:
            return {}

        beat_counts = self.df["Beat"].value_counts()
        total = beat_counts.sum()
        total_beats = len(beat_counts)
        if total == 0 or total_beats == 0:
            return {}

        cumulative_pct = beat_counts.cumsum() / total
        top_20_idx = max(int(total_beats * 0.2) - 1, 0)
        share_top_20 = float(cumulative_pct.iloc[top_20_idx]) if top_20_idx < len(cumulative_pct) else 1.0
        beats_for_50_idx = int(cumulative_pct.searchsorted(0.50, side="left"))
        pct_beats_for_50 = (beats_for_50_idx + 1) / total_beats

        return {
            "top_20_percent_holds": round(share_top_20 * 100, 1),
            "beats_for_50_percent": round(pct_beats_for_50 * 100, 1),
            "total_active_beats": int(total_beats),
            "top_3_hottest_beats": beat_counts.head(3).index.astype(str).tolist(),
        }

    def get_summary_stats(self) -> Dict[str, Any]:
        if not self._require_data():
            return {}

        monthly = (
            self.df.set_index("Date")
            .resample("M")
            .size()
            .tail(12)
            .rename_axis("month")
            .reset_index(name="count")
        )
        monthly_counts = {d["month"].strftime("%Y-%m"): int(d["count"]) for d in monthly.to_dict("records")}

        hourly_counts = {int(k): int(v) for k, v in self.df["hour"].value_counts().sort_index().to_dict().items()}

        stats = {
            "total_incidents": int(len(self.df)),
            "date_range": {
                "start": self.df["Date"].min().strftime("%Y-%m-%d"),
                "end": self.df["Date"].max().strftime("%Y-%m-%d"),
            },
            "top_types": {k: int(v) for k, v in self.df["Primary Type"].value_counts().head(7).to_dict().items()},
            "arrest_rate": float(self.df["Arrest"].mean()),
            "hourly_counts": hourly_counts,
            "monthly_counts": monthly_counts,
            "pareto_concentration": self.compute_pareto_stats(),
        }
        return stats

    # --- Hotspots ---
    def compute_hotspots(self, crime_type: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self._require_data():
            return []

        target = self.df
        if crime_type and crime_type != "ALL":
            target = target[target["Primary Type"] == crime_type]

        if target.empty:
            return []

        if len(target) > self.dbscan_sample_limit:
            target = target.sample(self.dbscan_sample_limit, random_state=42)

        coords = target[["lat_rad", "lon_rad"]].to_numpy(dtype="float64")
        epsilon_rad = self.dbscan_eps_miles / self.earth_radius_miles

        db = DBSCAN(
            eps=epsilon_rad,
            min_samples=self.dbscan_min_samples,
            metric="haversine",
            algorithm="ball_tree",
            n_jobs=-1,
        )
        labels = db.fit_predict(coords)
        target = target.assign(cluster=labels)

        clusters = target[target["cluster"] != -1].groupby("cluster")
        results = []
        for cluster_id, data in clusters:
            results.append(
                {
                    "cluster_id": int(cluster_id),
                    "count": int(len(data)),
                    "center_lat": float(data["Latitude"].mean()),
                    "center_lon": float(data["Longitude"].mean()),
                    "crime_type": crime_type if crime_type else "MIXED",
                    # Approx radius using std dev in degrees converted to feet
                    "radius_approx_ft": float(data[["Latitude", "Longitude"]].std().mean() * 364000),
                }
            )

        return sorted(results, key=lambda x: x["count"], reverse=True)[:100]

    # --- Network ---
    def build_network(self, crime_type: str = "ROBBERY") -> Dict[str, Any]:
        if not self._require_data():
            return {"nodes": [], "edges": [], "summary": {}}

        cache_key = f"network:{crime_type}:{self.spatial_radius_miles}:{self.temporal_window_days}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        sub = self.df[self.df["Primary Type"] == crime_type].copy()
        if sub.empty:
            return {"nodes": [], "edges": [], "summary": {}}

        if len(sub) > self.network_node_limit:
            sub = sub.sort_values("Date", ascending=False).head(self.network_node_limit).reset_index(drop=True)
        else:
            sub = sub.reset_index(drop=True)

        r_rad = self.spatial_radius_miles / self.earth_radius_miles
        tree = BallTree(sub[["lat_rad", "lon_rad"]].to_numpy(dtype="float64"), metric="haversine")
        neighbors = tree.query_radius(sub[["lat_rad", "lon_rad"]].to_numpy(dtype="float64"), r=r_rad)

        edges = []
        dates = sub["Date"].to_numpy()
        ids = sub["ID"].astype(str).to_numpy()
        time_window_ns = np.timedelta64(self.temporal_window_days, "D")

        for i, idxs in enumerate(neighbors):
            if len(idxs) < 2:
                continue
            t_i = dates[i]
            t_neighbors = dates[idxs]
            diffs = np.abs(t_neighbors - t_i)
            valid = idxs[diffs <= time_window_ns]
            for j in valid:
                if i < j:
                    edges.append((ids[i], ids[j]))

        G = nx.Graph()
        G.add_edges_from(edges)

        components = list(nx.connected_components(G))
        largest_cc_size = len(max(components, key=len)) if components else 0
        centrality = nx.degree_centrality(G) if G.number_of_nodes() > 0 else {}

        node_meta = (
            sub.assign(id=sub["ID"].astype(str))
            .set_index("id")[["Date", "Latitude", "Longitude", "Description"]]
            .to_dict("index")
        )

        visible_nodes = set().union(*components[:10]) if components else set()
        api_nodes = []
        for nid in visible_nodes:
            meta = node_meta.get(nid, {})
            api_nodes.append(
                {
                    "id": nid,
                    "lat": meta.get("Latitude"),
                    "lng": meta.get("Longitude"),
                    "date": str(meta.get("Date")),
                    "desc": meta.get("Description"),
                    "centrality": centrality.get(nid, 0.0),
                }
            )

        api_edges = [{"source": u, "target": v} for u, v in G.edges() if u in visible_nodes and v in visible_nodes]

        result = {
            "summary": {
                "connected_nodes": int(G.number_of_nodes()),
                "edge_count": int(G.number_of_edges()),
                "components_count": len(components),
                "largest_component_size": largest_cc_size,
            },
            "nodes": api_nodes,
            "edges": api_edges,
        }
        self.cache[cache_key] = result
        return result


# Singleton instance
analyzer = CrimeAnalyzer(config.DATA_PATH)
