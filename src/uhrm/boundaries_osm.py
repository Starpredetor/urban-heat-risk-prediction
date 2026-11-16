"""Provisional study-area boundaries from OpenStreetMap (SRS FR-P1.8, Stage A).

These are a quick starting point. Which polygon is the right one for each city
is OI-1; Stage B replaces them with authoritative boundaries verified in QGIS.
Every saved boundary gets a sidecar `.meta.json` recording where it came from
(FR-P1.2).

Nominatim usage policy: identify the application and send at most one request
per second. `search` does both.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import shapely
from shapely.geometry import mapping, shape

from uhrm.regions import approx_area_km2

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "uhrm-student-project/0.1 (research prototype)"
LICENCE = "© OpenStreetMap contributors, ODbL 1.0"

_last_call = 0.0


def search(query: str, *, limit: int = 10, country: str = "in") -> list[dict]:
    """GeoJSON features (with polygons) matching `query`."""
    global _last_call
    import requests

    wait = 1.1 - (time.monotonic() - _last_call)
    if wait > 0:
        time.sleep(wait)
    resp = requests.get(NOMINATIM_URL, headers={"User-Agent": USER_AGENT}, timeout=60,
                        params={"q": query, "countrycodes": country, "format": "geojson",
                                "polygon_geojson": 1, "limit": limit})
    _last_call = time.monotonic()
    resp.raise_for_status()
    return resp.json()["features"]


def show(features: list[dict]) -> None:
    """Print candidates with their approximate area, to choose between them."""
    for i, f in enumerate(features):
        p = f["properties"]
        g = shape(f["geometry"])
        area = approx_area_km2(g) if g.geom_type in ("Polygon", "MultiPolygon") else float("nan")
        print(f"[{i}] {p.get('osm_type')} {p.get('osm_id')}  {p.get('category')}/{p.get('type')}  "
              f"{g.geom_type}  ~{area:.0f} km2  {p.get('display_name', '')[:70]}")


def pick(features: list[dict], osm_ids) -> list[dict]:
    """The candidates whose osm_id is in `osm_ids`; every id must be found."""
    ids = list(osm_ids)
    chosen = [f for f in features if f["properties"].get("osm_id") in ids]
    missing = set(ids) - {f["properties"].get("osm_id") for f in chosen}
    if missing:
        raise KeyError(f"osm ids not among the candidates: {sorted(missing)}")
    return chosen


def pick_largest_by_prefix(features: list[dict], prefixes) -> list[dict]:
    """For each name prefix, the largest polygon whose display name starts with it.

    Lets a notebook choose boundaries without a manual step. Points and
    buildings are skipped because only polygons qualify. Raises LookupError if a
    prefix matches nothing, so the caller can fall back to choosing ids by hand.
    """
    chosen = []
    for prefix in prefixes:
        matches = [f for f in features
                   if f["properties"].get("display_name", "").lower().startswith(prefix.lower())
                   and shape(f["geometry"]).geom_type in ("Polygon", "MultiPolygon")]
        if not matches:
            raise LookupError(f"no polygon whose name starts with {prefix!r}; set the osm ids by hand")
        chosen.append(max(matches, key=lambda f: approx_area_km2(shape(f["geometry"]))))
    return chosen


def save_boundary(features: list[dict], path, *, query: str) -> Path:
    """Union the chosen polygons and save them as GeoJSON plus a provenance sidecar."""
    if not features:
        raise ValueError("no features to save")
    geoms = [shape(f["geometry"]) for f in features]
    if any(g.geom_type not in ("Polygon", "MultiPolygon") for g in geoms):
        raise ValueError("every chosen feature must be a polygon, not a point or line")
    union = shapely.make_valid(shapely.union_all(geoms))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    collection = {"type": "FeatureCollection",
                  "features": [{"type": "Feature", "properties": {}, "geometry": mapping(union)}]}
    path.write_text(json.dumps(collection), encoding="utf-8")

    meta = {
        "status": "provisional (OpenStreetMap via Nominatim); replace in Stage B (FR-P1.8)",
        "query": query,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "licence": LICENCE,
        "approx_area_km2": round(approx_area_km2(union), 1),
        "sources": [{"osm_type": f["properties"].get("osm_type"),
                     "osm_id": f["properties"].get("osm_id"),
                     "display_name": f["properties"].get("display_name")} for f in features],
    }
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path
