"""Phase 1: study-area boundaries (SRS FR-1.1 to FR-1.6).

Plan step                                   Function
1 download administrative boundaries        search, show, pick, pick_largest_by_prefix
3 check the coordinate reference system     geojson_crs
4 repair invalid geometries                 validity, repair
5 save both boundaries as GeoJSON           save_boundary
6 create a combined study-area boundary     combine
(2 "open them in QGIS" is done by the team; GeoJSON in WGS84 opens directly.)

Boundaries come from OpenStreetMap via Nominatim (SRS assumption A1). Nominatim's
policy: identify the application and send at most one request per second.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import mapping, shape

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "urban-heat-student-project/1.0 (research)"
LICENCE = "© OpenStreetMap contributors, ODbL 1.0"
WGS84 = "EPSG:4326"
POLYGONAL = ("Polygon", "MultiPolygon")

_last_call = 0.0


# --- step 1: download ---------------------------------------------------------

def search(query: str, *, limit: int = 10, country: str = "in") -> list[dict]:
    """GeoJSON features (with polygon geometry) from OpenStreetMap matching `query`."""
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


def approx_area_km2(geom_lonlat) -> float:
    """Approximate area of a lon/lat geometry (within ~1%), for sanity checks."""
    lat = geom_lonlat.centroid.y
    kx, ky = 111.32 * math.cos(math.radians(lat)), 110.57
    return float(shapely.transform(geom_lonlat, lambda c: np.column_stack([c[:, 0] * kx, c[:, 1] * ky])).area)


def show(features: list[dict]) -> None:
    for i, f in enumerate(features):
        p, g = f["properties"], shape(f["geometry"])
        area = approx_area_km2(g) if g.geom_type in POLYGONAL else float("nan")
        print(f"[{i}] {p.get('osm_type')} {p.get('osm_id')}  {p.get('category')}/{p.get('type')}  "
              f"{g.geom_type}  ~{area:.0f} km2  {p.get('display_name', '')[:70]}")


def pick(features: list[dict], osm_ids) -> list[dict]:
    ids = list(osm_ids)
    chosen = [f for f in features if f["properties"].get("osm_id") in ids]
    missing = set(ids) - {f["properties"].get("osm_id") for f in chosen}
    if missing:
        raise KeyError(f"osm ids not among the candidates: {sorted(missing)}")
    return chosen


def pick_largest_by_prefix(features: list[dict], prefixes) -> list[dict]:
    """For each prefix, the largest polygon whose name starts with it (points and
    buildings never qualify). LookupError if a prefix matches no polygon."""
    chosen = []
    for prefix in prefixes:
        matches = [f for f in features
                   if f["properties"].get("display_name", "").lower().startswith(prefix.lower())
                   and shape(f["geometry"]).geom_type in POLYGONAL]
        if not matches:
            raise LookupError(f"no polygon whose name starts with {prefix!r}; choose osm ids by hand")
        chosen.append(max(matches, key=lambda f: approx_area_km2(shape(f["geometry"]))))
    return chosen


# --- step 3: coordinate reference system ----------------------------------------

def geojson_crs(path) -> str:
    """CRS of a GeoJSON file. RFC 7946 GeoJSON is always WGS84 (EPSG:4326); an old-style
    "crs" member is honoured if present."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    name = (data.get("crs") or {}).get("properties", {}).get("name", "")
    if not name or name.upper().endswith(("CRS84", "4326")):
        return WGS84
    return name


# --- step 4: validity and repair -----------------------------------------------------

def validity(geom) -> str:
    """'valid', or the reason the geometry is invalid."""
    return "valid" if geom.is_valid else shapely.is_valid_reason(geom)


def repair(geom):
    """Make a geometry valid and keep only its polygon parts."""
    fixed = shapely.make_valid(geom)
    if fixed.geom_type in POLYGONAL:
        return fixed
    parts = [g for g in getattr(fixed, "geoms", []) if g.geom_type in POLYGONAL]
    if not parts:
        raise ValueError("geometry has no polygon part after repair")
    return shapely.union_all(parts)


# --- step 5: save; reading back ----------------------------------------------------------

def save_boundary(geom_lonlat, path, *, name: str, provenance: dict) -> Path:
    """Write one polygon as GeoJSON (WGS84) plus a `.meta.json` recording its source."""
    if geom_lonlat.geom_type not in POLYGONAL:
        raise ValueError("a boundary must be a polygon")
    if not geom_lonlat.is_valid:
        raise ValueError(f"refusing to save an invalid geometry: {validity(geom_lonlat)}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"name": name}, "geometry": mapping(geom_lonlat)}]}),
        encoding="utf-8")
    meta = {"name": name, "crs": WGS84, "approx_area_km2": round(approx_area_km2(geom_lonlat), 1),
            "saved_utc": datetime.now(timezone.utc).isoformat(), **provenance}
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def boundary_from_features(features: list[dict], *, name: str, query: str) -> tuple:
    """Union the chosen OSM polygons, repair if needed. Returns (geometry, provenance)."""
    if not features:
        raise ValueError("no features chosen")
    geoms = [shape(f["geometry"]) for f in features]
    if any(g.geom_type not in POLYGONAL for g in geoms):
        raise ValueError("every chosen feature must be a polygon")
    union = shapely.union_all(geoms)
    before = validity(union)
    geom = repair(union) if before != "valid" else union
    provenance = {"source": "OpenStreetMap via Nominatim", "licence": LICENCE, "query": query,
                  "validity_before_repair": before, "repaired": before != "valid",
                  "osm_features": [{"osm_type": f["properties"].get("osm_type"),
                                    "osm_id": f["properties"].get("osm_id"),
                                    "display_name": f["properties"].get("display_name")} for f in features]}
    return geom, provenance


def load(path):
    """A boundary file as one shapely geometry in WGS84."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    feats = data["features"] if data.get("type") == "FeatureCollection" else [data]
    return shapely.union_all([shape(f["geometry"] if "geometry" in f else f) for f in feats])


def to_crs(geom, src: str, dst: str):
    """Reproject a shapely geometry (needs pyproj)."""
    from pyproj import Transformer
    t = Transformer.from_crs(src, dst, always_xy=True)
    return shapely.transform(geom, lambda c: np.column_stack(t.transform(c[:, 0], c[:, 1])))


# --- step 6: combined boundary ---------------------------------------------------------------

def combine(paths, out_path) -> Path:
    """Union of the city boundaries, saved as combined_study_area.geojson."""
    geom = repair(shapely.union_all([load(p) for p in paths]))
    return save_boundary(geom, out_path, name="combined_study_area",
                         provenance={"source": "union of " + ", ".join(Path(p).name for p in paths)})
