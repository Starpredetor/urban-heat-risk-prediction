"""Boundary saving and provenance, without the network (SRS FR-P1.2, FR-P1.3)."""

import json

import pytest
from shapely.geometry import Point, box, mapping

from uhrm.boundaries_osm import pick, save_boundary


def feature(osm_id, geom, name="x"):
    return {"type": "Feature", "geometry": mapping(geom),
            "properties": {"osm_type": "relation", "osm_id": osm_id, "display_name": name}}


def test_save_unions_polygons_and_writes_provenance(tmp_path):
    a = feature(1, box(72.80, 19.00, 72.90, 19.10), "District A")
    b = feature(2, box(72.90, 19.00, 73.00, 19.10), "District B")
    out = save_boundary([a, b], tmp_path / "b" / "mumbai.geojson", query="Mumbai districts")

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["type"] == "FeatureCollection" and len(data["features"]) == 1
    xs = [c[0] for c in data["features"][0]["geometry"]["coordinates"][0]]
    assert min(xs) == pytest.approx(72.80) and max(xs) == pytest.approx(73.00)   # one merged polygon

    meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["query"] == "Mumbai districts"
    assert [s["osm_id"] for s in meta["sources"]] == [1, 2]
    assert "provisional" in meta["status"] and "ODbL" in meta["licence"]
    assert meta["approx_area_km2"] > 0


def test_point_features_are_refused(tmp_path):
    with pytest.raises(ValueError, match="polygon"):
        save_boundary([feature(1, Point(72.8, 19.0))], tmp_path / "p.geojson", query="q")


def test_empty_selection_is_refused(tmp_path):
    with pytest.raises(ValueError):
        save_boundary([], tmp_path / "e.geojson", query="q")


def test_pick_returns_requested_and_fails_loudly_on_unknown_ids():
    a, b = feature(1, box(0, 0, 1, 1)), feature(2, box(1, 0, 2, 1))
    assert pick([a, b], [2]) == [b]
    with pytest.raises(KeyError):
        pick([a, b], [3])
