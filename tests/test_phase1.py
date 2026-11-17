"""Phase 1: boundaries and the common grid (SRS FR-1.1 to FR-1.7). No network needed."""

import json

import numpy as np
import pytest
from shapely.geometry import Point, Polygon, box, mapping

from urban_heat import boundaries as b
from urban_heat.grid import Grid, load_grids, save_grids


def feature(osm_id, geom, name):
    return {"type": "Feature", "geometry": mapping(geom),
            "properties": {"osm_type": "relation", "osm_id": osm_id, "display_name": name}}


# --- step 1: choosing boundaries ------------------------------------------------------

def test_pick_largest_by_prefix_ignores_points_and_takes_largest():
    small = feature(1, box(72.80, 19.00, 72.82, 19.02), "Mumbai City District")
    big = feature(2, box(72.80, 19.00, 72.90, 19.10), "mumbai city district, Maharashtra")
    sub = feature(3, box(72.80, 19.10, 72.90, 19.20), "Mumbai Suburban District")
    office = feature(4, Point(72.83, 18.94), "Mumbai City office")
    chosen = b.pick_largest_by_prefix([small, big, sub, office], ["Mumbai City", "Mumbai Suburban"])
    assert [f["properties"]["osm_id"] for f in chosen] == [2, 3]
    with pytest.raises(LookupError):
        b.pick_largest_by_prefix([office], ["Mumbai City"])


def test_pick_by_id_fails_loudly():
    a = feature(1, box(0, 0, 1, 1), "a")
    assert b.pick([a], [1]) == [a]
    with pytest.raises(KeyError):
        b.pick([a], [2])


def test_approx_area():
    expected = 111.32 * np.cos(np.radians(19.05)) * 0.1 * 110.57 * 0.1
    assert b.approx_area_km2(box(72.8, 19.0, 72.9, 19.1)) == pytest.approx(expected, rel=0.01)


# --- step 3: CRS --------------------------------------------------------------------

def test_geojson_crs(tmp_path):
    p = tmp_path / "a.geojson"
    p.write_text(json.dumps({"type": "FeatureCollection", "features": []}))
    assert b.geojson_crs(p) == "EPSG:4326"                      # RFC 7946 default
    p.write_text(json.dumps({"type": "FeatureCollection", "features": [],
                             "crs": {"type": "name", "properties": {"name": "EPSG:32643"}}}))
    assert b.geojson_crs(p) == "EPSG:32643"


# --- step 4: repair --------------------------------------------------------------------

def test_invalid_geometry_is_reported_and_repaired():
    bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
    assert b.validity(bowtie) != "valid"
    fixed = b.repair(bowtie)
    assert fixed.is_valid and fixed.area > 0


def test_boundary_from_features_records_repair():
    bowtie = Polygon([(72.8, 19.0), (72.9, 19.1), (72.9, 19.0), (72.8, 19.1)])
    geom, prov = b.boundary_from_features([feature(7, bowtie, "Bowtie")], name="x", query="q")
    assert geom.is_valid and prov["repaired"] and prov["osm_features"][0]["osm_id"] == 7


# --- steps 5-6: save and combine ------------------------------------------------------------

def test_save_load_and_combine(tmp_path):
    m = b.save_boundary(box(72.80, 18.90, 72.95, 19.25), tmp_path / "mumbai.geojson",
                        name="Mumbai", provenance={"source": "test"})
    n = b.save_boundary(box(73.00, 19.00, 73.06, 19.15), tmp_path / "navi_mumbai.geojson",
                        name="Navi Mumbai", provenance={"source": "test"})
    meta = json.loads(m.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["crs"] == "EPSG:4326" and meta["source"] == "test" and meta["approx_area_km2"] > 0
    assert b.load(m).equals(box(72.80, 18.90, 72.95, 19.25))

    c = b.combine([m, n], tmp_path / "combined_study_area.geojson")
    combined = b.load(c)
    assert combined.area == pytest.approx(b.load(m).area + b.load(n).area)   # disjoint cities
    assert b.geojson_crs(c) == "EPSG:4326"


def test_save_refuses_points_and_invalid(tmp_path):
    with pytest.raises(ValueError):
        b.save_boundary(Point(0, 0), tmp_path / "p.geojson", name="p", provenance={})
    with pytest.raises(ValueError, match="invalid"):
        b.save_boundary(Polygon([(0, 0), (2, 2), (2, 0), (0, 2)]), tmp_path / "x.geojson", name="x",
                        provenance={})


# --- step 7: common grid ------------------------------------------------------------------

def test_grid_covers_geometry_snapped_with_margin():
    g = Grid.covering(box(1500, 2500, 3500, 4500), res=1000, margin_m=1000, crs="EPSG:32643")
    assert (g.xmin, g.ymax, g.width, g.height) == (0.0, 6000.0, 5, 5)
    assert g.bounds == (0.0, 1000.0, 5000.0, 6000.0)
    assert g.transform == (1000.0, 0.0, 0.0, 0.0, -1000.0, 6000.0)


def test_cell_mask_uses_area_overlap():
    g = Grid(crs="EPSG:32643", xmin=0.0, ymax=2000.0, res=1000.0, width=2, height=2)
    frac = g.overlap_fraction(box(0, 1000, 1500, 2000))      # top-left full, top-right half
    np.testing.assert_allclose(frac, [[1.0, 0.5], [0.0, 0.0]])
    np.testing.assert_array_equal(g.cell_mask(box(0, 1000, 1500, 2000), 0.5), [[True, True], [False, False]])
    with pytest.raises(ValueError):
        g.cell_mask(box(0, 0, 1, 1), 0.0)


def test_grids_round_trip(tmp_path):
    grids = {"modis_1km": Grid("EPSG:32643", 0.0, 6000.0, 1000.0, 5, 5),
             "landsat_30m": Grid("EPSG:32643", 0.0, 6000.0, 30.0, 167, 167)}
    save_grids(grids, tmp_path / "grids.json")
    assert load_grids(tmp_path / "grids.json") == grids
