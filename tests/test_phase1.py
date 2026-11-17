"""Phase 1 outputs and the map diagnostics (SRS FR-P1.x)."""

import json

import numpy as np
import pytest
from shapely.geometry import box, mapping

from uhrm import phase1
from uhrm.diagnostics import qc_breakdown, registration_offset
from uhrm.modis import QcRule
from uhrm.project import load_project
from uhrm.regions import Grid


# --- registration ------------------------------------------------------------------

def coast(n=24):
    """Irregular coast (sea on the left) plus an island: like a real coastline, it
    pins the alignment in both directions."""
    y, x = np.mgrid[0:n, 0:n]
    land = x > 10 + 4 * np.sin(y / 2.5)
    land[3:6, 2:5] = True                      # island
    return land


def test_aligned_data_scores_best_at_zero_shift():
    land = coast()
    res = registration_offset(land.copy(), land)
    assert res["aligned"] and res["agreement_at_zero"] == 1.0


def test_misaligned_data_is_detected_with_its_offset():
    land = coast()
    shifted = np.zeros_like(land)
    shifted[:, 2:] = land[:, :-2]              # data pushed 2 cells east
    res = registration_offset(shifted, land)
    assert not res["aligned"] and res["best_shift_rows_cols"] == (0, 2)


def test_registration_tolerates_inland_cloud_gaps():
    land = coast()
    has_value = land.copy()
    has_value[5:9, 15:19] = False              # a cloud over land
    assert registration_offset(has_value, land)["aligned"]


# --- QC breakdown ------------------------------------------------------------------

def test_qc_breakdown_counts_reasons_and_alternatives():
    dn = np.array([15000, 15000, 15000, 15000, 0], dtype=np.uint16)
    qc = np.array([0b00, 0b01, 0b10000001, 0b11000000, 0], dtype=np.uint16)
    #             good  other  other+<=3K   good,>3K      no value
    region = np.ones(5, bool)
    b = qc_breakdown(dn, qc, region, QcRule())
    assert b["cells"] == 5 and b["with_lst_value"] == 4
    assert b["mandatory_qa (cells with a value)"]["01 other quality"] == 2
    assert b["kept_by_current_rule"] == 1
    assert b["kept_if_other_quality_allowed"] == 2
    assert b["kept_if_other_quality_and_error_le_3K"] == 3


# --- coastal flags and phase 1 outputs -----------------------------------------------------

def test_coastal_cells():
    water = np.array([[1.0, 0.0, 0.0],
                      [1.0, 0.0, 0.2],
                      [0.0, 0.0, 0.0]])
    c = phase1.coastal_cells(water, 0.5)
    np.testing.assert_array_equal(c, [[False, True, False],     # touches water cell
                                      [False, True, True],      # touches water / has 20% water
                                      [True, False, False]])    # touches water cell above


def write_boundary(path, geom_lonlat):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": mapping(geom_lonlat)}]}), encoding="utf-8")


def setup_project(tmp_path):
    p = load_project("offline", storage_root=tmp_path)
    write_boundary(p.boundary_path(p.area("mumbai")), box(72.80, 18.95, 72.90, 19.20))
    write_boundary(p.boundary_path(p.area("navi_mumbai")), box(73.00, 19.00, 73.05, 19.10))
    geoms = {"mumbai": box(0, 0, 4000, 6000), "navi_mumbai": box(5000, 1000, 6000, 3000)}
    return p, geoms


def test_build_report_and_save(tmp_path):
    p, geoms = setup_project(tmp_path)
    grid = Grid(crs="EPSG:32643", xmin=0.0, ymax=6000.0, res=1000.0, width=6, height=6)
    water = np.zeros(grid.shape, dtype=np.float32)
    water[:, 0] = 1.0                                # westmost column is sea
    p1 = phase1.build(p, geoms, water=water, grid=grid)
    rep = phase1.report(p1)
    assert rep["per_city"]["mumbai"]["cells"] == 24
    assert rep["per_city"]["mumbai"]["water_dominated_cells"] == 6
    assert rep["per_city"]["mumbai"]["land_cells"] == 18
    assert rep["per_city"]["mumbai"]["coastal_land_cells"] == 6      # column next to the sea
    assert rep["overlap_between_cities"]["mumbai&navi_mumbai"] == 0
    assert p1.analysis_grid.res == 30.0

    out = phase1.save(p1, p)
    for key in ("grid", "grid_30m", "combined", "cells", "report"):
        assert out[key].exists()
    combined = json.loads(out["combined"].read_text(encoding="utf-8"))
    assert len(combined["features"]) == 1

    import rasterio
    with rasterio.open(out["cells"]) as src:
        assert list(src.descriptions) == ["city_id", "water_fraction", "coastal"]
        assert src.read(1)[0, 0] == 1 and src.read(2)[0, 0] == 1.0


def test_save_refuses_a_changed_grid(tmp_path):
    p, geoms = setup_project(tmp_path)
    g1 = Grid(crs="EPSG:32643", xmin=0.0, ymax=6000.0, res=1000.0, width=6, height=6)
    phase1.save(phase1.build(p, geoms, grid=g1), p)
    g2 = Grid(crs="EPSG:32643", xmin=-1000.0, ymax=6000.0, res=1000.0, width=7, height=6)
    with pytest.raises(RuntimeError, match="grid changed"):
        phase1.save(phase1.build(p, geoms, grid=g2), p)


def test_build_without_water_still_works_and_says_so(tmp_path):
    p, geoms = setup_project(tmp_path)
    grid = Grid(crs="EPSG:32643", xmin=0.0, ymax=6000.0, res=1000.0, width=6, height=6)
    p1 = phase1.build(p, geoms, grid=grid)
    assert p1.land is None and "not available" in phase1.report(p1)["per_city"]
    with pytest.raises(ValueError, match="does not match"):
        phase1.build(p, geoms, water=np.zeros((2, 2)), grid=grid)
