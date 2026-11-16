"""Grid and city-mask geometry, against hand-computed cases (SRS Phase 1, V1).

These tests use projected coordinates directly, so they need no pyproj.
"""

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from uhrm.regions import Grid, approx_area_km2, cell_report, load_grid, refined, repair, save_grid

GRID = Grid(crs="EPSG:32643", xmin=1000.0, ymax=5000.0, res=1000.0, width=3, height=3)


def test_grid_geometry_properties():
    assert GRID.shape == (3, 3)
    assert GRID.bounds == (1000.0, 2000.0, 4000.0, 5000.0)
    assert GRID.transform == (1000.0, 0.0, 1000.0, 0.0, -1000.0, 5000.0)


def test_covering_snaps_to_multiples_of_resolution():
    g = Grid.covering([box(1500, 2500, 3500, 4500)], res=1000, buffer_m=0, crs="EPSG:32643")
    assert (g.xmin, g.ymax, g.width, g.height) == (1000.0, 5000.0, 3, 3)


def test_covering_adds_the_buffer():
    g = Grid.covering([box(1500, 2500, 3500, 4500)], res=1000, buffer_m=1000, crs="EPSG:32643")
    assert (g.xmin, g.ymax, g.width, g.height) == (0.0, 6000.0, 5, 5)
    # every geometry sits fully inside the grid
    gx0, gy0, gx1, gy1 = g.bounds
    assert gx0 <= 1500 and gy0 <= 2500 and gx1 >= 3500 and gy1 >= 4500


def test_covering_two_cities_spans_both():
    a, b = box(1000, 4000, 2000, 5000), box(7000, 1000, 8000, 2000)
    g = Grid.covering([a, b], res=1000, buffer_m=0, crs="EPSG:32643")
    assert g.bounds == (1000.0, 1000.0, 8000.0, 5000.0)


def test_overlap_fraction_whole_cells():
    # polygon exactly covers rows 0-1, cols 0-1
    frac = GRID.overlap_fraction(box(1000, 3000, 3000, 5000))
    expected = np.zeros((3, 3), dtype=np.float32)
    expected[:2, :2] = 1.0
    np.testing.assert_allclose(frac, expected)


def test_overlap_fraction_partial_cell():
    # left half of the top-left cell
    frac = GRID.overlap_fraction(box(1000, 4000, 1500, 5000))
    assert frac[0, 0] == pytest.approx(0.5)
    assert frac.sum() == pytest.approx(0.5)


def test_cell_mask_threshold():
    half = box(1000, 4000, 1500, 5000)
    assert GRID.cell_mask(half, 0.5)[0, 0]          # >= threshold counts
    assert not GRID.cell_mask(half, 0.6)[0, 0]
    with pytest.raises(ValueError):
        GRID.cell_mask(half, 0.0)


def test_polygon_outside_grid_gives_empty_mask():
    assert not GRID.cell_mask(box(9000, 9000, 9500, 9500)).any()


def test_cell_report_counts_and_flags_overlap():
    a = GRID.cell_mask(box(1000, 3000, 3000, 5000))   # 4 cells
    b = GRID.cell_mask(box(3000, 2000, 4000, 4000))   # 2 cells, disjoint from a
    rep = cell_report(GRID, {"a": a, "b": b})
    assert rep["cells_per_city"] == {"a": 4, "b": 2}
    assert rep["overlap_between_cities"] == {"a&b": 0}
    assert rep["grid_cells"] == 9

    c = GRID.cell_mask(box(2000, 3000, 3000, 4000))   # inside a's area
    assert cell_report(GRID, {"a": a, "c": c})["overlap_between_cities"]["a&c"] == 1


def test_refined_grid_nests_exactly_in_the_coarse_grid():
    fine = refined(GRID, 2)
    assert fine.shape == (6, 6) and fine.res == 500.0
    assert fine.bounds == GRID.bounds                      # same extent
    assert (fine.xmin, fine.ymax) == (GRID.xmin, GRID.ymax)
    with pytest.raises(ValueError):
        refined(GRID, 0)


def test_grid_json_round_trip(tmp_path):
    save_grid(GRID, tmp_path / "sub" / "grid.json")
    assert load_grid(tmp_path / "sub" / "grid.json") == GRID


def test_repair_fixes_a_self_intersecting_polygon():
    bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
    assert not bowtie.is_valid
    fixed = repair(bowtie)
    assert fixed.is_valid and fixed.area > 0


def test_approx_area_is_close_for_a_known_box():
    # 0.1 x 0.1 degrees at 19 N: 111.32*cos(19 deg)*0.1 by 110.57*0.1 km
    expected = 111.32 * np.cos(np.radians(19.05)) * 0.1 * 110.57 * 0.1
    got = approx_area_km2(box(72.8, 19.0, 72.9, 19.1))
    assert got == pytest.approx(expected, rel=0.01)
