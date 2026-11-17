"""Phase 4 features on hand-made inputs (SRS FR-4.1 to FR-4.3)."""

from datetime import date

import numpy as np
import pytest

from urban_heat import features as ft
from urban_heat.grid import Grid


def test_indices_formulas():
    sr = {"red": np.array([0.1]), "nir": np.array([0.4]), "green": np.array([0.05]), "swir1": np.array([0.3])}
    out = ft.indices(sr)
    assert out["ndvi"][0] == pytest.approx((0.4 - 0.1) / 0.5)
    assert out["ndbi"][0] == pytest.approx((0.3 - 0.4) / 0.7)
    assert out["ndwi"][0] == pytest.approx((0.05 - 0.4) / 0.45)


def test_normalized_difference_guards():
    assert np.isnan(ft.normalized_difference(np.array([0.0]), np.array([0.0]))[0])     # zero denominator
    assert np.isnan(ft.normalized_difference(np.array([np.nan]), np.array([0.2]))[0])   # missing input


def test_nanmedian_stack_ignores_nan_and_counts():
    a = [np.array([[1.0, np.nan]]), np.array([[3.0, np.nan]]), np.array([[2.0, 5.0]])]
    med, count = ft.nanmedian_stack(a)
    assert med[0, 0] == 2.0 and med[0, 1] == 5.0
    assert count.tolist() == [[3, 1]]


def test_nanmedian_stack_cuda_matches_cpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    rng = np.random.default_rng(1)
    arrs = [np.where(rng.random((50, 60)) > 0.3, rng.random((50, 60)), np.nan).astype(np.float32) for _ in range(7)]
    cpu, c1 = ft.nanmedian_stack(arrs)
    gpu, c2 = ft.nanmedian_stack(arrs, torch.device("cuda"))
    np.testing.assert_array_equal(c1, c2)
    np.testing.assert_allclose(cpu, gpu, equal_nan=True)


def test_group_landcover():
    classes = np.array([50, 10, 95, 80, 60, 0, 90, 40])
    assert ft.group_landcover(classes).tolist() == [1, 2, 2, 3, 4, 0, 2, 2]   # built-up, veg, veg, water, other, none


def test_major_water_distance_ignores_small_ponds():
    c = np.full((40, 40), 50)
    c[:, :5] = 80                 # a "sea" strip: 40 x 5 pixels
    c[20, 30] = 80                # a one-pixel pond
    # at 100 m pixels: sea = 200 pixels = 2 km2 (major), pond = 0.01 km2 (not major)
    dist, major = ft.major_water_distance(c, 100.0, min_km2=1.0)
    assert major[:, :5].all() and not major[20, 30]
    assert dist[0, 5] == pytest.approx(100.0) and dist[0, 4] == 0.0
    assert dist[20, 30] == pytest.approx((30 - 4) * 100.0)       # distance to the sea, not the pond


def test_aggregate_non_nesting_grids():
    src = Grid("EPSG:32643", 0.0, 90.0, 30.0, 3, 3)               # 3 x 3 pixels of 30 m
    dst = Grid("EPSG:32643", 0.0, 100.0, 50.0, 2, 2)              # 2 x 2 cells of 50 m
    vals = np.arange(9, dtype=float).reshape(3, 3)
    mean, count = ft.aggregate(vals, src, dst)
    # pixel centres at 15, 45, 75 m: cell col 0 gets x=15,45; col 1 gets x=75. y centres 75,45,15 -> rows 0,1,1
    assert count.tolist() == [[2, 1], [4, 2]]
    assert mean[0, 0] == pytest.approx((0 + 1) / 2) and mean[1, 1] == pytest.approx((5 + 8) / 2)


def test_period_means_follow_modis_periods():
    days = [f"2024{m:02d}{d:02d}" for m, d in [(12, 25), (12, 26), (12, 27), (12, 31)]]
    daily = np.array([1.0, 2.0, 4.0, 6.0]).reshape(4, 1, 1)
    out = ft.period_means(daily, days, [date(2024, 12, 18), date(2024, 12, 26)])
    assert out[0, 0, 0] == pytest.approx(1.0)          # only 12-25 falls in [12-18, 12-26)
    assert out[1, 0, 0] == pytest.approx(4.0)          # 12-26..12-31: mean of 2, 4, 6 (window cut at 1 Jan)
