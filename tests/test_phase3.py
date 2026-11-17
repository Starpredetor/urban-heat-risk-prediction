"""Phase 3 preprocessing on hand-made inputs (SRS FR-3.1 to FR-3.9)."""

from datetime import date

import numpy as np
import pytest
from shapely.geometry import box

from urban_heat import acquisition as acq, preprocessing as pp
from urban_heat.grid import Grid


# --- Landsat: official QA flags and scale factors ------------------------------------------

@pytest.mark.parametrize("qa, clear", [
    (0b0000000, True),          # clear
    (0b1000000, True),          # bit 6 "clear" flag set: still clear
    (0b10000000, True),         # bit 7 water: kept (water has a real temperature)
    (0b00001, False),           # fill
    (0b00010, False),           # dilated cloud
    (0b00100, False),           # cirrus
    (0b01000, False),           # cloud
    (0b10000, False),           # cloud shadow
])
def test_landsat_qa(qa, clear):
    assert bool(pp.landsat_clear(np.array([qa]))[0]) is clear


def test_landsat_scaling_masking_and_clipping():
    def bands(st, sr, qa):
        b = {v: np.array([sr, sr, sr]) for v in pp.LANDSAT_SR_BANDS.values()}
        b.update(ST_B10=np.array([st, st, st]), QA_PIXEL=np.array([qa, qa, qa]))
        return b
    inside = np.array([True, True, False])
    out = pp.landsat_scale(bands(st=45000, sr=20000, qa=0), inside)
    # ST: 45000*0.00341802+149 = 302.811 K = 29.66 C ; SR: 20000*0.0000275-0.2 = 0.35
    assert out["st_celsius"][0] == pytest.approx(45000 * 0.00341802 + 149 - 273.15, abs=1e-3)
    assert out["red"][0] == pytest.approx(0.35, abs=1e-6)
    assert np.isnan(out["st_celsius"][2]) and np.isnan(out["nir"][2])        # outside the cities
    cloudy = pp.landsat_scale(bands(st=45000, sr=20000, qa=0b01000), inside)
    assert np.isnan(cloudy["st_celsius"]).all()
    bad_sr = pp.landsat_scale(bands(st=45000, sr=50000, qa=0), inside)     # DN above 43636
    assert np.isnan(bad_sr["red"][0]) and np.isfinite(bad_sr["st_celsius"][0])
    nodata = pp.landsat_scale(bands(st=0, sr=20000, qa=0), inside)
    assert np.isnan(nodata["st_celsius"][0])


# --- WorldCover: categorical resampling --------------------------------------------------------

def test_block_mode_keeps_real_classes():
    a = np.array([[10, 10, 50, 50, 50, 80],
                  [10, 50, 50, 80, 80, 80],
                  [50, 10, 10, 80, 80, 50]], dtype=np.uint8)
    np.testing.assert_array_equal(pp.block_mode(a, 3), [[10, 80]])


# --- city masks ----------------------------------------------------------------------------------

def test_city_id_raster():
    g = Grid("EPSG:32643", 0.0, 2000.0, 1000.0, 3, 2)
    ids = pp.city_id_raster(g, {"mumbai": box(0, 0, 1000, 2000), "navi_mumbai": box(2000, 1000, 3000, 2000)})
    np.testing.assert_array_equal(ids, [[1, 0, 2], [1, 0, 0]])


# --- MODIS: QC, Celsius, clip, missing dates ----------------------------------------------------

def test_process_modis_and_valid_fractions(tmp_path, monkeypatch):
    monkeypatch.setattr(pp, "P_MODIS", tmp_path)
    g = Grid("EPSG:32643", 0.0, 2000.0, 1000.0, 2, 2)
    raw = acq.write_tif(tmp_path / "raw.tif", {"LST_Day_1km": np.array([[15000, 15000], [15500, 0]]),
                                               "QC_Day": np.array([[0, 2], [1, 0]])}, g, "uint16", {})
    ids = np.array([[1, 1], [2, 0]], dtype=np.uint8)
    path, lst = pp.process_modis(raw, date(2024, 4, 6), g, ids)
    assert path.name == "MOD11A2_LST_C_20240406.tif"
    assert lst[0, 0] == pytest.approx(26.85, abs=1e-4)          # kept, converted to C
    assert np.isnan(lst[0, 1])                                    # cloud flag
    assert lst[1, 0] == pytest.approx(36.85, abs=1e-4)           # "other quality", error <= 1 K: kept
    assert np.isnan(lst[1, 1])                                    # outside both cities (and no data)
    rows = pp.valid_fractions(date(2024, 4, 6), lst, ids)
    assert rows[0] == {"date": "2024-04-06", "city": "mumbai", "cells": 2, "valid_cells": 1,
                       "valid_fraction": 0.5, "raw_file": "present"}
    absent = pp.valid_fractions(date(2001, 6, 18), None, ids)
    assert absent[1]["valid_cells"] == 0 and absent[1]["raw_file"] == "absent in archive"


# --- ERA5-Land: coarse values copied, sea cells replaced by the nearest land cell ---------------

def test_era5_lookup_uses_containing_cell_or_nearest_land():
    # a 1 km grid around 72.95E, 19.05N; ERA5 points 0.1 deg apart, the western one is sea
    g = Grid("EPSG:32643", 285000.0, 2108000.0, 1000.0, 3, 1)     # three cells west->east
    lons, lats = [72.8, 72.9, 73.0], [19.05, 19.05, 19.05]
    land = [False, True, True]
    use, substituted = pp.era5_cell_lookup(g, lons, lats, land)
    assert use.shape == (1, 3)
    assert set(use.ravel().tolist()) <= {1, 2}                    # never the sea point
    assert not substituted.all() or use.ravel().tolist() == [1, 1, 1]


def test_landsat_scaling_is_identical_on_cuda_and_cpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    rng = np.random.default_rng(0)
    shape = (300, 400)
    bands = {v: rng.integers(5000, 46000, shape) for v in pp.LANDSAT_SR_BANDS.values()}
    bands["ST_B10"] = rng.integers(0, 50000, shape)
    bands["QA_PIXEL"] = rng.choice([21824, 21826, 22080, 23888, 1], shape)   # clear, cloud, water, shadow, fill
    inside = rng.random(shape) > 0.3
    cpu = pp.landsat_scale(bands, inside)
    gpu = pp.landsat_scale(bands, inside, torch.device("cuda"))
    for k in cpu:
        np.testing.assert_array_equal(np.isnan(cpu[k]), np.isnan(gpu[k]))
        np.testing.assert_allclose(cpu[k], gpu[k], rtol=0, atol=1e-5, equal_nan=True)
