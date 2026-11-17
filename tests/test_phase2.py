"""Phase 2 pieces that need no network, plus the MODIS QC used by the first milestone."""

import csv
from datetime import date

import numpy as np
import pytest

from urban_heat import acquisition as acq, config, modis_qc
from urban_heat.grid import Grid


# --- MODIS calendar -------------------------------------------------------------------

def test_modis_dates():
    d = acq.modis_dates(2024, 2024)
    assert len(d) == 46 and d[0] == date(2024, 1, 1) and d[-1] == date(2024, 12, 26)
    assert acq.modis_dates(2023, 2023)[-1] == date(2023, 12, 27)
    assert len(acq.modis_dates(2001, 2025)) == 46 * 25


# --- catalogue: exactly the plan's columns, no duplicates ------------------------------------

def test_catalog_columns_and_dedup(tmp_path):
    cat = tmp_path / "catalog.csv"
    f = config.RAW_MODIS / "MOD11A2_20240101.tif"
    acq.catalog_add("MOD11A2", "2024-01-01", "NASA MODIS", f, catalog=cat)
    acq.catalog_add("MOD11A2", "2024-01-01", "NASA MODIS", f, catalog=cat)   # same file again
    rows = list(csv.DictReader(open(cat, encoding="utf-8")))
    assert list(rows[0]) == ["dataset", "date", "city", "source", "file_path"]
    assert len(rows) == 1 and rows[0]["city"] == "mumbai;navi_mumbai"
    assert rows[0]["file_path"] == "data/raw/modis/MOD11A2_20240101.tif"


# --- retry and resume ------------------------------------------------------------------------

def test_with_retry_backs_off_then_succeeds():
    calls, sleeps = [], []
    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("temporary")
        return "ok"
    assert acq.with_retry(flaky, sleep=sleeps.append, log=lambda *_: None) == "ok"
    assert sleeps == [2.0, 4.0]


def test_with_retry_never_retries_lookup_error():
    calls = []
    def absent():
        calls.append(1)
        raise LookupError("no such date")
    with pytest.raises(LookupError):
        acq.with_retry(absent, sleep=lambda s: None)
    assert len(calls) == 1


def test_run_items_skips_existing_and_records_outcomes(tmp_path):
    (tmp_path / "a").write_text("x")
    def go(item):
        if item == "missing":
            raise LookupError()
        if item == "bad":
            raise ValueError("permanent")
        (tmp_path / item).write_text("x")
    s = acq.run_items(["a", "b", "missing", "bad"], lambda i: tmp_path / i, go, log=lambda *_: None)
    assert s == {"done": ["b"], "skipped": ["a"], "missing": ["missing"], "failed": ["bad"]}


def test_run_items_runs_in_parallel_and_catalog_stays_consistent(tmp_path):
    import threading
    import time
    active, peak, lock = [0], [0], threading.Lock()
    cat = tmp_path / "catalog.csv"

    def go(i):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        (tmp_path / f"f{i}").write_text("x")
        acq.catalog_add("T", str(i), "src", tmp_path / f"f{i}", catalog=cat)
        with lock:
            active[0] -= 1

    s = acq.run_items(range(40), lambda i: tmp_path / f"f{i}", go, workers=16, log=lambda *_: None)
    assert len(s["done"]) == 40 and peak[0] > 1                 # really concurrent
    rows = list(csv.DictReader(open(cat, encoding="utf-8")))
    assert len(rows) == 40 and len({r["file_path"] for r in rows}) == 40   # no lost or duplicate rows


# --- grids -----------------------------------------------------------------------------------

def test_era5_grid_uses_the_native_lattice():
    g = acq.era5_grid((72.77, 18.88, 73.08, 19.29))
    # cell edges on x.x5 deg, cell centres on x.x0 deg
    assert g.xmin == pytest.approx(72.75) and g.ymax == pytest.approx(19.35)
    assert (g.width, g.height) == (4, 5)
    x0, y0, x1, y1 = g.bounds
    assert x0 <= 72.77 and x1 >= 73.08 and y0 <= 18.88 and y1 >= 19.29


def test_subgrids_cover_exactly_once():
    g = Grid("EPSG:32643", 0.0, 7000.0, 1000.0, 5, 7)
    cover = np.zeros(g.shape, int)
    for r, c, sub in acq.subgrids(g, 3):
        assert sub.xmin == g.xmin + c * g.res and sub.ymax == g.ymax - r * g.res
        cover[r:r + sub.height, c:c + sub.width] += 1
    assert (cover == 1).all()


def test_write_tif_round_trip(tmp_path):
    import rasterio
    g = Grid("EPSG:32643", 263000.0, 2135000.0, 1000.0, 3, 2)
    p = acq.write_tif(tmp_path / "x.tif", {"LST_Day_1km": np.full(g.shape, 15000), "QC_Day": np.zeros(g.shape)},
                      g, "uint16", {"collection": "test"})
    with rasterio.open(p) as src:
        assert src.crs.to_string() == "EPSG:32643" and src.descriptions == ("LST_Day_1km", "QC_Day")
        assert src.transform.c == 263000.0 and src.transform.f == 2135000.0 and src.res == (1000.0, 1000.0)
        assert src.read(1)[0, 0] == 15000 and src.tags()["collection"] == "test"
    assert not (tmp_path / "x.partial.tif").exists()


# --- MODIS QC (first milestone) ---------------------------------------------------------------

@pytest.mark.parametrize("dn, qc, kept", [
    (15000, 0b00000000, True),      # good, <=1 K
    (15000, 0b01000000, True),      # good, <=2 K
    (15000, 0b10000000, False),     # error <=3 K
    (15000, 0b00000001, True),      # other quality, <=1 K  (urban cells; kept by default)
    (15000, 0b01000001, True),      # other quality, <=2 K
    (15000, 0b10000001, False),     # other quality, <=3 K
    (15000, 0b00000010, False),     # not produced: cloud
    (15000, 0b00000011, False),     # not produced: other reasons
    (0, 0, False),                  # no data (QC 0 looks "good" but DN 0 is fill)
    (7499, 0, False), (7500, 0, True),
])
def test_quality_mask(dn, qc, kept):
    assert bool(modis_qc.quality_mask(np.array([dn]), np.array([qc]))[0]) is kept


def test_strict_rule_still_available():
    assert not modis_qc.quality_mask(np.array([15000]), np.array([0b01]), require_good_qa=True)[0]


def test_lst_celsius():
    out = modis_qc.lst_celsius(np.array([15000, 15000]), np.array([0, 2]))
    assert out[0] == pytest.approx(26.85, abs=1e-4) and np.isnan(out[1])    # 15000*0.02-273.15


def test_qc_summary():
    s = modis_qc.qc_summary(np.array([15000, 15000, 0]), np.array([0, 1, 0]), np.ones(3, bool))
    assert s == {"cells": 3, "no_lst_value": 1, "qa_good": 1, "qa_other_quality": 1,
                 "error_le_1K": 2, "error_le_2K": 0, "error_gt_2K": 0, "kept": 2}
