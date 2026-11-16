"""Phase 2 orchestration with fake backends: no network, no Earth Engine (SRS FR-X.3)."""

import csv
from datetime import date

import numpy as np
import pytest

from uhrm import acquire, phase2
from uhrm.backends.base import ModisComposite, RasterSet
from uhrm.project import load_project
from uhrm.regions import Grid

GRID = Grid(crs="EPSG:32643", xmin=0.0, ymax=3000.0, res=1000.0, width=3, height=3)


def zeros(grid, v=0.0):
    return np.full(grid.shape, v)


class FakeLst:
    def __init__(self, absent=()):
        self.absent = set(absent)

    def fetch_composite(self, d, grid):
        if d in self.absent:
            raise LookupError("absent")
        z = np.full(grid.shape, 15000, dtype=np.uint16)
        return ModisComposite(date=d, lst_dn=z, qc=np.zeros_like(z), source="fake:lst",
                              meta={"collection": "FAKE/LST"})


class FakeRefl:
    def __init__(self):
        self.shapes = []

    def fetch_composite(self, d, grid):
        self.shapes.append(grid.shape)
        return RasterSet(bands={"sur_refl_b01": zeros(grid), "valid": zeros(grid, 1)},
                         date=d, source="fake:refl", meta={"collection": "FAKE/REFL"})


class FakeEra5:
    def __init__(self):
        self.windows = []

    def fetch_window(self, start, end, grid):
        self.windows.append((start, end))
        return RasterSet(bands={"temperature_2m": zeros(grid, 300.0)}, date=start,
                         source="fake:era5", meta={"collection": "FAKE/ERA5"})


class FakeWorldCover:
    def fetch_fractions(self, year, grid):
        return RasterSet(bands={"c10": zeros(grid, 0.5), "c50": zeros(grid, 0.5)},
                         source="fake:wc", meta={"collection": f"FAKE/WC{year}"})


class FakeTerrain:
    def fetch(self, grid):
        return RasterSet(bands={"elevation_mean": zeros(grid, 10)}, source="fake:srtm",
                         meta={"collection": "FAKE/SRTM"})


def fake_lst_writer(path, comp, grid):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"lst")


def fake_bands_writer(path, rs, grid, dtype="float32"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"{dtype}|{grid.shape}|{sorted(rs.bands)}".encode())


def make_backends(**kw):
    return phase2.Backends(modis_lst=kw.get("lst", FakeLst()), modis_reflectance=kw.get("refl", FakeRefl()),
                           era5=kw.get("era5", FakeEra5()), worldcover=FakeWorldCover(),
                           terrain=FakeTerrain())


def run(tmp_path, backends=None, years=(2024,), **kw):
    project = load_project("offline", storage_root=tmp_path)
    logs = []
    result = phase2.run_phase2(project, GRID, years, backends or make_backends(),
                               write_lst=fake_lst_writer, write_bands=fake_bands_writer,
                               log=logs.append, **kw)
    return project, result, logs


def catalog_rows(tmp_path):
    with open(tmp_path / "catalog.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --- the 8-day calendar -----------------------------------------------------------

def test_expected_dates_46_per_year_with_known_endpoints():
    d2023 = acquire.expected_composite_dates(2023, 2023)
    assert len(d2023) == 46 and d2023[0] == date(2023, 1, 1) and d2023[-1] == date(2023, 12, 27)
    assert acquire.expected_composite_dates(2024, 2024)[-1] == date(2024, 12, 26)   # leap year
    assert len(acquire.expected_composite_dates(2023, 2024)) == 92


def test_milestone_date_is_a_real_composite_start():
    assert date(2024, 4, 6) in acquire.expected_composite_dates(2024, 2024)


def test_last_composite_of_a_year_ends_on_1_january():
    assert acquire.composite_window(date(2024, 4, 6)) == (date(2024, 4, 6), date(2024, 4, 14))
    assert acquire.composite_window(date(2023, 12, 27)) == (date(2023, 12, 27), date(2024, 1, 1))


def test_present_dates_ignores_empty_and_foreign_files(tmp_path):
    (tmp_path / "MOD11A2_20240101.tif").write_bytes(b"x")
    (tmp_path / "MOD11A2_20240109.tif").write_bytes(b"")          # interrupted write
    (tmp_path / "MOD11A2_notadate.tif").write_bytes(b"x")
    assert acquire.present_dates(tmp_path, "MOD11A2") == {date(2024, 1, 1)}


def test_coverage_report_lists_missing_dates():
    expected = acquire.expected_composite_dates(2024, 2024)
    rep = acquire.coverage_report(expected, set(expected) - {date(2024, 4, 6)})
    assert rep[2024]["expected"] == 46 and rep[2024]["found"] == 45
    assert rep[2024]["missing"] == [date(2024, 4, 6)]


# --- run_phase2 -------------------------------------------------------------------

def test_full_run_fetches_everything_and_catalogues_it(tmp_path):
    refl, era5 = FakeRefl(), FakeEra5()
    project, result, _ = run(tmp_path, make_backends(refl=refl, era5=era5))

    s = result["summaries"]
    assert [len(s[k]["fetched"]) for k in ("modis_lst", "modis_reflectance", "era5_land")] == [46, 46, 46]
    assert len(s["worldcover"]["fetched"]) == 2 and len(s["terrain"]["fetched"]) == 1
    assert len(catalog_rows(tmp_path)) == 46 * 3 + 2 + 1
    assert (tmp_path / "raw" / "worldcover" / "worldcover_2020_fractions.tif").exists()
    assert (tmp_path / "raw" / "terrain" / "srtm_terrain.tif").exists()
    assert result["coverage"]["modis_lst"][2024]["found"] == 46

    # reflectance is requested on the 2x finer grid; ERA5 windows follow the composites
    assert set(refl.shapes) == {(6, 6)}
    assert era5.windows[0] == (date(2024, 1, 1), date(2024, 1, 9))
    assert era5.windows[-1] == (date(2024, 12, 26), date(2025, 1, 1))


def test_reflectance_is_written_as_int32_and_era5_as_float32(tmp_path):
    run(tmp_path)
    assert (tmp_path / "raw" / "modis_reflectance" / "MOD09A1_20240101.tif").read_bytes().startswith(b"int32|(6, 6)")
    assert (tmp_path / "raw" / "era5" / "ERA5L_20240101.tif").read_bytes().startswith(b"float32|(3, 3)")


def test_rerun_skips_everything_and_adds_no_catalogue_rows(tmp_path):
    run(tmp_path)
    before = len(catalog_rows(tmp_path))
    _, result, _ = run(tmp_path)
    assert all(not v["fetched"] for v in result["summaries"].values())
    assert len(result["summaries"]["modis_lst"]["skipped"]) == 46
    assert len(catalog_rows(tmp_path)) == before


def test_which_limits_the_datasets(tmp_path):
    _, result, _ = run(tmp_path, which=("terrain",))
    assert list(result["summaries"]) == ["terrain"] and result["coverage"] == {}
    assert not (tmp_path / "raw" / "modis").exists()


def test_gap_in_a_series_shows_up_in_coverage_and_is_logged(tmp_path):
    missing_day = date(2024, 4, 6)
    _, result, logs = run(tmp_path, make_backends(lst=FakeLst(absent=[missing_day])))
    assert result["summaries"]["modis_lst"]["missing"] == [missing_day]
    assert result["coverage"]["modis_lst"][2024]["found"] == 45
    assert any("years with gaps" in line and "2024" in line for line in logs)


def test_bad_arguments_are_rejected(tmp_path):
    project = load_project("offline", storage_root=tmp_path)
    with pytest.raises(ValueError, match="unknown datasets"):
        phase2.run_phase2(project, GRID, [2024], make_backends(), which=("landsat",))
    with pytest.raises(ValueError, match="years must lie within"):
        phase2.run_phase2(project, GRID, [1999], make_backends())
    with pytest.raises(ValueError, match="years must lie within"):
        phase2.run_phase2(project, GRID, [], make_backends())


def test_rasterset_requires_matching_band_shapes():
    with pytest.raises(ValueError):
        RasterSet(bands={"a": np.zeros((2, 2)), "b": np.zeros((3, 3))}, source="x")
    with pytest.raises(ValueError):
        RasterSet(bands={}, source="x")
