"""Phase 3 preprocessing: hand-computed cases plus one end-to-end cube build from
real GeoTIFFs on a tiny grid (SRS V1, FR-P3.x)."""

from datetime import date

import numpy as np
import pytest

from uhrm import acquire
from uhrm.backends.base import ModisComposite, RasterSet
from uhrm.preprocess import (ReflectanceRule, block_mean, build_cube, clear_reflectance,
                             era5_to_celsius, indices, load_cube, normalized_difference,
                             save_cube, stateqa_clear)
from uhrm.project import load_project
from uhrm.regions import Grid, refined

RULE = ReflectanceRule()


# --- StateQA ----------------------------------------------------------------------

@pytest.mark.parametrize("qa, clear", [
    (0b00, True),            # clear
    (0b01, False),           # cloudy
    (0b10, False),           # mixed
    (0b11, False),           # "not set": treated as not clear (conservative)
    (0b100, False),          # cloud shadow
    (1 << 8, True),          # cirrus small (class 1) allowed by default
    (2 << 8, False),         # cirrus average
    (3 << 8, False),         # cirrus high
    (1 << 10, True),         # bit 10 is deliberately ignored
])
def test_stateqa_truth_table(qa, clear):
    assert bool(stateqa_clear(np.array([qa]))[0]) is clear


def test_cirrus_limit_is_configurable():
    assert not stateqa_clear(np.array([1 << 8]), max_cirrus_class=0)[0]


def raw_bands(qa=0, dn=1000, valid=1, shape=(1,)):
    full = lambda v: np.full(shape, v)  # noqa: E731
    return {"sur_refl_b01": full(dn), "sur_refl_b02": full(dn), "sur_refl_b04": full(dn),
            "sur_refl_b06": full(dn), "StateQA": full(qa), "valid": full(valid)}


def test_clear_reflectance_scales_and_masks():
    assert clear_reflectance(raw_bands(), RULE)["red"][0] == pytest.approx(0.1)
    assert np.isnan(clear_reflectance(raw_bands(qa=1), RULE)["nir"][0])          # cloud
    assert np.isnan(clear_reflectance(raw_bands(valid=0), RULE)["nir"][0])       # masked pixel
    assert np.isnan(clear_reflectance(raw_bands(dn=-28672), RULE)["nir"][0])     # fill value
    # masked pixel arrives as DN 0 and QA 0 ("clear"): only the valid band catches it
    assert np.isnan(clear_reflectance(raw_bands(dn=0, qa=0, valid=0), RULE)["red"][0])


# --- aggregation and indices ---------------------------------------------------------

def test_block_mean_ignores_nan_and_applies_min_count():
    a = np.array([[1, 3, np.nan, np.nan],
                  [5, 7, np.nan, 8],
                  [1, 1, 1, 1],
                  [1, 1, 1, np.nan]], dtype=float)
    mean, count = block_mean(a, 2, min_count=2)
    assert mean[0, 0] == pytest.approx(4.0) and count[0, 0] == 4
    assert np.isnan(mean[0, 1]) and count[0, 1] == 1                # only 1 clear sub-pixel
    assert mean[1, 1] == pytest.approx(1.0) and count[1, 1] == 3
    with pytest.raises(ValueError):
        block_mean(np.zeros((3, 4)), 2)


def test_normalized_difference_and_indices():
    assert normalized_difference(np.array([0.4]), np.array([0.1]))[0] == pytest.approx(0.6)
    assert np.isnan(normalized_difference(np.array([0.0]), np.array([0.0]))[0])
    out = indices({"red": np.array([0.1]), "nir": np.array([0.4]),
                   "green": np.array([0.05]), "swir1": np.array([0.3])})
    assert out["ndvi"][0] == pytest.approx(0.6)            # (0.4-0.1)/(0.5)
    assert out["ndbi"][0] == pytest.approx(-1 / 7)          # (0.3-0.4)/(0.7)
    assert out["ndwi"][0] == pytest.approx(-0.35 / 0.45)    # (0.05-0.4)/(0.45)


def test_era5_kelvin_to_celsius_only_for_temperatures():
    out = era5_to_celsius({"temperature_2m": np.array([300.0]), "u_component_of_wind_10m": np.array([3.0])})
    assert out["temperature_2m_c"][0] == pytest.approx(26.85, abs=1e-4)
    assert out["u_component_of_wind_10m"][0] == 3.0 and "temperature_2m" not in out


# --- end to end ---------------------------------------------------------------------------

GRID = Grid(crs="EPSG:32643", xmin=0.0, ymax=2000.0, res=1000.0, width=2, height=2)
D1, D2 = date(2024, 4, 6), date(2024, 4, 14)


def build_fixture(tmp_path):
    project = load_project("offline", storage_root=tmp_path)
    raw = tmp_path / "raw"
    fine = refined(GRID, 2)

    # WorldCover: top-right cell is water, the rest land
    water = np.array([[0.0, 0.9], [0.1, 0.0]])
    acquire.write_bands_geotiff(raw / "worldcover" / "worldcover_2021_fractions.tif",
                                RasterSet(bands={"c50": 1 - water, "c80": water}, source="t"), GRID)
    acquire.write_bands_geotiff(raw / "terrain" / "srtm_terrain.tif",
                                RasterSet(bands={"elevation_mean": np.full(GRID.shape, 12.0)}, source="t"), GRID)

    # LST on D1 only (D2 missing): 3 valid cells, bottom-right cloudy
    dn = np.array([[15000, 15500], [15250, 15000]], dtype=np.uint16)
    qc = np.array([[0, 0], [0, 0b10]], dtype=np.uint16)
    acquire.write_geotiff(acquire.composite_path(raw / "modis", D1),
                          ModisComposite(date=D1, lst_dn=dn, qc=qc, source="t"), GRID)

    # Reflectance on D1: NIR 0.4, red 0.1 everywhere; one cloudy 500 m pixel in cell (0,0)
    qa = np.zeros(fine.shape)
    qa[0, 0] = 1
    refl = {"sur_refl_b01": np.full(fine.shape, 1000), "sur_refl_b02": np.full(fine.shape, 4000),
            "sur_refl_b04": np.full(fine.shape, 500), "sur_refl_b06": np.full(fine.shape, 3000),
            "StateQA": qa, "valid": np.ones(fine.shape)}
    acquire.write_bands_geotiff(acquire.dated_path(raw / "modis_reflectance", "MOD09A1", D1),
                                RasterSet(bands=refl, source="t"), fine, "int32")

    # ERA5 on D1 and D2
    for d in (D1, D2):
        acquire.write_bands_geotiff(acquire.dated_path(raw / "era5", "ERA5L", d),
                                    RasterSet(bands={"temperature_2m": np.full(GRID.shape, 303.15),
                                                     "u_component_of_wind_10m": np.full(GRID.shape, 2.0)},
                                              source="t"), GRID)
    return project


def test_build_cube_end_to_end(tmp_path):
    project = build_fixture(tmp_path)
    cube = build_cube(project, GRID, [2024], log=lambda *_: None)
    t1, t2 = list(cube["dates"]).index(D1.isoformat()), list(cube["dates"]).index(D2.isoformat())

    assert cube["lst_c"].shape == (46, 2, 2)
    np.testing.assert_allclose(cube["lst_c"][t1], [[26.85, 36.85], [31.85, np.nan]], atol=1e-3)
    assert np.isnan(cube["lst_c"][t2]).all()                       # missing file -> NaN, not an error
    assert cube["present_lst"][t1] and not cube["present_lst"][t2]

    # land mask from WorldCover water fraction; valid fraction is over land cells only
    np.testing.assert_array_equal(cube["land"], [[True, False], [True, True]])
    assert cube["lst_valid_fraction"][t1] == pytest.approx(2 / 3)   # (0,0),(1,0) valid of 3 land
    assert cube["usable"][t1] and not cube["usable"][t2]

    # indices: NDVI = (0.4-0.1)/0.5 = 0.6; cell (0,0) still has 3 clear sub-pixels
    np.testing.assert_allclose(cube["ndvi"][t1], 0.6, atol=1e-5)
    assert cube["refl_clear_subpixels"][t1][0, 0] == 3 and cube["refl_clear_subpixels"][t1][1, 1] == 4
    assert np.isnan(cube["ndvi"][t2]).all()

    assert cube["era5_temperature_2m_c"][t2][0, 0] == pytest.approx(30.0, abs=1e-4)
    assert cube["era5_u_component_of_wind_10m"][t1][0, 0] == pytest.approx(2.0)
    assert cube["terrain_elevation_mean"][0, 0] == pytest.approx(12.0)


def test_save_and_load_round_trip_with_manifest(tmp_path):
    import json
    project = build_fixture(tmp_path)
    cube = build_cube(project, GRID, [2024], log=lambda *_: None)
    path = save_cube(cube, GRID, project, tmp_path / "processed" / "cube_2024.npz")
    back = load_cube(path)
    np.testing.assert_array_equal(back["dates"], cube["dates"])
    np.testing.assert_allclose(back["lst_c"], cube["lst_c"], equal_nan=True)
    manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert manifest["variables"]["lst_c"]["unit"] == "degC"
    assert manifest["summary"]["usable_dates"] == 1
    assert manifest["summary"]["missing_files"]["lst"] == 45
    assert manifest["gap_filling"].startswith("none")


def test_build_cube_requires_worldcover(tmp_path):
    project = load_project("offline", storage_root=tmp_path)
    with pytest.raises(FileNotFoundError, match="WorldCover"):
        build_cube(project, GRID, [2024], log=lambda *_: None)
