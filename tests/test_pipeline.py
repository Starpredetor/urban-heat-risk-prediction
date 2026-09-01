"""The checks that matter.

Each of these caught a real bug that produced plausible-looking WRONG numbers.
That is the point: none of these mistakes crash. They quietly give you a result
that is incorrect, which is far more dangerous in a research project.
"""

import numpy as np
import pytest
import torch

from uhrm.cube import fill_missing_years, ndbi, ndvi, ndwi
from uhrm.model import HeatForecaster
from uhrm.sensors import (
    landsat_is_clear,
    landsat_reflectance,
    landsat_temperature,
    sentinel2_is_clear,
    sentinel2_reflectance,
)


# --- Converting stored numbers into real units -------------------------------

def test_temperature_conversion():
    """Stored 44000 -> 44000 x 0.00341802 + 149.0 = 299.4 K = 26.2 C"""
    assert landsat_temperature(np.array([44000.0]))[0] == pytest.approx(26.24, abs=0.01)


def test_impossible_temperatures_are_dropped():
    """Empty and broken pixels must not be treated as real readings."""
    result = landsat_temperature(np.array([0.0, 44000.0, 65535.0]))
    assert np.isnan(result[0]) and np.isfinite(result[1]) and np.isnan(result[2])


def test_landsat_reflectance_conversion():
    """Stored 10000 -> 10000 x 0.0000275 - 0.2 = 0.075"""
    assert landsat_reflectance(np.array([10000.0]))[0] == pytest.approx(0.075, abs=1e-6)


def test_impossible_reflectance_is_dropped():
    """Reflectance outside 0-1 is physically impossible.

    This is the bug that produced an NDVI of -12 on real Mumbai images: a
    broken pixel scales to near zero, which makes the ratio blow up.
    """
    assert np.isnan(landsat_reflectance(np.array([0.0]))[0])
    assert np.isnan(landsat_reflectance(np.array([65535.0]))[0])


def test_sentinel2_2022_change_is_handled():
    """ESA changed the file format in January 2022.

    The same patch of ground is stored as 2000 before and 3000 after. Both must
    convert to 0.20. Getting this wrong shifts every reading after 2022 by 0.1
    with no error message at all.
    """
    before = sentinel2_reflectance(np.array([2000.0]), year=2021)[0]
    after = sentinel2_reflectance(np.array([3000.0]), year=2023)[0]
    assert before == pytest.approx(0.20, abs=1e-6)
    assert after == pytest.approx(0.20, abs=1e-6)


def test_ignoring_the_2022_change_would_shift_by_a_tenth():
    """Shows exactly how large the silent error would have been."""
    stored = 3000.0
    correct = sentinel2_reflectance(np.array([stored]), year=2023)[0]
    if_we_forgot = stored / 10000.0
    assert if_we_forgot - correct == pytest.approx(0.10, abs=1e-6)


# --- The indices --------------------------------------------------------------

def test_ndvi_detects_vegetation():
    """Plants reflect infrared strongly: high infrared, low red -> high NDVI."""
    leafy = ndvi(np.array([0.40]), np.array([0.05]))[0]
    concrete = ndvi(np.array([0.22]), np.array([0.20]))[0]
    assert leafy > 0.7 > concrete


def test_ndbi_detects_buildings():
    assert ndbi(np.array([0.30]), np.array([0.20]))[0] == pytest.approx(0.2)


def test_ndwi_detects_water():
    """Water absorbs infrared, so green stays high while infrared drops."""
    sea = ndwi(np.array([0.20]), np.array([0.05]))[0]
    land = ndwi(np.array([0.08]), np.array([0.30]))[0]
    assert sea > 0 > land


def test_indices_always_land_between_minus_one_and_one():
    """A normalised ratio cannot mathematically leave this range."""
    rng = np.random.default_rng(0)
    a, b = rng.uniform(0, 1, 5000), rng.uniform(0, 1, 5000)
    for values in (ndvi(a, b), ndbi(a, b), ndwi(a, b)):
        good = values[np.isfinite(values)]
        assert good.min() >= -1.0001 and good.max() <= 1.0001


def test_index_does_not_crash_on_empty_pixels():
    assert np.isnan(ndvi(np.array([0.0]), np.array([0.0]))[0])


# --- Cloud removal ------------------------------------------------------------

@pytest.mark.parametrize("bit", [0, 1, 2, 3, 4])
def test_landsat_cloud_flags_are_read(bit):
    """Bits 0-4 mean no data, nearby cloud, thin cloud, cloud, shadow."""
    assert not landsat_is_clear(np.array([1 << bit], dtype=np.uint16))[0]


def test_landsat_clear_pixel_is_kept():
    assert landsat_is_clear(np.array([21824], dtype=np.uint16))[0]


@pytest.mark.parametrize("code", [0, 1, 2, 3, 8, 9, 10])
def test_sentinel2_cloud_categories_are_dropped(code):
    assert not sentinel2_is_clear(np.array([float(code)]))[0]


@pytest.mark.parametrize("code", [4, 5, 6, 7, 11])
def test_sentinel2_keeps_ground_and_water(code):
    assert sentinel2_is_clear(np.array([float(code)]))[0]


def test_water_is_never_masked_out():
    """A deliberate choice: the sea has a real temperature, and it is our coastline."""
    assert landsat_is_clear(np.array([1 << 7], dtype=np.uint16))[0]
    assert sentinel2_is_clear(np.array([6.0]))[0]


# --- Filling cloudy gaps ------------------------------------------------------

def test_missing_year_is_filled_between_neighbours():
    series = np.array([[[10.0]], [[np.nan]], [[30.0]]], dtype=np.float32)
    assert fill_missing_years(series)[1, 0, 0] == pytest.approx(20.0)


def test_pixel_missing_in_every_year_stays_unknown():
    """We never invent a value where there is no observation at all."""
    series = np.full((3, 1, 1), np.nan, dtype=np.float32)
    assert np.all(np.isnan(fill_missing_years(series)))


# --- The model ----------------------------------------------------------------

def test_model_produces_a_map_of_the_right_size():
    model = HeatForecaster(n_layers=4, memory=8)
    out = model(torch.randn(2, 4, 4, 32, 32))      # 2 samples, 4 years, 4 layers
    assert out.shape == (2, 1, 32, 32)


def test_model_starts_by_guessing_no_change():
    """Untrained it should predict roughly 'same as last year', then learn."""
    model = HeatForecaster(n_layers=4, memory=8).eval()
    years = torch.randn(1, 4, 4, 16, 16)
    difference = (model(years) - years[:, -1, 0:1]).abs().max().item()
    assert difference < 0.05


def test_the_model_can_actually_learn():
    """Force it to memorise one example. If it cannot, the wiring is broken."""
    torch.manual_seed(0)
    model = HeatForecaster(n_layers=4, memory=16)
    years = torch.randn(2, 4, 4, 16, 16)
    answer = torch.randn(2, 1, 16, 16)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.05)

    start = (model(years) - answer).abs().mean().item()
    for _ in range(150):
        loss = (model(years) - answer).abs().mean()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    assert loss.item() < 0.1 * start


# --- Which satellites we accept -----------------------------------------------

def test_landsat_is_limited_to_the_right_satellites():
    """Landsat 5 and 7 are in the same archive but are NOT compatible.

    They name the thermal band "lwir" instead of "lwir11", so reading them
    fails outright. Landsat 7 also has diagonal stripes of missing data in
    every image since a 2003 hardware failure.
    """
    from uhrm.sensors import LANDSAT
    assert LANDSAT["platforms"] == ["landsat-8", "landsat-9"]
    assert LANDSAT["bands"]["thermal"] == "lwir11"


def test_sentinel2_accepts_any_platform():
    """2A/2B/2C are named inconsistently; filtering would silently return none."""
    from uhrm.sensors import SENTINEL2
    assert SENTINEL2["platforms"] == []
