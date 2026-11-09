"""Step 3 and 4: turn cleaned images into the four maps we model.

The four layers, matching the project plan:

    lst    Land Surface Temperature, degrees Celsius   (from Landsat)
    ndvi   how much vegetation                          (from Sentinel-2)
    ndbi   how built-up                                 (from Sentinel-2)
    ndwi   how much water                               (from Sentinel-2)

Stacked over the years this is the "multi-temporal dataset": one file per city,
each holding years x 4 layers x height x width.

Where Sentinel-2 has no images for a year (it only starts in 2017), the indices
fall back to Landsat. That is recorded per year in `index_source`, because
switching sensor mid-series can put a small step in a graph that looks like a
real change but is not.
"""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr

from uhrm.config import City, Settings
from uhrm.download import map_grid, yearly_average
from uhrm.sensors import SENTINEL2, landsat_for

log = logging.getLogger(__name__)

LAYERS = ["lst", "ndvi", "ndbi", "ndwi"]


def _ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a - b) / (a + b), the shape every index below takes.

    Using a ratio rather than a plain difference cancels out lighting: a sunlit
    and a shaded patch of the same grass give the same answer.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    total = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total != 0, (a - b) / total, np.nan).astype(np.float32)


def ndvi(nir, red):
    """Vegetation. Plants reflect infrared strongly, so leafy areas score high."""
    return _ratio(nir, red)


def ndbi(swir, nir):
    """Built-up. Concrete and tarmac reflect shortwave infrared strongly."""
    return _ratio(swir, nir)


def ndwi(green, nir):
    """Water. Water absorbs infrared, so the sea and creeks score high."""
    return _ratio(green, nir)


def _indices_from(bands: dict) -> dict[str, np.ndarray]:
    return {
        "ndvi": ndvi(bands["nir"], bands["red"]),
        "ndbi": ndbi(bands["swir"], bands["nir"]),
        "ndwi": ndwi(bands["green"], bands["nir"]),
    }


def build_city_cube(settings: Settings, city: City) -> xr.Dataset:
    """Download, clean, and stack every year for one city."""
    size = city.size_pixels
    log.info("%s: %d x %d pixels at %d m (%.0f x %.0f km), %s",
             city.name, size, size, city.resolution,
             city.radius_km, city.radius_km, city.crs)

    years = settings.years
    stack = {name: np.full((len(years), size, size), np.nan, np.float32) for name in LAYERS}
    clear_counts = np.zeros((len(years), size, size), dtype=np.int16)
    index_source = []

    for i, year in enumerate(years):
        # Landsat is the only satellite with a heat camera, so it always
        # supplies the temperature.
        landsat = yearly_average(landsat_for(year), settings, city, year)
        if landsat:
            stack["lst"][i] = landsat["thermal"]
            clear_counts[i] = landsat["n_clear"]

        # Sentinel-2 sees finer detail, so it supplies the indices when it can.
        source = "none"
        s2 = yearly_average(SENTINEL2, settings, city, year) \
            if year >= SENTINEL2["first_year"] else {}
        if s2:
            source = "sentinel2"
            for name, values in _indices_from(s2).items():
                stack[name][i] = values
        elif landsat:
            source = "landsat"
            for name, values in _indices_from(landsat).items():
                stack[name][i] = values
        index_source.append(source)

    for name in LAYERS:
        stack[name] = fill_missing_years(stack[name])

    left, bottom, right, top = map_grid(city).extent.boundingbox
    return xr.Dataset(
        {
            "data": (("year", "layer", "y", "x"),
                     np.stack([stack[n] for n in LAYERS], axis=1)),
            "n_clear": (("year", "y", "x"), clear_counts),
            "index_source": (("year",), np.array(index_source)),
        },
        coords={
            "year": np.array(years, dtype=np.int32),
            "layer": LAYERS,
            "y": top - (np.arange(size) + 0.5) * city.resolution,
            "x": left + (np.arange(size) + 0.5) * city.resolution,
        },
        attrs={"city": city.name, "crs": city.crs, "resolution": city.resolution,
               "lat": city.lat, "lon": city.lon, "radius_km": city.radius_km},
    )


def fill_missing_years(series: np.ndarray) -> np.ndarray:
    """Fill a pixel's missing years by joining the dots between known years.

    A pixel missing in EVERY year cannot be filled and stays unknown. We never
    invent a value where we have no observation at all.
    """
    n_years = series.shape[0]
    flat = series.reshape(n_years, -1).copy()
    steps = np.arange(n_years, dtype=np.float32)

    known = np.isfinite(flat)
    # Only touch pixels that are partly missing; most are complete.
    partial = np.flatnonzero(known.any(axis=0) & ~known.all(axis=0))
    for pixel in partial:
        column = flat[:, pixel]
        good = known[:, pixel]
        flat[:, pixel] = np.interp(steps, steps[good], column[good])

    return flat.reshape(series.shape)


def save_cube(cube: xr.Dataset, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cube.to_netcdf(path)
    log.info("saved %s", path)


def load_cube(path) -> xr.Dataset:
    return xr.open_dataset(path)


def layer(cube: xr.Dataset, name: str) -> np.ndarray:
    """Pull one named layer out as a (years, y, x) array."""
    return cube["data"].sel(layer=name).values
