"""Step 3 and 4: turn cleaned images into the four maps we model.

The four layers, matching the project plan:

    lst    Land Surface Temperature, degrees Celsius   (from Landsat)
    ndvi   how much vegetation                          (from Sentinel-2)
    ndbi   how built-up                                 (from Sentinel-2)
    ndwi   how much water                               (from Sentinel-2)

Stacked over the years, this is the "multi-temporal dataset" the model trains
on: years x 4 layers x height x width.
"""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr

from uhrm.config import Settings
from uhrm.download import map_grid, yearly_average
from uhrm.sensors import LANDSAT, SENTINEL2

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


def build_cube(settings: Settings) -> xr.Dataset:
    """Download, clean, and stack every year into one dataset."""
    grid = map_grid(settings)
    height, width = grid.shape
    log.info(
        "%s: %d x %d pixels at %d m (%.1f x %.1f km)",
        settings.city, height, width, settings.resolution,
        width * settings.resolution / 1000, height * settings.resolution / 1000,
    )

    years = settings.years
    stack = {name: np.full((len(years), height, width), np.nan, np.float32) for name in LAYERS}
    clear_counts = np.zeros((len(years), height, width), dtype=np.int16)

    for i, year in enumerate(years):
        log.info("--- %d ---", year)

        landsat = yearly_average(LANDSAT, settings, year)
        if landsat:
            stack["lst"][i] = landsat["thermal"]
            clear_counts[i] = landsat["n_clear"]

        s2 = yearly_average(SENTINEL2, settings, year)
        if s2:
            stack["ndvi"][i] = ndvi(s2["nir"], s2["red"])
            stack["ndbi"][i] = ndbi(s2["swir"], s2["nir"])
            stack["ndwi"][i] = ndwi(s2["green"], s2["nir"])

    for name in LAYERS:
        stack[name] = fill_missing_years(stack[name])

    left, bottom, right, top = grid.extent.boundingbox
    return xr.Dataset(
        {
            "data": (("year", "layer", "y", "x"),
                     np.stack([stack[n] for n in LAYERS], axis=1)),
            "n_clear": (("year", "y", "x"), clear_counts),
        },
        coords={
            "year": np.array(years, dtype=np.int32),
            "layer": LAYERS,
            "y": top - (np.arange(height) + 0.5) * settings.resolution,
            "x": left + (np.arange(width) + 0.5) * settings.resolution,
        },
        attrs={"city": settings.city, "crs": settings.crs,
               "resolution": settings.resolution},
    )


def fill_missing_years(series: np.ndarray) -> np.ndarray:
    """Fill a pixel's missing years by joining the dots between known years.

    A pixel that is missing in EVERY year cannot be filled and stays unknown.
    We never invent a value where we have no observation at all.
    """
    n_years = series.shape[0]
    flat = series.reshape(n_years, -1).copy()
    steps = np.arange(n_years, dtype=np.float32)

    for pixel in range(flat.shape[1]):
        column = flat[:, pixel]
        known = np.isfinite(column)
        if known.all() or not known.any():
            continue
        flat[:, pixel] = np.interp(steps, steps[known], column[known])

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
