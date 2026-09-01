"""Step 1 and 2 of the pipeline: get the images, and clean them.

For each satellite and each year:

    1. ask the archive which images cover our city in the chosen months
    2. keep only the least cloudy ones
    3. read them onto one fixed map grid, so pixels line up across years
    4. mark cloudy pixels as unknown
    5. take the MIDDLE value of each pixel across all images that year

Step 5 is the important one. A single image has holes where clouds were. Taking
the middle value (the median) of a dozen images fills those holes, and ignores a
leftover cloud edge that an average would let drag the result down.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import planetary_computer
import pystac_client
from odc.geo.geobox import GeoBox
from odc.stac import load as odc_load
from pyproj import Transformer

from uhrm.config import Settings
from uhrm.sensors import (
    hide_cloudy_pixels,
    landsat_is_clear,
    landsat_reflectance,
    landsat_temperature,
    sentinel2_is_clear,
    sentinel2_reflectance,
)

log = logging.getLogger(__name__)

# Free, no account needed. Hosts the same official ESA and USGS images.
ARCHIVE = "https://planetarycomputer.microsoft.com/api/stac/v1"


def map_grid(settings: Settings) -> GeoBox:
    """The one fixed grid every image is placed on.

    Without this, images from different satellite passes sit a few metres apart
    and "change over time" would partly be that misalignment instead of a real
    change on the ground.
    """
    to_map = Transformer.from_crs("EPSG:4326", settings.crs, always_xy=True)
    left, bottom, right, top = settings.bbox
    # All four corners: a rectangle in lon/lat is not a rectangle on a map.
    xs, ys = to_map.transform([left, right, left, right], [bottom, bottom, top, top])
    step = settings.resolution
    bounds = (
        math.floor(min(xs) / step) * step,
        math.floor(min(ys) / step) * step,
        math.ceil(max(xs) / step) * step,
        math.ceil(max(ys) / step) * step,
    )
    return GeoBox.from_bbox(bounds, crs=settings.crs, resolution=step, tight=True)


def find_images(sensor: dict, settings: Settings, year: int) -> list:
    """Ask the archive for this year's images, least cloudy first."""
    catalog = pystac_client.Client.open(ARCHIVE, modifier=planetary_computer.sign_inplace)
    first, last = min(settings.months), max(settings.months)
    end = f"{year + 1}-01-01" if last == 12 else f"{year}-{last + 1:02d}-01"

    filters = {"eo:cloud_cover": {"lt": settings.max_cloud}}
    if sensor["platforms"]:
        filters["platform"] = {"in": sensor["platforms"]}

    found = list(
        catalog.search(
            collections=[sensor["collection"]],
            bbox=list(settings.bbox),
            datetime=f"{year}-{first:02d}-01/{end}",
            query=filters,
        ).items()
    )
    found.sort(key=lambda item: item.properties.get("eo:cloud_cover", 100))
    return found[: settings.max_images_per_year]


def read_images(images: list, sensor: dict, settings: Settings) -> dict[str, np.ndarray]:
    """Read the images onto the shared grid. Returns our band name -> array."""
    data = odc_load(
        images,
        bands=list(sensor["bands"].values()),
        geobox=map_grid(settings),
        groupby="solar_day",           # images from the same day are one picture
        chunks={"x": 2048, "y": 2048},
        dtype="float32",
        nodata=float("nan"),
        # The quality layer holds category codes, so it must never be blurred.
        resampling={sensor["bands"]["quality"]: "nearest", "*": "average"},
    ).compute()
    return {ours: data[theirs].values for ours, theirs in sensor["bands"].items()}


def yearly_average(sensor: dict, settings: Settings, year: int) -> dict[str, np.ndarray]:
    """One cloud-free picture for this satellite, this year.

    Returns each band in real units, plus `n_clear` counting how many good
    observations each pixel actually had.
    """
    images = find_images(sensor, settings, year)
    if not images:
        log.warning("%s %d: no usable images found", sensor["name"], year)
        return {}
    log.info("%s %d: using %d images", sensor["name"], year, len(images))

    raw = read_images(images, sensor, settings)

    if sensor["has_thermal"]:
        clear = landsat_is_clear(raw["quality"])
        to_reflectance = landsat_reflectance
    else:
        clear = sentinel2_is_clear(raw["quality"])
        def to_reflectance(x):
            return sentinel2_reflectance(x, year)

    result = {}
    for band in ("blue", "green", "red", "nir", "swir"):
        cleaned = hide_cloudy_pixels(to_reflectance(raw[band]), clear)
        result[band] = _middle_value(cleaned)

    if sensor["has_thermal"]:
        cleaned = hide_cloudy_pixels(landsat_temperature(raw["thermal"]), clear)
        result["thermal"] = _middle_value(cleaned)

    result["n_clear"] = np.isfinite(
        hide_cloudy_pixels(to_reflectance(raw["red"]), clear)
    ).sum(axis=0).astype(np.int16)
    return result


def _middle_value(images: np.ndarray) -> np.ndarray:
    """Median across images, ignoring the cloudy (unknown) pixels.

    A pixel that was cloudy in every single image stays unknown -- we do not
    invent a value for it.
    """
    with np.errstate(invalid="ignore"):
        # An all-cloudy pixel warns about an empty average; that is expected.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.nanmedian(images, axis=0).astype(np.float32)
