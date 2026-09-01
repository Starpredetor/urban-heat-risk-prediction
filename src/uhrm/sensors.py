"""The two satellites we use, and how to read each one.

    Sentinel-2  ->  NDVI, NDBI, NDWI   (10 m detail, an image every ~5 days)
    Landsat     ->  Land Surface Temperature

Sentinel-2 carries no thermal camera, so it physically cannot measure
temperature. Landsat is the only source of LST here. Sentinel-2 is the better
source for the indices because it sees finer detail more often.
"""

from __future__ import annotations

import numpy as np

# Satellite files store whole numbers to save space. These constants turn them
# into real physical values. They come from the data providers, not from us.

LANDSAT = {
    "name": "landsat",
    "collection": "landsat-c2-l2",
    # our name -> the name used in the data files
    "bands": {
        "blue": "blue", "green": "green", "red": "red",
        "nir": "nir08", "swir": "swir16",
        "thermal": "lwir11", "quality": "qa_pixel",
    },
    "first_year": 2013,          # Landsat 8 launched in 2013
    "has_thermal": True,
    # Only Landsat 8 and 9. The archive also holds Landsat 5 and 7, but they
    # name their thermal band "lwir" instead of "lwir11", and Landsat 7 has had
    # diagonal stripes of missing data in every image since a 2003 hardware
    # failure. Mixing them in would break the read and corrupt the averages.
    "platforms": ["landsat-8", "landsat-9"],
}

SENTINEL2 = {
    "name": "sentinel2",
    "collection": "sentinel-2-l2a",
    "bands": {
        "blue": "B02", "green": "B03", "red": "B04",
        "nir": "B08", "swir": "B11",
        "quality": "SCL",
    },
    "first_year": 2017,          # worldwide coverage starts in 2017
    "has_thermal": False,
    # Empty means "accept any". Sentinel-2A/2B/2C are named inconsistently
    # across the archive, and an over-tight filter silently returns nothing.
    "platforms": [],
}


# --- turning stored numbers into real reflectance ----------------------------
#
# Reflectance is "what fraction of sunlight this surface bounces back", so it
# must land between 0 and 1. Anything outside that is a broken pixel, and we
# throw it away rather than pretend it is 0 or 1.

def landsat_reflectance(stored: np.ndarray) -> np.ndarray:
    """Landsat: value = stored x 0.0000275 - 0.2"""
    value = np.asarray(stored, dtype=np.float32) * 0.0000275 - 0.2
    return _only_sensible(value)


def sentinel2_reflectance(stored: np.ndarray, year: int) -> np.ndarray:
    """Sentinel-2: value = stored / 10000, but ESA changed this in 2022.

    From January 2022 the files carry an extra +1000 that has to be subtracted
    first. Ignoring it shifts every reading by 0.1 with no error message, which
    would quietly corrupt every index from 2022 onwards.
    """
    offset = -1000.0 if year >= 2022 else 0.0
    value = (np.asarray(stored, dtype=np.float32) + offset) / 10000.0
    return _only_sensible(value)


def _only_sensible(reflectance: np.ndarray) -> np.ndarray:
    """Replace impossible reflectance with 'unknown' (NaN)."""
    ok = (reflectance >= 0.0) & (reflectance <= 1.0)
    return np.where(ok, reflectance, np.nan).astype(np.float32)


def landsat_temperature(stored: np.ndarray) -> np.ndarray:
    """Landsat thermal band -> degrees Celsius.

    stored x 0.00341802 + 149.0 gives Kelvin; subtract 273.15 for Celsius.
    Ground surfaces below 0 C or above 70 C in pre-monsoon Mumbai are broken
    pixels, not real readings.
    """
    kelvin = np.asarray(stored, dtype=np.float32) * 0.00341802 + 149.0
    celsius = kelvin - 273.15
    return np.where((celsius >= 0) & (celsius <= 70), celsius, np.nan).astype(np.float32)


# --- finding the cloudy pixels ------------------------------------------------
#
# Both satellites ship a "quality" layer telling us what each pixel is. We use
# it to drop clouds and their shadows. We deliberately KEEP water: the sea has
# a real temperature, and dropping it would erase Mumbai's coastline.

def landsat_is_clear(quality: np.ndarray) -> np.ndarray:
    """Landsat packs several yes/no flags into one number, one per bit.

    Bit 0 = no data, 1 = nearby cloud, 2 = thin cloud, 3 = cloud, 4 = shadow.
    """
    q = np.nan_to_num(np.asarray(quality, dtype=np.float64), nan=0.0).astype(np.uint16)
    bad = np.zeros(q.shape, dtype=bool)
    for bit in (0, 1, 2, 3, 4):
        bad |= (q >> np.uint16(bit)) & np.uint16(1) == 1
    return ~bad


# Sentinel-2 labels every pixel with a single category number instead.
SCL_LABELS = {
    0: "no data", 1: "broken", 2: "shadow", 3: "cloud shadow",
    4: "vegetation", 5: "bare soil", 6: "water", 7: "unknown",
    8: "cloud", 9: "thick cloud", 10: "thin cloud", 11: "snow",
}
SCL_BAD = (0, 1, 2, 3, 8, 9, 10)


def sentinel2_is_clear(quality: np.ndarray) -> np.ndarray:
    """Drop the cloud, shadow and broken categories; keep land and water."""
    q = np.asarray(quality, dtype=np.float32)
    bad = ~np.isfinite(q)                       # outside the image
    labels = np.nan_to_num(q, nan=0.0).round().astype(np.int16)
    for code in SCL_BAD:
        bad |= labels == code
    return ~bad


def hide_cloudy_pixels(values: np.ndarray, is_clear: np.ndarray) -> np.ndarray:
    """Set cloudy pixels to 'unknown' so they are ignored later."""
    assert values.shape == is_clear.shape, "image and cloud mask must be the same size"
    return np.where(is_clear, values, np.nan).astype(np.float32)
