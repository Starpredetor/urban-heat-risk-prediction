"""Earth Engine backend for MOD11A2 (Stage A, Colab).

Needs an authenticated Earth Engine session (see `uhrm.colab.bootstrap`).
Pixels are requested directly on our model grid with `ee.data.computePixels`,
so one request returns one composite already aligned to the grid.

Resampling: Earth Engine reprojects with nearest neighbour by default. That is
what we want for a QC byte, and it never invents LST values.

NOTE: this module has been written against the Earth Engine API but could not
be run during development (it needs a live authenticated session). The first
real run is the first-milestone notebook; treat any error there as expected
feedback, not as a surprise.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np

from uhrm.backends.base import ModisComposite
from uhrm.regions import Grid

LST_BAND = "LST_Day_1km"
QC_BAND = "QC_Day"


class EarthEngineModis:
    def __init__(self, collection: str = "MODIS/061/MOD11A2"):
        import ee  # imported here so offline machines never need earthengine-api

        self._ee = ee
        self.collection = collection

    def _images(self):
        return self._ee.ImageCollection(self.collection)

    def list_composites(self, start: date, end: date) -> list[date]:
        times = (self._images()
                 .filterDate(start.isoformat(), end.isoformat())
                 .aggregate_array("system:time_start")
                 .getInfo())
        return sorted({datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date() for ms in times})

    def fetch_composite(self, composite_date: date, grid: Grid) -> ModisComposite:
        ee = self._ee
        day = self._images().filterDate(composite_date.isoformat(),
                                        (composite_date + timedelta(days=1)).isoformat())
        if day.size().getInfo() == 0:
            raise LookupError(f"no MOD11A2 composite starts on {composite_date}")

        request = {
            "expression": day.first().select([LST_BAND, QC_BAND]),
            "fileFormat": "NUMPY_NDARRAY",
            "grid": {
                "dimensions": {"width": grid.width, "height": grid.height},
                "affineTransform": {"scaleX": grid.res, "shearX": 0, "translateX": grid.xmin,
                                    "shearY": 0, "scaleY": -grid.res, "translateY": grid.ymax},
                "crsCode": grid.crs,
            },
        }
        pixels = ee.data.computePixels(request)

        # Masked pixels come back as 0 (fill); the quality mask in uhrm.modis
        # handles that through the DN valid range.
        lst = np.nan_to_num(np.asarray(pixels[LST_BAND], dtype=np.float64)).astype(np.uint16)
        qc = np.nan_to_num(np.asarray(pixels[QC_BAND], dtype=np.float64)).astype(np.uint16)
        if lst.shape != grid.shape:
            raise RuntimeError(f"Earth Engine returned {lst.shape}, expected {grid.shape}")

        return ModisComposite(date=composite_date, lst_dn=lst, qc=qc,
                              source=f"earthengine:{self.collection}",
                              meta={"collection": self.collection})
