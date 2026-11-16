"""Earth Engine backends (Stage A, Colab): MODIS LST and reflectance, ERA5-Land,
WorldCover fractions and terrain.

Needs an authenticated Earth Engine session (see `uhrm.colab.bootstrap`).
Pixels are requested directly on our grid with `ee.data.computePixels`, so one
request returns one layer already aligned to the grid.

Resampling: Earth Engine reprojects with nearest neighbour by default, which is
right for QC bytes and never invents values. Smooth fields (ERA5-Land) ask for
bilinear. Area fractions (WorldCover) and terrain are averaged server-side with
`reduceResolution`, so each 1 km cell reflects all the finer pixels inside it.

NOTE: this module was written against the Earth Engine API but could not be run
during development (it needs a live authenticated session). Treat any error on
the first real run as expected feedback, not a surprise. Band names and asset ids
come from config/project.yaml; verify them there if a call fails.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np

from uhrm.backends.base import ModisComposite, RasterSet
from uhrm.regions import Grid


class _EarthEngine:
    def __init__(self):
        import ee  # imported here so offline machines never need earthengine-api

        self._ee = ee

    @staticmethod
    def _transform(grid: Grid) -> list[float]:
        return [grid.res, 0, grid.xmin, 0, -grid.res, grid.ymax]

    def _pixels(self, image, grid: Grid, bands: list[str]) -> dict[str, np.ndarray]:
        """Request `bands` of `image` on `grid`. Masked pixels come back as NaN or 0."""
        request = {
            "expression": image.select(bands),
            "fileFormat": "NUMPY_NDARRAY",
            "grid": {
                "dimensions": {"width": grid.width, "height": grid.height},
                "affineTransform": {"scaleX": grid.res, "shearX": 0, "translateX": grid.xmin,
                                    "shearY": 0, "scaleY": -grid.res, "translateY": grid.ymax},
                "crsCode": grid.crs,
            },
        }
        pixels = self._ee.data.computePixels(request)
        out = {b: np.asarray(pixels[b], dtype=np.float64) for b in bands}
        for b, arr in out.items():
            if arr.shape != grid.shape:
                raise RuntimeError(f"Earth Engine returned {arr.shape} for {b}, expected {grid.shape}")
        return out

    def _day(self, collection: str, d: date):
        """The (at most one) image of `collection` starting on `d`; LookupError if none."""
        day = self._ee.ImageCollection(collection).filterDate(
            d.isoformat(), (d + timedelta(days=1)).isoformat())
        if day.size().getInfo() == 0:
            raise LookupError(f"no image in {collection} starts on {d}")
        return day.first()

    def _list_dates(self, collection: str, start: date, end: date) -> list[date]:
        times = (self._ee.ImageCollection(collection)
                 .filterDate(start.isoformat(), end.isoformat())
                 .aggregate_array("system:time_start").getInfo())
        return sorted({datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date() for ms in times})


class EarthEngineModis(_EarthEngine):
    """MOD11A2 daytime LST and its QC byte, as raw digital numbers."""

    LST_BAND, QC_BAND = "LST_Day_1km", "QC_Day"

    def __init__(self, collection: str = "MODIS/061/MOD11A2"):
        super().__init__()
        self.collection = collection

    def list_composites(self, start: date, end: date) -> list[date]:
        return self._list_dates(self.collection, start, end)

    def fetch_composite(self, composite_date: date, grid: Grid) -> ModisComposite:
        img = self._day(self.collection, composite_date)
        px = self._pixels(img, grid, [self.LST_BAND, self.QC_BAND])
        # Masked pixels come back as 0 (fill); the quality mask in uhrm.modis
        # handles that through the DN valid range.
        return ModisComposite(
            date=composite_date,
            lst_dn=np.nan_to_num(px[self.LST_BAND]).astype(np.uint16),
            qc=np.nan_to_num(px[self.QC_BAND]).astype(np.uint16),
            source=f"earthengine:{self.collection}", meta={"collection": self.collection})


class EarthEngineReflectance(_EarthEngine):
    """MOD09A1 8-day surface reflectance plus StateQA, raw, on a 500 m grid.

    An extra `valid` band (1 where Earth Engine considers the pixel unmasked)
    is added so Phase 3 can tell "masked" from a genuine zero. Fill values may
    still pass through as large negative numbers if Earth Engine does not mask
    them; Phase 3 screens the value range as well.
    """

    def __init__(self, collection: str, bands: tuple[str, ...]):
        super().__init__()
        self.collection, self.bands = collection, list(bands)

    def fetch_composite(self, composite_date: date, grid: Grid) -> RasterSet:
        img = self._day(self.collection, composite_date)
        image = img.select(self.bands).addBands(img.select(self.bands[0]).mask().toFloat().rename("valid"))
        px = self._pixels(image, grid, self.bands + ["valid"])
        return RasterSet(bands=px, date=composite_date, source=f"earthengine:{self.collection}",
                         meta={"collection": self.collection})


class EarthEngineEra5Land(_EarthEngine):
    """ERA5-Land hourly, averaged over the given UTC hours inside a date window."""

    def __init__(self, collection: str, bands: tuple[str, ...], hours_utc: tuple[int, ...]):
        super().__init__()
        self.collection, self.bands, self.hours = collection, list(bands), tuple(hours_utc)

    def fetch_window(self, start: date, end: date, grid: Grid) -> RasterSet:
        ee = self._ee
        col = (ee.ImageCollection(self.collection)
               .filterDate(start.isoformat(), end.isoformat())
               .filter(ee.Filter.calendarRange(min(self.hours), max(self.hours), "hour"))
               .select(self.bands))
        n = col.size().getInfo()
        if n == 0:
            raise LookupError(f"no ERA5-Land hours for {start} to {end} (beyond the data's end?)")
        mean = col.map(lambda i: i.resample("bilinear")).mean()
        px = self._pixels(mean, grid, self.bands)
        return RasterSet(bands=px, date=start, source=f"earthengine:{self.collection}",
                         meta={"collection": self.collection, "window": f"{start}/{end}",
                               "hours_utc": ",".join(map(str, self.hours)), "n_hours": n})


class EarthEngineWorldCover(_EarthEngine):
    """Land-cover area fraction of each class inside each grid cell (static)."""

    def __init__(self, versions: dict[int, str], classes: tuple[int, ...]):
        super().__init__()
        self.versions, self.classes = dict(versions), tuple(classes)

    def fetch_fractions(self, year: int, grid: Grid) -> RasterSet:
        ee = self._ee
        if year not in self.versions:
            raise LookupError(f"no WorldCover release configured for {year}")
        asset = self.versions[year]
        land = ee.ImageCollection(asset).first().select("Map")
        stack = ee.Image.cat([land.eq(c).rename(f"c{c}") for c in self.classes]).toFloat()
        # Averaging 10 m straight to 1 km makes Earth Engine materialise the whole
        # domain at 10 m ("Reprojection output too large"). Two steps keep each
        # request small: 10 m -> 100 m (100 pixels per cell), then 100 m -> grid.
        # Equal-area steps, so the result equals the direct mean.
        step = 100.0
        mid = Grid(crs=grid.crs, xmin=grid.xmin, ymax=grid.ymax, res=step,
                   width=int(round(grid.width * grid.res / step)),
                   height=int(round(grid.height * grid.res / step)))
        frac = (stack.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=1024)
                .reproject(crs=grid.crs, crsTransform=self._transform(mid))
                .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=1024)
                .reproject(crs=grid.crs, crsTransform=self._transform(grid)))
        px = self._pixels(frac, grid, [f"c{c}" for c in self.classes])
        return RasterSet(bands=px, source=f"earthengine:{asset}",
                         meta={"collection": asset, "year": year, "grid_res_m": grid.res})


class EarthEngineTerrain(_EarthEngine):
    """Mean elevation and relief (standard deviation) per grid cell (static)."""

    def __init__(self, asset: str):
        super().__init__()
        self.asset = asset

    def fetch(self, grid: Grid) -> RasterSet:
        ee = self._ee
        elevation = ee.Image(self.asset).select("elevation")
        reducer = ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True)
        stats = elevation.reduceResolution(reducer=reducer, maxPixels=4096).reproject(
            crs=grid.crs, crsTransform=self._transform(grid))
        px = self._pixels(stats, grid, ["elevation_mean", "elevation_stdDev"])
        return RasterSet(bands=px, source=f"earthengine:{self.asset}",
                         meta={"collection": self.asset, "grid_res_m": grid.res})
