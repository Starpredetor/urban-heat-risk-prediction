"""Phase 1 step 7: one projection and spatial grid for all later processing (SRS FR-1.7).

A Grid is a north-up raster: cell (row, col) covers
    x in [xmin + col*res, xmin + (col+1)*res),  y in (ymax - (row+1)*res, ymax - row*res].
Grids are snapped to whole multiples of their resolution, so the MODIS (1 km) and
Landsat (30 m) grids are reproducible from the boundaries alone.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import shapely


@dataclass(frozen=True)
class Grid:
    crs: str
    xmin: float
    ymax: float
    res: float
    width: int
    height: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(xmin, ymin, xmax, ymax)"""
        return (self.xmin, self.ymax - self.height * self.res, self.xmin + self.width * self.res, self.ymax)

    @property
    def transform(self) -> tuple[float, float, float, float, float, float]:
        """Affine (a, b, c, d, e, f) as used by rasterio."""
        return (self.res, 0.0, self.xmin, 0.0, -self.res, self.ymax)

    @classmethod
    def covering(cls, geom, *, res: float, margin_m: float, crs: str) -> "Grid":
        """Smallest snapped grid covering `geom` (projected, metres) plus a margin."""
        minx, miny, maxx, maxy = geom.bounds
        xmin = math.floor((minx - margin_m) / res) * res
        xmax = math.ceil((maxx + margin_m) / res) * res
        ymin = math.floor((miny - margin_m) / res) * res
        ymax = math.ceil((maxy + margin_m) / res) * res
        return cls(crs=crs, xmin=float(xmin), ymax=float(ymax), res=float(res),
                   width=int(round((xmax - xmin) / res)), height=int(round((ymax - ymin) / res)))

    def overlap_fraction(self, geom) -> np.ndarray:
        """Fraction (0..1) of each cell's area inside `geom`, shape (height, width)."""
        x0 = self.xmin + self.res * np.arange(self.width)
        top = self.ymax - self.res * np.arange(self.height)
        X, T = np.meshgrid(x0, top)
        boxes = shapely.box(X, T - self.res, X + self.res, T)
        frac = np.zeros(boxes.shape, dtype=np.float32)
        hit = shapely.intersects(boxes, geom)
        if hit.any():
            frac[hit] = shapely.area(shapely.intersection(boxes[hit], geom)) / self.res ** 2
        return frac

    def cell_mask(self, geom, min_overlap: float = 0.5) -> np.ndarray:
        """Cells at least `min_overlap` inside `geom` (used to clip to a city)."""
        if not 0.0 < min_overlap <= 1.0:
            raise ValueError("min_overlap must be in (0, 1]")
        return self.overlap_fraction(geom) >= min_overlap


def save_grids(grids: dict[str, Grid], path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: asdict(g) for k, g in grids.items()}, indent=2), encoding="utf-8")
    return path


def load_grids(path) -> dict[str, Grid]:
    return {k: Grid(**v) for k, v in json.loads(Path(path).read_text(encoding="utf-8")).items()}
