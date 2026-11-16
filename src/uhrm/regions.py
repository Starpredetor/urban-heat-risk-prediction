"""Study areas, the model grid and city masks (SRS Phase 1).

At 1 km a city is only hundreds of cells, so the model reads a buffered
"model domain" (FR-P1.4). The city polygons stay the masks for analysis and
for every metric. A cell belongs to a city when at least `min_overlap` of its
area lies inside the polygon (FR-P1.5).

Geometries here are shapely objects in the grid's projected CRS (metres),
except where a function says otherwise.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import shape


@dataclass(frozen=True)
class Grid:
    """A north-up raster grid: cell (row, col) covers
    x in [xmin + col*res, xmin + (col+1)*res), y in (ymax - (row+1)*res, ymax - row*res]."""

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
        return (self.xmin, self.ymax - self.height * self.res,
                self.xmin + self.width * self.res, self.ymax)

    @property
    def transform(self) -> tuple[float, float, float, float, float, float]:
        """Affine coefficients (a, b, c, d, e, f) as used by rasterio."""
        return (self.res, 0.0, self.xmin, 0.0, -self.res, self.ymax)

    @classmethod
    def covering(cls, geoms, *, res: float, buffer_m: float, crs: str) -> "Grid":
        """Smallest grid, snapped to multiples of `res`, covering all geoms plus a buffer."""
        minx, miny, maxx, maxy = shapely.union_all(list(geoms)).bounds
        xmin = math.floor((minx - buffer_m) / res) * res
        xmax = math.ceil((maxx + buffer_m) / res) * res
        ymin = math.floor((miny - buffer_m) / res) * res
        ymax = math.ceil((maxy + buffer_m) / res) * res
        return cls(crs=crs, xmin=float(xmin), ymax=float(ymax), res=float(res),
                   width=int(round((xmax - xmin) / res)),
                   height=int(round((ymax - ymin) / res)))

    def cell_boxes(self) -> np.ndarray:
        """Square polygon of every cell, shape (height, width)."""
        cols = self.xmin + self.res * np.arange(self.width)
        tops = self.ymax - self.res * np.arange(self.height)
        x0, top = np.meshgrid(cols, tops)
        return shapely.box(x0, top - self.res, x0 + self.res, top)

    def overlap_fraction(self, geom) -> np.ndarray:
        """Fraction (0..1) of each cell's area inside `geom`; float32 (height, width)."""
        boxes = self.cell_boxes()
        frac = np.zeros(boxes.shape, dtype=np.float32)
        hit = shapely.intersects(boxes, geom)
        if hit.any():
            inter = shapely.intersection(boxes[hit], geom)
            frac[hit] = (shapely.area(inter) / (self.res * self.res)).astype(np.float32)
        return frac

    def cell_mask(self, geom, min_overlap: float = 0.5) -> np.ndarray:
        """True for cells at least `min_overlap` inside `geom`."""
        if not 0.0 < min_overlap <= 1.0:
            raise ValueError("min_overlap must be in (0, 1]")
        return self.overlap_fraction(geom) >= min_overlap


def refined(grid: Grid, factor: int) -> Grid:
    """The same extent at `factor` times the resolution (cells nest exactly).

    Used for 500 m MODIS reflectance on the 1 km model grid: raw data is kept at
    its finer resolution and aggregated later with a validity mask.
    """
    if factor < 1:
        raise ValueError("factor must be >= 1")
    return Grid(crs=grid.crs, xmin=grid.xmin, ymax=grid.ymax, res=grid.res / factor,
                width=grid.width * factor, height=grid.height * factor)


def save_grid(grid: Grid, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(grid), indent=2), encoding="utf-8")
    return path


def load_grid(path) -> Grid:
    return Grid(**json.loads(Path(path).read_text(encoding="utf-8")))


def cell_report(grid: Grid, masks: dict[str, np.ndarray]) -> dict:
    """Cell counts per city (FR-P1.7), plus overlap between cities (should be 0)."""
    names = list(masks)
    overlap = {f"{a}&{b}": int((masks[a] & masks[b]).sum())
               for i, a in enumerate(names) for b in names[i + 1:]}
    return {
        "grid_shape": grid.shape,
        "grid_cells": int(grid.height * grid.width),
        "cell_size_m": grid.res,
        "cells_per_city": {n: int(m.sum()) for n, m in masks.items()},
        "overlap_between_cities": overlap,
    }


# --- geometry helpers --------------------------------------------------------

def _polygonal(geom):
    """Keep only polygon parts (make_valid can return a GeometryCollection)."""
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    parts = [g for g in getattr(geom, "geoms", []) if g.geom_type in ("Polygon", "MultiPolygon")]
    if not parts:
        raise ValueError("geometry has no polygon part")
    return shapely.union_all(parts)


def repair(geom):
    """Make a polygon valid (FR-P1.3)."""
    return _polygonal(shapely.make_valid(geom))


def project_geometry(geom, src_crs: str, dst_crs: str):
    """Reproject a shapely geometry. Needs pyproj."""
    from pyproj import Transformer
    t = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

    def fn(coords):
        x, y = t.transform(coords[:, 0], coords[:, 1])
        return np.column_stack([x, y])

    return shapely.transform(geom, fn)


def load_boundary(path, crs: str):
    """Read a GeoJSON (WGS84), repair it, and return it in `crs` (metres)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("type") == "FeatureCollection":
        geoms = [shape(f["geometry"]) for f in data["features"]]
    elif data.get("type") == "Feature":
        geoms = [shape(data["geometry"])]
    else:
        geoms = [shape(data)]
    geom = repair(shapely.union_all(geoms))
    return project_geometry(geom, "EPSG:4326", crs)


def approx_area_km2(geom_lonlat) -> float:
    """Rough area of a lon/lat geometry (within about 1%), to sanity-check boundaries."""
    lat = geom_lonlat.centroid.y
    kx, ky = 111.32 * math.cos(math.radians(lat)), 110.57

    def fn(coords):
        return np.column_stack([coords[:, 0] * kx, coords[:, 1] * ky])

    return float(shapely.transform(geom_lonlat, fn).area)
