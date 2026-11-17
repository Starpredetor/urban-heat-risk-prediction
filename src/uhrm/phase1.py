"""Phase 1: study areas, grids, city masks and land/water flags (SRS FR-P1.1 to FR-P1.8).

Outputs under the storage root:
    boundaries/combined_study_area.geojson   union of both cities (FR-P1.3)
    grid.json                                1 km model grid (FR-P1.4, FR-P1.6)
    grid_30m.json                            30 m Landsat analysis grid (FR-P1.6)
    phase1/cells.tif                         city_id, water_fraction, coastal  -> open in QGIS
    phase1/report.json                       cell counts per city (FR-P1.7)

Land/water comes from the WorldCover permanent-water fraction per 1 km cell
(FR-P1.5). MODIS has no LST over open sea and mixed coastal cells are a blend
of land and water temperature, so both are flagged here.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import mapping, shape

from uhrm import acquire, regions
from uhrm.backends.base import RasterSet

WATER_BAND = "c80"           # WorldCover permanent water bodies
COASTAL_MIN_WATER = 0.1      # a land cell with more water than this is "coastal"
ANALYSIS_RES_M = 30          # Landsat analysis grid
ANALYSIS_BUFFER_M = 1000


@dataclass
class Phase1:
    grid: regions.Grid
    analysis_grid: regions.Grid
    geoms: dict                     # slug -> projected polygon
    masks: dict                     # slug -> bool (H, W)
    water: np.ndarray | None        # permanent-water fraction (H, W), or None if not yet fetched
    land: np.ndarray | None
    coastal: np.ndarray | None


def model_grid(project, geoms: dict) -> regions.Grid:
    return regions.Grid.covering(list(geoms.values()), res=project.grid.model_resolution_m,
                                 buffer_m=project.grid.domain_buffer_km * 1000, crs=project.grid.crs)


def coastal_cells(water, max_water_fraction: float) -> np.ndarray:
    """Land cells that contain some water or touch a water-dominated cell."""
    water = np.asarray(water, dtype=float)
    wet = water > max_water_fraction
    near = np.zeros_like(wet)
    near[1:, :] |= wet[:-1, :]
    near[:-1, :] |= wet[1:, :]
    near[:, 1:] |= wet[:, :-1]
    near[:, :-1] |= wet[:, 1:]
    return ~wet & ((water > COASTAL_MIN_WATER) | near)


def build(project, geoms: dict, water=None, grid: regions.Grid | None = None) -> Phase1:
    grid = grid or model_grid(project, geoms)
    masks = {s: grid.cell_mask(g, project.grid.min_cell_overlap) for s, g in geoms.items()}
    analysis = regions.Grid.covering(list(geoms.values()), res=ANALYSIS_RES_M,
                                     buffer_m=ANALYSIS_BUFFER_M, crs=project.grid.crs)
    land = coastal = None
    if water is not None:
        water = np.asarray(water, dtype=np.float32)
        if water.shape != grid.shape:
            raise ValueError(f"water fraction {water.shape} does not match grid {grid.shape}")
        land = water <= project.preprocess.max_water_fraction
        coastal = coastal_cells(water, project.preprocess.max_water_fraction)
    return Phase1(grid, analysis, geoms, masks, water, land, coastal)


def report(p1: Phase1) -> dict:
    rep = regions.cell_report(p1.grid, p1.masks)
    rep["analysis_grid_30m_shape"] = p1.analysis_grid.shape
    if p1.water is not None:
        rep["per_city"] = {s: {"cells": int(m.sum()),
                               "land_cells": int((m & p1.land).sum()),
                               "coastal_land_cells": int((m & p1.coastal).sum()),
                               "water_dominated_cells": int((m & ~p1.land).sum())}
                           for s, m in p1.masks.items()}
        rep["domain_land_fraction"] = round(float(p1.land.mean()), 3)
    else:
        rep["per_city"] = "land/water not available yet (WorldCover not fetched)"
    return rep


def combine_boundaries(paths, out_path) -> Path:
    """Union of the city boundaries in lon/lat (FR-P1.3)."""
    geoms = []
    for p in paths:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        geoms += [shape(f["geometry"]) for f in data["features"]]
    union = regions.repair(shapely.union_all(geoms))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"name": "combined_study_area"}, "geometry": mapping(union)}]}),
        encoding="utf-8")
    return out_path


def load_water_fraction(project) -> np.ndarray | None:
    """Permanent-water fraction per cell from the latest WorldCover file, if fetched."""
    from uhrm.preprocess import read_raster

    files = sorted(project.path("raw", "worldcover").glob("worldcover_*_fractions.tif"))
    if not files:
        return None
    return np.asarray(read_raster(files[-1]).get(WATER_BAND), dtype=np.float32)


def check_grid_unchanged(grid: regions.Grid, path) -> None:
    """Raw data on disk was fetched on the saved grid. If boundaries change the grid,
    those files no longer line up, so refuse rather than mix two grids silently."""
    path = Path(path)
    if path.exists():
        old = regions.load_grid(path)
        if old != grid:
            raise RuntimeError(
                f"the model grid changed ({old.shape} -> {grid.shape}, origin "
                f"{old.xmin, old.ymax} -> {grid.xmin, grid.ymax}). Raw files under raw/ were fetched "
                f"on the old grid. Move raw/ and processed/ aside, delete grid.json, then re-run.")


def save(p1: Phase1, project) -> dict:
    check_grid_unchanged(p1.grid, project.path("grid.json"))
    regions.save_grid(p1.grid, project.path("grid.json"))
    regions.save_grid(p1.analysis_grid, project.path("grid_30m.json"))
    combined = combine_boundaries([project.boundary_path(a) for a in project.study_areas],
                                  project.path("boundaries", "combined_study_area.geojson"))

    city_id = np.zeros(p1.grid.shape, dtype=np.float32)
    for i, m in enumerate(p1.masks.values(), start=1):
        city_id[m] = i
    bands = {"city_id": city_id}
    if p1.water is not None:
        bands["water_fraction"] = p1.water
        bands["coastal"] = p1.coastal.astype(np.float32)
    cells = project.path("phase1", "cells.tif")
    acquire.write_bands_geotiff(cells, RasterSet(bands=bands, source="uhrm.phase1",
                                                 meta={"city_ids": json.dumps({i: s for i, s in enumerate(p1.masks, 1)})}),
                                p1.grid)
    rep_path = project.path("phase1", "report.json")
    rep_path.write_text(json.dumps(report(p1), indent=2, default=str), encoding="utf-8")
    return {"grid": project.path("grid.json"), "grid_30m": project.path("grid_30m.json"),
            "combined": combined, "cells": cells, "report": rep_path,
            "grid_def": asdict(p1.grid)}
