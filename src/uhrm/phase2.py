"""Phase 2: build the free dataset collection (SRS v2.0 Phase 2, FR-P2.1 to FR-P2.11).

The spec, step by step, and where each step happens here:

  1-2  MODIS MOD11A2 v6.1 daytime LST + QC, both cities      -> modis_lst
  3    Landsat 8/9 imagery for selected years (2013-)         -> landsat
       (surface reflectance for NDVI/NDBI/NDWI + surface temperature)
  4    WorldCover tiles covering both regions (2020, 2021)    -> worldcover_map (10 m classes)
                                                                 worldcover     (1 km class fractions)
  5    ERA5-Land weather for the same region and period       -> era5_land
  6    raw files stored separately from processed files       -> everything here goes under raw/
  7    CSV catalogue: dataset, date, city, source, file path   -> catalog.csv (+ checksum)
  -    verify the latest complete observations                -> end-year check on MODIS

Two further datasets are kept because later phases need them (SRS FR-P2.5):
  modis_reflectance  MOD09A1 8-day 500 m: NDVI/NDBI/NDWI for 2001-2012, before Landsat 8
  terrain            SRTM elevation for the Phase 4 features

Layout under the storage root:
    raw/modis/               MOD11A2_YYYYMMDD.tif                 1 km model grid
    raw/modis_reflectance/   MOD09A1_YYYYMMDD.tif                 500 m
    raw/era5/                ERA5L_YYYYMMDD.tif                   1 km grid (~9 km native)
    raw/landsat/             landsat_<year>_m3-5.tif              30 m analysis grid
    raw/worldcover/          worldcover_<year>_map_10m.tif        10 m analysis extent
                             worldcover_<year>_fractions.tif      1 km model grid
    raw/terrain/             srtm_terrain.tif                     1 km model grid
    catalog.csv

Everything is resumable: finished files are skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from uhrm import acquire, regions

ALL = ("modis_lst", "landsat", "worldcover_map", "worldcover", "era5_land",
       "modis_reflectance", "terrain")


@dataclass
class Backends:
    """The data sources, as objects with the methods used below (FR-X.3)."""

    modis_lst: Any            # fetch_composite(date, grid) -> ModisComposite; list_composites(start, end)
    modis_reflectance: Any    # fetch_composite(date, grid) -> RasterSet
    era5: Any                 # fetch_window(start, end, grid) -> RasterSet
    worldcover: Any           # fetch_fractions(year, grid), fetch_map(year, grid) -> RasterSet
    terrain: Any              # fetch(grid) -> RasterSet
    landsat: Any = None       # fetch_composite(year, grid) -> RasterSet


def earthengine_backends(project) -> Backends:
    from uhrm.backends import earthengine as ee_backends

    d = project.datasets
    ls = d.landsat
    return Backends(
        modis_lst=ee_backends.EarthEngineModis(project.modis.collection),
        modis_reflectance=ee_backends.EarthEngineReflectance(d.reflectance.collection, d.reflectance.bands),
        era5=ee_backends.EarthEngineEra5Land(d.era5_land.collection, d.era5_land.bands, d.era5_land.hours_utc),
        worldcover=ee_backends.EarthEngineWorldCover(d.worldcover.versions, d.worldcover.classes),
        terrain=ee_backends.EarthEngineTerrain(d.terrain.asset),
        landsat=ee_backends.EarthEngineLandsat(ls.collections, ls.bands, ls.months, ls.max_scene_cloud),
    )


def verify_end_year(backend, year: int) -> dict:
    """How many MOD11A2 composites exist for `year` (the spec's 'verify the latest
    complete observations before fixing the final end date')."""
    found = backend.list_composites(date(year, 1, 1), date(year + 1, 1, 1))
    return {"year": year, "composites_available": len(found),
            "expected": acquire.COMPOSITES_PER_YEAR,
            "complete": len(found) >= acquire.COMPOSITES_PER_YEAR,
            "last_available": found[-1].isoformat() if found else None}


def run_phase2(project, grid, years, backends: Backends, *, which=ALL, analysis_grid=None,
               write_lst=acquire.write_geotiff, write_bands=acquire.write_bands_geotiff,
               log=print, **loop_kw) -> dict:
    """Fetch the datasets in `which` for `years`. Returns {"summaries", "coverage", "end_check"}.

    `analysis_grid` is Phase 1's 30 m grid (loaded from grid_30m.json if not given);
    it is needed only for `landsat` and `worldcover_map`.
    `loop_kw` goes to the acquisition loop (retries, base_delay, sleep, progress_every).
    """
    unknown = set(which) - set(ALL)
    if unknown:
        raise ValueError(f"unknown datasets {sorted(unknown)}; choose from {ALL}")
    years = sorted(set(years))
    lo, hi = project.modis.start_date.year, project.modis.end_date.year
    if not years or years[0] < lo or years[-1] > hi:
        raise ValueError(f"years must lie within {lo}-{hi}")

    if {"landsat", "worldcover_map"} & set(which) and analysis_grid is None:
        path = project.path("grid_30m.json")
        if not path.exists():
            raise FileNotFoundError("grid_30m.json not found: run Phase 1 (the grid step) first")
        analysis_grid = regions.load_grid(path)

    catalog = project.path("catalog.csv")
    raw = project.path("raw")
    dates = [d for d in acquire.expected_composite_dates(years[0], years[-1]) if d.year in set(years)]
    summaries: dict[str, dict] = {}
    coverage: dict[str, dict] = {}
    end_check = None

    def run(name, **kw):
        summaries[name] = acquire.acquire_series(catalog_path=catalog, log=log, **kw, **loop_kw)
        log(f"[{name}] {acquire.brief(summaries[name])}")

    if "modis_lst" in which:
        if hasattr(backends.modis_lst, "list_composites"):
            end_check = verify_end_year(backends.modis_lst, hi)
            log(f"[end check] {end_check}")
        out = raw / "modis"
        summaries["modis_lst"] = acquire.acquire_modis(
            backends.modis_lst, grid, dates, out, catalog, writer=write_lst, log=log, **loop_kw)
        log(f"[modis_lst] {acquire.brief(summaries['modis_lst'])}")
        coverage["modis_lst"] = acquire.coverage_report(dates, acquire.present_dates(out, "MOD11A2"))

    if "landsat" in which:
        ls = project.datasets.landsat
        tag = f"m{min(ls.months)}-{max(ls.months)}"
        ls_years = ls.years_within(years)
        if not ls_years:
            log(f"[landsat] no selected Landsat years within {years[0]}-{years[-1]} (starts {ls.start_year})")
        run("landsat", dataset="Landsat8/9 C2L2", region="analysis_grid_30m", keys=ls_years,
            fetch=lambda y: backends.landsat.fetch_composite(y, analysis_grid),
            path_for=lambda y: raw / "landsat" / f"landsat_{y}_{tag}.tif",
            writer=lambda p, rs: write_bands(p, rs, analysis_grid, "float32"))

    wc_years = sorted(project.datasets.worldcover.versions)
    if "worldcover_map" in which:
        grid10 = regions.refined(analysis_grid, int(round(analysis_grid.res / 10)))
        run("worldcover_map", dataset="WorldCover map", region="analysis_extent_10m", keys=wc_years,
            fetch=lambda y: backends.worldcover.fetch_map(y, grid10),
            path_for=lambda y: raw / "worldcover" / f"worldcover_{y}_map_10m.tif",
            writer=lambda p, rs: write_bands(p, rs, grid10, "uint8"))

    if "worldcover" in which:
        run("worldcover", dataset="WorldCover fractions", keys=wc_years,
            fetch=lambda y: backends.worldcover.fetch_fractions(y, grid),
            path_for=lambda y: raw / "worldcover" / f"worldcover_{y}_fractions.tif",
            writer=lambda p, rs: write_bands(p, rs, grid, "float32"))

    if "era5_land" in which:
        out = raw / "era5"
        run("era5_land", dataset="ERA5-Land", keys=dates,
            fetch=lambda d: backends.era5.fetch_window(*acquire.composite_window(d), grid),
            path_for=lambda d: acquire.dated_path(out, "ERA5L", d),
            writer=lambda p, rs: write_bands(p, rs, grid, "float32"))
        coverage["era5_land"] = acquire.coverage_report(dates, acquire.present_dates(out, "ERA5L"))

    if "modis_reflectance" in which:
        grid_fine = regions.refined(grid, project.datasets.reflectance.grid_refinement)
        out = raw / "modis_reflectance"
        run("modis_reflectance", dataset="MOD09A1", region="model_domain_500m", keys=dates,
            fetch=lambda d: backends.modis_reflectance.fetch_composite(d, grid_fine),
            path_for=lambda d: acquire.dated_path(out, "MOD09A1", d),
            writer=lambda p, rs: write_bands(p, rs, grid_fine, "int32"))
        coverage["modis_reflectance"] = acquire.coverage_report(dates, acquire.present_dates(out, "MOD09A1"))

    if "terrain" in which:
        run("terrain", dataset="SRTM", keys=["srtm"],
            fetch=lambda _: backends.terrain.fetch(grid),
            path_for=lambda _: raw / "terrain" / "srtm_terrain.tif",
            writer=lambda p, rs: write_bands(p, rs, grid, "float32"))

    for name, rep in coverage.items():
        found = sum(r["found"] for r in rep.values())
        total = sum(r["expected"] for r in rep.values())
        gaps = {y: r["expected"] - r["found"] for y, r in rep.items() if r["found"] < r["expected"]}
        log(f"[{name}] on disk: {found}/{total} composites" + (f"; years with gaps: {gaps}" if gaps else ""))

    return {"summaries": summaries, "coverage": coverage, "end_check": end_check}
