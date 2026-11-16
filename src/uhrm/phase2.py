"""Phase 2: collect the free datasets (SRS FR-P2.1 to FR-P2.8, FR-X.4).

One call, `run_phase2`, fetches every Stage A dataset for the model domain,
writes raw GeoTIFFs under the storage root, catalogues them, and reports
expected-versus-found composites per year. Everything is resumable: finished
files are skipped, so re-running after a Colab disconnect continues.

Layout under the storage root:
    raw/modis/               MOD11A2_YYYYMMDD.tif     LST DN + QC_Day        (1 km)
    raw/modis_reflectance/   MOD09A1_YYYYMMDD.tif     reflectance DN + QA    (500 m)
    raw/era5/                ERA5L_YYYYMMDD.tif       8-day window means     (1 km grid)
    raw/worldcover/          worldcover_<year>_fractions.tif  class fractions (1 km)
    raw/terrain/             srtm_terrain.tif         elevation mean, std    (1 km)
    catalog.csv              one row per file, with checksum
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from uhrm import acquire, regions

ALL = ("modis_lst", "modis_reflectance", "era5_land", "worldcover", "terrain")


@dataclass
class Backends:
    """The data sources, as objects with the methods used below (FR-X.3)."""

    modis_lst: Any            # fetch_composite(date, grid) -> ModisComposite
    modis_reflectance: Any    # fetch_composite(date, grid) -> RasterSet
    era5: Any                 # fetch_window(start, end, grid) -> RasterSet
    worldcover: Any           # fetch_fractions(year, grid) -> RasterSet
    terrain: Any              # fetch(grid) -> RasterSet


def earthengine_backends(project) -> Backends:
    from uhrm.backends import earthengine as ee_backends

    d = project.datasets
    return Backends(
        modis_lst=ee_backends.EarthEngineModis(project.modis.collection),
        modis_reflectance=ee_backends.EarthEngineReflectance(d.reflectance.collection, d.reflectance.bands),
        era5=ee_backends.EarthEngineEra5Land(d.era5_land.collection, d.era5_land.bands, d.era5_land.hours_utc),
        worldcover=ee_backends.EarthEngineWorldCover(d.worldcover.versions, d.worldcover.classes),
        terrain=ee_backends.EarthEngineTerrain(d.terrain.asset),
    )


def run_phase2(project, grid, years, backends: Backends, *, which=ALL,
               write_lst=acquire.write_geotiff, write_bands=acquire.write_bands_geotiff,
               log=print, **loop_kw) -> dict:
    """Fetch the datasets in `which` for `years`. Returns {"summaries", "coverage"}.

    `loop_kw` goes to the acquisition loop (retries, base_delay, sleep, progress_every).
    """
    unknown = set(which) - set(ALL)
    if unknown:
        raise ValueError(f"unknown datasets {sorted(unknown)}; choose from {ALL}")
    years = sorted(set(years))
    lo, hi = project.modis.start_date.year, project.modis.end_date.year
    if not years or years[0] < lo or years[-1] > hi:
        raise ValueError(f"years must lie within {lo}-{hi}")

    catalog = project.path("catalog.csv")
    grid_fine = regions.refined(grid, project.datasets.reflectance.grid_refinement)
    dates = acquire.expected_composite_dates(years[0], years[-1])
    dates = [d for d in dates if d.year in set(years)]
    summaries: dict[str, dict] = {}
    coverage: dict[str, dict] = {}

    def run(name, **kw):
        summaries[name] = acquire.acquire_series(catalog_path=catalog, log=log, **kw, **loop_kw)
        log(f"[{name}] {acquire.brief(summaries[name])}")

    if "modis_lst" in which:
        out = project.path("raw", "modis")
        summaries["modis_lst"] = acquire.acquire_modis(
            backends.modis_lst, grid, dates, out, catalog, writer=write_lst, log=log, **loop_kw)
        log(f"[modis_lst] {acquire.brief(summaries['modis_lst'])}")
        coverage["modis_lst"] = acquire.coverage_report(dates, acquire.present_dates(out, "MOD11A2"))

    if "modis_reflectance" in which:
        out = project.path("raw", "modis_reflectance")
        run("modis_reflectance", dataset="MOD09A1", region="model_domain_500m", keys=dates,
            fetch=lambda d: backends.modis_reflectance.fetch_composite(d, grid_fine),
            path_for=lambda d: acquire.dated_path(out, "MOD09A1", d),
            writer=lambda p, rs: write_bands(p, rs, grid_fine, "int32"))
        coverage["modis_reflectance"] = acquire.coverage_report(dates, acquire.present_dates(out, "MOD09A1"))

    if "era5_land" in which:
        out = project.path("raw", "era5")
        run("era5_land", dataset="ERA5-Land", keys=dates,
            fetch=lambda d: backends.era5.fetch_window(*acquire.composite_window(d), grid),
            path_for=lambda d: acquire.dated_path(out, "ERA5L", d),
            writer=lambda p, rs: write_bands(p, rs, grid, "float32"))
        coverage["era5_land"] = acquire.coverage_report(dates, acquire.present_dates(out, "ERA5L"))

    if "worldcover" in which:
        out = project.path("raw", "worldcover")
        run("worldcover", dataset="WorldCover", keys=sorted(project.datasets.worldcover.versions),
            fetch=lambda y: backends.worldcover.fetch_fractions(y, grid),
            path_for=lambda y: out / f"worldcover_{y}_fractions.tif",
            writer=lambda p, rs: write_bands(p, rs, grid, "float32"))

    if "terrain" in which:
        out = project.path("raw", "terrain")
        run("terrain", dataset="SRTM", keys=["srtm"],
            fetch=lambda _: backends.terrain.fetch(grid),
            path_for=lambda _: out / "srtm_terrain.tif",
            writer=lambda p, rs: write_bands(p, rs, grid, "float32"))

    for name, rep in coverage.items():
        found = sum(r["found"] for r in rep.values())
        total = sum(r["expected"] for r in rep.values())
        gaps = {y: r["expected"] - r["found"] for y, r in rep.items() if r["found"] < r["expected"]}
        log(f"[{name}] on disk: {found}/{total} composites" + (f"; years with gaps: {gaps}" if gaps else ""))

    return {"summaries": summaries, "coverage": coverage}
