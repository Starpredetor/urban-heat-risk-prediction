"""Phase 2: build the free dataset collection (SRS FR-2.1 to FR-2.8).

Plan step                                                     Function
1-2 MODIS MOD11A2 v6.1 daytime LST + QC, both cities           download_modis
3   Landsat 8/9 imagery for selected years                     download_landsat
4   WorldCover tiles covering both regions                     download_worldcover
5   ERA5-Land weather for the same region and period           download_era5
6   original downloads kept apart from processed files         everything is written under data/raw/
7   CSV catalogue: dataset, date, city, source, file path      catalog_add -> data/catalog.csv
    verify the latest complete observations                    verify_end_year

Downloads are untouched source values (digital numbers and QC/QA bands). Scaling,
quality masking and clipping are Phase 3. Every download is resumable: an existing
file is skipped, and a failed request is retried before being reported.
"""

from __future__ import annotations

import csv
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from urban_heat import config
from urban_heat.grid import Grid

CATALOG_COLUMNS = ["dataset", "date", "city", "source", "file_path"]
DEFAULT_WORKERS = 32          # Earth Engine allows roughly 40 concurrent requests per project
NO_DATA_MARKER = -9999.0      # stands in for masked pixels in physical-value requests
_catalog_lock = threading.Lock()
BOTH_CITIES = "mumbai;navi_mumbai"
COMPOSITES_PER_YEAR = 46                       # MOD11A2 starts on day 1, 9, ..., 361


# --- Earth Engine helpers ------------------------------------------------------------

def init_ee(project: str = config.EE_PROJECT):
    import ee
    ee.Initialize(project=project)
    return ee


def _grid_request(image, grid: Grid) -> dict:
    return {"expression": image, "fileFormat": "NUMPY_NDARRAY",
            "grid": {"dimensions": {"width": grid.width, "height": grid.height},
                     "affineTransform": {"scaleX": grid.res, "shearX": 0, "translateX": grid.xmin,
                                         "shearY": 0, "scaleY": -grid.res, "translateY": grid.ymax},
                     "crsCode": grid.crs}}


def compute_pixels(ee, image, grid: Grid, bands, *, keep_nan: bool = False) -> dict[str, np.ndarray]:
    """`bands` of `image` sampled on `grid` (nearest neighbour).
    Masked pixels become 0 (the products' own no-data value for integer rasters),
    or stay NaN with keep_nan=True (for physical values where 0 is meaningful)."""
    image = image.select(list(bands))
    if keep_nan:
        # Earth Engine returns masked pixels as 0 in NumPy output, so fill them with
        # an impossible marker first and turn that into NaN here.
        arr = ee.data.computePixels(_grid_request(image.unmask(NO_DATA_MARKER), grid))
        return {b: np.where(np.asarray(arr[b], dtype=np.float64) == NO_DATA_MARKER, np.nan,
                            np.asarray(arr[b], dtype=np.float64)) for b in bands}
    arr = ee.data.computePixels(_grid_request(image, grid))
    return {b: np.nan_to_num(np.asarray(arr[b], dtype=np.float64)) for b in bands}


def subgrids(grid: Grid, size: int):
    """(row0, col0, sub_grid) tiles of at most size x size cells."""
    for r in range(0, grid.height, size):
        for c in range(0, grid.width, size):
            yield r, c, Grid(grid.crs, grid.xmin + c * grid.res, grid.ymax - r * grid.res, grid.res,
                             min(size, grid.width - c), min(size, grid.height - r))


def compute_pixels_tiled(ee, image, grid: Grid, bands, size: int) -> dict[str, np.ndarray]:
    """compute_pixels in tiles, for grids too large for one request."""
    out = {b: np.zeros(grid.shape) for b in bands}
    for r, c, sub in subgrids(grid, size):
        part = compute_pixels(ee, image, sub, bands)
        for b in bands:
            out[b][r:r + sub.height, c:c + sub.width] = part[b]
    return out


# --- files and catalogue -------------------------------------------------------------------

def write_tif(path, bands: dict[str, np.ndarray], grid: Grid, dtype: str, tags: dict) -> Path:
    import rasterio
    from rasterio import Affine

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial.tif")           # write then rename: no half-written files
    with rasterio.open(tmp, "w", driver="GTiff", width=grid.width, height=grid.height,
                       count=len(bands), dtype=dtype, crs=grid.crs, transform=Affine(*grid.transform),
                       compress="deflate") as dst:
        for i, (name, a) in enumerate(bands.items(), start=1):
            dst.write(np.asarray(a).astype(dtype), i)
            dst.set_band_description(i, name)
        dst.update_tags(**{k: str(v) for k, v in tags.items()})
    tmp.replace(path)
    return path


def catalog_add(dataset: str, when: str, source: str, file_path, catalog=None) -> None:
    """Append one row; a file already catalogued is not added twice. Thread-safe."""
    with _catalog_lock:
        _catalog_add(dataset, when, source, file_path, catalog)


def _catalog_add(dataset, when, source, file_path, catalog) -> None:
    catalog = Path(catalog or config.CATALOG)
    rel = Path(file_path).resolve().relative_to(config.ROOT).as_posix() \
        if Path(file_path).resolve().is_relative_to(config.ROOT) else str(file_path)
    rows = []
    if catalog.exists():
        with open(catalog, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if any(r["file_path"] == rel for r in rows):
            return
    catalog.parent.mkdir(parents=True, exist_ok=True)
    new = not catalog.exists()
    with open(catalog, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CATALOG_COLUMNS)
        if new:
            w.writeheader()
        w.writerow({"dataset": dataset, "date": when, "city": BOTH_CITIES, "source": source, "file_path": rel})


def with_retry(fn, *, retries: int = 3, base_delay: float = 2.0, sleep=time.sleep, log=print):
    """Call fn(); retry with exponential backoff; re-raise after the last attempt.
    LookupError means 'does not exist' and is never retried."""
    for attempt in range(retries):
        try:
            return fn()
        except LookupError:
            raise
        except Exception as e:
            if attempt + 1 == retries:
                raise
            log(f"  attempt {attempt + 1} failed ({e}); retrying")
            sleep(base_delay * 2 ** attempt)


def run_items(items, path_for, fetch_and_write, *, workers: int = DEFAULT_WORKERS,
              log=print, label="") -> dict:
    """Fetch+write every item whose file does not exist yet, `workers` at a time in
    parallel threads (each with retry). Returns {"done", "skipped", "missing", "failed"}."""
    summary = {"done": [], "skipped": [], "missing": [], "failed": []}
    items = list(items)
    todo = [it for it in items if not Path(path_for(it)).exists()]
    summary["skipped"] = [it for it in items if it not in todo]
    if not todo:
        return summary
    log(f"  {label}: {len(todo)} to download, {len(summary['skipped'])} already on disk, {workers} threads")
    started, finished = time.monotonic(), 0

    def one(item):
        try:
            with_retry(lambda: fetch_and_write(item), log=log)
            return item, "done"
        except LookupError:
            return item, "missing"
        except Exception as e:
            log(f"  {label} {item}: failed after retries: {e}")
            return item, "failed"

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for fut in as_completed([pool.submit(one, it) for it in todo]):
            item, outcome = fut.result()
            summary[outcome].append(item)
            finished += 1
            if finished % 50 == 0 or finished == len(todo):
                rate = finished / max(time.monotonic() - started, 1e-9)
                log(f"  {label}: {finished}/{len(todo)} ({rate:.1f}/s, ~{(len(todo) - finished) / rate:.0f} s left)")
    for k in ("done", "missing", "failed"):
        summary[k].sort(key=str)
    return summary


def brief(summary: dict) -> str:
    s = ", ".join(f"{k} {len(v)}" for k, v in summary.items())
    for k in ("missing", "failed"):
        if summary[k]:
            s += f"\n  {k}: {summary[k][:10]}{' ...' if len(summary[k]) > 10 else ''}"
    return s


# --- MODIS (steps 1-2) ------------------------------------------------------------------------

def modis_dates(start_year: int, end_year: int) -> list[date]:
    """Every MOD11A2 composite start date (day 1, 9, ..., 361 of each year)."""
    return [date(y, 1, 1) + timedelta(days=8 * i) for y in range(start_year, end_year + 1)
            for i in range(COMPOSITES_PER_YEAR)]


def modis_path(d: date) -> Path:
    return config.RAW_MODIS / f"MOD11A2_{d:%Y%m%d}.tif"


def verify_end_year(ee, year: int = config.END_YEAR) -> dict:
    """How many MOD11A2 composites the archive has for `year` (plan: verify the latest
    complete observations before fixing the final end date)."""
    times = (ee.ImageCollection(config.MODIS_COLLECTION)
             .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
             .aggregate_array("system:time_start").getInfo())
    found = sorted(datetime.fromtimestamp(t / 1000, timezone.utc).date() for t in times)
    return {"year": year, "composites": len(found), "expected": COMPOSITES_PER_YEAR,
            "complete": len(found) >= COMPOSITES_PER_YEAR,
            "last": found[-1].isoformat() if found else None}


def fetch_modis(ee, d: date, grid: Grid) -> dict[str, np.ndarray]:
    day = ee.ImageCollection(config.MODIS_COLLECTION).filterDate(d.isoformat(), (d + timedelta(days=1)).isoformat())
    if day.size().getInfo() == 0:
        raise LookupError(f"no MOD11A2 composite starts on {d}")
    return compute_pixels(ee, day.first(), grid, config.MODIS_BANDS)


def download_modis(ee, grid: Grid, dates, *, workers: int = DEFAULT_WORKERS, log=print) -> dict:
    def go(d):
        bands = fetch_modis(ee, d, grid)
        path = write_tif(modis_path(d), bands, grid, "uint16",
                         {"collection": config.MODIS_COLLECTION, "composite_start": d.isoformat(),
                          "note": "raw DN; LST scale 0.02 K, 0 = no data"})
        catalog_add("MOD11A2", d.isoformat(), f"NASA MODIS via Earth Engine ({config.MODIS_COLLECTION})", path)
    return run_items(dates, modis_path, go, workers=workers, log=log, label="MODIS")


# --- Landsat (step 3) ---------------------------------------------------------------------------

def landsat_scenes(ee, year: int, grid: Grid) -> list[tuple[str, str]]:
    """(collection, scene id) for every Landsat 8/9 scene over the grid in the selected
    months of `year` with scene cloud cover below the threshold."""
    x0, y0, x1, y1 = grid.bounds
    region = ee.Geometry.Rectangle([x0, y0, x1, y1], grid.crs, False)
    out = []
    for coll in config.LANDSAT_COLLECTIONS:
        c = (ee.ImageCollection(coll).filterBounds(region)
             .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
             .filter(ee.Filter.calendarRange(min(config.LANDSAT_MONTHS), max(config.LANDSAT_MONTHS), "month"))
             .filter(ee.Filter.lt("CLOUD_COVER", config.LANDSAT_MAX_SCENE_CLOUD)))
        out += [(coll, sid) for sid in c.aggregate_array("system:index").getInfo()]
    return sorted(out, key=lambda t: t[1][-8:])         # by acquisition date


def landsat_path(scene_id: str) -> Path:
    return config.RAW_LANDSAT / f"{scene_id}.tif"


def download_landsat(ee, grid: Grid, years, *, workers: int = DEFAULT_WORKERS, log=print) -> dict:
    """Each selected scene's reflectance, thermal and QA_PIXEL bands on the 30 m grid."""
    scenes = []
    for y in years:
        found = landsat_scenes(ee, y, grid)
        log(f"  Landsat {y}: {len(found)} scenes (months {config.LANDSAT_MONTHS}, cloud < {config.LANDSAT_MAX_SCENE_CLOUD}%)")
        scenes += found

    def go(item):
        coll, sid = item
        img = ee.Image(f"{coll}/{sid}")
        bands = compute_pixels_tiled(ee, img, grid, config.LANDSAT_BANDS, 600)
        acq = f"{sid[-8:-4]}-{sid[-4:-2]}-{sid[-2:]}"
        path = write_tif(landsat_path(sid), bands, grid, "uint16",
                         {"collection": coll, "scene": sid, "acquired": acq,
                          "note": "raw DN; SR = DN*0.0000275-0.2, ST_K = DN*0.00341802+149, 0 = no data"})
        catalog_add("Landsat C2 L2", acq, f"USGS Landsat via Earth Engine ({coll})", path)
    return run_items(scenes, lambda it: landsat_path(it[1]), go, workers=workers, log=log, label="Landsat")


# --- WorldCover (step 4) -------------------------------------------------------------------------

def worldcover_path(year: int) -> Path:
    return config.RAW_WORLDCOVER / f"worldcover_{year}_10m.tif"


def download_worldcover(ee, grid10: Grid, *, workers: int = DEFAULT_WORKERS, log=print) -> dict:
    """The 10 m WorldCover class map over the study extent for each release."""
    def go(year):
        img = ee.ImageCollection(config.WORLDCOVER[year]).first()
        bands = compute_pixels_tiled(ee, img, grid10, ["Map"], 1000)
        path = write_tif(worldcover_path(year), bands, grid10, "uint8",
                         {"collection": config.WORLDCOVER[year], "year": year, "note": "class codes 10-100"})
        catalog_add("ESA WorldCover", str(year), f"ESA WorldCover via Earth Engine ({config.WORLDCOVER[year]})", path)
    return run_items(sorted(config.WORLDCOVER), worldcover_path, go, workers=workers, log=log, label="WorldCover")


# --- ERA5-Land (step 5) -----------------------------------------------------------------------------

def era5_grid(lonlat_bounds) -> Grid:
    """ERA5-Land's own 0.1 deg lattice (cell edges at x.x5 deg) covering the bounds."""
    r = config.ERA5_RES_DEG
    x0, y0, x1, y1 = lonlat_bounds
    xmin = math.floor((x0 - r / 2) / r) * r + r / 2
    xmax = math.ceil((x1 - r / 2) / r) * r + r / 2
    ymin = math.floor((y0 - r / 2) / r) * r + r / 2
    ymax = math.ceil((y1 - r / 2) / r) * r + r / 2
    return Grid("EPSG:4326", round(xmin, 6), round(ymax, 6), r,
                int(round((xmax - xmin) / r)), int(round((ymax - ymin) / r)))


def era5_path(year: int) -> Path:
    return config.RAW_ERA5 / f"era5_land_daily_{year}.csv"


def download_era5(ee, grid_ll: Grid, years, *, workers: int = DEFAULT_WORKERS, log=print) -> dict:
    """Daily ERA5-Land values at every native cell over the study area, one CSV per year:
    date, lon, lat, then one column per variable (units as delivered: K, m/s)."""
    def go(year):
        col = (ee.ImageCollection(config.ERA5_COLLECTION)
               .filterDate(f"{year}-01-01", f"{year + 1}-01-01").select(list(config.ERA5_BANDS)))
        days = [datetime.fromtimestamp(t / 1000, timezone.utc).date()
                for t in col.aggregate_array("system:time_start").getInfo()]
        if not days:
            raise LookupError(f"no ERA5-Land days in {year}")
        # Earth Engine allows at most 1024 bands per request, and NumPy refuses
        # responses whose field list is very long, so request one variable and at
        # most 100 days at a time.
        px = {}
        for band in config.ERA5_BANDS:
            for i in range(0, len(days), 100):
                chunk = days[i:i + 100]
                names = [f"{d:%Y%m%d}_{band}" for d in chunk]
                sub = col.filterDate(chunk[0].isoformat(), (chunk[-1] + timedelta(days=1)).isoformat())
                stack = sub.select(band).toBands().rename(names)
                px.update(compute_pixels(ee, stack, grid_ll, names, keep_nan=True))
        rows = []
        for r in range(grid_ll.height):
            for c in range(grid_ll.width):
                lon = grid_ll.xmin + (c + 0.5) * grid_ll.res
                lat = grid_ll.ymax - (r + 0.5) * grid_ll.res
                for d in days:
                    vals = [px[f"{d:%Y%m%d}_{b}"][r, c] for b in config.ERA5_BANDS]
                    # ERA5-Land covers land only: sea cells have no value -> empty field
                    rows.append([d.isoformat(), round(lon, 3), round(lat, 3)] +
                                ["" if not np.isfinite(v) else v for v in vals])
        path = era5_path(year)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".partial.csv")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["date", "lon", "lat", *config.ERA5_BANDS])
            w.writerows(rows)
        tmp.replace(path)
        catalog_add("ERA5-Land daily", str(year), f"Copernicus ERA5-Land via Earth Engine ({config.ERA5_COLLECTION})", path)
    return run_items(years, era5_path, go, workers=workers, log=log, label="ERA5-Land")
