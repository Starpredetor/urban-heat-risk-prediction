"""Phase 3: preprocess all satellite data (SRS FR-3.1 to FR-3.9).

Plan step                                              Where
1 apply the MODIS quality-control mask                 process_modis (modis_qc.quality_mask)
2 remove invalid and poor-quality observations         process_modis, landsat_clear
3 apply the product-specific scale factors             modis_qc, landsat_scale
4 convert temperature into Celsius                     MODIS LST, Landsat ST, ERA5-Land
5 clip all imagery to the two city boundaries          city masks; values outside both cities = no data
6 reproject to a common coordinate system              everything in EPSG:32643 (ERA5-Land from EPSG:4326)
7 resample compatible layers to a common grid          MODIS/ERA5 -> 1 km grid; Landsat/WorldCover -> 30 m grid
8 handle missing observations carefully                nothing is gap-filled; NaN = no valid observation;
                                                       per-date valid fractions in modis_valid_fraction.csv
9 save processed images with consistent dates          data/processed/<dataset>/..._YYYYMMDD.tif

Landsat Collection 2 Level-2 (official scale factors and QA_PIXEL flags):
    SR = DN * 0.0000275 - 0.2   (valid DN 7273-43636)
    ST_K = DN * 0.00341802 + 149; ST_C = ST_K - 273.15   (DN 0 = no data)
    QA_PIXEL bits rejected: 0 fill, 1 dilated cloud, 2 cirrus, 3 cloud, 4 cloud shadow
"""

from __future__ import annotations

import csv
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np

from urban_heat import acquisition as acq, boundaries as b, config, modis_qc
from urban_heat.grid import Grid
from urban_heat.logs import progress

log = logging.getLogger("phase3")

P_MODIS, P_LANDSAT, P_WORLDCOVER, P_ERA5, P_MASKS = (
    config.PROCESSED / n for n in ("modis", "landsat", "worldcover", "era5", "masks"))

LANDSAT_SR_BANDS = {"blue": "SR_B2", "green": "SR_B3", "red": "SR_B4", "nir": "SR_B5",
                    "swir1": "SR_B6", "swir2": "SR_B7"}
SR_VALID_DN = (7273, 43636)
QA_REJECT_BITS = 0b11111                  # bits 0-4
ERA5_KELVIN = ("temperature_2m", "dewpoint_temperature_2m", "skin_temperature")
CITY_IDS = {"mumbai": 1, "navi_mumbai": 2}


# --- step 5: city masks -----------------------------------------------------------------

def city_geoms() -> dict:
    return {k: b.to_crs(b.load(config.BOUNDARIES / f"{k}.geojson"), b.WGS84, config.CRS) for k in config.CITIES}


def city_id_raster(grid: Grid, geoms: dict) -> np.ndarray:
    """0 outside both cities, 1 Mumbai, 2 Navi Mumbai (cell at least half inside)."""
    ids = np.zeros(grid.shape, dtype=np.uint8)
    for k, g in geoms.items():
        ids[grid.cell_mask(g)] = CITY_IDS[k]
    return ids


def _read_band(path, name: str) -> np.ndarray:
    import rasterio
    with rasterio.open(path) as src:
        return src.read(list(src.descriptions).index(name) + 1)


def read(path) -> dict[str, np.ndarray]:
    import rasterio
    with rasterio.open(path) as src:
        return {(src.descriptions[i - 1] or f"b{i}"): src.read(i) for i in range(1, src.count + 1)}


# --- MODIS ------------------------------------------------------------------------------

def modis_out(d: date) -> Path:
    return P_MODIS / f"MOD11A2_LST_C_{d:%Y%m%d}.tif"


def process_modis(raw_path, d: date, grid: Grid, ids: np.ndarray) -> tuple[Path, np.ndarray]:
    bands = read(raw_path)
    lst = modis_qc.lst_celsius(bands["LST_Day_1km"], bands["QC_Day"])
    lst[ids == 0] = np.nan                                     # clip to the two cities
    path = acq.write_tif(modis_out(d), {"lst_celsius": lst}, grid, "float32",
                         {"composite_start": d, "qc": "LST produced (good/other quality), error <= 2 K",
                          "clipped_to": "mumbai, navi_mumbai", "nodata": "NaN"})
    return path, lst


def valid_fractions(d: date, lst: np.ndarray | None, ids: np.ndarray) -> list[dict]:
    rows = []
    for k, i in CITY_IDS.items():
        cells = int((ids == i).sum())
        valid = int(np.isfinite(lst[ids == i]).sum()) if lst is not None else 0
        rows.append({"date": d.isoformat(), "city": k, "cells": cells, "valid_cells": valid,
                     "valid_fraction": round(valid / cells, 4) if cells else 0.0,
                     "raw_file": "present" if lst is not None else "absent in archive"})
    return rows


def run_modis(grid: Grid, ids: np.ndarray, *, workers: int = 8) -> Path:
    dates = acq.modis_dates(config.START_YEAR, config.END_YEAR)
    t0 = time.monotonic()
    log.info("MODIS: %d composites, CPU (%d threads; the 1 km grid is too small to gain from the GPU)", len(dates), workers)

    def one(d):
        raw = acq.modis_path(d)
        if not raw.exists():
            return valid_fractions(d, None, ids)
        if modis_out(d).exists():
            lst = read(modis_out(d))["lst_celsius"]
        else:
            _, lst = process_modis(raw, d, grid, ids)
        return valid_fractions(d, lst, ids)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = [r for rs in progress(pool.map(one, dates), total=len(dates), desc="MODIS QC + scale + clip",
                                     unit="composite") for r in rs]
    out = config.PROCESSED / "modis_valid_fraction.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    absent = sorted({r['date'] for r in rows if r['raw_file'] != 'present'})
    for city in CITY_IDS:
        v = [r['valid_fraction'] for r in rows if r['city'] == city and r['raw_file'] == 'present']
        log.info("MODIS %s: mean valid fraction %.2f over %d composites", city, float(np.mean(v)), len(v))
    log.info("MODIS: done in %.0f s; absent in archive: %s; summary %s", time.monotonic() - t0, absent,
             out.relative_to(config.ROOT))
    return out


# --- Landsat ----------------------------------------------------------------------------

def landsat_clear(qa) -> np.ndarray:
    """True where QA_PIXEL bits 0-4 (fill, dilated cloud, cirrus, cloud, shadow) are all 0."""
    return (np.asarray(qa).astype(np.int64) & QA_REJECT_BITS) == 0


def landsat_scale(bands: dict, inside: np.ndarray, device=None) -> dict[str, np.ndarray]:
    """Scaled, quality-masked, clipped Landsat layers: st_celsius plus six reflectances.
    Runs on `device` (a torch CUDA device) when given, otherwise with NumPy on the CPU;
    both give the same numbers (float64 arithmetic, float32 output)."""
    if device is not None and getattr(device, "type", "cpu") == "cuda":
        return _landsat_scale_torch(bands, inside, device)
    ok = landsat_clear(bands["QA_PIXEL"]) & inside
    st = np.asarray(bands["ST_B10"], dtype=np.float64)
    out = {"st_celsius": np.where(ok & (st > 0), st * 0.00341802 + 149.0 - 273.15, np.nan)}
    lo, hi = SR_VALID_DN
    for name, band in LANDSAT_SR_BANDS.items():
        dn = np.asarray(bands[band], dtype=np.float64)
        out[name] = np.where(ok & (dn >= lo) & (dn <= hi), dn * 0.0000275 - 0.2, np.nan)
    return {k: v.astype(np.float32) for k, v in out.items()}


def _landsat_scale_torch(bands: dict, inside: np.ndarray, device) -> dict[str, np.ndarray]:
    import torch

    def gpu(a, dtype):
        return torch.from_numpy(np.ascontiguousarray(a)).to(device=device, dtype=dtype, non_blocking=True)

    nan = torch.tensor(float("nan"), dtype=torch.float64, device=device)
    ok = ((gpu(bands["QA_PIXEL"], torch.int32) & QA_REJECT_BITS) == 0) & gpu(inside, torch.bool)
    st = gpu(bands["ST_B10"], torch.float64)
    out = {"st_celsius": torch.where(ok & (st > 0), st * 0.00341802 + 149.0 - 273.15, nan)}
    lo, hi = SR_VALID_DN
    for name, band in LANDSAT_SR_BANDS.items():
        dn = gpu(bands[band], torch.float64)
        out[name] = torch.where(ok & (dn >= lo) & (dn <= hi), dn * 0.0000275 - 0.2, nan)
    return {k: v.to(torch.float32).cpu().numpy() for k, v in out.items()}


def landsat_out(scene: str) -> Path:
    return P_LANDSAT / f"{scene}_processed_{scene[-8:]}.tif"


def run_landsat(grid: Grid, ids: np.ndarray, *, workers: int = 4, device=None) -> list[dict]:
    inside = ids > 0
    scenes = sorted(config.RAW_LANDSAT.glob("L*.tif"))
    t0 = time.monotonic()
    log.info("Landsat: %d scenes, arithmetic on %s, %d I/O threads", len(scenes),
             getattr(device, "type", "cpu").upper(), workers)

    def one(raw):
        scene = raw.stem
        out = landsat_out(scene)
        if not out.exists():
            t = time.monotonic()
            bands = read(raw)
            t_read = time.monotonic() - t
            t = time.monotonic()
            layers = landsat_scale(bands, inside, device)
            t_calc = time.monotonic() - t
            acq.write_tif(out, layers, grid, "float32",
                          {"scene": scene, "acquired": scene[-8:], "clipped_to": "mumbai, navi_mumbai",
                           "nodata": "NaN", "qa_rejected": "fill, dilated cloud, cirrus, cloud, shadow"})
            st = layers["st_celsius"]
            log.info("Landsat %s: read %.1f s, scale+mask %.2f s on %s, clear in cities %.0f%%", scene, t_read,
                     t_calc, getattr(device, "type", "cpu"), 100 * float(np.isfinite(st[inside]).mean()))
        else:
            st = read(out)["st_celsius"]
        # Coverage (scene footprint) and cloud are different things: a scene from a
        # neighbouring track may cover only a corner of the cities.
        qa = _read_band(raw, "QA_PIXEL")
        covered = inside & ((qa.astype(np.int64) & 1) == 0)          # QA bit 0 = fill (outside footprint)
        clear = np.isfinite(st) & inside
        n_in, n_cov = int(inside.sum()), int(covered.sum())
        return {"scene": scene, "date": scene[-8:],
                "coverage_of_cities": round(n_cov / n_in, 4),
                "clear_share_of_covered": round(int(clear.sum()) / n_cov, 4) if n_cov else 0.0,
                "clear_fraction_in_cities": round(int(clear.sum()) / n_in, 4)}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(progress(pool.map(one, scenes), total=len(scenes), desc="Landsat QA + scale + clip",
                             unit="scene"))
    with open(config.PROCESSED / "landsat_clear_fraction.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["scene", "date", "coverage_of_cities", "clear_share_of_covered",
                                          "clear_fraction_in_cities"])
        w.writeheader()
        w.writerows(rows)
    log.info("Landsat: done in %.0f s (%d scenes)", time.monotonic() - t0, len(rows))
    return rows


# --- WorldCover --------------------------------------------------------------------------

def block_mode(a: np.ndarray, f: int) -> np.ndarray:
    """Most common value in each f x f block (categorical resampling, no invented classes).
    Ties go to the smallest class code."""
    h, w = a.shape
    blocks = a[: h - h % f, : w - w % f].reshape(h // f, f, w // f, f).transpose(0, 2, 1, 3).reshape(h // f, w // f, f * f)
    classes = np.unique(blocks)
    counts = np.stack([(blocks == c).sum(axis=2) for c in classes])
    return classes[np.argmax(counts, axis=0)].astype(a.dtype)


def run_worldcover(grid30: Grid, ids: np.ndarray) -> list[Path]:
    paths = []
    for year in progress(sorted(config.WORLDCOVER), desc="WorldCover to 30 m", unit="map"):
        out = P_WORLDCOVER / f"worldcover_{year}_30m.tif"
        if not out.exists():
            m10 = read(acq.worldcover_path(year))["Map"]
            f = int(round(grid30.res / config.WORLDCOVER_RES_M))
            m30 = block_mode(m10, f)
            m30[ids == 0] = 0
            acq.write_tif(out, {"landcover": m30}, grid30, "uint8",
                          {"year": year, "resampling": f"mode of {f}x{f} 10 m pixels", "nodata": 0,
                           "clipped_to": "mumbai, navi_mumbai"})
        paths.append(out)
    log.info("WorldCover: %d maps at 30 m (mode of 3x3 10 m pixels, CPU)", len(paths))
    return paths


# --- ERA5-Land ----------------------------------------------------------------------------

def era5_cell_lookup(grid1km: Grid, lons, lats, land) -> tuple[np.ndarray, np.ndarray]:
    """For every 1 km cell: index of the ERA5-Land point to use, and whether it was
    substituted. A cell takes the ERA5 cell it falls in; if that is sea (no ERA5-Land
    data), the nearest land ERA5 cell. Values are copied, never interpolated."""
    from pyproj import Transformer
    t = Transformer.from_crs(grid1km.crs, "EPSG:4326", always_xy=True)
    cols = grid1km.xmin + (np.arange(grid1km.width) + 0.5) * grid1km.res
    rows = grid1km.ymax - (np.arange(grid1km.height) + 0.5) * grid1km.res
    X, Y = np.meshgrid(cols, rows)
    lon, lat = t.transform(X, Y)
    lons, lats, land = map(np.asarray, (lons, lats, land))
    d2 = (lon[..., None] - lons) ** 2 + (lat[..., None] - lats) ** 2      # ERA5 points are cell centres
    own = np.argmin(d2, axis=-1)                                          # the cell containing it
    d2_land = np.where(land, d2, np.inf)
    nearest_land = np.argmin(d2_land, axis=-1)
    use = np.where(land[own], own, nearest_land)
    return use, ~land[own]


def run_era5(grid: Grid, ids: np.ndarray) -> list[Path]:
    paths, lookup = [], None
    for year in progress(list(range(config.START_YEAR, config.END_YEAR + 1)), desc="ERA5-Land to 1 km grid",
                         unit="year"):
        raw = acq.era5_path(year)
        if not raw.exists():
            log.warning("ERA5-Land %d: raw file missing", year)
            continue
        with open(raw, encoding="utf-8") as f:
            recs = list(csv.DictReader(f))
        points = sorted({(float(r["lon"]), float(r["lat"])) for r in recs})
        index = {p: i for i, p in enumerate(points)}
        days = sorted({r["date"] for r in recs})
        dpos = {d: i for i, d in enumerate(days)}
        values = {v: np.full((len(days), len(points)), np.nan) for v in config.ERA5_BANDS}
        for r in recs:
            i, t = index[(float(r["lon"]), float(r["lat"]))], dpos[r["date"]]
            for v in config.ERA5_BANDS:
                if r[v] != "":
                    values[v][t, i] = float(r[v])
        if lookup is None:
            land = np.isfinite(values["temperature_2m"]).any(axis=0)
            use, substituted = era5_cell_lookup(grid, [p[0] for p in points], [p[1] for p in points], land)
            lookup = (points, use)
            sub_cells = int((substituted & (ids > 0)).sum())
            log.info("ERA5-Land: %d/%d native cells are land; %d city cells take the nearest land cell",
                     int(land.sum()), len(points), sub_cells)
            acq.write_tif(P_MASKS / "era5_nearest_land_substituted_1km.tif",
                          {"substituted": (substituted & (ids > 0)).astype(np.uint8)}, grid, "uint8",
                          {"meaning": "1 = the ERA5-Land cell over this city cell is sea; nearest land cell used"})
        assert lookup[0] == points, "ERA5-Land points changed between years"
        for v in config.ERA5_BANDS:
            out = P_ERA5 / f"era5_{v}_{year}.tif"
            if out.exists():
                continue
            arr = values[v][:, lookup[1]]                                 # (days, H, W)
            if v in ERA5_KELVIN:
                arr = arr - 273.15
            arr[:, ids == 0] = np.nan
            unit = "degC" if v in ERA5_KELVIN else "m/s"
            acq.write_tif(out, {d.replace("-", ""): arr[t] for t, d in enumerate(days)}, grid, "float32",
                          {"variable": v, "unit": unit, "year": year, "bands": "one per day (YYYYMMDD)",
                           "resampling": "copied from containing ERA5-Land cell (nearest land cell at the coast)",
                           "clipped_to": "mumbai, navi_mumbai"})
            paths.append(out)
    log.info("ERA5-Land: %d files written (K -> C for temperatures, CPU)", len(paths))
    return paths


# --- all ---------------------------------------------------------------------------------

def run_all() -> None:
    from urban_heat.device import get_device
    from urban_heat.grid import load_grids
    t0 = time.monotonic()
    device = get_device()
    grids = load_grids(config.DATA / "grids.json")
    g1, g30 = grids["modis_1km"], grids["landsat_30m"]
    geoms = city_geoms()
    ids1, ids30 = city_id_raster(g1, geoms), city_id_raster(g30, geoms)
    acq.write_tif(P_MASKS / "city_id_1km.tif", {"city_id": ids1}, g1, "uint8", {"ids": "1 mumbai, 2 navi_mumbai"})
    acq.write_tif(P_MASKS / "city_id_30m.tif", {"city_id": ids30}, g30, "uint8", {"ids": "1 mumbai, 2 navi_mumbai"})
    log.info("city cells: 1 km Mumbai %d, Navi Mumbai %d | 30 m Mumbai %d, Navi Mumbai %d",
             int((ids1 == 1).sum()), int((ids1 == 2).sum()), int((ids30 == 1).sum()), int((ids30 == 2).sum()))
    run_modis(g1, ids1)
    run_landsat(g30, ids30, device=device)
    run_worldcover(g30, ids30)
    run_era5(g1, ids1)
    log.info("Phase 3 finished in %.0f s", time.monotonic() - t0)
