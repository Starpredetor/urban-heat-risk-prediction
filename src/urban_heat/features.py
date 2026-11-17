"""Phase 4: generate environmental features (SRS FR-4.1 to FR-4.3).

Plan item                                   Output (data/processed/features/)
NDVI, NDBI, NDWI                            landsat_<year>_30m.tif (+ _1km)  per selected year
Land cover: built-up, vegetation, water,    static_30m.tif: landcover_group
  other                                     static_1km.tif: share of each group per 1 km cell
Elevation                                   static_30m.tif / static_1km.tif: elevation_m
Distance to major water bodies              static_30m.tif / static_1km.tif: dist_major_water_m
Built-up land-cover proportion              static_1km.tif: share_built_up
ERA5-Land temperature and wind              era5_8day_<year>_1km.tif: one band per MODIS 8-day period
Feature stack per city and period           feature_summary.csv (per city, per period)

Formulas (Landsat surface reflectance from Phase 3):
    NDVI = (NIR - Red) / (NIR + Red)      NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)
    NDWI = (Green - NIR) / (Green + NIR)
An observation period for Landsat is the March-May season of a selected year: each
feature is the per-pixel median over that season's clear observations, with the
number of observations kept as clear_obs. ERA5-Land is averaged over each MODIS
8-day composite period so it lines up with the LST series.
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

import numpy as np

from urban_heat import acquisition as acq, config, preprocessing as pp
from urban_heat.grid import Grid
from urban_heat.logs import progress

log = logging.getLogger("phase4")
F = config.FEATURES
GROUP_CODES = {name: i for i, name in enumerate(config.LANDCOVER_GROUPS, start=1)}   # 1..4, 0 = outside


# --- indices ---------------------------------------------------------------------------

def normalized_difference(a, b) -> np.ndarray:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    den = a + b
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(np.abs(den) > 1e-6, (a - b) / den, np.nan)
    out[~((out >= -1) & (out <= 1))] = np.nan
    return out.astype(np.float32)


def indices(sr: dict) -> dict[str, np.ndarray]:
    return {"ndvi": normalized_difference(sr["nir"], sr["red"]),
            "ndbi": normalized_difference(sr["swir1"], sr["nir"]),
            "ndwi": normalized_difference(sr["green"], sr["nir"])}


def nanmedian_stack(arrays, device=None) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel median over a list of 2-D arrays ignoring NaN, and the count of
    finite values. Uses PyTorch (CUDA when `device` is a CUDA device). For an even
    count, torch returns the lower of the two middle values."""
    import torch
    dev = device if device is not None else torch.device("cpu")
    t = torch.stack([torch.from_numpy(np.asarray(a, np.float32)) for a in arrays]).to(dev)
    count = (~torch.isnan(t)).sum(dim=0)
    med = torch.nanmedian(t, dim=0).values
    return med.cpu().numpy().astype(np.float32), count.cpu().numpy().astype(np.int16)


# --- land cover, water, aggregation --------------------------------------------------------

def group_landcover(classes) -> np.ndarray:
    """WorldCover class codes -> plan groups (1 built-up, 2 vegetation, 3 water, 4 other, 0 none)."""
    classes = np.asarray(classes)
    out = np.zeros(classes.shape, dtype=np.uint8)
    for name, codes in config.LANDCOVER_GROUPS.items():
        out[np.isin(classes, codes)] = GROUP_CODES[name]
    return out


def major_water_distance(classes30, res_m: float, min_km2: float = config.MAJOR_WATER_MIN_KM2):
    """Distance (m) from every pixel to the nearest major water body: connected
    permanent-water areas (WorldCover 80) of at least `min_km2`. Returns (distance, major_mask)."""
    from scipy import ndimage
    water = np.asarray(classes30) == 80
    labels, n = ndimage.label(water, structure=np.ones((3, 3)))
    if n == 0:
        return np.full(water.shape, np.nan, np.float32), water
    sizes = np.bincount(labels.ravel())
    min_px = min_km2 * 1e6 / res_m ** 2
    major = np.isin(labels, np.nonzero(sizes >= min_px)[0][1:])        # label 0 is background
    if not major.any():
        return np.full(water.shape, np.nan, np.float32), major
    return (ndimage.distance_transform_edt(~major) * res_m).astype(np.float32), major


def aggregate(values, src: Grid, dst: Grid, valid=None) -> tuple[np.ndarray, np.ndarray]:
    """Mean of finite source pixels whose centre falls in each destination cell, and
    how many there were. Works for grids that do not nest (30 m into 1 km)."""
    values = np.asarray(values, np.float64)
    ok = np.isfinite(values) if valid is None else (np.asarray(valid, bool) & np.isfinite(values))
    r, c = np.nonzero(ok)
    x = src.xmin + (c + 0.5) * src.res
    y = src.ymax - (r + 0.5) * src.res
    dc = np.floor((x - dst.xmin) / dst.res).astype(int)
    dr = np.floor((dst.ymax - y) / dst.res).astype(int)
    inside = (dc >= 0) & (dc < dst.width) & (dr >= 0) & (dr < dst.height)
    flat = dr[inside] * dst.width + dc[inside]
    total = np.bincount(flat, weights=values[r[inside], c[inside]], minlength=dst.width * dst.height)
    count = np.bincount(flat, minlength=dst.width * dst.height)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / count, np.nan)
    return mean.reshape(dst.shape).astype(np.float32), count.reshape(dst.shape)


# --- ERA5-Land onto MODIS 8-day periods ---------------------------------------------------------

def period_means(daily: np.ndarray, days: list[str], period_starts) -> np.ndarray:
    """Mean over each MODIS composite period [start, start+8 days), cut at year end.
    daily: (n_days, H, W) with `days` as YYYYMMDD strings."""
    from datetime import date, timedelta
    as_date = [date(int(d[:4]), int(d[4:6]), int(d[6:])) for d in days]
    out = []
    for s in period_starts:
        end = min(s + timedelta(days=8), date(s.year + 1, 1, 1))
        sel = [i for i, d in enumerate(as_date) if s <= d < end]
        with np.errstate(all="ignore"):
            out.append(np.nanmean(daily[sel], axis=0) if sel else np.full(daily.shape[1:], np.nan))
    return np.stack(out).astype(np.float32)


def _read_daily(var: str, year: int):
    import rasterio
    with rasterio.open(pp.P_ERA5 / f"era5_{var}_{year}.tif") as src:
        return src.read().astype(np.float32), list(src.descriptions)


def era5_8day(year: int, grid1: Grid) -> Path:
    out = F / f"era5_8day_{year}_1km.tif"
    if out.exists():
        return out
    starts = acq.modis_dates(year, year)
    layers = {}
    for var, name in [("temperature_2m", "air_temp_2m_c"), ("dewpoint_temperature_2m", "dewpoint_2m_c"),
                      ("skin_temperature", "skin_temp_c")]:
        daily, days = _read_daily(var, year)
        layers[name] = period_means(daily, days, starts)
    u, days = _read_daily("u_component_of_wind_10m", year)
    v, _ = _read_daily("v_component_of_wind_10m", year)
    layers["wind_speed_10m_ms"] = period_means(np.hypot(u, v), days, starts)   # speed of each daily-mean wind
    bands = {f"{name}_{s:%Y%m%d}": layers[name][t] for name in layers for t, s in enumerate(starts)}
    return acq.write_tif(out, bands, grid1, "float32",
                         {"year": year, "periods": "MODIS 8-day composites (band suffix = period start)",
                          "variables": ", ".join(layers), "units": "degC, degC, degC, m/s",
                          "wind": "mean over the period of daily wind speed from daily-mean u and v"})


# --- Landsat yearly features ---------------------------------------------------------------------

def landsat_year(year: int, grid30: Grid, grid1: Grid, ids30: np.ndarray, device=None) -> Path | None:
    out30, out1 = F / f"landsat_{year}_30m.tif", F / f"landsat_{year}_1km.tif"
    if out30.exists() and out1.exists():
        return out30
    scenes = sorted(pp.P_LANDSAT.glob(f"L*_{year}*_processed_{year}*.tif"))
    if not scenes:
        log.warning("Landsat %d: no processed scenes", year)
        return None
    stacks = {k: [] for k in ("ndvi", "ndbi", "ndwi", "st_celsius")}
    for p in progress(scenes, desc=f"Landsat {year}: indices per scene", unit="scene"):
        b = pp.read(p)
        for k, v in indices(b).items():
            stacks[k].append(v)
        stacks["st_celsius"].append(b["st_celsius"])
    t = time.monotonic()
    med = {}
    for k, arrs in stacks.items():
        med[k], count = nanmedian_stack(arrs, device)
    log.info("Landsat %d: median of %d scenes on %s in %.1f s", year, len(scenes),
             getattr(device, "type", "cpu"), time.monotonic() - t)
    inside = ids30 > 0
    for k in med:
        med[k][~inside] = np.nan
    count = np.where(inside, count, 0).astype(np.int16)
    acq.write_tif(out30, {**med, "clear_obs": count}, grid30, "float32",
                  {"year": year, "season": "March-May", "scenes": len(scenes),
                   "statistic": "per-pixel median of clear observations", "clear_obs": "observations used"})
    agg = {k: aggregate(v, grid30, grid1)[0] for k, v in med.items()}
    acq.write_tif(out1, agg, grid1, "float32",
                  {"year": year, "aggregation": "mean of 30 m pixels whose centre falls in the 1 km cell"})
    return out30


# --- static layers ----------------------------------------------------------------------------------

def static_layers(grid30: Grid, grid1: Grid, ids30: np.ndarray, ids1: np.ndarray) -> tuple[Path, Path]:
    out30, out1 = F / "static_30m.tif", F / "static_1km.tif"
    raw10 = pp.read(acq.worldcover_path(config.LANDCOVER_YEAR))["Map"]
    f = int(round(grid30.res / config.WORLDCOVER_RES_M))
    classes30_full = pp.block_mode(raw10, f)                         # whole extent, incl. the sea
    dist, major = major_water_distance(classes30_full, grid30.res)
    log.info("major water bodies (>= %.1f km2): %.1f km2 of water in the grid", config.MAJOR_WATER_MIN_KM2,
             major.sum() * grid30.res ** 2 / 1e6)
    inside = ids30 > 0
    groups = np.where(inside, group_landcover(classes30_full), 0).astype(np.uint8)
    elev = np.full(grid30.shape, np.nan, np.float32)
    if acq.elevation_path().exists():
        elev = pp.read(acq.elevation_path())["elevation"].astype(np.float32)
    else:
        log.warning("elevation not downloaded yet (scripts/build_features.py fetches it); elevation left empty")
    layers30 = {"landcover_group": groups.astype(np.float32),
                "elevation_m": np.where(inside, elev, np.nan),
                "dist_major_water_m": np.where(inside, dist, np.nan)}
    acq.write_tif(out30, layers30, grid30, "float32",
                  {"landcover_group": "1 built-up, 2 vegetation, 3 water, 4 other, 0 outside",
                   "landcover_source": f"ESA WorldCover {config.LANDCOVER_YEAR}",
                   "major_water": f"connected permanent water >= {config.MAJOR_WATER_MIN_KM2} km2"})
    layers1 = {f"share_{name}": aggregate((groups == code).astype(np.float32), grid30, grid1, valid=inside)[0]
               for name, code in GROUP_CODES.items()}
    layers1["elevation_m"] = aggregate(layers30["elevation_m"], grid30, grid1)[0]
    layers1["dist_major_water_m"] = aggregate(layers30["dist_major_water_m"], grid30, grid1)[0]
    for k in layers1:
        layers1[k][ids1 == 0] = np.nan
    acq.write_tif(out1, layers1, grid1, "float32",
                  {"share_*": "fraction of the cell's in-city 30 m pixels in each land-cover group",
                   "aggregation": "mean of 30 m pixels whose centre falls in the 1 km cell"})
    return out30, out1


# --- per-city summary ----------------------------------------------------------------------------

def summarize(ids30, ids1) -> Path:
    rows = []

    def add(period, path, ids, skip=()):
        for name, a in pp.read(path).items():
            if name in skip:
                continue
            for city, i in pp.CITY_IDS.items():
                v = a[ids == i].astype(np.float64)
                ok = np.isfinite(v)
                rows.append({"city": city, "period": period, "feature": name, "valid_share": round(ok.mean(), 4),
                             "mean": round(float(v[ok].mean()), 4) if ok.any() else "",
                             "min": round(float(v[ok].min()), 4) if ok.any() else "",
                             "max": round(float(v[ok].max()), 4) if ok.any() else ""})

    add("static", F / "static_1km.tif", ids1)
    for p in sorted(F.glob("landsat_*_1km.tif")):
        add(f"{p.stem.split('_')[1]} Mar-May", p, ids1)
    out = F / "feature_summary.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return out


def run_all() -> None:
    from urban_heat.device import get_device
    from urban_heat.grid import load_grids
    t0 = time.monotonic()
    F.mkdir(parents=True, exist_ok=True)
    device = get_device()
    grids = load_grids(config.DATA / "grids.json")
    g1, g30 = grids["modis_1km"], grids["landsat_30m"]
    ids1 = pp.read(pp.P_MASKS / "city_id_1km.tif")["city_id"]
    ids30 = pp.read(pp.P_MASKS / "city_id_30m.tif")["city_id"]
    static_layers(g30, g1, ids30, ids1)
    log.info("static layers written")
    for y in progress(list(config.LANDSAT_YEARS), desc="Landsat yearly features", unit="year"):
        landsat_year(y, g30, g1, ids30, device)
    for y in progress(list(range(config.START_YEAR, config.END_YEAR + 1)), desc="ERA5-Land 8-day features",
                      unit="year"):
        era5_8day(y, g1)
    log.info("ERA5-Land 8-day features written for %d-%d", config.START_YEAR, config.END_YEAR)
    log.info("summary: %s", summarize(ids30, ids1).relative_to(config.ROOT))
    log.info("Phase 4 finished in %.0f s", time.monotonic() - t0)
