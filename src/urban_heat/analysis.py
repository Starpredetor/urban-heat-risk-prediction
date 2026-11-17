"""Phase 5: historical urban heat analysis (SRS FR-5.1 to FR-5.7).

Plan step                                              Output (outputs/ and data/processed/analysis/)
1 historical MODIS temperature maps                    historical_maps/modis_lst_mam_<year>.tif, ..._mam_2001_2025_mean.tif
2 detailed Landsat temperature maps, selected years    historical_maps/landsat_st_mam_<year>.tif
3 annual and pre-monsoon statistics                    analysis/city_year_stats.csv
4 persistent high-temperature regions                  historical_maps/hotspot_frequency.tif
5 vegetation versus built-up                           analysis/lst_by_builtup_share.csv, landsat_lst_vs_indices.csv
6 Mumbai versus Navi Mumbai                            analysis/city_comparison.csv
7 change in the spatial distribution of heat           historical_maps/lst_trend_c_per_decade.tif,
                                                       historical_maps/relative_lst_change.tif

Definitions (SRS assumptions A11-A13):
- A cell's annual / March-May (MAM) LST for a year = mean of its valid 8-day composites in
  that period; the number of composites is kept. Nothing is gap-filled.
- Hotspot: MAM LST at or above the city's 90th percentile in that year (city-relative,
  so overall warm years do not make every cell a hotspot). Persistent: hotspot in at
  least half of the years with data.
- Trend: ordinary least-squares slope of MAM LST against year, per cell with >= 15 years,
  in C per decade. Spatial change: (cell - city mean) in 2021-2025 minus the same in
  2001-2005, i.e. where heat moved relative to the city, separate from overall warming.
"""

from __future__ import annotations

import csv
import logging
import time
import warnings
from datetime import date
from pathlib import Path

import numpy as np

from urban_heat import acquisition as acq, config, features as ft, preprocessing as pp
from urban_heat.logs import progress

log = logging.getLogger("phase5")
# cells with no valid data in a period give NaN by design; silence numpy's "Mean of empty slice"
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
A = config.PROCESSED / "analysis"
MAPS = config.OUTPUTS / "historical_maps"
MAM = (3, 4, 5)
HOTSPOT_PCT = 90
PERSISTENT_SHARE = 0.5
MIN_TREND_YEARS = 15
EARLY, LATE = (2001, 2005), (2021, 2025)


# --- loading ----------------------------------------------------------------------------

def load_modis_stack() -> tuple[np.ndarray, list[date]]:
    """All processed MODIS LST composites as (T, H, W) degrees C, with their start dates."""
    dates = [d for d in acq.modis_dates(config.START_YEAR, config.END_YEAR) if pp.modis_out(d).exists()]
    stack = np.stack([pp.read(pp.modis_out(d))["lst_celsius"] for d in
                      progress(dates, desc="Load MODIS LST", unit="composite")]).astype(np.float32)
    return stack, dates


# --- step 1/3: per-year maps and statistics ---------------------------------------------------

def period_mean(stack, dates, year: int, months=None) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell mean of valid composites in `year` (optionally only `months`), and their count."""
    sel = [i for i, d in enumerate(dates) if d.year == year and (months is None or d.month in months)]
    if not sel:
        return np.full(stack.shape[1:], np.nan, np.float32), np.zeros(stack.shape[1:], np.int16)
    sub = stack[sel]
    count = np.isfinite(sub).sum(axis=0).astype(np.int16)
    with np.errstate(all="ignore"):
        return np.nanmean(sub, axis=0).astype(np.float32), count


def yearly_maps(stack, dates) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    years = range(config.START_YEAR, config.END_YEAR + 1)
    return {"annual": {y: period_mean(stack, dates, y) for y in years},
            "mam": {y: period_mean(stack, dates, y, MAM) for y in years}}


def city_values(a, ids, city_id):
    v = a[ids == city_id]
    return v[np.isfinite(v)]


def city_year_stats(maps, ids) -> list[dict]:
    rows = []
    for season, per_year in maps.items():
        for y, (m, n) in per_year.items():
            for city, i in pp.CITY_IDS.items():
                v = city_values(m, ids, i)
                c = n[ids == i]
                rows.append({"year": y, "season": "March-May" if season == "mam" else "annual", "city": city,
                             "mean_c": round(float(v.mean()), 3) if v.size else "",
                             "p10_c": round(float(np.percentile(v, 10)), 3) if v.size else "",
                             "p90_c": round(float(np.percentile(v, 90)), 3) if v.size else "",
                             "cells_with_data": int(v.size), "cells": int((ids == i).sum()),
                             "mean_composites_per_cell": round(float(c.mean()), 2)})
    return rows


# --- step 4: hotspots ------------------------------------------------------------------------------

def hotspot_frequency(mam_maps: dict, ids) -> tuple[np.ndarray, np.ndarray]:
    """Share of years each cell was a hotspot (>= its city's 90th percentile), and the
    number of years with data."""
    hits = np.zeros(ids.shape, np.int32)
    years_with = np.zeros(ids.shape, np.int32)
    for y, (m, _) in mam_maps.items():
        for i in pp.CITY_IDS.values():
            v = city_values(m, ids, i)
            if v.size < 10:
                continue
            thr = np.percentile(v, HOTSPOT_PCT)
            cell = (ids == i) & np.isfinite(m)
            years_with[cell] += 1
            hits[cell & (m >= thr)] += 1
    with np.errstate(all="ignore"):
        freq = np.where(years_with > 0, hits / years_with, np.nan).astype(np.float32)
    return freq, years_with


# --- step 7: change over time ------------------------------------------------------------------------

def per_cell_trend(mam_maps: dict, min_years: int = MIN_TREND_YEARS) -> tuple[np.ndarray, np.ndarray]:
    """OLS slope of MAM LST against year per cell (C per decade) and the years used."""
    years = np.array(sorted(mam_maps), dtype=np.float64)
    y = np.stack([mam_maps[int(t)][0] for t in years]).astype(np.float64)        # (Y, H, W)
    ok = np.isfinite(y)
    n = ok.sum(axis=0)
    x = np.where(ok, years[:, None, None], 0.0)
    yv = np.where(ok, y, 0.0)
    with np.errstate(all="ignore"):
        xm, ym = x.sum(0) / n, yv.sum(0) / n
        sxy = (np.where(ok, (years[:, None, None] - xm) * (y - ym), 0.0)).sum(0)
        sxx = (np.where(ok, (years[:, None, None] - xm) ** 2, 0.0)).sum(0)
        slope = sxy / sxx * 10.0
    return np.where(n >= min_years, slope, np.nan).astype(np.float32), n.astype(np.int16)


def relative_change(mam_maps: dict, ids) -> np.ndarray:
    """(cell - city mean) in the late window minus the same in the early window."""
    def rel(window):
        with np.errstate(all="ignore"):
            m = np.nanmean(np.stack([mam_maps[y][0] for y in range(window[0], window[1] + 1)]), axis=0)
        out = np.full(m.shape, np.nan, np.float32)
        for i in pp.CITY_IDS.values():
            sel = ids == i
            out[sel] = m[sel] - np.nanmean(m[sel])
        return out
    return (rel(LATE) - rel(EARLY)).astype(np.float32)


# --- step 5: vegetation vs built-up -------------------------------------------------------------------

def lst_by_builtup(mam_mean, ids) -> list[dict]:
    """Multi-year MAM LST of 1 km cells grouped by built-up share (0-20%, ..., 80-100%)."""
    share = pp.read(config.FEATURES / "static_1km.tif")["share_built_up"]
    rows = []
    edges = [0, 0.2, 0.4, 0.6, 0.8, 1.0001]
    for city, i in pp.CITY_IDS.items():
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (ids == i) & (share >= lo) & (share < hi) & np.isfinite(mam_mean)
            v = mam_mean[sel]
            rows.append({"city": city, "built_up_share": f"{int(lo * 100)}-{min(int(hi * 100), 100)}%",
                         "cells": int(sel.sum()), "mean_lst_c": round(float(v.mean()), 3) if v.size else "",
                         "median_lst_c": round(float(np.median(v)), 3) if v.size else ""})
    return rows


def landsat_lst_vs_indices(ids30) -> list[dict]:
    """Per selected year and city: correlation of Landsat surface temperature with NDVI,
    NDBI and NDWI (land pixels only), and mean temperature per land-cover group."""
    groups = pp.read(config.FEATURES / "static_30m.tif")["landcover_group"].astype(int)
    rows = []
    for y in config.LANDSAT_YEARS:
        path = config.FEATURES / f"landsat_{y}_30m.tif"
        if not path.exists():
            continue
        f = pp.read(path)
        for city, i in pp.CITY_IDS.items():
            land = (ids30 == i) & (groups != ft.GROUP_CODES["water"]) & np.isfinite(f["st_celsius"])
            row = {"year": y, "city": city, "land_pixels": int(land.sum())}
            for k in ("ndvi", "ndbi", "ndwi"):
                ok = land & np.isfinite(f[k])
                row[f"r_st_{k}"] = round(float(np.corrcoef(f["st_celsius"][ok], f[k][ok])[0, 1]), 3) if ok.sum() > 100 else ""
            for name, code in ft.GROUP_CODES.items():
                v = f["st_celsius"][(ids30 == i) & (groups == code) & np.isfinite(f["st_celsius"])]
                row[f"mean_st_{name}_c"] = round(float(v.mean()), 3) if v.size else ""
            rows.append(row)
    return rows


# --- writing ---------------------------------------------------------------------------------------------

def _csv(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def run_all() -> None:
    from urban_heat.grid import load_grids
    t0 = time.monotonic()
    MAPS.mkdir(parents=True, exist_ok=True)
    grids = load_grids(config.DATA / "grids.json")
    g1, g30 = grids["modis_1km"], grids["landsat_30m"]
    ids1 = pp.read(pp.P_MASKS / "city_id_1km.tif")["city_id"]
    ids30 = pp.read(pp.P_MASKS / "city_id_30m.tif")["city_id"]
    log.info("Phase 5 runs on the CPU: the MODIS stack is small (%d x %d cells per composite)", g1.height, g1.width)

    stack, dates = load_modis_stack()
    log.info("MODIS: %d composites loaded (%s to %s)", len(dates), dates[0], dates[-1])
    maps = yearly_maps(stack, dates)

    # step 1: historical maps
    for y, (m, n) in progress(maps["mam"].items(), desc="MODIS March-May maps", unit="year"):
        acq.write_tif(MAPS / f"modis_lst_mam_{y}.tif", {"lst_mam_mean_c": m, "composites_used": n.astype(np.float32)},
                      g1, "float32", {"year": y, "season": "March-May", "statistic": "mean of valid 8-day composites"})
    with np.errstate(all="ignore"):
        mam_mean = np.nanmean(np.stack([m for m, _ in maps["mam"].values()]), axis=0).astype(np.float32)
    acq.write_tif(MAPS / f"modis_lst_mam_{config.START_YEAR}_{config.END_YEAR}_mean.tif", {"lst_mam_mean_c": mam_mean},
                  g1, "float32", {"years": f"{config.START_YEAR}-{config.END_YEAR}", "season": "March-May"})

    # step 2: Landsat detailed maps
    for y in config.LANDSAT_YEARS:
        p = config.FEATURES / f"landsat_{y}_30m.tif"
        if p.exists():
            f = pp.read(p)
            acq.write_tif(MAPS / f"landsat_st_mam_{y}.tif", {"st_mam_median_c": f["st_celsius"], "clear_obs": f["clear_obs"]},
                          g30, "float32", {"year": y, "season": "March-May", "statistic": "median of clear scenes",
                                           "note": "thermal band is 100 m native, delivered on a 30 m grid"})
    log.info("historical maps written to %s", MAPS.relative_to(config.ROOT))

    # step 3 / 6: statistics and city comparison
    stats = city_year_stats(maps, ids1)
    _csv(A / "city_year_stats.csv", stats)
    comp = []
    for season in ("March-May", "annual"):
        for city in pp.CITY_IDS:
            v = [r["mean_c"] for r in stats if r["season"] == season and r["city"] == city and r["mean_c"] != ""]
            comp.append({"season": season, "city": city, "years": len(v), "mean_of_years_c": round(float(np.mean(v)), 3),
                         "warmest_year_c": round(float(np.max(v)), 3), "coolest_year_c": round(float(np.min(v)), 3)})
    _csv(A / "city_comparison.csv", comp)

    # step 4: hotspots
    freq, years_with = hotspot_frequency(maps["mam"], ids1)
    acq.write_tif(MAPS / "hotspot_frequency.tif", {"hotspot_share_of_years": freq, "years_with_data": years_with.astype(np.float32),
                                                   "persistent_hotspot": (freq >= PERSISTENT_SHARE).astype(np.float32)},
                  g1, "float32", {"hotspot": f"MAM LST >= city {HOTSPOT_PCT}th percentile in that year",
                                  "persistent": f"hotspot in >= {int(PERSISTENT_SHARE * 100)}% of years"})
    for city, i in pp.CITY_IDS.items():
        log.info("%s: %d persistent hotspot cells of %d", city, int(((ids1 == i) & (freq >= PERSISTENT_SHARE)).sum()),
                 int((ids1 == i).sum()))

    # step 7: change over time
    trend, n_years = per_cell_trend(maps["mam"])
    acq.write_tif(MAPS / "lst_trend_c_per_decade.tif", {"trend_c_per_decade": trend, "years_used": n_years.astype(np.float32)},
                  g1, "float32", {"method": f"OLS of March-May LST on year, cells with >= {MIN_TREND_YEARS} years"})
    rel = relative_change(maps["mam"], ids1)
    acq.write_tif(MAPS / "relative_lst_change.tif", {"relative_change_c": rel}, g1, "float32",
                  {"definition": f"(cell - city mean) {LATE[0]}-{LATE[1]} minus same {EARLY[0]}-{EARLY[1]}, March-May"})

    # step 5: vegetation vs built-up
    _csv(A / "lst_by_builtup_share.csv", lst_by_builtup(mam_mean, ids1))
    rows = landsat_lst_vs_indices(ids30)
    if rows:
        _csv(A / "landsat_lst_vs_indices.csv", rows)
    log.info("Phase 5 finished in %.0f s; tables in %s", time.monotonic() - t0, A.relative_to(config.ROOT))
