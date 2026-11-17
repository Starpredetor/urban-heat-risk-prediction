"""Phase 3: turn raw downloads into one clean, aligned cube (SRS FR-P3.1 to FR-P3.6).

    LST          MOD11A2 QC mask + scale -> degrees C                 (uhrm.modis)
    reflectance  MOD09A1 StateQA cloud screen + scale, 500 m -> 1 km mean
                 of clear sub-pixels, then NDVI / NDBI / NDWI
    weather      ERA5-Land temperatures K -> C; winds kept as u, v (m/s)
    static       WorldCover class fractions, SRTM elevation

Nothing is gap-filled. Every variable keeps NaN where there was no valid
observation, and each date gets an LST valid fraction over land cells plus a
`usable` flag (FR-P3.4). Missing raw files become all-NaN dates, never errors.

MOD09A1 StateQA bits used (0-based):
    0-1 cloud state (00 clear)   2 cloud shadow   8-9 cirrus (0 none .. 3 high)
Bit 10 (internal cloud flag) is NOT used: sources document opposite meanings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from uhrm import acquire, modis

REFLECTANCE_BANDS = {"red": "sur_refl_b01", "nir": "sur_refl_b02",
                     "green": "sur_refl_b04", "swir1": "sur_refl_b06"}
ERA5_KELVIN = ("temperature_2m", "dewpoint_temperature_2m", "skin_temperature")
WATER_CLASS = 80          # WorldCover "permanent water bodies"


@dataclass(frozen=True)
class ReflectanceRule:
    scale: float = 0.0001
    valid_dn_range: tuple[int, int] = (-100, 16000)
    max_cirrus_class: int = 1
    min_clear_subpixels: int = 2


# --- reflectance ----------------------------------------------------------------

def stateqa_clear(qa, max_cirrus_class: int = 1) -> np.ndarray:
    """True where StateQA says clear, no shadow, cirrus at most `max_cirrus_class`."""
    q = np.asarray(qa).astype(np.int64) & 0xFFFF
    return ((q & 0b11) == 0) & (((q >> 2) & 1) == 0) & (((q >> 8) & 0b11) <= max_cirrus_class)


def clear_reflectance(bands: dict, rule: ReflectanceRule) -> dict[str, np.ndarray]:
    """Scaled reflectance (0..1) per named band at native resolution; NaN where not
    clear, masked, or any band is out of its valid DN range."""
    ok = stateqa_clear(bands["StateQA"], rule.max_cirrus_class)
    if "valid" in bands:
        ok &= np.asarray(bands["valid"]) > 0.5
    lo, hi = rule.valid_dn_range
    raw = {k: np.asarray(bands[v], dtype=np.float64) for k, v in REFLECTANCE_BANDS.items()}
    for arr in raw.values():
        ok &= (arr >= lo) & (arr <= hi)
    return {k: np.where(ok, arr * rule.scale, np.nan).astype(np.float32) for k, arr in raw.items()}


def block_mean(arr, factor: int, min_count: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Mean of each factor x factor block over finite values only.

    Returns (mean, count). Blocks with fewer than `min_count` finite values are NaN.
    Used to aggregate 500 m reflectance into the 1 km cells it nests in exactly.
    """
    a = np.asarray(arr, dtype=np.float64)
    h, w = a.shape
    if h % factor or w % factor:
        raise ValueError(f"shape {a.shape} is not divisible by {factor}")
    blocks = a.reshape(h // factor, factor, w // factor, factor)
    finite = np.isfinite(blocks)
    count = finite.sum(axis=(1, 3))
    total = np.where(finite, blocks, 0.0).sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count >= max(min_count, 1), total / count, np.nan)
    return mean.astype(np.float32), count.astype(np.int16)


def normalized_difference(a, b) -> np.ndarray:
    """(a - b) / (a + b); NaN where the denominator is ~0 or the result is outside [-1, 1]."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    den = a + b
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(np.abs(den) > 1e-6, (a - b) / den, np.nan)
    out[~((out >= -1) & (out <= 1))] = np.nan
    return out.astype(np.float32)


def indices(refl_1km: dict) -> dict[str, np.ndarray]:
    """NDVI = (NIR-Red)/(NIR+Red); NDBI = (SWIR1-NIR)/(SWIR1+NIR); NDWI = (Green-NIR)/(Green+NIR)."""
    return {"ndvi": normalized_difference(refl_1km["nir"], refl_1km["red"]),
            "ndbi": normalized_difference(refl_1km["swir1"], refl_1km["nir"]),
            "ndwi": normalized_difference(refl_1km["green"], refl_1km["nir"])}


# --- weather --------------------------------------------------------------------

def era5_to_celsius(bands: dict) -> dict[str, np.ndarray]:
    """Temperatures K -> C; everything else unchanged. Names gain a `_c` suffix when converted."""
    out = {}
    for name, arr in bands.items():
        arr = np.asarray(arr, dtype=np.float32)
        if name in ERA5_KELVIN:
            out[name + "_c"] = arr - np.float32(273.15)
        else:
            out[name] = arr
    return out


# --- reading ----------------------------------------------------------------------

def read_raster(path) -> dict[str, np.ndarray]:
    """All bands of a GeoTIFF keyed by band description (b1, b2, ... if none)."""
    import rasterio

    with rasterio.open(path) as src:
        return {(src.descriptions[i - 1] or f"b{i}"): src.read(i) for i in range(1, src.count + 1)}


# --- the cube -----------------------------------------------------------------------

def build_cube(project, grid, years, *, read=read_raster, log=print) -> dict:
    """Assemble every processed variable on the model grid for `years`.

    Returns a dict of numpy arrays (time-varying ones shaped (T, H, W)) plus
    `dates`, `lst_valid_fraction`, `usable`, and per-dataset presence flags.
    """
    pre = project.preprocess
    rule = pre.reflectance
    factor = project.datasets.reflectance.grid_refinement
    years = sorted(set(years))
    dates = [d for d in acquire.expected_composite_dates(years[0], years[-1]) if d.year in set(years)]
    T, (H, W) = len(dates), grid.shape
    nan = lambda: np.full((T, H, W), np.nan, dtype=np.float32)  # noqa: E731
    raw = project.path("raw")

    # static layers ------------------------------------------------------------
    wc_files = sorted((raw / "worldcover").glob("worldcover_*_fractions.tif"))
    if not wc_files:
        raise FileNotFoundError("no WorldCover fractions on disk; run Phase 2 (worldcover) first")
    wc = read(wc_files[-1])                                 # latest release
    water = np.asarray(wc.get(f"c{WATER_CLASS}", np.zeros(grid.shape)), dtype=np.float32)
    land = water <= pre.max_water_fraction
    terrain_path = raw / "terrain" / "srtm_terrain.tif"
    terrain = read(terrain_path) if terrain_path.exists() else {}

    out = {"lst_c": nan(), "ndvi": nan(), "ndbi": nan(), "ndwi": nan(),
           "refl_clear_subpixels": np.zeros((T, H, W), dtype=np.int16)}
    era5_vars: dict[str, np.ndarray] = {}
    present = {k: np.zeros(T, dtype=bool) for k in ("lst", "reflectance", "era5")}

    for t, d in enumerate(dates):
        p = acquire.composite_path(raw / "modis", d)
        if p.exists():
            dn, qc = list(read(p).values())[:2]
            out["lst_c"][t] = modis.process_composite(dn, qc, project.modis.qc)
            present["lst"][t] = True

        p = acquire.dated_path(raw / "modis_reflectance", "MOD09A1", d)
        if p.exists():
            fine = clear_reflectance(read(p), rule)
            coarse, count = {}, None
            for k, arr in fine.items():
                coarse[k], count = block_mean(arr, factor, rule.min_clear_subpixels)
            for k, arr in indices(coarse).items():
                out[k][t] = arr
            out["refl_clear_subpixels"][t] = count
            present["reflectance"][t] = True

        p = acquire.dated_path(raw / "era5", "ERA5L", d)
        if p.exists():
            for k, arr in era5_to_celsius(read(p)).items():
                era5_vars.setdefault(k, nan())[t] = arr
            present["era5"][t] = True

        if (t + 1) % 100 == 0:
            log(f"processed {t + 1}/{T} dates")

    out.update({f"era5_{k}": v for k, v in era5_vars.items()})
    n_land = max(int(land.sum()), 1)
    valid_frac = (np.isfinite(out["lst_c"]) & land).reshape(T, -1).sum(axis=1) / n_land
    out.update({
        "dates": np.array([d.isoformat() for d in dates]),
        "land": land,
        "lst_valid_fraction": valid_frac.astype(np.float32),
        "usable": valid_frac >= pre.min_valid_fraction,
        "present_lst": present["lst"], "present_reflectance": present["reflectance"],
        "present_era5": present["era5"],
        "worldcover_release": np.array(wc_files[-1].stem),
    })
    out.update({f"wc_{k}": np.asarray(v, dtype=np.float32) for k, v in wc.items()})
    out.update({f"terrain_{k}": np.asarray(v, dtype=np.float32) for k, v in terrain.items()})
    return out


def summarize(cube: dict) -> dict:
    """Numbers to judge the cube by, including the valid-fraction distribution that
    justifies the `min_valid_fraction` threshold (FR-P3.4)."""
    vf = cube["lst_valid_fraction"]
    dates = [date.fromisoformat(s) for s in cube["dates"]]
    by_month = {}
    for m in range(1, 13):
        sel = np.array([d.month == m for d in dates])
        if sel.any():
            by_month[m] = round(float(np.mean(vf[sel])), 3)
    return {
        "dates": len(dates),
        "land_cells": int(cube["land"].sum()),
        "usable_dates": int(cube["usable"].sum()),
        "missing_files": {k: int((~cube[f"present_{k}"]).sum()) for k in ("lst", "reflectance", "era5")},
        "lst_valid_fraction_quantiles": {q: round(float(np.quantile(vf, q)), 3)
                                         for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        "mean_lst_valid_fraction_by_month": by_month,
        "lst_c_range": [round(float(np.nanmin(cube["lst_c"])), 1), round(float(np.nanmax(cube["lst_c"])), 1)]
        if np.isfinite(cube["lst_c"]).any() else None,
        "ndvi_range": [round(float(np.nanmin(cube["ndvi"])), 3), round(float(np.nanmax(cube["ndvi"])), 3)]
        if np.isfinite(cube["ndvi"]).any() else None,
    }


def save_cube(cube: dict, grid, project, path) -> Path:
    """Compressed .npz plus a manifest describing variables, units, rules and grid (FR-P3.6)."""
    from dataclasses import asdict

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **cube)
    units = {"lst_c": "degC", "ndvi": "index", "ndbi": "index", "ndwi": "index",
             "refl_clear_subpixels": "count of clear 500 m pixels (0-4)",
             "lst_valid_fraction": "fraction of land cells", "usable": "bool"}
    units.update({k: ("degC" if k.endswith("_c") else "m/s" if "wind" in k else "")
                  for k in cube if k.startswith("era5_")})
    manifest = {
        "file": path.name,
        "grid": asdict(grid),
        "variables": {k: {"shape": list(np.shape(v)), "dtype": str(np.asarray(v).dtype),
                          "unit": units.get(k, "")} for k, v in cube.items()},
        "rules": {"modis_qc": asdict(project.modis.qc), "reflectance": asdict(project.preprocess.reflectance),
                  "max_water_fraction": project.preprocess.max_water_fraction,
                  "min_valid_fraction": project.preprocess.min_valid_fraction},
        "gap_filling": "none; NaN marks missing or rejected observations",
        "summary": summarize(cube),
    }
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, default=str),
                                                  encoding="utf-8")
    return path


def load_cube(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
