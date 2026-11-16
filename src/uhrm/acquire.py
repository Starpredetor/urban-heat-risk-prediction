"""Download datasets to disk and catalogue them (SRS FR-P2.6 to FR-P2.8, FR-X.4).

Raw files are written once and never modified. A finished file is skipped on
re-run, so an interrupted run (a Colab disconnect) just continues. Failed
fetches retry with exponential backoff, then are logged and skipped
(NFR-3.2) rather than stopping the run.

`acquire_series` is the one loop; MODIS LST, reflectance, ERA5-Land windows and
the static layers are all runs of it with a different fetch function.
"""

from __future__ import annotations

import csv
import hashlib
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from uhrm.backends.base import ModisBackend, ModisComposite, RasterSet
from uhrm.regions import Grid

CATALOG_COLUMNS = ["dataset", "date", "region", "source", "path", "sha256",
                   "exported_utc", "collection"]

COMPOSITES_PER_YEAR = 46   # MOD11A2 / MOD09A1 8-day composites start at day 1, 9, ..., 361


# --- MODIS 8-day calendar -----------------------------------------------------

def expected_composite_dates(start_year: int, end_year: int) -> list[date]:
    """Every 8-day composite start date from `start_year` to `end_year` inclusive."""
    return [date(y, 1, 1) + timedelta(days=8 * i)
            for y in range(start_year, end_year + 1) for i in range(COMPOSITES_PER_YEAR)]


def composite_window(d: date) -> tuple[date, date]:
    """[start, end) covered by the composite starting on `d`. The last composite
    of a year is shorter: it ends on 1 January."""
    return d, min(d + timedelta(days=8), date(d.year + 1, 1, 1))


def dated_path(out_dir, prefix: str, d: date) -> Path:
    return Path(out_dir) / f"{prefix}_{d:%Y%m%d}.tif"


def composite_path(out_dir, d: date) -> Path:
    return dated_path(out_dir, "MOD11A2", d)


def present_dates(out_dir, prefix: str) -> set[date]:
    """Dates for which a finished file `<prefix>_YYYYMMDD.tif` exists."""
    found = set()
    for p in Path(out_dir).glob(f"{prefix}_*.tif"):
        stamp = p.stem.split("_")[-1]
        if p.stat().st_size > 0 and len(stamp) == 8 and stamp.isdigit():
            found.add(date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:])))
    return found


def coverage_report(expected: list[date], present: set[date]) -> dict[int, dict]:
    """Per year: how many expected composites are on disk, and which are missing
    (FR-P2 completion criterion)."""
    report: dict[int, dict] = {}
    for d in expected:
        row = report.setdefault(d.year, {"expected": 0, "found": 0, "missing": []})
        row["expected"] += 1
        if d in present:
            row["found"] += 1
        else:
            row["missing"].append(d)
    return report


# --- writers ------------------------------------------------------------------

def write_geotiff(path: Path, comp: ModisComposite, grid: Grid) -> None:
    """Two-band uint16 GeoTIFF: band 1 = LST DN, band 2 = QC_Day. Needs rasterio."""
    import rasterio
    from rasterio import Affine

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=grid.height, width=grid.width,
                       count=2, dtype="uint16", crs=grid.crs,
                       transform=Affine(*grid.transform), compress="deflate") as dst:
        dst.write(comp.lst_dn.astype("uint16"), 1)
        dst.write(comp.qc.astype("uint16"), 2)
        dst.set_band_description(1, "LST_Day_1km (DN, scale 0.02 K, 0 = fill)")
        dst.set_band_description(2, "QC_Day")
        dst.update_tags(composite_date=comp.date.isoformat(), source=comp.source,
                        exported_utc=datetime.now(timezone.utc).isoformat(), **comp.meta)


def write_bands_geotiff(path: Path, rs: RasterSet, grid: Grid, dtype: str = "float32") -> None:
    """Multi-band GeoTIFF with one band per RasterSet entry, named in the band
    descriptions. Integer dtypes store masked values as 0. Needs rasterio."""
    import rasterio
    from rasterio import Affine

    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(rs.bands)
    with rasterio.open(path, "w", driver="GTiff", height=grid.height, width=grid.width,
                       count=len(names), dtype=dtype, crs=grid.crs,
                       transform=Affine(*grid.transform), compress="deflate") as dst:
        for i, name in enumerate(names, start=1):
            arr = np.asarray(rs.bands[name])
            if np.issubdtype(np.dtype(dtype), np.integer):
                arr = np.nan_to_num(arr.astype(np.float64))
            dst.write(arr.astype(dtype), i)
            dst.set_band_description(i, name)
        tags = {"source": rs.source, "exported_utc": datetime.now(timezone.utc).isoformat(), **rs.meta}
        if rs.date:
            tags["composite_date"] = rs.date.isoformat()
        dst.update_tags(**{k: str(v) for k, v in tags.items()})


# --- the loop -----------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _append_catalog(catalog_path: Path, row: dict) -> None:
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    new = not catalog_path.exists()
    with open(catalog_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CATALOG_COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)


def _label(key) -> str:
    return key.isoformat() if hasattr(key, "isoformat") else str(key)


def acquire_series(fetch, keys, path_for, writer, catalog_path, *, dataset: str,
                   region: str = "model_domain", retries: int = 3, base_delay: float = 2.0,
                   sleep=time.sleep, log=print, progress_every: int = 25) -> dict:
    """For each key: skip if its file exists, else fetch, write, and catalogue.

    `fetch(key)` returns an object with `.source` and `.meta`; it raises
    LookupError when that key does not exist (recorded, not retried) and any
    other exception for a transient failure (retried with backoff, then
    recorded as failed). `writer(path, item)` writes the file.
    Returns {"fetched", "skipped", "missing", "failed"} lists of keys.
    """
    catalog_path = Path(catalog_path)
    summary = {"fetched": [], "skipped": [], "missing": [], "failed": []}
    started = time.monotonic()

    for key in keys:
        path = Path(path_for(key))
        if path.exists() and path.stat().st_size > 0:
            summary["skipped"].append(key)
            continue

        item = None
        for attempt in range(retries):
            try:
                item = fetch(key)
                break
            except LookupError:
                summary["missing"].append(key)        # does not exist: retrying is pointless
                break
            except Exception as e:                    # transient: back off and retry
                log(f"{dataset} {_label(key)}: attempt {attempt + 1}/{retries} failed: {e}")
                if attempt + 1 < retries:
                    sleep(base_delay * 2 ** attempt)
        else:
            summary["failed"].append(key)
            continue
        if item is None:
            continue

        writer(path, item)
        _append_catalog(catalog_path, {
            "dataset": dataset, "date": _label(key), "region": region,
            "source": item.source, "path": str(path), "sha256": _sha256(path),
            "exported_utc": datetime.now(timezone.utc).isoformat(),
            "collection": item.meta.get("collection", ""),
        })
        summary["fetched"].append(key)
        if progress_every and len(summary["fetched"]) % progress_every == 0:
            log(f"{dataset}: {len(summary['fetched'])} fetched, latest {_label(key)}, "
                f"{time.monotonic() - started:.0f}s elapsed")

    return summary


def acquire_modis(backend: ModisBackend, grid: Grid, dates, out_dir, catalog_path, *,
                  region: str = "model_domain", retries: int = 3, base_delay: float = 2.0,
                  writer=write_geotiff, sleep=time.sleep, log=print,
                  progress_every: int = 25) -> dict:
    """MOD11A2 LST composites on the model grid."""
    out_dir = Path(out_dir)
    return acquire_series(
        lambda d: backend.fetch_composite(d, grid), dates,
        lambda d: composite_path(out_dir, d), lambda p, c: writer(p, c, grid), catalog_path,
        dataset="MOD11A2", region=region, retries=retries, base_delay=base_delay,
        sleep=sleep, log=log, progress_every=progress_every)


def brief(summary: dict) -> str:
    """One-line summary of an acquire_series result, listing anything that went wrong."""
    line = ", ".join(f"{k} {len(v)}" for k, v in summary.items())
    for problem in ("missing", "failed"):
        if summary[problem]:
            shown = ", ".join(_label(k) for k in summary[problem][:8])
            more = "..." if len(summary[problem]) > 8 else ""
            line += f"\n  {problem}: {shown}{more}"
    return line
