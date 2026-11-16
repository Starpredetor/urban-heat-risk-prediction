"""Download MODIS composites to disk and catalogue them (SRS FR-P2.6 to FR-P2.8, FR-X.4).

Raw files are written once and never modified. A finished file is skipped on
re-run, so an interrupted run (a Colab disconnect) just continues. Failed
fetches retry with exponential backoff, then are logged and skipped
(NFR-3.2) rather than stopping the run.
"""

from __future__ import annotations

import csv
import hashlib
import time
from datetime import date, datetime, timezone
from pathlib import Path

from uhrm.backends.base import ModisBackend, ModisComposite
from uhrm.regions import Grid

CATALOG_COLUMNS = ["dataset", "date", "region", "source", "path", "sha256",
                   "exported_utc", "collection"]


def composite_path(out_dir: Path, composite_date: date) -> Path:
    return out_dir / f"MOD11A2_{composite_date:%Y%m%d}.tif"


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


def acquire_modis(backend: ModisBackend, grid: Grid, dates, out_dir, catalog_path, *,
                  region: str = "model_domain", retries: int = 3, base_delay: float = 2.0,
                  writer=write_geotiff, sleep=time.sleep, log=print) -> dict:
    """Fetch each date, write it, catalogue it. Returns {fetched, skipped, missing, failed}."""
    out_dir, catalog_path = Path(out_dir), Path(catalog_path)
    summary = {"fetched": [], "skipped": [], "missing": [], "failed": []}

    for d in dates:
        path = composite_path(out_dir, d)
        if path.exists() and path.stat().st_size > 0:
            summary["skipped"].append(d)
            continue

        comp = None
        for attempt in range(retries):
            try:
                comp = backend.fetch_composite(d, grid)
                break
            except LookupError:
                summary["missing"].append(d)          # does not exist: retrying is pointless
                break
            except Exception as e:                    # transient: back off and retry
                log(f"{d}: attempt {attempt + 1}/{retries} failed: {e}")
                if attempt + 1 < retries:
                    sleep(base_delay * 2 ** attempt)
        else:
            summary["failed"].append(d)
            continue
        if comp is None:
            continue

        writer(path, comp, grid)
        _append_catalog(catalog_path, {
            "dataset": "MOD11A2", "date": d.isoformat(), "region": region,
            "source": comp.source, "path": str(path), "sha256": _sha256(path),
            "exported_utc": datetime.now(timezone.utc).isoformat(),
            "collection": comp.meta.get("collection", ""),
        })
        summary["fetched"].append(d)

    return summary
