"""Phase 2 bulk download, run from a terminal (same code as notebooks/02_data_acquisition.ipynb).

    .venv\\Scripts\\python.exe scripts\\download_data.py
    .venv\\Scripts\\python.exe scripts\\download_data.py --workers 32 --datasets modis era5
    .venv\\Scripts\\python.exe scripts\\download_data.py --years 2024-2025 --datasets modis

Downloads run in parallel threads. Earth Engine accepts roughly 40 concurrent
requests per project; more threads than that only cause "too many requests"
errors and retries. Finished files are skipped, so an interrupted run resumes
where it stopped (Ctrl+C is safe; half-written files are never left under the
final name).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from urban_heat import acquisition as acq, boundaries as b, config, logs  # noqa: E402
from urban_heat.grid import Grid, load_grids  # noqa: E402

DATASETS = ("modis", "landsat", "worldcover", "era5")


def parse_years(text: str) -> list[int]:
    a, _, z = text.partition("-")
    return list(range(int(a), int(z or a) + 1))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=acq.DEFAULT_WORKERS,
                   help=f"parallel threads (default {acq.DEFAULT_WORKERS})")
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument("--years", default=f"{config.START_YEAR}-{config.END_YEAR}",
                   help="MODIS / ERA5-Land years, e.g. 2001-2025 (Landsat uses config.LANDSAT_YEARS)")
    args = p.parse_args(argv)
    years = parse_years(args.years)
    log = logs.setup("phase2_download")
    info = log.info

    config.ensure_dirs()
    ee = acq.init_ee()
    grids = load_grids(config.DATA / "grids.json")
    g1km, g30 = grids["modis_1km"], grids["landsat_30m"]
    info(f"Earth Engine project {config.EE_PROJECT}; {args.workers} threads; years {years[0]}-{years[-1]}")
    info("end-year check: %s", acq.verify_end_year(ee, config.END_YEAR))

    t0 = time.monotonic()
    if "modis" in args.datasets:
        info("\n[MODIS MOD11A2]")
        info(acq.brief(acq.download_modis(ee, g1km, acq.modis_dates(years[0], years[-1]), workers=args.workers, log=info)))
    if "landsat" in args.datasets:
        info("\n[Landsat 8/9]")
        info(acq.brief(acq.download_landsat(ee, g30, config.LANDSAT_YEARS, workers=args.workers, log=info)))
    if "worldcover" in args.datasets:
        info("\n[ESA WorldCover]")
        f = int(round(g30.res / config.WORLDCOVER_RES_M))
        g10 = Grid(g30.crs, g30.xmin, g30.ymax, config.WORLDCOVER_RES_M, g30.width * f, g30.height * f)
        info(acq.brief(acq.download_worldcover(ee, g10, workers=args.workers, log=info)))
    if "era5" in args.datasets:
        info("\n[ERA5-Land]")
        g_ll = acq.era5_grid(b.load(config.BOUNDARIES / "combined_study_area.geojson").bounds)
        info(acq.brief(acq.download_era5(ee, g_ll, years, workers=args.workers, log=info)))

    info(f"\nfinished in {time.monotonic() - t0:.0f} s")
    for sub in ("modis", "landsat", "worldcover", "era5"):
        files = [f for f in (config.RAW / sub).glob("*") if ".partial" not in f.name]
        info(f"data/raw/{sub}: {len(files)} files, {sum(f.stat().st_size for f in files) / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
