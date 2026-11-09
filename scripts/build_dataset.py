"""STEP 1 - Download satellite images and build the dataset.

    python scripts/build_dataset.py                 all cities
    python scripts/build_dataset.py --city Mumbai   just one

Downloads Landsat and Sentinel-2 images, removes clouds, and saves one file
per city into data/cubes/. Already-built cities are skipped unless you pass
--rebuild, so an interrupted run can simply be started again.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uhrm.config import load_settings
from uhrm.cube import build_city_cube, save_cube


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", help="build only this city")
    parser.add_argument("--rebuild", action="store_true",
                        help="rebuild cities that already have a file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("rasterio", "urllib3", "odc", "aiohttp", "botocore"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    settings = load_settings()
    cities = [settings.city(args.city)] if args.city else settings.cities

    print(f"Building {len(cities)} city/cities, "
          f"{settings.start_year}-{settings.end_year}, "
          f"{settings.radius_summary()}")
    print()

    for n, city in enumerate(cities, 1):
        path = settings.cube_path(city)
        if path.exists() and not args.rebuild:
            print(f"[{n}/{len(cities)}] {city.name}: already built, skipping "
                  f"(use --rebuild to redo)")
            continue

        print(f"[{n}/{len(cities)}] {city.name}")
        started = time.time()
        cube = build_city_cube(settings, city)
        save_cube(cube, path)
        good = float((cube["n_clear"].values > 0).mean()) * 100
        print(f"    done in {time.time() - started:.0f}s, "
              f"{good:.1f}% of pixels had a clear view")
        print()

    print("All done. Next:  python scripts/train.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
