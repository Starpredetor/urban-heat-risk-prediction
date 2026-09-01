"""STEP 1 - Download satellite images and build the dataset.

    python scripts/build_dataset.py

Downloads Landsat and Sentinel-2 images of the city, removes clouds, and saves
one dataset file: data/cube.nc
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uhrm.config import load_settings
from uhrm.cube import build_cube, save_cube


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("rasterio", "urllib3", "odc", "aiohttp"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    settings = load_settings()
    cube = build_cube(settings)
    save_cube(cube, settings.cube_path)

    good = float((cube["n_clear"].values > 0).mean()) * 100
    print()
    print(f"  Done. {cube.sizes['year']} years x {len(cube['layer'])} layers "
          f"x {cube.sizes['y']} x {cube.sizes['x']} pixels")
    print(f"  {good:.1f}% of pixels had at least one clear view")
    print(f"  Saved to {settings.cube_path}")


if __name__ == "__main__":
    main()
