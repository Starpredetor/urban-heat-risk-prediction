"""STEP 2 - Train the model and compare it to simple methods.

    python scripts/train.py

Reads data/cube.nc, trains the ConvLSTM, and prints a comparison table.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uhrm.config import load_settings
from uhrm.cube import load_cube
from uhrm.train import run


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = load_settings()

    if not settings.cube_path.exists():
        print(f"No dataset found at {settings.cube_path}")
        print("Run this first:  python scripts/build_dataset.py")
        return 1

    run(load_cube(settings.cube_path), settings, settings.out_dir / "results")
    print(f"\n  Saved to {settings.out_dir / 'results'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
