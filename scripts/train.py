"""STEP 3 - Train one model on every city, and compare it to simple methods.

    python scripts/train.py

Reads every city file in data/cubes/, pools them together, trains the ConvLSTM,
and prints a comparison table.
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

    built = [c for c in settings.cities if settings.cube_path(c).exists()]
    missing = [c.name for c in settings.cities if c not in built]

    if not built:
        print("No city data found. Run this first:")
        print("  python scripts/build_dataset.py")
        return 1
    if missing:
        print(f"Note: no data yet for {', '.join(missing)} - training without them.")

    cubes = [load_cube(settings.cube_path(c)) for c in built]
    run(cubes, settings, settings.out_dir / "results")

    print()
    print(f"  Saved to {settings.out_dir / 'results'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
