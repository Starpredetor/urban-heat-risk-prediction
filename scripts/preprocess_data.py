"""Phase 3: preprocess all downloaded data (same code as notebooks/03_preprocessing.ipynb).

    .venv\\Scripts\\python.exe scripts\\preprocess_data.py

Reads data/raw/, writes data/processed/. Re-running skips files already processed.
Logs to the console and to outputs/logs/phase3_preprocess_<timestamp>.log.
Landsat scaling/masking runs on the CUDA GPU when available (URBAN_HEAT_DEVICE=cpu forces CPU).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from urban_heat import config, logs, preprocessing  # noqa: E402

if __name__ == "__main__":
    log = logs.setup("phase3_preprocess")
    config.ensure_dirs()
    try:
        preprocessing.run_all()
    except Exception:
        log.exception("Phase 3 failed")
        raise
    for sub in ("masks", "modis", "landsat", "worldcover", "era5"):
        files = list((config.PROCESSED / sub).glob("*.tif"))
        log.info("data/processed/%s: %d files, %.1f MB", sub, len(files), sum(f.stat().st_size for f in files) / 1e6)
    log.info("summaries: data/processed/modis_valid_fraction.csv, data/processed/landsat_clear_fraction.csv")
