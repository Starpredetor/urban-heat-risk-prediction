"""Phase 4: generate environmental features (same code as notebooks/04_feature_engineering.ipynb).

    .venv\\Scripts\\python.exe scripts\\build_features.py

First fetches SRTM elevation once (Earth Engine, needs internet), then builds every
feature layer into data/processed/features/. Re-running skips finished files.
Logs to outputs/logs/phase4_features_<timestamp>.log.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from urban_heat import acquisition as acq, config, features, logs  # noqa: E402
from urban_heat.grid import load_grids  # noqa: E402

if __name__ == "__main__":
    log = logs.setup("phase4_features")
    config.ensure_dirs()
    try:
        if not acq.elevation_path().exists():
            log.info("downloading SRTM elevation (once)")
            ee = acq.init_ee()
            g30 = load_grids(config.DATA / "grids.json")["landsat_30m"]
            log.info(acq.brief(acq.download_elevation(ee, g30, log=log.info)))
        features.run_all()
    except Exception:
        log.exception("Phase 4 failed")
        raise
    files = sorted(config.FEATURES.glob("*"))
    log.info("data/processed/features: %d files, %.1f MB", len(files), sum(f.stat().st_size for f in files) / 1e6)
