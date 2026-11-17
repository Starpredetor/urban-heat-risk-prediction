"""Phase 5: historical urban heat analysis (same code as notebooks/05_historical_analysis.ipynb).

    .venv\\Scripts\\python.exe scripts\\historical_analysis.py

Writes maps to outputs/historical_maps/ and tables to data/processed/analysis/.
The figures are drawn by the notebook. Logs to outputs/logs/phase5_analysis_<timestamp>.log.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from urban_heat import analysis, config, logs  # noqa: E402

if __name__ == "__main__":
    log = logs.setup("phase5_analysis")
    config.ensure_dirs()
    try:
        analysis.run_all()
    except Exception:
        log.exception("Phase 5 failed")
        raise
