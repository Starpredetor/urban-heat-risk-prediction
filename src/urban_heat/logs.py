"""Logging for every stage: to the console and to outputs/logs/<stage>_<timestamp>.log."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from urban_heat import config

LOG_DIR = config.OUTPUTS / "logs"
FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup(stage: str, level: int = logging.INFO) -> logging.Logger:
    """Configure root logging for a stage; returns that stage's logger. Safe to call twice."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"{stage}_{datetime.now():%Y%m%d_%H%M%S}.log"
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):                       # avoid duplicate handlers on re-run
        root.removeHandler(h)
    fmt = logging.Formatter(FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(path, encoding="utf-8")):
        h.setFormatter(fmt)
        root.addHandler(h)
    for noisy in ("urllib3", "google", "googleapiclient", "rasterio", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    log = logging.getLogger(stage)
    log.info("log file: %s", Path(path).relative_to(config.ROOT))
    return log
