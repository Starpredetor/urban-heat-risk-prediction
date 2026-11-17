"""Logging and progress bars for every stage.

- Log lines go to the console and to outputs/logs/<stage>_<timestamp>.log.
- `progress(...)` shows a live bar (percentage, count, rate, time remaining). Console
  log lines are printed above the bars so neither garbles the other, and the bar's
  progress is also written to the log file every 10%.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from urban_heat import config

LOG_DIR = config.OUTPUTS / "logs"
FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_progress_log = logging.getLogger("progress")


class _TqdmConsoleHandler(logging.StreamHandler):
    """Writes log lines through tqdm so active progress bars stay intact."""

    def emit(self, record):
        try:
            from tqdm import tqdm
            tqdm.write(self.format(record), file=sys.stdout)
        except Exception:
            self.handleError(record)


def setup(stage: str, level: int = logging.INFO) -> logging.Logger:
    """Configure root logging for a stage; returns that stage's logger. Safe to call twice."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"{stage}_{datetime.now():%Y%m%d_%H%M%S}.log"
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):                       # avoid duplicate handlers on re-run
        root.removeHandler(h)
    fmt = logging.Formatter(FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
    for h in (_TqdmConsoleHandler(), logging.FileHandler(path, encoding="utf-8")):
        h.setFormatter(fmt)
        root.addHandler(h)
    for noisy in ("urllib3", "google", "googleapiclient", "rasterio", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    log = logging.getLogger(stage)
    log.info("log file: %s", Path(path).relative_to(config.ROOT))
    return log


class progress:
    """Progress bar for a known number of steps.

        with progress(total=n, desc="MODIS download", unit="file") as bar:
            ...; bar.update(1)

    or wrap an iterable: `for x in progress(items, desc=...)`. The bar shows
    percentage, count, rate and estimated time remaining; every 10% (and at the end)
    a line is also logged so the log file records progress.
    """

    def __init__(self, iterable=None, *, total=None, desc="", unit="item"):
        from tqdm.auto import tqdm
        if total is None and iterable is not None and hasattr(iterable, "__len__"):
            total = len(iterable)
        self._iterable = iterable
        self.total, self.desc = total, desc
        self._bar = tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True, mininterval=0.5,
                         file=sys.stdout, leave=True)
        self._next_log = 0.1

    def update(self, n: int = 1) -> None:
        self._bar.update(n)
        if self.total:
            frac = self._bar.n / self.total
            if frac >= self._next_log or self._bar.n == self.total:
                rate = self._bar.format_dict.get("rate") or 0
                left = (self.total - self._bar.n) / rate if rate else float("nan")
                _progress_log.info("%s: %d/%d (%.0f%%), %.2f per s, ~%.0f s left", self.desc, self._bar.n,
                                   self.total, 100 * frac, rate, left)
                while self._next_log <= frac:
                    self._next_log += 0.1

    def set_postfix(self, **kw) -> None:
        self._bar.set_postfix(kw, refresh=False)

    def close(self) -> None:
        self._bar.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __iter__(self):
        try:
            for item in self._iterable:
                yield item
                self.update(1)
        finally:
            self.close()
