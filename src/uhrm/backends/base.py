"""The interface every MODIS backend implements (SRS FR-X.3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Protocol

import numpy as np

from uhrm.regions import Grid


@dataclass
class ModisComposite:
    """One MOD11A2 8-day composite on the model grid, as raw digital numbers.

    Raw values are kept untouched (SRS FR-P2.6); scaling and quality masking
    happen later in `uhrm.modis`.
    """

    date: date                  # composite start date
    lst_dn: np.ndarray          # uint16 (height, width); 0 = fill / masked
    qc: np.ndarray              # uint16 (height, width), QC_Day byte
    source: str                 # e.g. "earthengine:MODIS/061/MOD11A2"
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.lst_dn.shape != self.qc.shape:
            raise ValueError("lst_dn and qc must have the same shape")


class ModisBackend(Protocol):
    def list_composites(self, start: date, end: date) -> list[date]:
        """Composite start dates available in [start, end)."""

    def fetch_composite(self, composite_date: date, grid: Grid) -> ModisComposite:
        """One composite on `grid`. Raises LookupError if that date does not exist."""
