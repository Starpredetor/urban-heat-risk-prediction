"""MODIS MOD11A2 quality filtering and scaling (SRS FR-P3.1).

The product stores land-surface temperature as an integer (DN) with a scale
factor of 0.02 kelvin, plus a QC byte. A DN of 0 is the fill value.

QC_Day bit layout used here (NASA MOD11A2 v6.1 specification):

    bits 0-1  mandatory QA   00 LST produced, good quality
                             01 LST produced, other quality
                             10 not produced, cloud effects
                             11 not produced, other reasons
    bits 6-7  LST error      00 <=1 K   01 <=2 K   10 <=3 K   11 >3 K

Bits 2-5 are not used: documentation sources disagree on their meaning.

Note on masked pixels: Earth Engine returns 0 for masked pixels in BOTH bands,
and QC 0 reads as "good quality". The DN valid range therefore has to be part
of the quality mask, or masked pixels would look valid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LST_SCALE = 0.02
KELVIN = 273.15


@dataclass(frozen=True)
class QcRule:
    """Which pixels to trust. Defaults match config/project.yaml."""

    require_good_mandatory_qa: bool = True
    max_lst_error_class: int = 1               # 0 <=1 K, 1 <=2 K, 2 <=3 K, 3 any
    valid_dn_range: tuple[int, int] = (7500, 65535)
    plausible_celsius: tuple[float, float] = (0.0, 75.0)

    def __post_init__(self):
        if not 0 <= self.max_lst_error_class <= 3:
            raise ValueError("max_lst_error_class must be 0..3")
        lo, hi = self.valid_dn_range
        if not 0 <= lo < hi:
            raise ValueError("valid_dn_range must be increasing")
        if not self.plausible_celsius[0] < self.plausible_celsius[1]:
            raise ValueError("plausible_celsius must be increasing")


def mandatory_qa(qc) -> np.ndarray:
    """Bits 0-1 of QC_Day."""
    return np.asarray(qc).astype(np.uint16) & 0b11


def lst_error_class(qc) -> np.ndarray:
    """Bits 6-7 of QC_Day."""
    return (np.asarray(qc).astype(np.uint16) >> 6) & 0b11


def quality_mask(lst_dn, qc, rule: QcRule) -> np.ndarray:
    """True where the observation is trustworthy under `rule`."""
    dn = np.asarray(lst_dn).astype(np.int64)
    lo, hi = rule.valid_dn_range
    ok = (dn >= lo) & (dn <= hi)
    if rule.require_good_mandatory_qa:
        ok &= mandatory_qa(qc) == 0
    ok &= lst_error_class(qc) <= rule.max_lst_error_class
    return ok


def dn_to_celsius(lst_dn) -> np.ndarray:
    """LST_C = DN * 0.02 - 273.15 (no masking)."""
    return (np.asarray(lst_dn, dtype=np.float64) * LST_SCALE - KELVIN).astype(np.float32)


def process_composite(lst_dn, qc, rule: QcRule) -> np.ndarray:
    """Quality-masked daytime LST in degrees Celsius; NaN where invalid.

    Invalid pixels are never filled here (SRS FR-P3.4).
    """
    valid = quality_mask(lst_dn, qc, rule)
    celsius = dn_to_celsius(lst_dn)
    lo, hi = rule.plausible_celsius
    valid &= (celsius >= lo) & (celsius <= hi)
    return np.where(valid, celsius, np.nan).astype(np.float32)
