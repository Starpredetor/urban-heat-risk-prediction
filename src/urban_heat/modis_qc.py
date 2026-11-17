"""MODIS MOD11A2 quality filtering and temperature scaling (SRS FR-3.1 to FR-3.4,
used first by the first-milestone map).

QC_Day bits (MOD11A2 v6.1 product specification):
    bits 0-1  mandatory QA   00 LST produced, good quality; 01 other quality;
                             10 not produced (cloud); 11 not produced (other)
    bits 6-7  LST error      00 <=1 K, 01 <=2 K, 10 <=3 K, 11 >3 K
Bits 2-5 are not used: documentation sources disagree on them.
Default rule (SRS assumption A4): keep pixels where LST was produced (QA 00 or 01)
with LST error <= 2 K. "Other quality" (01) is kept because MODIS assigns it to
most dense urban cells (uncertain emissivity): on 2024-04-06 it covered 59% of
city cells against 13% elsewhere, so dropping it would blank out both cities.
The 2 K error bound applies to every kept pixel either way.

Scale: LST_K = DN * 0.02; DN 0 means no data; valid DN 7500-65535.
"""

from __future__ import annotations

import numpy as np

LST_SCALE = 0.02
VALID_DN = (7500, 65535)


def mandatory_qa(qc) -> np.ndarray:
    return np.asarray(qc).astype(np.int64) & 0b11


def lst_error_class(qc) -> np.ndarray:
    return (np.asarray(qc).astype(np.int64) >> 6) & 0b11


def quality_mask(dn, qc, *, require_good_qa: bool = False, max_error_class: int = 1) -> np.ndarray:
    """True where the pixel has an LST value that passes the QC rule."""
    dn = np.asarray(dn).astype(np.int64)
    ok = (dn >= VALID_DN[0]) & (dn <= VALID_DN[1])
    m = mandatory_qa(qc)
    ok &= m <= 1                                  # LST produced (00 good or 01 other quality)
    if require_good_qa:
        ok &= m == 0
    return ok & (lst_error_class(qc) <= max_error_class)


def lst_celsius(dn, qc, **rule) -> np.ndarray:
    """Quality-filtered daytime LST in degrees C; NaN where rejected or no data."""
    c = np.asarray(dn, dtype=np.float64) * LST_SCALE - 273.15
    return np.where(quality_mask(dn, qc, **rule), c, np.nan).astype(np.float32)


def qc_summary(dn, qc, region) -> dict:
    """Why cells in `region` were kept or rejected."""
    region = np.asarray(region, bool)
    d, q = np.asarray(dn)[region].astype(np.int64), np.asarray(qc)[region]
    has = (d >= VALID_DN[0]) & (d <= VALID_DN[1])
    m, e = mandatory_qa(q), lst_error_class(q)
    return {"cells": int(region.sum()), "no_lst_value": int((~has).sum()),
            "qa_good": int((has & (m == 0)).sum()), "qa_other_quality": int((has & (m == 1)).sum()),
            "error_le_1K": int((has & (e == 0)).sum()), "error_le_2K": int((has & (e == 1)).sum()),
            "error_gt_2K": int((has & (e >= 2)).sum()),
            "kept": int(quality_mask(d, q).sum())}
