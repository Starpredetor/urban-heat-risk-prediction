"""Checks that tell a wrong map from a correct-but-sparse one.

registration_offset  Is the MODIS data aligned with the land? Shifts the "has an
                     LST value" mask against the WorldCover land mask and finds
                     the best agreement. A correct grid peaks at (0, 0).
qc_breakdown         Why were cells in a region rejected? Counts each QC reason,
                     and how many cells alternative rules would keep.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from uhrm.modis import QcRule, lst_error_class, mandatory_qa, quality_mask


def registration_offset(has_value, land, max_shift: int = 3) -> dict:
    """Shift `has_value` by up to `max_shift` cells in each direction and score its
    agreement with `land` (fraction of overlapping cells that match).

    MODIS has no LST over open sea, so the coastline is in both masks; if the data
    were misplaced, a non-zero shift would match better than (0, 0).
    """
    v, l = np.asarray(has_value, bool), np.asarray(land, bool)
    if v.shape != l.shape:
        raise ValueError("masks must have the same shape")
    H, W = v.shape
    scores = {}
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            # compare v[y, x] with l[y - dy, x - dx]
            vy, ly = slice(max(dy, 0), H + min(dy, 0)), slice(max(-dy, 0), H + min(-dy, 0))
            vx, lx = slice(max(dx, 0), W + min(dx, 0)), slice(max(-dx, 0), W + min(-dx, 0))
            scores[(dy, dx)] = float(np.mean(v[vy, vx] == l[ly, lx]))
    # A straight stretch of coast cannot reveal a shift along itself, so several
    # shifts can tie. Among (near-)ties, prefer the smallest shift.
    top = max(scores.values())
    ties = [s for s, v in scores.items() if v >= top - 1e-6]
    best = min(ties, key=lambda s: (s[0] ** 2 + s[1] ** 2, s))
    return {"best_shift_rows_cols": best, "best_agreement": round(top, 4),
            "agreement_at_zero": round(scores[(0, 0)], 4), "aligned": best == (0, 0)}


def qc_breakdown(lst_dn, qc, region, rule: QcRule) -> dict:
    """Counts of QC outcomes for cells inside `region`, plus what alternative rules keep."""
    region = np.asarray(region, bool)
    dn = np.asarray(lst_dn)[region].astype(np.int64)
    q = np.asarray(qc)[region]
    has_value = (dn >= rule.valid_dn_range[0]) & (dn <= rule.valid_dn_range[1])
    mand, err = mandatory_qa(q), lst_error_class(q)
    names_m = ["00 good", "01 other quality", "10 cloud", "11 other reason"]
    names_e = ["<=1K", "<=2K", "<=3K", ">3K"]
    keep = lambda r: int(quality_mask(dn, q, r).sum())  # noqa: E731
    return {
        "cells": int(region.sum()),
        "with_lst_value": int(has_value.sum()),
        "mandatory_qa (cells with a value)": {names_m[i]: int(((mand == i) & has_value).sum()) for i in range(4)},
        "lst_error (cells with a value)": {names_e[i]: int(((err == i) & has_value).sum()) for i in range(4)},
        "kept_by_current_rule": keep(rule),
        "kept_if_other_quality_allowed": keep(replace(rule, require_good_mandatory_qa=False)),
        "kept_if_other_quality_and_error_le_3K": keep(replace(rule, require_good_mandatory_qa=False,
                                                              max_lst_error_class=2)),
    }
