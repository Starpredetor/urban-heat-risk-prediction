"""Phase 5 analysis functions on hand-made inputs (SRS FR-5.1 to FR-5.7)."""

from datetime import date

import numpy as np
import pytest

from urban_heat import analysis as an


def test_period_mean_uses_only_valid_composites_in_the_period():
    dates = [date(2020, 3, 5), date(2020, 4, 6), date(2020, 7, 11), date(2021, 4, 7)]
    stack = np.array([[[30.0]], [[np.nan]], [[25.0]], [[40.0]]])
    m, n = an.period_mean(stack, dates, 2020, an.MAM)
    assert m[0, 0] == 30.0 and n[0, 0] == 1           # April composite was missing; July is not MAM
    m, n = an.period_mean(stack, dates, 2020)
    assert m[0, 0] == pytest.approx(27.5) and n[0, 0] == 2
    m, n = an.period_mean(stack, dates, 2019)
    assert np.isnan(m[0, 0]) and n[0, 0] == 0


def test_hotspot_frequency_is_city_relative():
    ids = np.array([[1] * 10 + [2] * 10])
    mam = {}
    for y in range(4):
        m = np.concatenate([np.arange(10) + 30.0, np.arange(10) + 20.0])   # Navi cells cooler overall
        mam[2000 + y] = (m[None, :], None)
    freq, years = an.hotspot_frequency(mam, ids)
    # in each city only the hottest cell (>= 90th percentile) is a hotspot, every year
    assert freq[0, 9] == 1.0 and freq[0, 19] == 1.0
    assert freq[0, 0] == 0.0 and freq[0, 10] == 0.0
    assert years[0, 0] == 4


def test_per_cell_trend_in_degrees_per_decade():
    mam = {2000 + t: (np.array([[30 + 0.05 * t, 30.0]]), None) for t in range(20)}
    trend, n = an.per_cell_trend(mam, min_years=15)
    assert trend[0, 0] == pytest.approx(0.5) and trend[0, 1] == pytest.approx(0.0)
    assert n[0, 0] == 20
    short = {y: v for y, v in list(mam.items())[:10]}
    assert np.isnan(an.per_cell_trend(short, min_years=15)[0][0, 0])     # too few years


def test_relative_change_removes_overall_warming():
    ids = np.array([[1, 1]])
    mam = {}
    for y in range(2001, 2026):
        warm = 2.0 if y >= 2021 else 0.0                                # whole city warms by 2 C
        shift = 1.0 if y >= 2021 else 0.0                               # cell 0 additionally warms 1 C
        mam[y] = (np.array([[30.0 + warm + shift, 30.0 + warm]]), None)
    rel = an.relative_change(mam, ids)
    # city mean change (2.5 C) removed: cell 0 gained +0.5 relative, cell 1 lost 0.5
    assert rel[0, 0] == pytest.approx(0.5) and rel[0, 1] == pytest.approx(-0.5)
