"""Acquisition loop behaviour with a fake backend: no network, no Earth Engine (SRS FR-X.3)."""

import csv
from datetime import date

import numpy as np

from uhrm.acquire import CATALOG_COLUMNS, acquire_modis, composite_path
from uhrm.backends.base import ModisComposite
from uhrm.regions import Grid

GRID = Grid(crs="EPSG:32643", xmin=0.0, ymax=2000.0, res=1000.0, width=2, height=2)
D1, D2, D3 = date(2024, 4, 5), date(2024, 4, 13), date(2024, 4, 21)


class FakeBackend:
    """Fails the first `fail_first` calls per date; dates in `absent` do not exist."""

    def __init__(self, fail_first=0, absent=(), always_fail=()):
        self.fail_first, self.absent, self.always_fail = fail_first, set(absent), set(always_fail)
        self.calls = {}

    def list_composites(self, start, end):
        return [D1, D2, D3]

    def fetch_composite(self, composite_date, grid):
        n = self.calls[composite_date] = self.calls.get(composite_date, 0) + 1
        if composite_date in self.absent:
            raise LookupError("no such composite")
        if composite_date in self.always_fail or n <= self.fail_first:
            raise ConnectionError("temporary failure")
        z = np.full(grid.shape, 15000, dtype=np.uint16)
        return ModisComposite(date=composite_date, lst_dn=z, qc=np.zeros_like(z),
                              source="fake", meta={"collection": "FAKE/1"})


def fake_writer(path, comp, grid):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(comp.lst_dn.tobytes())


def run(backend, tmp_path, dates=(D1, D2), sleeps=None, **kw):
    sleeps = [] if sleeps is None else sleeps
    summary = acquire_modis(backend, GRID, dates, tmp_path / "raw", tmp_path / "catalog.csv",
                            writer=fake_writer, sleep=sleeps.append, log=lambda *_: None, **kw)
    return summary, sleeps


def read_catalog(tmp_path):
    with open(tmp_path / "catalog.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_fetches_writes_and_catalogues(tmp_path):
    summary, _ = run(FakeBackend(), tmp_path)
    assert summary["fetched"] == [D1, D2]
    assert composite_path(tmp_path / "raw", D1).exists()
    rows = read_catalog(tmp_path)
    assert [r["date"] for r in rows] == ["2024-04-05", "2024-04-13"]
    assert set(rows[0]) == set(CATALOG_COLUMNS)
    assert len(rows[0]["sha256"]) == 64 and rows[0]["collection"] == "FAKE/1"


def test_rerun_skips_finished_files_and_does_not_duplicate_catalogue(tmp_path):
    run(FakeBackend(), tmp_path)
    backend = FakeBackend()
    summary, _ = run(backend, tmp_path)
    assert summary["skipped"] == [D1, D2] and summary["fetched"] == []
    assert backend.calls == {}                        # never asked the backend again
    assert len(read_catalog(tmp_path)) == 2


def test_transient_failure_is_retried_with_exponential_backoff(tmp_path):
    backend = FakeBackend(fail_first=2)
    summary, sleeps = run(backend, tmp_path, dates=(D1,), base_delay=2.0)
    assert summary["fetched"] == [D1]
    assert backend.calls[D1] == 3
    assert sleeps == [2.0, 4.0]


def test_persistent_failure_is_logged_not_raised_and_run_continues(tmp_path):
    backend = FakeBackend(always_fail=[D1])
    summary, sleeps = run(backend, tmp_path, dates=(D1, D2), retries=3)
    assert summary["failed"] == [D1]
    assert summary["fetched"] == [D2]                 # the next date still ran
    assert backend.calls[D1] == 3
    assert sleeps == [2.0, 4.0]                       # no pointless sleep after the last attempt
    assert not composite_path(tmp_path / "raw", D1).exists()


def test_missing_composite_is_recorded_and_not_retried(tmp_path):
    backend = FakeBackend(absent=[D1])
    summary, sleeps = run(backend, tmp_path, dates=(D1, D2))
    assert summary["missing"] == [D1] and summary["fetched"] == [D2]
    assert backend.calls[D1] == 1 and sleeps == []
