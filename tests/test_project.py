"""Project configuration loading and validation (SRS FR-X.2)."""

from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from uhrm.project import check, load_project


def test_offline_mode_loads_with_both_cities():
    p = load_project("offline")
    assert [a.slug for a in p.study_areas] == ["mumbai", "navi_mumbai"]
    assert p.mode.backend == "local_files"
    assert p.mode.storage_root.name == "data"
    assert p.grid.crs == "EPSG:32643"
    assert p.modis.start_date == date(2001, 1, 1)


def test_colab_mode_uses_earth_engine_and_drive():
    p = load_project("colab")
    assert p.mode.backend == "earthengine"
    assert "drive" in str(p.mode.storage_root).lower()


def test_modes_share_the_same_science_config():
    a, b = load_project("offline"), load_project("colab")
    assert a.study_areas == b.study_areas and a.grid == b.grid and a.modis == b.modis


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        load_project("cluster")


def test_paths_resolve_under_storage_root(tmp_path):
    p = load_project("offline", storage_root=tmp_path)
    assert p.path("grid.json") == tmp_path / "grid.json"
    assert p.boundary_path(p.area("mumbai")) == tmp_path / "boundaries" / "mumbai.geojson"
    with pytest.raises(KeyError):
        p.area("delhi")


def test_check_rejects_duplicate_slugs():
    p = load_project("offline")
    dup = replace(p, study_areas=[p.study_areas[0], p.study_areas[0]])
    with pytest.raises(ValueError, match="slug"):
        check(dup)


def test_check_rejects_wrong_modis_scale():
    p = load_project("offline")
    with pytest.raises(ValueError, match="0.02"):
        check(replace(p, modis=replace(p.modis, scale=0.03)))


def test_check_rejects_milestone_date_outside_range():
    p = load_project("offline")
    with pytest.raises(ValueError, match="first_milestone_date"):
        check(replace(p, modis=replace(p.modis, first_milestone_date=date(1999, 1, 1))))


def test_check_rejects_bad_overlap_threshold():
    p = load_project("offline")
    with pytest.raises(ValueError, match="min_cell_overlap"):
        check(replace(p, grid=replace(p.grid, min_cell_overlap=0.0)))
