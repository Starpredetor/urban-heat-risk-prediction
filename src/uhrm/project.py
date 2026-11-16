"""Project configuration for UHRM v2 (SRS v2.1).

One code base, two modes (FR-X.2). `config/project.yaml` holds the science
choices; `config/modes/<mode>.yaml` says where the code runs (Colab or offline)
and where data lives. Nothing else should hard-code a path or a backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath

import yaml

from uhrm.modis import QcRule

ROOT = Path(__file__).resolve().parents[2]
MODES = ("colab", "offline")
BACKENDS = ("earthengine", "planetary_computer", "local_files")


@dataclass(frozen=True)
class StudyArea:
    name: str
    slug: str
    boundary: str            # relative to the storage root


@dataclass(frozen=True)
class GridConfig:
    crs: str
    model_resolution_m: int
    domain_buffer_km: float
    min_cell_overlap: float


@dataclass(frozen=True)
class ModisConfig:
    collection: str
    start_date: date
    end_date: date
    first_milestone_date: date
    scale: float
    qc: QcRule


@dataclass(frozen=True)
class Mode:
    name: str
    backend: str
    storage_root: Path
    ee_project: str | None


@dataclass(frozen=True)
class Project:
    study_areas: list[StudyArea]
    grid: GridConfig
    modis: ModisConfig
    mode: Mode

    def path(self, *parts: str) -> Path:
        """A path under the storage root (Drive in Colab, data/ offline)."""
        return self.mode.storage_root.joinpath(*parts)

    def boundary_path(self, area: StudyArea) -> Path:
        return self.path(area.boundary)

    def area(self, slug: str) -> StudyArea:
        for a in self.study_areas:
            if a.slug == slug:
                return a
        raise KeyError(f"no study area {slug!r}; have {[a.slug for a in self.study_areas]}")


def _as_date(value, name: str) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as e:
        raise ValueError(f"{name}: {value!r} is not a date (YYYY-MM-DD)") from e


def _storage_root(value: str) -> Path:
    # A Colab path like /content/drive/... is absolute on Linux but not on
    # Windows, so accept either notion of absolute.
    if PurePosixPath(value).is_absolute() or Path(value).is_absolute():
        return Path(value)
    return ROOT / value


def load_project(mode: str = "offline", *, config_path=None, storage_root=None) -> Project:
    """Load config/project.yaml plus config/modes/<mode>.yaml and validate both."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    cfg_path = Path(config_path) if config_path else ROOT / "config" / "project.yaml"
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    mode_raw = yaml.safe_load((ROOT / "config" / "modes" / f"{mode}.yaml").read_text(encoding="utf-8"))

    areas = [StudyArea(name=a["name"], slug=a["slug"], boundary=a["boundary"])
             for a in raw["study_areas"]]
    g = raw["grid"]
    m = raw["modis"]
    q = m["qc"]

    project = Project(
        study_areas=areas,
        grid=GridConfig(crs=g["crs"], model_resolution_m=int(g["model_resolution_m"]),
                        domain_buffer_km=float(g["domain_buffer_km"]),
                        min_cell_overlap=float(g["min_cell_overlap"])),
        modis=ModisConfig(
            collection=m["collection"],
            start_date=_as_date(m["start_date"], "modis.start_date"),
            end_date=_as_date(m["end_date"], "modis.end_date"),
            first_milestone_date=_as_date(m["first_milestone_date"], "modis.first_milestone_date"),
            scale=float(m["scale"]),
            qc=QcRule(require_good_mandatory_qa=bool(q["require_good_mandatory_qa"]),
                      max_lst_error_class=int(q["max_lst_error_class"]),
                      valid_dn_range=tuple(q["valid_dn_range"]),
                      plausible_celsius=tuple(q["plausible_celsius"]))),
        mode=Mode(name=mode, backend=mode_raw["backend"],
                  storage_root=Path(storage_root) if storage_root else _storage_root(mode_raw["storage_root"]),
                  ee_project=mode_raw.get("ee_project")),
    )
    check(project)
    return project


def check(p: Project) -> None:
    """Catch configuration mistakes that would otherwise give wrong results quietly."""
    if not p.study_areas:
        raise ValueError("at least one study area is required")
    slugs = [a.slug for a in p.study_areas]
    if len(slugs) != len(set(slugs)):
        raise ValueError("two study areas share a slug")
    if p.mode.backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {p.mode.backend!r}")
    if p.modis.start_date >= p.modis.end_date:
        raise ValueError("modis.start_date must be before modis.end_date")
    if not p.modis.start_date <= p.modis.first_milestone_date <= p.modis.end_date:
        raise ValueError("modis.first_milestone_date must lie inside the date range")
    if p.grid.model_resolution_m <= 0 or p.grid.domain_buffer_km < 0:
        raise ValueError("grid resolution must be positive and buffer non-negative")
    if not 0.0 < p.grid.min_cell_overlap <= 1.0:
        raise ValueError("grid.min_cell_overlap must be in (0, 1]")
    if abs(p.modis.scale - 0.02) > 1e-9:
        raise ValueError("MOD11A2 scale factor is 0.02 K per DN; refusing to run with another value")
