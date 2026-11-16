"""Colab bootstrap: the one function every notebook calls first (SRS FR-X.5).

It mounts Drive, installs dependencies if missing, signs in to Earth Engine
and returns the loaded project config in "colab" mode. Notebooks stay thin;
the logic lives here so the same code works offline (FR-X.2).

Earth Engine project id: set the Colab secret `EE_PROJECT` (key icon in the
left sidebar), or pass `ee_project=`. Never paste credentials into a cell
(FR-X.6, NFR-6.1).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

from uhrm.project import ROOT, Project, load_project

REQUIREMENTS = ROOT / "requirements-colab.txt"
DRIVE_MOUNT = "/content/drive"


def in_colab() -> bool:
    return importlib.util.find_spec("google.colab") is not None


def _install_missing() -> None:
    needed = {"ee": "earthengine-api", "shapely": "shapely", "pyproj": "pyproj",
              "rasterio": "rasterio", "yaml": "pyyaml", "requests": "requests"}
    if all(importlib.util.find_spec(mod) for mod in needed):
        return
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(REQUIREMENTS)],
                   check=True)


def _ee_project(explicit: str | None, from_config: str | None) -> str | None:
    if explicit or from_config:
        return explicit or from_config
    try:
        from google.colab import userdata
        return userdata.get("EE_PROJECT")
    except Exception:
        return None


def _init_earth_engine(project_id: str | None) -> None:
    import ee

    try:
        ee.Initialize(project=project_id)
    except Exception:
        ee.Authenticate()            # opens the Google sign-in flow
        ee.Initialize(project=project_id)


def bootstrap(*, ee_project: str | None = None, mount_drive: bool = True,
              earth_engine: bool = True) -> Project:
    """Prepare a Colab session and return the project in colab mode."""
    if not in_colab():
        raise RuntimeError("bootstrap() is for Colab. Offline, use uhrm.project.load_project('offline').")

    if mount_drive and not Path(DRIVE_MOUNT, "MyDrive").exists():
        from google.colab import drive
        drive.mount(DRIVE_MOUNT)

    _install_missing()
    project = load_project("colab")
    project.mode.storage_root.mkdir(parents=True, exist_ok=True)

    if earth_engine:
        _init_earth_engine(_ee_project(ee_project, project.mode.ee_project))

    _report(project)
    return project


def _report(project: Project) -> None:
    print(f"mode: {project.mode.name} | backend: {project.mode.backend}")
    print(f"storage: {project.mode.storage_root}")
    try:
        import torch
        print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU runtime)")
    except ImportError:
        print("torch not installed (not needed until Phase 7)")
    print("python:", sys.version.split()[0])
