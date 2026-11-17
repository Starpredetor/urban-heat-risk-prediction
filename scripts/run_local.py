"""Run the Stage A pipeline on this machine instead of Colab.

Same code, same Earth Engine data, same results as the notebook; files go to
data/ in the repository instead of Google Drive.

One-time setup (in the project's .venv):
    .venv\\Scripts\\python.exe -m pip install earthengine-api
    set EE_PROJECT=your-earth-engine-project-id        (PowerShell: $env:EE_PROJECT="...")

Steps, in order (each is safe to re-run; finished work is reused):
    python scripts/run_local.py check          Earth Engine access and MODIS coverage
    python scripts/run_local.py boundaries     Mumbai and Navi Mumbai (provisional, OpenStreetMap)
    python scripts/run_local.py grid           model grid, city masks, cell counts (+ PNG)
    python scripts/run_local.py milestone      one MODIS LST map per city (+ PNG)
    python scripts/run_local.py phase2 --years 2024
    python scripts/run_local.py phase2 --years 2001-2025
    python scripts/run_local.py phase2 --years 2024 --which worldcover
    python scripts/run_local.py phase3 --years 2024    clean, aligned cube (+ PNGs)

The first Earth Engine call opens a browser to sign in to Google.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")                      # write PNGs; no window needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from uhrm import acquire, modis, phase2, regions  # noqa: E402
from uhrm import boundaries_osm as osm  # noqa: E402
from uhrm.colab import _init_earth_engine  # noqa: E402
from uhrm.project import Project, load_project  # noqa: E402


def setup(ee_project: str | None = None, earth_engine: bool = True) -> Project:
    project = load_project("local")
    project.mode.storage_root.mkdir(parents=True, exist_ok=True)
    if earth_engine:
        _init_earth_engine(ee_project or project.mode.ee_project or os.environ.get("EE_PROJECT"))
    print(f"mode: local | storage: {project.mode.storage_root}")
    return project


def figures(project: Project) -> Path:
    out = project.path("figures")
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_layout(project: Project):
    grid = regions.load_grid(project.path("grid.json"))
    geoms = {a.slug: regions.load_boundary(project.boundary_path(a), project.grid.crs)
             for a in project.study_areas}
    masks = {s: grid.cell_mask(g, project.grid.min_cell_overlap) for s, g in geoms.items()}
    return grid, geoms, masks


def outline(ax, geoms, colour):
    for g in geoms.values():
        for poly in getattr(g, "geoms", [g]):
            ax.plot(*poly.exterior.xy, colour, lw=0.8)


# --- steps -------------------------------------------------------------------

def cmd_check(args):
    setup(args.ee_project)
    import ee
    project = load_project("local")
    col = ee.ImageCollection(project.modis.collection)
    t = col.aggregate_array("system:time_start")
    lo, hi = t.reduce(ee.Reducer.min()).getInfo(), t.reduce(ee.Reducer.max()).getInfo()
    as_date = lambda ms: datetime.fromtimestamp(ms / 1000, timezone.utc).date()  # noqa: E731
    print("Earth Engine OK")
    print("MOD11A2 composites:", col.size().getInfo(), "| bands:", col.first().bandNames().getInfo())
    print("coverage:", as_date(lo), "to", as_date(hi))


def cmd_boundaries(args):
    project = setup(earth_engine=False)
    paths = {a.slug: project.boundary_path(a) for a in project.study_areas}
    if args.refresh or not all(p.exists() for p in paths.values()):
        navi = osm.search("Navi Mumbai")
        mumbai = osm.search("Mumbai City district") + osm.search("Mumbai Suburban district")
        print("--- Navi Mumbai candidates"); osm.show(navi)
        print("--- Mumbai candidates"); osm.show(mumbai)
        navi_pick = osm.pick(navi, args.navi_ids) if args.navi_ids else osm.pick_largest_by_prefix(navi, ["Navi Mumbai"])
        mumbai_pick = (osm.pick(mumbai, args.mumbai_ids) if args.mumbai_ids
                       else osm.pick_largest_by_prefix(mumbai, ["Mumbai City", "Mumbai Suburban"]))
        osm.save_boundary(navi_pick, paths["navi_mumbai"], query="Navi Mumbai")
        osm.save_boundary(mumbai_pick, paths["mumbai"], query="Mumbai City + Mumbai Suburban districts")
    else:
        print("Using saved boundaries (--refresh to fetch again).")
    for slug, p in paths.items():
        meta = json.loads(p.with_suffix(".meta.json").read_text(encoding="utf-8"))
        print(f"{slug}: ~{meta['approx_area_km2']:.0f} km2 from", [s["display_name"][:40] for s in meta["sources"]])
        if not 20 < meta["approx_area_km2"] < 3000:
            sys.exit(f"{slug}: area looks wrong; re-run with --refresh and --{slug.replace('_', '-')}-ids")


def cmd_grid(args):
    """Phase 1: grid, city masks, land/water and coastal flags, QGIS outputs."""
    from uhrm import phase1, plots
    project = setup(args.ee_project, earth_engine=not args.no_earth_engine)
    geoms = {a.slug: regions.load_boundary(project.boundary_path(a), project.grid.crs) for a in project.study_areas}
    grid = phase1.model_grid(project, geoms)
    phase1.check_grid_unchanged(grid, project.path("grid.json"))
    regions.save_grid(grid, project.path("grid.json"))
    if not args.no_earth_engine:      # land/water from WorldCover (FR-P1.5); reused by Phase 2
        phase2.run_phase2(project, grid, [project.modis.first_milestone_date.year],
                          phase2.earthengine_backends(project), which=("worldcover",))
    p1 = phase1.build(project, geoms, phase1.load_water_fraction(project), grid)
    saved = phase1.save(p1, project)
    for k, v in phase1.report(p1).items():
        print(f"{k}: {v}")
    path = figures(project) / "phase1_grid.png"
    plots.phase1_map(p1).savefig(path, dpi=120, bbox_inches="tight")
    print("saved", path)
    print("open in QGIS:", saved["cells"], "and", saved["combined"])


def cmd_milestone(args):
    project = setup(args.ee_project)
    grid, geoms, masks = load_layout(project)
    backend = phase2.earthengine_backends(project).modis_lst
    start = project.modis.first_milestone_date
    dates = backend.list_composites(start, start + timedelta(days=40))
    if not dates:
        sys.exit(f"no composites found after {start}")
    day = dates[0]
    out_dir = project.path("raw", "modis")
    print(acquire.brief(acquire.acquire_modis(backend, grid, [day], out_dir, project.path("catalog.csv"))))

    import rasterio
    from uhrm import diagnostics, phase1, plots
    with rasterio.open(acquire.composite_path(out_dir, day)) as src:
        dn, qc = src.read(1), src.read(2)
    lst = modis.process_composite(dn, qc, project.modis.qc)
    water = phase1.load_water_fraction(project)
    if water is None:
        sys.exit("no WorldCover water fraction yet: run the `grid` step first")
    land = water <= project.preprocess.max_water_fraction

    has_value = dn >= project.modis.qc.valid_dn_range[0]
    print("registration:", diagnostics.registration_offset(has_value, land))
    for slug, m in masks.items():
        v = lst[m & land]; ok = np.isfinite(v)
        stats = (f"LST {v[ok].min():.1f} to {v[ok].max():.1f} C, mean {v[ok].mean():.1f} C"
                 if ok.any() else "no valid cells")
        print(f"\n{slug}: {int((m & land).sum())} land cells, {ok.mean():.0%} valid; {stats}")
        print(json.dumps(diagnostics.qc_breakdown(dn, qc, m & land, project.modis.qc), indent=1))

    path = figures(project) / f"milestone_lst_{day:%Y%m%d}.png"
    plots.lst_map(lst, grid, geoms, water, f"MOD11A2 daytime LST (C), composite starting {day}").savefig(
        path, dpi=120, bbox_inches="tight")
    print("saved", path)


def parse_years(text: str) -> list[int]:
    """'2024' -> [2024]; '2001-2025' -> 2001..2025; '2003,2010' -> [2003, 2010]."""
    years = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            years += list(range(int(a), int(b) + 1))
        else:
            years.append(int(part))
    return sorted(set(years))


def cmd_phase2(args):
    project = setup(args.ee_project)
    grid = regions.load_grid(project.path("grid.json"))
    t0 = time.time()
    phase2.run_phase2(project, grid, parse_years(args.years), phase2.earthengine_backends(project),
                      which=tuple(args.which))
    print(f"finished in {time.time() - t0:.0f} s")


def cmd_phase3(args):
    from uhrm import preprocess
    project = setup(earth_engine=False)
    grid, geoms, masks = load_layout(project)
    years = parse_years(args.years)
    t0 = time.time()
    cube = preprocess.build_cube(project, grid, years)
    path = preprocess.save_cube(cube, grid, project,
                                project.path("processed", f"cube_{years[0]}_{years[-1]}.npz"))
    for k, v in preprocess.summarize(cube).items():
        print(f"{k}: {v}")
    print(f"saved {path} ({path.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f} s")

    # valid-fraction distribution: the evidence for min_valid_fraction (FR-P3.4)
    dates = [datetime.fromisoformat(s) for s in cube["dates"]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(dates, cube["lst_valid_fraction"], ".", ms=3)
    axes[0].axhline(project.preprocess.min_valid_fraction, color="r", lw=1, label="min_valid_fraction")
    axes[0].set_title("LST valid fraction of land cells per composite"); axes[0].legend()
    axes[1].hist(cube["lst_valid_fraction"], bins=20)
    axes[1].set_title("distribution")
    fig.savefig(figures(project) / "phase3_valid_fraction.png", dpi=120, bbox_inches="tight")

    # mean pre-monsoon LST map per city as a visual sanity check
    pre = np.array([d.month in (3, 4, 5) for d in dates])
    if pre.any():
        with np.errstate(all="ignore"):
            mean_lst = np.nanmean(cube["lst_c"][pre], axis=0)
        x0, y0, x1, y1 = grid.bounds
        fig, ax = plt.subplots(figsize=(7, 8))
        im = ax.imshow(mean_lst, extent=(x0, x1, y0, y1), cmap="inferno")
        fig.colorbar(im, ax=ax, shrink=0.7)
        outline(ax, geoms, "c-")
        ax.set_title(f"Mean March-May daytime LST (C), {years[0]}-{years[-1]}")
        fig.savefig(figures(project) / "phase3_mean_premonsoon_lst.png", dpi=120, bbox_inches="tight")
    print("figures saved in", figures(project))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ee-project", help="Earth Engine project id (default: env EE_PROJECT)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check").set_defaults(fn=cmd_check)
    b = sub.add_parser("boundaries")
    b.add_argument("--refresh", action="store_true")
    b.add_argument("--mumbai-ids", type=int, nargs="+")
    b.add_argument("--navi-mumbai-ids", dest="navi_ids", type=int, nargs="+")
    b.set_defaults(fn=cmd_boundaries)
    gr = sub.add_parser("grid")
    gr.add_argument("--no-earth-engine", action="store_true",
                    help="skip fetching WorldCover (no land/water flags)")
    gr.set_defaults(fn=cmd_grid)
    sub.add_parser("milestone").set_defaults(fn=cmd_milestone)
    ph = sub.add_parser("phase2")
    ph.add_argument("--years", default="2024", help="e.g. 2024, 2001-2025, 2003,2010")
    ph.add_argument("--which", nargs="+", default=list(phase2.ALL), choices=list(phase2.ALL))
    ph.set_defaults(fn=cmd_phase2)
    p3 = sub.add_parser("phase3")
    p3.add_argument("--years", default="2024", help="e.g. 2024 or 2001-2025 (must already be downloaded)")
    p3.set_defaults(fn=cmd_phase3)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
