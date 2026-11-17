"""Shared figure helpers: one graph per figure, labelled axes, units on every colour
bar, and a legend for the city outlines.

Map coordinates are UTM zone 43N (EPSG:32643) shown in kilometres:
x = easting (km east of the zone's false origin), y = northing (km north of the equator).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

CITY_STYLE = {"mumbai": ("Mumbai boundary", "#00e5ff"), "navi_mumbai": ("Navi Mumbai boundary", "#76ff03")}

# ESA WorldCover classes with the official colour palette.
WORLDCOVER_CLASSES = {
    10: ("Tree cover", "#006400"), 20: ("Shrubland", "#ffbb22"), 30: ("Grassland", "#ffff4c"),
    40: ("Cropland", "#f096ff"), 50: ("Built-up", "#fa0000"), 60: ("Bare / sparse vegetation", "#b4b4b4"),
    70: ("Snow and ice", "#f0f0f0"), 80: ("Permanent water", "#0064c8"),
    90: ("Herbaceous wetland", "#0096a0"), 95: ("Mangroves", "#00cf75"), 100: ("Moss and lichen", "#fae6a0"),
}


def extent_km(grid):
    x0, y0, x1, y1 = grid.bounds
    return (x0 / 1000, x1 / 1000, y0 / 1000, y1 / 1000)


def draw_cities(ax, geoms_utm: dict, *, legend: bool = True, lw: float = 1.4, extra_handles=()):
    """City outlines (geometries in metres, drawn in km) plus a legend."""
    handles = list(extra_handles)
    for key, geom in geoms_utm.items():
        label, colour = CITY_STYLE.get(key, (key, "white"))
        for poly in getattr(geom, "geoms", [geom]):
            x, y = poly.exterior.xy
            ax.plot(np.asarray(x) / 1000, np.asarray(y) / 1000, color=colour, lw=lw)
        handles.append(Line2D([], [], color=colour, lw=2, label=label))
    if legend:
        ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.9)


def map_axes(ax, title: str):
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Easting, UTM zone 43N (km)")
    ax.set_ylabel("Northing, UTM zone 43N (km)")
    ax.set_aspect("equal")


def raster_map(data, grid, geoms_utm, *, title, cbar_label, cmap="inferno", vmin=None, vmax=None,
               nodata_note="white = no valid value", figsize=(8, 9), extra_handles=()):
    """One map in its own figure. Returns (fig, ax)."""
    fig, ax = plt.subplots(figsize=figsize)
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("white")
    im = ax.imshow(np.ma.masked_invalid(data), extent=extent_km(grid), cmap=cm, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, shrink=0.75)
    cb.set_label(cbar_label)
    draw_cities(ax, geoms_utm, extra_handles=[Patch(facecolor="white", edgecolor="grey", label=nodata_note),
                                              *extra_handles])
    map_axes(ax, title)
    fig.tight_layout()
    return fig, ax


def worldcover_map(classes, grid, geoms_utm, *, title, figsize=(9, 9)):
    """Land-cover map with a named legend for the classes that occur."""
    present = [c for c in WORLDCOVER_CLASSES if (classes == c).any()]
    codes = sorted(WORLDCOVER_CLASSES)
    cmap = ListedColormap([WORLDCOVER_CLASSES[c][1] for c in codes])
    cmap.set_bad("white")
    norm = BoundaryNorm([c - 0.5 for c in codes] + [codes[-1] + 0.5], cmap.N)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(np.ma.masked_where(classes == 0, classes), extent=extent_km(grid), cmap=cmap, norm=norm,
              interpolation="nearest")
    total = (classes > 0).sum()
    handles = [Patch(color=WORLDCOVER_CLASSES[c][1],
                     label=f"{WORLDCOVER_CLASSES[c][0]} ({100 * (classes == c).sum() / total:.1f}%)")
               for c in present]
    handles.append(Patch(facecolor="white", edgecolor="grey", label="outside both cities"))
    draw_cities(ax, geoms_utm, legend=False)
    ax.legend(handles=handles + [Line2D([], [], color=CITY_STYLE[k][1], lw=2, label=CITY_STYLE[k][0])
                                 for k in geoms_utm],
              loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9, title="Land-cover class (share of city area)")
    map_axes(ax, title)
    fig.tight_layout()
    return fig, ax


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
