"""Shared figures for the notebook and scripts/run_local.py.

Every map draws the city outlines and, when available, the coastline (the 0.5
WorldCover water-fraction contour), so it is visible at a glance whether the
data sits on the land it should.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def extent(grid):
    x0, y0, x1, y1 = grid.bounds
    return (x0, x1, y0, y1)


def outlines(ax, geoms, colour="c", lw=1.0):
    for g in geoms.values():
        for poly in getattr(g, "geoms", [g]):
            ax.plot(*poly.exterior.xy, color=colour, lw=lw)


def coastline(ax, grid, water, colour="w"):
    if water is not None:
        ax.contour(water, levels=[0.5], extent=extent(grid), origin="upper",
                   colors=colour, linewidths=0.9, linestyles="--")


def km_axes(ax):
    ax.set_xlabel("UTM 43N easting (m)")
    ax.set_ylabel("northing (m)")


def phase1_map(p1):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    city = np.zeros(p1.grid.shape)
    for i, m in enumerate(p1.masks.values(), start=1):
        city[m] = i
    axes[0].imshow(city, extent=extent(p1.grid), cmap="viridis", interpolation="nearest")
    coastline(axes[0], p1.grid, p1.water)
    outlines(axes[0], p1.geoms, "w")
    axes[0].set_title("1 km cells per city (dashed: coastline from WorldCover)")
    if p1.water is not None:
        im = axes[1].imshow(p1.water, extent=extent(p1.grid), cmap="Blues", vmin=0, vmax=1)
        fig.colorbar(im, ax=axes[1], shrink=0.7, label="permanent-water fraction")
        axes[1].contour(p1.coastal.astype(float), levels=[0.5], extent=extent(p1.grid),
                        origin="upper", colors="orange", linewidths=0.8)
        axes[1].set_title("Water fraction; orange: coastal land cells")
    outlines(axes[1], p1.geoms, "k")
    for ax in axes:
        km_axes(ax)
    plt.tight_layout()
    return fig


def lst_map(lst, grid, geoms, water=None, title="Daytime LST (C)"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    im = axes[0].imshow(lst, extent=extent(grid), cmap="inferno", interpolation="nearest")
    fig.colorbar(im, ax=axes[0], shrink=0.7, label="C")
    axes[0].set_title(title)
    axes[1].imshow(np.isfinite(lst), extent=extent(grid), cmap="gray", interpolation="nearest")
    axes[1].set_title("valid cells (white) after quality filter")
    for ax in axes:
        coastline(ax, grid, water, "deepskyblue")
        outlines(ax, geoms, "c")
        km_axes(ax)
    plt.tight_layout()
    return fig
