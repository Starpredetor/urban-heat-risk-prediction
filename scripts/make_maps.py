"""STEP 3 - Make the pictures for your report and presentation.

    python scripts/make_maps.py

Reads data/cube.nc and writes PNG images into data/maps/ that you can drop
straight into a report or slides.
"""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # save files, do not open windows
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uhrm.config import load_settings
from uhrm.cube import layer, load_cube


def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {path.name}")


def heat_map(cube, out, city):
    """The headline picture: where the city is hot."""
    lst = layer(cube, "lst")
    years = [int(y) for y in cube["year"].values]

    fig, ax = plt.subplots(figsize=(7, 7))
    # 'inferno' goes dark->bright with heat and stays readable in greyscale
    # and for colour-blind viewers.
    img = ax.imshow(lst[-1], cmap="inferno")
    ax.set_title(f"{city} surface temperature, {years[-1]}\n"
                 f"(March-May average, from Landsat)", fontsize=13)
    ax.axis("off")
    bar = fig.colorbar(img, ax=ax, shrink=0.8)
    bar.set_label("degrees Celsius")
    save(fig, out / "1_heat_map.png")


def hot_vs_green(cube, out, city):
    """Side by side: the built-up map and the heat map look the same."""
    lst, ndbi, ndvi = layer(cube, "lst"), layer(cube, "ndbi"), layer(cube, "ndvi")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, data, title, cmap in [
        (axes[0], ndvi[-1], "Vegetation (NDVI)\ngreen = plants", "YlGn"),
        (axes[1], ndbi[-1], "Built-up (NDBI)\nbrown = concrete", "copper"),
        (axes[2], lst[-1], "Temperature\nbright = hot", "inferno"),
    ]:
        img = ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
        fig.colorbar(img, ax=ax, shrink=0.7)
    fig.suptitle(f"{city}: concrete areas are hot, green areas are cool",
                 fontsize=14, y=1.02)
    save(fig, out / "2_why_it_is_hot.png")


def heat_island_chart(cube, out, city):
    """A bar chart of the headline finding."""
    lst, ndbi, ndvi, ndwi = (layer(cube, n) for n in ("lst", "ndbi", "ndvi", "ndwi"))

    land = ndwi[-1] < 0                       # ignore the sea
    built = land & (ndbi[-1] > 0)
    green = land & (ndvi[-1] > 0.4)
    water = ndwi[-1] > 0.2

    groups = {
        "Built-up\n(concrete)": np.nanmean(lst[-1][built]),
        "Vegetated\n(parks, trees)": np.nanmean(lst[-1][green]),
        "Water\n(sea, creeks)": np.nanmean(lst[-1][water]),
    }
    difference = groups["Built-up\n(concrete)"] - groups["Vegetated\n(parks, trees)"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(groups.keys(), groups.values(),
                  color=["#c0392b", "#27ae60", "#2980b9"])
    for bar, value in zip(bars, groups.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.4,
                f"{value:.1f} C", ha="center", fontsize=12, weight="bold")
    ax.set_ylabel("average surface temperature (C)")
    ax.set_title(f"{city} urban heat island\n"
                 f"concrete is {difference:.1f} C hotter than green space",
                 fontsize=13)
    ax.set_ylim(0, max(groups.values()) * 1.2)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, out / "3_heat_island.png")
    return difference


def year_by_year(cube, out, city):
    """Every year side by side, to show this is multi-temporal."""
    lst = layer(cube, "lst")
    years = [int(y) for y in cube["year"].values]

    low, high = np.nanpercentile(lst, [2, 98])
    cols = 5
    rows = int(np.ceil(len(years) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.1 * rows))

    for ax, data, year in zip(axes.ravel(), lst, years):
        ax.imshow(data, cmap="inferno", vmin=low, vmax=high)
        ax.set_title(f"{year}   avg {np.nanmean(data):.1f} C", fontsize=10)
        ax.axis("off")
    for ax in axes.ravel()[len(years):]:
        ax.axis("off")

    fig.suptitle(f"{city} surface temperature, {years[0]}-{years[-1]} "
                 f"(all on the same colour scale)", fontsize=14, y=1.0)
    save(fig, out / "4_all_years.png")


def temperature_trend(cube, out, city):
    """A simple line chart of city-average temperature per year."""
    lst = layer(cube, "lst")
    years = [int(y) for y in cube["year"].values]
    averages = [float(np.nanmean(y)) for y in lst]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(years, averages, marker="o", color="#c0392b", linewidth=2)
    ax.set_xlabel("year")
    ax.set_ylabel("average surface temperature (C)")
    ax.set_title(f"{city} city-wide average, March-May", fontsize=13)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, out / "5_trend.png")
    return averages


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = load_settings()

    if not settings.cube_path.exists():
        print(f"No dataset found at {settings.cube_path}")
        print("Run this first:  python scripts/build_dataset.py")
        return 1

    cube = load_cube(settings.cube_path)
    out = settings.out_dir / "maps"
    out.mkdir(parents=True, exist_ok=True)

    print("Writing pictures to", out)
    heat_map(cube, out, settings.city)
    hot_vs_green(cube, out, settings.city)
    difference = heat_island_chart(cube, out, settings.city)
    year_by_year(cube, out, settings.city)
    averages = temperature_trend(cube, out, settings.city)

    years = [int(y) for y in cube["year"].values]
    print()
    print("  NUMBERS YOU CAN QUOTE")
    print(f"    Study area      {settings.city}, {years[0]}-{years[-1]}, March-May")
    print(f"    Resolution      {settings.resolution} m per pixel, "
          f"{cube.sizes['y']} x {cube.sizes['x']} pixels")
    print(f"    Heat island     concrete is {difference:.1f} C hotter than green space")
    print(f"    Hottest year    {years[int(np.argmax(averages))]} "
          f"({max(averages):.1f} C average)")
    print(f"    Coolest year    {years[int(np.argmin(averages))]} "
          f"({min(averages):.1f} C average)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
