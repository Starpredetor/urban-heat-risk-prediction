"""Settings, loaded from config/settings.yaml.

One file, one dataclass. Everything the project decides is here, so you can
point at a single place and say "these are the choices we made".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class City:
    """One study area: a square centred on a city."""

    name: str
    lat: float
    lon: float
    radius_km: float
    resolution: int

    @property
    def crs(self) -> str:
        """The right map projection for this city, worked out from its longitude.

        India spans several UTM zones -- Mumbai and Delhi are in zone 43N,
        Chennai and Hyderabad in 44N. Using one city's projection for all of
        them would stretch the others. Each city gets its own.
        """
        zone = int((self.lon + 180) // 6) + 1
        return f"EPSG:{32600 + zone if self.lat >= 0 else 32700 + zone}"

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """A square around the city centre, in longitude/latitude."""
        half = self.radius_km / 2
        d_lat = half / 111.0
        # Lines of longitude squeeze together away from the equator, so the
        # same distance covers more degrees the further north you go.
        d_lon = half / (111.0 * math.cos(math.radians(self.lat)))
        return (self.lon - d_lon, self.lat - d_lat,
                self.lon + d_lon, self.lat + d_lat)

    @property
    def size_pixels(self) -> int:
        """Width and height in pixels. The same for every city, so the model
        can be trained on all of them together."""
        return int(round(self.radius_km * 1000 / self.resolution))

    @property
    def slug(self) -> str:
        return self.name.lower().replace(" ", "_")


@dataclass
class Settings:
    cities: list[City]
    resolution: int
    start_year: int
    end_year: int
    months: list[int]
    max_cloud: int
    max_images_per_year: int

    years_in: int
    patch: int
    train_years: tuple[int, int]
    val_years: tuple[int, int]
    test_years: tuple[int, int]
    epochs: int
    batch_size: int
    learning_rate: float
    seed: int

    out_dir: Path

    @property
    def years(self) -> list[int]:
        return list(range(self.start_year, self.end_year + 1))

    def cube_path(self, city: City) -> Path:
        return self.out_dir / "cubes" / f"{city.slug}.nc"

    def radius_summary(self) -> str:
        c = self.cities[0]
        return (f"{c.radius_km:.0f} x {c.radius_km:.0f} km per city "
                f"({c.size_pixels} x {c.size_pixels} pixels at {c.resolution} m)")

    def city(self, name: str) -> City:
        for c in self.cities:
            if c.slug == name.lower().replace(" ", "_"):
                return c
        raise KeyError(f"no city named {name!r}; have {[c.name for c in self.cities]}")

    def years_of(self, split: str) -> list[int]:
        start, end = getattr(self, f"{split}_years")
        return list(range(start, end + 1))

    def check(self) -> None:
        """Catch the mistakes that would otherwise give wrong results quietly."""
        assert self.cities, "at least one city must be listed"
        assert self.start_year <= self.end_year, "start_year must not be after end_year"
        assert self.resolution > 0, "resolution must be positive"

        names = [c.slug for c in self.cities]
        assert len(names) == len(set(names)), "two cities have the same name"

        train, val, test = (self.years_of(s) for s in ("train", "val", "test"))

        # The big one. If a year the model trained on also appears in the test
        # set, the score is meaningless -- it is being graded on an answer it
        # was already shown. Same if training years come after test years: that
        # is predicting the past from the future.
        assert not set(train) & set(test), "train and test years overlap"
        assert not set(train) & set(val), "train and validation years overlap"
        assert max(train) < min(test), "training years must come before test years"

        # The first `years_in` years can only ever be model input, never a
        # target, because a target needs that much history behind it.
        first_target = self.years[self.years_in]
        assert min(train) >= first_target, (
            f"earliest year that can be predicted is {first_target} "
            f"(needs {self.years_in} years of history), but training starts at {min(train)}"
        )


def load_settings(path: str | Path = "config/settings.yaml") -> Settings:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    resolution = raw["resolution"]
    radius = raw["radius_km"]
    cities = [
        City(name=c["name"], lat=c["lat"], lon=c["lon"],
             radius_km=radius, resolution=resolution)
        for c in raw["cities"]
    ]

    settings = Settings(
        cities=cities,
        resolution=resolution,
        start_year=raw["start_year"],
        end_year=raw["end_year"],
        months=raw["months"],
        max_cloud=raw["max_cloud"],
        max_images_per_year=raw["max_images_per_year"],
        years_in=raw["years_in"],
        patch=raw["patch"],
        train_years=tuple(raw["train_years"]),
        val_years=tuple(raw["val_years"]),
        test_years=tuple(raw["test_years"]),
        epochs=raw["epochs"],
        batch_size=raw["batch_size"],
        learning_rate=raw["learning_rate"],
        seed=raw["seed"],
        out_dir=ROOT / raw.get("out_dir", "data"),
    )
    settings.check()
    return settings
