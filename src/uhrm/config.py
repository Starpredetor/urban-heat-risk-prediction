"""Settings, loaded from config/settings.yaml.

One file, one dataclass. Everything the pipeline needs is here so you can point
at a single place and say "these are the choices we made".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Repository root, so paths work no matter which folder you run from.
ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Settings:
    # --- study area ---
    city: str
    bbox: tuple[float, float, float, float]   # lon/lat: left, bottom, right, top
    crs: str                                   # map projection, e.g. UTM zone 43N
    resolution: int                            # metres per pixel

    # --- which images to use ---
    start_year: int
    end_year: int
    months: list[int]                          # season to average over
    max_cloud: int                             # skip images cloudier than this (%)
    max_images_per_year: int                   # cap, to keep downloads sensible

    # --- model ---
    years_in: int                              # how many past years the model sees
    patch: int                                 # training patch size in pixels
    train_years: tuple[int, int]
    val_years: tuple[int, int]
    test_years: tuple[int, int]
    epochs: int
    batch_size: int
    learning_rate: float
    seed: int

    out_dir: Path = field(default=ROOT / "data")

    @property
    def years(self) -> list[int]:
        return list(range(self.start_year, self.end_year + 1))

    @property
    def cube_path(self) -> Path:
        return self.out_dir / "cube.nc"

    def years_of(self, split: str) -> list[int]:
        start, end = getattr(self, f"{split}_years")
        return list(range(start, end + 1))

    def check(self) -> None:
        """Catch the mistakes that would otherwise produce wrong results quietly."""
        assert self.start_year <= self.end_year, "start_year must not be after end_year"
        assert self.resolution > 0, "resolution must be positive"

        train, val, test = (self.years_of(s) for s in ("train", "val", "test"))

        # The big one. If a year the model trained on also appears in the test
        # set, the score is meaningless -- it is being graded on an answer it
        # was shown. Same if training years come after test years: that is
        # predicting the past from the future.
        assert not set(train) & set(test), "train and test years overlap"
        assert not set(train) & set(val), "train and validation years overlap"
        assert max(train) < min(test), "training years must come before test years"

        # The first `years_in` years can only ever be model input, never a
        # target, because a target needs that many years of history behind it.
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

    settings = Settings(
        city=raw["city"],
        bbox=tuple(raw["bbox"]),
        crs=raw["crs"],
        resolution=raw["resolution"],
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
