"""Step 5: feed the stacked maps to the model in bite-sized pieces.

Each city map is over 1000 pixels square, too big to learn from in one go. So we
cut every city into small squares ("patches") and pool them together.

Pooling matters. One city gives only a handful of predictable years, which is
far too few. Six cities pooled give six times the examples, and it forces the
model to learn what heat and concrete look like in general rather than
memorising one city's street layout.

Each training example is:
    input   the last N years of all four layers, for one 64x64 square
    answer  next year's temperature for that same square
"""

from __future__ import annotations

import json

import numpy as np
import torch
from torch.utils.data import Dataset

from uhrm.config import Settings
from uhrm.cube import LAYERS


class Scaler:
    """Puts every layer on a comparable footing before training.

    Temperature runs 27-57, NDVI runs -1 to 1. Without rescaling, the model
    would treat temperature as a thousand times more important simply because
    its numbers are bigger.

    The averages come from the TRAINING years only, across all cities. Using
    all years would let the model peek at the test period, and the score would
    look better than it really is.
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    @classmethod
    def fit(cls, cubes: list, n_train_years: int) -> "Scaler":
        n_layers = len(LAYERS)
        sums = np.zeros(n_layers)
        squares = np.zeros(n_layers)
        counts = np.zeros(n_layers)

        for cube in cubes:
            train = cube["data"].values[:n_train_years]        # (years, layer, y, x)
            for i in range(n_layers):
                values = train[:, i]
                values = values[np.isfinite(values)]
                sums[i] += values.sum()
                squares[i] += (values.astype(np.float64) ** 2).sum()
                counts[i] += values.size

        assert counts.all(), "a layer has no valid pixels in the training years"
        mean = sums / counts
        variance = np.maximum(squares / counts - mean**2, 1e-12)
        return cls(mean.astype(np.float32), np.sqrt(variance).astype(np.float32))

    def apply(self, block: np.ndarray) -> np.ndarray:
        return (block - self.mean[:, None, None]) / self.std[:, None, None]

    @property
    def lst_std(self) -> float:
        """Converts the model's error back into degrees Celsius."""
        return float(self.std[LAYERS.index("lst")])

    @property
    def lst_mean(self) -> float:
        return float(self.mean[LAYERS.index("lst")])

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"layers": LAYERS, "mean": self.mean.tolist(), "std": self.std.tolist()},
            indent=2), encoding="utf-8")


class HeatPatches(Dataset):
    """Squares cut from every city's maps, pooled into one training set."""

    def __init__(self, cubes: list, settings: Settings, split: str, scaler: Scaler):
        self.settings = settings
        self.scaler = scaler
        self.lst = LAYERS.index("lst")

        self.cities = [str(c.attrs["city"]) for c in cubes]
        self.data = [np.nan_to_num(c["data"].values.astype(np.float32)) for c in cubes]
        self.years = [int(y) for y in cubes[0]["year"].values]

        size = settings.patch
        wanted = set(settings.years_of(split))
        targets = [i for i, y in enumerate(self.years)
                   if y in wanted and i >= settings.years_in]

        # (which city, which year, top-left corner of the square)
        self.samples = []
        for city_index, block in enumerate(self.data):
            _, _, height, width = block.shape
            for row in _starts(height, size):
                for col in _starts(width, size):
                    for t in targets:
                        self.samples.append((city_index, t, row, col))

        assert self.samples, f"no examples for the {split} years"

    @property
    def target_years(self) -> list[int]:
        return sorted({self.years[t] for _, t, _, _ in self.samples})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i):
        city, t, row, col = self.samples[i]
        size, history = self.settings.patch, self.settings.years_in
        block = self.data[city]

        past = block[t - history: t, :, row: row + size, col: col + size]
        past = np.stack([self.scaler.apply(year) for year in past])

        answer = block[t, self.lst, row: row + size, col: col + size]
        answer = (answer - self.scaler.lst_mean) / self.scaler.lst_std

        return torch.from_numpy(past.copy()), torch.from_numpy(answer.copy())[None]

    def last_known_temperature(self, i) -> np.ndarray:
        """Last year's temperature -- what the 'no change' guess would predict."""
        city, t, row, col = self.samples[i]
        size = self.settings.patch
        frame = self.data[city][t - 1, self.lst, row: row + size, col: col + size]
        return (frame - self.scaler.lst_mean) / self.scaler.lst_std


def _starts(extent: int, size: int) -> list[int]:
    """Top-left corners, stepping half a patch, always covering the far edge."""
    step = size // 2
    starts = list(range(0, extent - size + 1, step)) or [0]
    if starts[-1] != extent - size:
        starts.append(extent - size)
    return starts
