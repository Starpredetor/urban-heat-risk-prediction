"""Step 5: feed the stacked maps to the model in bite-sized pieces.

The full map is 674 x 675 pixels, too big to learn from in one go. So we cut it
into small squares ("patches") and show the model many of them.

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

    The averages are worked out from the TRAINING years only. Using all years
    would let the model peek at the test period, and the score would look
    better than it really is.
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    @classmethod
    def fit(cls, cube_data: np.ndarray, n_train_years: int) -> "Scaler":
        train = cube_data[:n_train_years]                     # (years, layer, y, x)
        mean = np.array([np.nanmean(train[:, i]) for i in range(train.shape[1])], np.float32)
        std = np.array([max(float(np.nanstd(train[:, i])), 1e-6)
                        for i in range(train.shape[1])], np.float32)
        return cls(mean, std)

    def apply(self, block: np.ndarray) -> np.ndarray:
        return (block - self.mean[:, None, None]) / self.std[:, None, None]

    @property
    def lst_std(self) -> float:
        """Used to convert the model's error back into degrees Celsius."""
        return float(self.std[LAYERS.index("lst")])

    @property
    def lst_mean(self) -> float:
        return float(self.mean[LAYERS.index("lst")])

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"mean": self.mean.tolist(), "std": self.std.tolist()}, indent=2), encoding="utf-8")


class HeatPatches(Dataset):
    """Cut the map into squares and serve them to the model."""

    def __init__(self, cube, settings: Settings, split: str, scaler: Scaler):
        self.settings = settings
        self.scaler = scaler

        self.data = np.nan_to_num(cube["data"].values.astype(np.float32))
        self.years = [int(y) for y in cube["year"].values]
        self.lst = LAYERS.index("lst")

        size = settings.patch
        _, _, height, width = self.data.shape

        # Which years can we ask the model to predict? Only ones in this split
        # that also have enough history behind them.
        wanted = set(settings.years_of(split))
        targets = [i for i, y in enumerate(self.years)
                   if y in wanted and i >= settings.years_in]

        # Top-left corner of every square, stepping half a patch at a time.
        step = size // 2
        rows = list(range(0, height - size + 1, step)) or [0]
        cols = list(range(0, width - size + 1, step)) or [0]
        if rows[-1] != height - size:
            rows.append(height - size)
        if cols[-1] != width - size:
            cols.append(width - size)

        self.samples = [(t, r, c) for t in targets for r in rows for c in cols]
        assert self.samples, f"no training examples for the {split} years"

    @property
    def target_years(self) -> list[int]:
        return sorted({self.years[t] for t, _, _ in self.samples})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i):
        t, row, col = self.samples[i]
        size, history = self.settings.patch, self.settings.years_in

        past = self.data[t - history: t, :, row: row + size, col: col + size]
        past = np.stack([self.scaler.apply(year) for year in past])

        answer = self.data[t, self.lst, row: row + size, col: col + size]
        answer = (answer - self.scaler.lst_mean) / self.scaler.lst_std

        return torch.from_numpy(past.copy()), torch.from_numpy(answer.copy())[None]

    def last_known_temperature(self, i) -> np.ndarray:
        """Last year's temperature -- what the 'no change' guess would predict."""
        t, row, col = self.samples[i]
        size = self.settings.patch
        frame = self.data[t - 1, self.lst, row: row + size, col: col + size]
        return (frame - self.scaler.lst_mean) / self.scaler.lst_std
