"""Checks on the settings file.

The split check is the single most important test in the project. If the years
the model trains on overlap the years it is tested on, the score is meaningless
-- the model is being graded on answers it was already shown.
"""

import pytest
import yaml

from uhrm.config import ROOT, Settings, load_settings


def make(**changes) -> Settings:
    """A copy of the real settings with one or two things changed."""
    raw = yaml.safe_load((ROOT / "config/settings.yaml").read_text(encoding="utf-8"))
    values = dict(
        city=raw["city"], bbox=tuple(raw["bbox"]), crs=raw["crs"],
        resolution=raw["resolution"], start_year=raw["start_year"],
        end_year=raw["end_year"], months=raw["months"], max_cloud=raw["max_cloud"],
        max_images_per_year=raw["max_images_per_year"], years_in=raw["years_in"],
        patch=raw["patch"], train_years=tuple(raw["train_years"]),
        val_years=tuple(raw["val_years"]), test_years=tuple(raw["test_years"]),
        epochs=raw["epochs"], batch_size=raw["batch_size"],
        learning_rate=raw["learning_rate"], seed=raw["seed"],
    )
    values.update(changes)
    return Settings(**values)


def test_the_shipped_settings_are_valid():
    settings = load_settings()
    assert settings.city == "Mumbai"
    assert settings.years[0] == 2017


def test_training_and_test_years_must_not_overlap():
    with pytest.raises(AssertionError, match="overlap"):
        make(train_years=(2021, 2024), test_years=(2024, 2025)).check()


def test_training_must_come_before_testing():
    """Otherwise we would be predicting the past using the future."""
    with pytest.raises(AssertionError, match="before test"):
        make(train_years=(2024, 2025), val_years=(2023, 2023),
             test_years=(2021, 2022)).check()


def test_cannot_predict_a_year_without_enough_history():
    """With years_in=4, the first four years can only be input, never a target."""
    with pytest.raises(AssertionError, match="history"):
        make(train_years=(2018, 2019)).check()
