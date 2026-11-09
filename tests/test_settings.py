"""Checks on the settings file.

The split check is the single most important test in the project. If the years
the model trains on overlap the years it is tested on, the score is meaningless
-- the model is being graded on answers it was already shown.
"""

import math

import pytest

from uhrm.config import Settings, load_settings


def make(**changes) -> Settings:
    """A copy of the real settings with one or two things changed."""
    s = load_settings()
    values = dict(
        cities=s.cities, resolution=s.resolution,
        start_year=s.start_year, end_year=s.end_year, months=s.months,
        max_cloud=s.max_cloud, max_images_per_year=s.max_images_per_year,
        years_in=s.years_in, patch=s.patch,
        train_years=s.train_years, val_years=s.val_years, test_years=s.test_years,
        epochs=s.epochs, batch_size=s.batch_size,
        learning_rate=s.learning_rate, seed=s.seed, out_dir=s.out_dir,
    )
    values.update(changes)
    return Settings(**values)


# --- Cities -------------------------------------------------------------------

def test_the_shipped_settings_are_valid():
    settings = load_settings()
    assert len(settings.cities) == 6
    assert settings.city("Mumbai").name == "Mumbai"


def test_every_city_gets_the_right_map_projection():
    """India spans two UTM zones. Using one for all would stretch the others."""
    zones = {c.name: c.crs for c in load_settings().cities}
    assert zones["Mumbai"] == "EPSG:32643"       # zone 43N
    assert zones["Delhi"] == "EPSG:32643"
    assert zones["Ahmedabad"] == "EPSG:32643"
    assert zones["Nagpur"] == "EPSG:32644"       # zone 44N
    assert zones["Hyderabad"] == "EPSG:32644"
    assert zones["Chennai"] == "EPSG:32644"


def test_every_city_is_the_same_pixel_size():
    """Required so patches from different cities can be pooled for training."""
    sizes = {c.size_pixels for c in load_settings().cities}
    assert len(sizes) == 1, f"cities have different sizes: {sizes}"


def test_city_boxes_are_centred_and_the_right_width():
    """Longitude lines squeeze together further north, so the degrees differ
    between Chennai and Delhi even though the distance is the same."""
    for city in load_settings().cities:
        left, bottom, right, top = city.bbox
        assert left < city.lon < right and bottom < city.lat < top
        width_km = (right - left) * 111.0 * math.cos(math.radians(city.lat))
        assert abs(width_km - city.radius_km) < 1.0


def test_northern_cities_need_more_degrees_for_the_same_distance():
    settings = load_settings()
    delhi = settings.city("Delhi")
    chennai = settings.city("Chennai")
    span = lambda c: c.bbox[2] - c.bbox[0]      # noqa: E731
    assert span(delhi) > span(chennai)


def test_duplicate_cities_rejected():
    settings = load_settings()
    with pytest.raises(AssertionError, match="same name"):
        make(cities=settings.cities + [settings.cities[0]]).check()


def test_unknown_city_name_reports_what_is_available():
    with pytest.raises(KeyError, match="no city named"):
        load_settings().city("Kolkata")


# --- The train/test split -----------------------------------------------------

def test_training_and_test_years_must_not_overlap():
    with pytest.raises(AssertionError, match="overlap"):
        make(train_years=(2016, 2024), test_years=(2023, 2025)).check()


def test_training_must_come_before_testing():
    """Otherwise we would be predicting the past using the future."""
    with pytest.raises(AssertionError, match="before test"):
        make(train_years=(2023, 2025), val_years=(2022, 2022),
             test_years=(2016, 2021)).check()


def test_cannot_predict_a_year_without_enough_history():
    """With years_in=3, the first three years can only be input, never a target."""
    with pytest.raises(AssertionError, match="history"):
        make(train_years=(2013, 2014)).check()
