"""Colab bootstrap helpers that can be checked outside Colab."""

import pytest

from uhrm.colab import _ee_project, _init_earth_engine, in_colab


def test_missing_project_id_gives_a_clear_error_before_touching_earth_engine():
    # No `ee` import happens, so this runs on machines without earthengine-api.
    with pytest.raises(RuntimeError, match="EE_PROJECT"):
        _init_earth_engine(None)
    with pytest.raises(RuntimeError, match="console.cloud.google.com"):
        _init_earth_engine("")


def test_explicit_project_wins_over_config_and_secret():
    assert _ee_project("explicit-id", "config-id") == "explicit-id"
    assert _ee_project(None, "config-id") == "config-id"


def test_no_secret_available_outside_colab_returns_none():
    assert not in_colab()
    assert _ee_project(None, None) is None
