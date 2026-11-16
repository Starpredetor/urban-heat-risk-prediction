"""Interchangeable data backends (SRS FR-X.3).

Callers depend on the interface in `base`, never on a particular backend, so
moving from Colab (Earth Engine) to offline (local files) changes only the
backend chosen in the mode config.
"""
