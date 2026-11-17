"""Project-wide settings and paths (SRS sections 2.3 and 5).

Folder layout follows the plan exactly:
    data/boundaries, data/raw/{modis,landsat,worldcover,era5}, data/processed, data/sequences
    models/{random_forest,convlstm}
    outputs/{historical_maps,predicted_maps,charts,evaluation}
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
BOUNDARIES = DATA / "boundaries"
RAW = DATA / "raw"
RAW_MODIS, RAW_LANDSAT, RAW_WORLDCOVER, RAW_ERA5 = (RAW / n for n in ("modis", "landsat", "worldcover", "era5"))
PROCESSED = DATA / "processed"
SEQUENCES = DATA / "sequences"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"

# Study areas (Phase 1). Keys are file names in data/boundaries.
CITIES = {"mumbai": "Mumbai", "navi_mumbai": "Navi Mumbai"}

# One projection and grid for all later processing (Phase 1 step 7, assumption A2).
CRS = "EPSG:32643"          # UTM zone 43N covers both cities
MODIS_RES_M = 1000          # MODIS MOD11A2 grid
LANDSAT_RES_M = 30          # Landsat analysis grid
GRID_MARGIN_M = 2000        # margin around the combined study area

# Earth Engine Cloud project (assumption A3). Override with the EE_PROJECT environment variable.
EE_PROJECT = os.environ.get("EE_PROJECT", "bright-proxy-514913-a3")

# --- Phase 2: dataset collection ------------------------------------------------
START_YEAR, END_YEAR = 2001, 2025      # historical training period (end year verified in Phase 2)
CATALOG = DATA / "catalog.csv"

MODIS_COLLECTION = "MODIS/061/MOD11A2"                       # Version 6.1
MODIS_BANDS = ("LST_Day_1km", "QC_Day")                     # daytime LST + its QC

LANDSAT_COLLECTIONS = ("LANDSAT/LC08/C02/T1_L2", "LANDSAT/LC09/C02/T1_L2")
LANDSAT_BANDS = ("SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "ST_B10", "QA_PIXEL")
LANDSAT_YEARS = (2014, 2017, 2020, 2023, 2025)              # "selected years"
LANDSAT_MONTHS = (3, 4, 5)                                  # March-May, the plan's main season
LANDSAT_MAX_SCENE_CLOUD = 30                                # percent, per scene

WORLDCOVER = {2020: "ESA/WorldCover/v100", 2021: "ESA/WorldCover/v200"}
WORLDCOVER_RES_M = 10

ERA5_COLLECTION = "ECMWF/ERA5_LAND/DAILY_AGGR"              # daily aggregates of the hourly data
ERA5_BANDS = ("temperature_2m", "dewpoint_temperature_2m", "skin_temperature",
              "u_component_of_wind_10m", "v_component_of_wind_10m")
ERA5_RES_DEG = 0.1                                          # native grid, cell centres on 0.1 deg


def ensure_dirs() -> None:
    """Create the plan's folder structure if missing."""
    for d in (BOUNDARIES, RAW_MODIS, RAW_LANDSAT, RAW_WORLDCOVER, RAW_ERA5, PROCESSED, SEQUENCES,
              MODELS / "random_forest", MODELS / "convlstm",
              OUTPUTS / "historical_maps", OUTPUTS / "predicted_maps", OUTPUTS / "charts",
              OUTPUTS / "evaluation"):
        d.mkdir(parents=True, exist_ok=True)
