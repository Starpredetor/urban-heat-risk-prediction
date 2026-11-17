# Software Requirements Specification

## Urban Heat Island Analysis and Spatiotemporal Prediction Using Multi-Source Satellite Data and Deep Learning: A Comparative Study of Mumbai and Navi Mumbai

**Version:** 3.0 (local, phase-wise)
**Date:** 17 November 2026
**Prepared by:** Shubham Yedve (project team)
**Institution:** Ramrao Adik Institute of Technology (D.Y. Patil University), Navi Mumbai
**Document standard:** Adapted from IEEE Std 830-1998 (condensed)

### Revision history

| Version | Date | Summary |
|---|---|---|
| 1.0 | 31 Aug 2026 | Initial SRS. |
| 2.0-2.1 | Nov 2026 | Two-city redesign; Colab prototype stage. |
| 3.0 | 17 Nov 2026 | **Fresh start.** The project follows the twelve-phase plan exactly and runs entirely on the local machine. Earlier code, Colab support and datasets outside the plan were removed. |

---

## 1. Introduction

### 1.1 Purpose
This document specifies a system that analyses how land-surface temperature (LST) and urban development have changed across Mumbai and Navi Mumbai, identifies consistently hotter areas and the environmental factors associated with them, and uses deep learning to predict near-term LST maps.

### 1.2 Objectives
1. **Historical urban heat analysis:** temperature maps for Mumbai and Navi Mumbai, hotspots, and change over the years.
2. **Environmental factor analysis:** how vegetation, built-up areas, water bodies and weather relate to LST.
3. **Future urban heat prediction:** a ConvLSTM that learns how temperature patterns change over time and predicts upcoming temperature maps.
4. **Interactive prediction dashboard:** select Mumbai or Navi Mumbai, inspect historical heat maps, view predicted maps and hotspots.

### 1.3 Prediction target (normative)
**Future daytime land-surface temperature, MODIS MOD11A2, for the next 1-3 8-day periods, given the preceding 12 observations.**
- Satellite-derived LST is not air temperature and not a person's actual heat exposure.
- Predictions are validated against later observed MODIS LST.
- An 8-24 day prediction is a near-term forecast. It is not a claim about the climate of 2030; a 2030 projection would be a separate scenario-based extension.

### 1.4 Research contribution
A multi-source spatiotemporal urban heat prediction framework for Mumbai and Navi Mumbai that integrates satellite LST, environmental indices, land cover and meteorological variables in a ConvLSTM forecaster, and evaluates both city-specific prediction and cross-city generalisation. The contribution is the combination: two-region prediction, multi-source integration, spatiotemporal learning, cross-city generalisation, environmental feature experiments, and future hotspot identification. ConvLSTM alone is not the novelty.

### 1.5 Definitions
| Term | Meaning |
|---|---|
| LST | Land Surface Temperature |
| MOD11A2 | MODIS 8-day 1 km LST product, Version 6.1 |
| NDVI / NDBI / NDWI | Vegetation / built-up / water indices |
| ConvLSTM | Convolutional Long Short-Term Memory network |
| Persistence | Baseline predicting that the next observation equals the latest one |
| ΔLST | LST_predicted − LST_latest |

### 1.6 Data sources
| Dataset | Period | Purpose |
|---|---|---|
| NASA MODIS MOD11A2 v6.1 | 2000-latest (training 2001-2025) | Main LST time series |
| Landsat 8/9 Collection 2 Level-2 | 2013-latest | Detailed temperature maps; reflectance for NDVI, NDBI, NDWI |
| ESA WorldCover | 2020, 2021 | Land-cover classification |
| Copernicus ERA5-Land | 2000-latest | Weather variables |

All are free. Landsat 8/9 thermal is 100 m native, distributed on a 30 m grid; resampling is not true 30 m thermal detail. ERA5-Land (~9 km) is broad weather context, not neighbourhood weather.

---

## 2. Overall Description

### 2.1 Operating environment
- **Everything runs locally** on the team's Windows machine: Python 3.12 virtual environment (`.venv`), NVIDIA RTX 4060 (8 GB) for training, QGIS for GIS checks.
- **Data access:** the Google Earth Engine Python API (one sign-in reaches all four sources). Downloads are saved as local files; nothing after Phase 2 needs the network.
- No Colab, cloud deployment, React, FastAPI, PostGIS or Kubernetes.

### 2.2 Technology stack
| Component | Technology |
|---|---|
| Programming | Python |
| Satellite data | MODIS, Landsat 8/9 |
| Land cover | ESA WorldCover |
| Weather | ERA5-Land |
| GIS | QGIS, GeoPandas |
| Image processing | Rasterio, GDAL |
| Traditional ML | Scikit-learn |
| Deep learning | PyTorch |
| Prediction model | ConvLSTM |
| Dashboard | Streamlit |
| Maps | Folium |
| Charts | Plotly |
| Storage | GeoTIFF, CSV, NumPy |
| Development | Local Python and Jupyter notebooks |

### 2.3 Project folder structure
```
data/
  boundaries/        mumbai.geojson, navi_mumbai.geojson, combined_study_area.geojson
  raw/               modis/, landsat/, worldcover/, era5/
  processed/
  sequences/
notebooks/
  01_study_area.ipynb            06_random_forest.ipynb
  02_data_acquisition.ipynb      07_sequence_preparation.ipynb
  03_preprocessing.ipynb         08_convlstm_training.ipynb
  04_feature_engineering.ipynb   09_prediction_evaluation.ipynb
  05_historical_analysis.ipynb   10_city_comparison.ipynb
models/              random_forest/, convlstm/
outputs/             historical_maps/, predicted_maps/, charts/, evaluation/
dashboard/app.py
src/                 shared code used by the notebooks
requirements.txt
README.md
```

### 2.4 First working milestone (before any bulk download)
Generate one correct MODIS LST map for Mumbai and one for Navi Mumbai:
1. Obtain both city boundary files.
2. Download one valid MODIS MOD11A2 observation date.
3. Apply quality filtering and temperature scaling.
4. Clip the data to each city boundary.
5. Display both maps in QGIS.
6. Verify that the temperature values and spatial coverage are sensible.

---

## 3. Functional Requirements by Phase

### Phase 1: Define the study areas (Weeks 1-2)
**Objective:** consistent geographical boundaries for both regions.
- FR-1.1 Two separate study-area polygons: Mumbai (Greater Mumbai municipal region) and Navi Mumbai (municipal/planning area, chosen boundary documented precisely). No rectangles around the cities.
- FR-1.2 Download municipal or administrative boundaries.
- FR-1.3 Check their coordinate reference system.
- FR-1.4 Repair invalid geometries if required.
- FR-1.5 Save both boundaries as GeoJSON.
- FR-1.6 Create a combined study-area boundary.
- FR-1.7 Use the same projection and spatial grid for all later processing.
- **Output:** `data/boundaries/mumbai.geojson`, `navi_mumbai.geojson`, `combined_study_area.geojson`.
- **Completion:** both boundaries display correctly in QGIS.

### Phase 2: Build the free dataset collection (Weeks 1-2)
- FR-2.1 Download MODIS MOD11A2 Version 6.1 for both city boundaries.
- FR-2.2 Retrieve daytime LST and its quality-control information.
- FR-2.3 Retrieve Landsat 8/9 imagery for selected years.
- FR-2.4 Download WorldCover tiles covering both regions.
- FR-2.5 Retrieve ERA5-Land weather variables for the same study regions and time periods.
- FR-2.6 Store original downloaded files separately from processed files.
- FR-2.7 Maintain a CSV catalogue with dataset name, date, city, source and file path.
- FR-2.8 Use 2001-2025 as the historical training period where data availability permits; verify the latest complete observations before fixing the final end date.
- **Output:** a locally organised historical dataset for both cities.

### Phase 3: Preprocess all satellite data (Weeks 3-4)
- FR-3.1 Apply the MODIS quality-control mask.
- FR-3.2 Remove invalid and poor-quality temperature observations.
- FR-3.3 Apply the product-specific scale factors.
- FR-3.4 Convert temperature into Celsius.
- FR-3.5 Clip all imagery to the two city boundaries.
- FR-3.6 Reproject datasets to a common coordinate system.
- FR-3.7 Resample compatible layers to a common grid.
- FR-3.8 Handle missing observations carefully.
- FR-3.9 Save the processed images with consistent dates.
- Landsat Collection 2 scale factors: ST_K = DN × 0.00341802 + 149; ST_C = ST_K − 273.15; SR = DN × 0.0000275 − 0.2. Use the official quality flags and product documentation; do not assume every pixel is valid.
- **Output:** clean, aligned raster images for Mumbai and Navi Mumbai.

### Phase 4: Generate environmental features (Weeks 3-4)
- FR-4.1 NDVI = (NIR − Red)/(NIR + Red); NDBI = (SWIR1 − NIR)/(SWIR1 + NIR); NDWI = (Green − NIR)/(Green + NIR).
- FR-4.2 Land cover: built-up, vegetation, water and other classes.
- FR-4.3 Where data quality permits: elevation; distance to major water bodies; built-up land-cover proportion; ERA5-Land temperature and wind variables.
- **Output:** a feature stack for each city and observation period.

### Phase 5: Historical urban heat analysis (Weeks 5-6)
- FR-5.1 Generate historical MODIS temperature maps.
- FR-5.2 Generate detailed Landsat temperature maps for selected years.
- FR-5.3 Calculate annual and pre-monsoon temperature statistics.
- FR-5.4 Identify persistent high-temperature regions.
- FR-5.5 Compare vegetation and built-up regions.
- FR-5.6 Compare Mumbai with Navi Mumbai.
- FR-5.7 Analyse how the spatial distribution of heat changes over time.
- March-May is the main seasonal comparison period; the full MODIS series is retained for model training.
- **Outputs:** historical LST maps, hotspot maps, city comparison charts, temperature trend graphs, NDVI/NDBI versus LST analysis. This is the research baseline.

### Phase 6: Machine-learning baseline (Weeks 7-8)
- FR-6.1 Random Forest regression. Inputs: NDVI, NDBI, NDWI, land-cover proportions, weather features, previous temperature observations where appropriate. Target: observed MODIS daytime LST.
- FR-6.2 Evaluate with MAE = (1/n)Σ|yᵢ − ŷᵢ|, RMSE = √((1/n)Σ(yᵢ − ŷᵢ)²) and R².
- FR-6.3 Training and test data separated by time; neighbouring pixels from the same observation are never split randomly between sets.
- **Output:** a working Random Forest baseline and evaluation report.

### Phase 7: ConvLSTM prediction model (Weeks 9-11)
- FR-7.1 Inputs: previous 12 MODIS observations at 1 km; channels: daytime LST, NDVI or vegetation proxy, built-up proportion, weather, month/season encoding; initial patch 64 × 64 pixels; PyTorch.
- FR-7.2 Outputs: the next 1-3 observations.
- FR-7.3 Channels not available at the 8-day cadence are aligned temporally with a documented aggregation method; coarse weather is not treated as fine-scale data.
- FR-7.4 Sliding windows over the chronological series; no window includes observations from the future relative to its target.
- FR-7.5 Chronological split, e.g. training 2001-2019, validation 2020-2022, testing 2023-2025, adjusted after checking data availability and valid observations.
- FR-7.6 Evaluate MAE, RMSE, R², spatial error maps, performance by city and across seasons.
- FR-7.7 Compare against Random Forest and persistence.
- **Output:** a trained model that produces future temperature maps.

### Phase 8: Two-city research experiments (Weeks 12-13)
- FR-8.1 Experiment A: train and test on Mumbai and Navi Mumbai separately.
- FR-8.2 Experiment B: one model trained on both cities.
- FR-8.3 Experiment C: train on Mumbai and test on Navi Mumbai, then the reverse.
- FR-8.4 Results, positive or negative, are reported as research findings.

### Phase 9: Future prediction and hotspot analysis (Weeks 12-13)
- FR-9.1 Predicted temperature for the next 8, 16 and 24 days.
- FR-9.2 Predicted hotspot regions, using consistent, documented thresholds for relative heat categories.
- FR-9.3 ΔLST = LST_predicted − LST_latest.
- FR-9.4 Predicted hotspots are labelled as model outputs, not confirmed future events.

### Phase 10: Interactive dashboard (Week 14)
- FR-10.1 Streamlit pages: Overview (introduction and city selection), Historical Analysis, Environmental Analysis, Prediction, City Comparison, Model Evaluation.
- FR-10.2 Uses Python, Streamlit, Folium, Plotly, Rasterio, GeoPandas, PyTorch, Scikit-learn.
- FR-10.3 Deep-learning processing stays offline; the dashboard loads already-generated predictions.

### Phase 11: Testing and research validation (Weeks 15-16)
| Test | Verifies |
|---|---|
| Data quality | Missing and invalid satellite pixels are handled |
| Preprocessing | Maps are aligned correctly |
| Model | Predicted temperature is close to observed temperature |
| Baseline comparison | Deep learning adds value over simpler methods |
| Temporal | Future observations are not leaked into training |
| City transfer | Model generalises across the two regions |
| Dashboard | Maps, charts and predictions display correctly |

Ablation: ConvLSTM with temperature only; temperature + vegetation; temperature + vegetation + built-up + weather.

### Phase 12: Documentation and IP preparation (from Week 1)
Maintain: dataset sources and licences; preprocessing methodology; formulas; model architecture; training configurations; experiment results; prediction examples; model evaluation; system architecture; screenshots and version history.
- Copyright covers original code, documentation and diagrams, not the idea. A patent requires novelty, inventive step and industrial applicability, with attention to Section 3(k) of the Indian Patents Act. Investigate patentability separately, with a prior-art search and the college IP cell, before any public disclosure. Do not design the project around a patent being granted.

---

## 4. Schedule (16 weeks)
| Weeks | Work |
|---|---|
| 1-2 | Study area and dataset acquisition |
| 3-4 | Preprocessing and feature engineering |
| 5-6 | Historical analysis |
| 7-8 | Machine-learning baseline |
| 9-11 | Deep-learning development |
| 12-13 | Prediction experiments: forecasting, cross-city transfer, ablation |
| 14 | Dashboard |
| 15-16 | Testing and documentation |

If data collection or training takes longer, reduce dashboard polish rather than validation. Near-term forecasting is proven first; a longer-term (e.g. 2030) scenario projection comes only after.

---

## 5. Implementation assumptions (where the plan leaves a choice open)
These record decisions needed to carry out the plan; they add no features.

| # | Plan step | Assumption |
|---|---|---|
| A1 | Phase 1, boundaries | Provisional boundaries from OpenStreetMap: Greater Mumbai = Mumbai City + Mumbai Suburban districts; Navi Mumbai = the OSM administrative boundary. Provenance saved beside each file; the team verifies in QGIS. |
| A2 | Phase 1, "same projection and grid" | UTM zone 43N (EPSG:32643), which covers both cities. A 1 km grid for MODIS and a 30 m grid for Landsat, both covering the combined study area. |
| A3 | Phase 2, data access | Google Earth Engine Python API, signed in with the account that owns the project's Earth Engine Cloud project. |
| A4 | Phase 3, MODIS QC rule | Keep a pixel when LST was produced (QC_Day bits 0-1 = 00 good or 01 other quality) and its LST error is ≤ 2 K (bits 6-7 ≤ 01). "Other quality" is kept because MODIS assigns it to most dense urban cells: on the first-milestone date it covered 59% of city cells against 13% elsewhere, and dropping it left only 33-40% of each city valid (96% when kept). The strict rule remains available as an option. |

| A5 | Phase 3, clipping | Processed rasters keep the common grid; cells outside both city boundaries are no-data (NaN, or 0 for class maps). A city-id mask (1 Mumbai, 2 Navi Mumbai) is saved per grid. |
| A6 | Phase 3, ERA5-Land onto the 1 km grid | Each city cell copies the value of the ERA5-Land cell it falls in (no interpolation, so no false fine-scale detail). ERA5-Land has no data over the sea; city cells under a sea ERA5 cell take the nearest land cell, and these cells are recorded in a mask. |
| A7 | Phase 3, WorldCover onto the 30 m grid | Mode (most common class) of the 3 x 3 nested 10 m pixels. |
| A8 | Phase 4, land-cover groups | From WorldCover 2021: built-up = 50; vegetation = 10, 20, 30, 40, 90, 95 (tree, shrub, grass, crop, herbaceous wetland, mangroves); water = 80; other = 60, 70, 100. |
| A9 | Phase 4, observation period and "major" water | Landsat features per selected year = per-pixel median of clear March-May observations, with the observation count kept. A major water body is connected permanent water of at least 1 km2. Elevation from SRTM 30 m (fetched in Phase 4, as the plan lists it there). |
| A11 | Phase 5, yearly and seasonal statistics | A cell's annual / March-May LST for a year = mean of its valid 8-day composites in that period (no gap-filling); the number of composites used is kept. |
| A12 | Phase 5, hotspots | Hotspot = a cell at or above its own city's 90th percentile of March-May LST that year (city-relative); persistent hotspot = hotspot in at least half of the years with data. |
| A13 | Phase 5, change over time | Trend = least-squares slope of March-May LST on year per cell with at least 15 years (C/decade). Spatial change = (cell − city mean) 2021-2025 minus the same in 2001-2005, separating where heat moved from overall warming. |
| A10 | Phase 4, ERA5-Land features | Daily values averaged over each MODIS 8-day period (cut at year end), so weather lines up with LST. Wind speed = period mean of daily speed from daily-mean u and v. Variables: 2 m air temperature, dewpoint, skin temperature, 10 m wind speed. |

## 6. Known issues to resolve during the phases
| Issue | Phase | Note |
|---|---|---|
| A 64 × 64 patch at 1 km (64 × 64 km) is larger than either city | 7 | Patch extent must be decided when sequences are built. |
| Landsat indices exist only from 2013, but training starts in 2001 | 4, 7 | The plan's "NDVI or vegetation proxy" must be chosen for 2001-2012. |
| Monsoon cloud leaves many MODIS composites mostly empty | 3 | Handled under "handle missing observations carefully". |
| **MODIS Terra orbit drift** | 5, 7 | Found in Phase 5: the March-May overpass over the study area was stable at about 10:40 local solar time from 2002 to 2021, then 10:33 (2022), 10:19 (2023), 10:04 (2024), 9:40 (2025) (`data/processed/analysis/terra_overpass_time.csv`). Earlier overpass means cooler measured LST, so 2022-2025 are biased low by the satellite. City-mean March-May trends: -0.80 (Mumbai) and -1.19 (Navi Mumbai) C/decade for 2001-2025, but -0.42 and -0.75 for the stable years 2002-2021; the remaining cooling is unexplained and unconfirmed. City-relative results (hotspots, relative change) are unaffected. **Phase 7's test years (2023-2025) fall in the drift period; how to handle this must be decided before training.** |
