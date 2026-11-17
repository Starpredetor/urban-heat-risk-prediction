# Software Requirements Specification

## Urban Heat Island Analysis and Spatiotemporal Prediction Using Multi-Source Satellite Data and Deep Learning: A Comparative Study of Mumbai and Navi Mumbai

**Version:** 2.1 (Finalized, phase-wise, Colab-first)
**Date:** 16 November 2026
**Prepared by:** Shubham Yedve (project team)
**Institution:** Ramrao Adik Institute of Technology (D.Y. Patil University), Navi Mumbai
**Document standard:** Adapted from IEEE Std 830-1998 (condensed)

### Revision history

| Version | Date | Summary |
|---|---|---|
| 1.0 | 31 Aug 2026 | Initial SRS: heat-risk mapping for one metropolitan area. |
| 1.1 | 9 Nov 2026 | Aligned with the built Landsat + Sentinel-2 pipeline; added LST anomaly target, land-cover classification, foundation-model comparison, remote GPU training. |
| 2.0 | 16 Nov 2026 | **Project redefined** as a two-region (Mumbai, Navi Mumbai) spatiotemporal urban heat analysis and prediction system. MODIS MOD11A2 becomes the primary prediction dataset; Landsat is used for detailed analysis. Requirements are organised by twelve delivery phases. Future prediction is a core objective. The risk index, land-cover classifier, foundation-model comparison and web-API stack are removed from scope. See Appendix C. |
| 2.1 | 16 Nov 2026 | **Two execution stages.** Stage A prototypes in Google Colab with data pulled through Google Earth Engine and stored on Drive; a freeze gate (G1) exports and catalogues all data as files; Stage B finalises offline (authoritative GIS boundaries and mapping in QGIS, additional data, ConvLSTM fine-tuning, dashboard, validation). Added cross-cutting requirements FR-X.1 to FR-X.9. See Appendix F. |

---

## 1. Introduction

### 1.1 Purpose

This document specifies the requirements for a system that analyses how land-surface temperature (LST) and urban development have changed across Mumbai and Navi Mumbai, identifies persistently hot areas and the environmental factors associated with them, and predicts near-term LST maps with a deep learning model. It is intended for the project team, the faculty guide, the review panel, and future contributors.

### 1.2 Scope and Objectives

The system, referred to as **UHRM** (the package name `uhrm` is retained from earlier versions; the system no longer computes a risk index), has three objectives and one delivery vehicle:

1. **Analyse the past.** Show how LST and urban development have changed across Mumbai and Navi Mumbai.
2. **Understand the present.** Identify consistently hotter areas and the environmental factors (vegetation, built-up area, water, land cover, weather) associated with them.
3. **Predict the future.** Use a ConvLSTM to predict the spatial pattern of LST for the next 1-3 MODIS 8-day periods, under clearly stated assumptions, and compare it with simpler methods.
4. **Interactive dashboard.** Let a user choose a city, inspect historical maps, and view predicted maps and hotspots.

**Out of scope for this version:** air-temperature or personal heat-exposure prediction, long-range (for example 2030) climate or land-use projection, a Heat Risk Index, a supervised land-cover classifier, real-time nowcasting, sensor networks, mobile apps, a separate web API or database tier, and cloud deployment.

### 1.3 Prediction Target (normative)

The prediction target is **satellite-observed daytime land-surface (skin) temperature from MODIS MOD11A2, in °C, at 1 km, as an 8-day composite.** This is not air temperature, not a human thermal-comfort measure, and not actual heat exposure. Therefore:

- Every predicted product shall be labelled "predicted LST (model output)" with its horizon and issue date.
- Predictions shall be validated only against later observed MODIS LST (FR-P7, FR-P11).
- The initial horizon is the next 1-3 periods (8, 16 and 24 days). This is a near-term forecast. Nothing in this document claims the model predicts the climate or urban heat of a future decade. A scenario-based projection is a separate extension (Section 12).

### 1.4 Research Contribution

The project proposes a multi-source spatiotemporal urban heat prediction framework for Mumbai and Navi Mumbai. It integrates satellite LST, environmental indices, land cover and meteorological variables in a ConvLSTM forecaster, and evaluates both city-specific prediction and cross-city generalisation. The contribution lies in the combination, not in any single component: ConvLSTM and urban heat prediction already exist in the literature. It comprises:

| Element | Why it matters |
|---|---|
| Two-region comparison | Mumbai and Navi Mumbai share a coastal setting but differ in urban layout and development |
| Multi-source integration | Satellite LST, land cover, vegetation and weather in one model |
| Spatiotemporal learning | Spatial patterns and temporal change learned together |
| Cross-city generalisation test | Whether a model trained in one city predicts the other |
| Input ablation | Measures what vegetation, built-up area and weather each add |
| Hotspot outputs | Predicted maps converted to spatially interpretable hotspots |

Positive and negative results are both reportable findings.

### 1.5 Definitions and Acronyms

| Term | Meaning |
|---|---|
| LST | Land Surface Temperature |
| MOD11A2 | MODIS/Terra 8-day LST and emissivity product, 1 km, Version 6.1 |
| QC | Quality-control flags supplied with a product |
| NDVI / NDBI / NDWI | Normalized Difference Vegetation / Built-up / Water Index |
| ConvLSTM | Convolutional Long Short-Term Memory network |
| ERA5-Land | Copernicus hourly land reanalysis, about 9 km native resolution |
| WorldCover | ESA 10 m global land-cover map (2020 and 2021 releases) |
| Persistence | Baseline predicting that the next observation equals the latest one |
| Seasonal climatology | Baseline predicting the mean of the same 8-day composite window over the training years |
| LST anomaly | LST minus its seasonal climatology |
| Model domain | The buffered rectangular region used as model input (FR-P1.4) |
| City mask | The polygon-derived mask of one city's pixels, used for analysis and metrics |
| Hotspot | A pixel at or above a documented, fixed temperature threshold |
| ΔLST | Predicted LST minus latest observed LST |
| MAE / RMSE / R² | Regression error metrics |

### 1.6 References

- IEEE Std 830-1998, Recommended Practice for Software Requirements Specifications.
- NASA LP DAAC, MOD11A2 Version 6.1 product documentation and user guide.
- USGS Landsat 8/9 Collection 2 Level-2 surface temperature and surface reflectance documentation.
- ESA WorldCover product documentation and user manual.
- Copernicus Climate Change Service, ERA5-Land dataset documentation.
- Shi et al., Convolutional LSTM Network (NeurIPS 2015).
- Patents Act 1970, Section 3(k), and IP India's computer-related-inventions examination guidelines (Phase 12).

### 1.7 Phase Overview

Requirements in Section 4 are organised by the twelve phases below. Each phase produces a working output and has completion criteria; the next phase starts only when they are met. Weeks are counted from the start of Phase 1.

| Phase | Name | Weeks | Output |
|---|---|---|---|
| 1 | Define the study areas | 1-2 | Boundary files, common grids, model domain |
| 2 | Build the free dataset collection | 1-2 | Raw data and catalogue |
| 3 | Preprocess all satellite data | 3-4 | Clean, aligned rasters |
| 4 | Generate environmental features | 3-4 | Feature stacks |
| 5 | Historical urban heat analysis | 5-6 | Maps, hotspots, charts (research baseline) |
| 6 | Machine-learning baseline | 7-8 | Random Forest and evaluation report |
| 7 | ConvLSTM prediction model | 9-11 | Trained forecaster |
| 8 | Two-city experiments | 12-13 | Individual, combined and transfer results |
| 9 | Future prediction and hotspots | 12-13 | Predicted maps, ΔLST, hotspots |
| 10 | Interactive dashboard | 14 | Streamlit application |
| 11 | Testing and research validation | 15-16 | Test report, ablation study |
| 12 | Documentation and IP preparation | 1-16 (continuous), final 15-16 | Documents, IP review |

**First milestone (gate before bulk download):** one correct MODIS LST map for Mumbai and one for Navi Mumbai, from a single observation date, quality-filtered, scaled, clipped, and checked in QGIS (Phases 1-3 in miniature).

### 1.8 Execution Stages

The work runs in two stages so that prototyping is fast and the final results are reproducible offline.

| Stage | Weeks | Where | Purpose |
|---|---|---|---|
| **A: Prototype** | 1-11 | Google Colab; data via Google Earth Engine; storage on Google Drive | Prove the pipeline end to end, get the data and libraries with minimal setup, and learn which model and inputs work (Phases 1-7 at prototype level) |
| **Gate G1: Prototype freeze** | end of Week 11 | n/a | Export and catalogue all data as files; record prototype results as provisional |
| **B: Finalisation** | 12-16 | Offline (laptop, QGIS); no Earth Engine dependency | Authoritative boundaries and final GIS mapping, additional data, ConvLSTM fine-tuning, two-city experiments, final predictions, dashboard, validation, documentation |

How each phase splits across the stages:

| Phase | Stage A (Colab prototype) | Stage B (offline final) |
|---|---|---|
| 1 Study areas | Provisional boundaries from open sources; grids and model domain | Authoritative boundaries repaired and verified in QGIS; final cell masks |
| 2 Data | Pull MODIS, Landsat, WorldCover, ERA5-Land through Earth Engine; export to Drive | Add further data if needed; frozen files only |
| 3-6 Preprocess, features, analysis, baselines | Run on prototype data | Re-run on the frozen dataset and final boundaries |
| 7 ConvLSTM | Prototype training on a Colab GPU; single seed allowed | Fine-tuning and final multi-seed runs |
| 8-9 Experiments, predictions | Smoke test only | Full runs and final products |
| 10-11 Dashboard, validation | n/a | Full |
| 12 Documentation | Continuous | Continuous; final pack |

**Gate G1 criteria:** (1) the pipeline runs from raw data to a ConvLSTM-versus-baselines evaluation in Colab; (2) Phase 1-6 completion criteria are met on prototype data; (3) open items OI-1, OI-2, OI-3, OI-4 and OI-6 are resolved or consciously deferred; (4) every prototype dataset is exported and catalogued as files (FR-X.4); (5) all logic is in `src/` and notebooks only orchestrate (FR-X.2); (6) prototype results are labelled provisional (FR-X.7).

---

## 2. Overall Description

### 2.1 Product Perspective

UHRM is an offline research pipeline plus a lightweight dashboard. All heavy computation (preprocessing, training, prediction) runs offline and writes files; the dashboard only reads them.

```
 Boundaries ─┐
 MODIS LST ──┤
 Landsat ────┼─► Preprocess ─► Features ─► Historical analysis ─┐
 WorldCover ─┤        │                                         │
 ERA5-Land ──┘        └─► Sequences ─► Random Forest ──┐        │
                                    └─► ConvLSTM ──────┼─► Predicted maps ─► Streamlit
                                       (persistence,   │   + evaluation      dashboard
                                        climatology)───┘
```

### 2.2 Product Functions

- **F1** Study-area definition and gridding for two cities.
- **F2** Acquisition and cataloguing of free public datasets.
- **F3** Preprocessing: quality masking, scaling, clipping, reprojection, alignment.
- **F4** Feature derivation: NDVI, NDBI, NDWI, land-cover fractions, terrain, weather aggregates.
- **F5** Historical analysis: maps, hotspots, trends, city comparison, factor relationships.
- **F6** Baseline and deep-learning prediction of near-term LST.
- **F7** Two-city experiments: individual, combined, cross-city transfer.
- **F8** Prediction products: predicted maps, ΔLST, hotspots.
- **F9** Dashboard for exploration of the above.
- **F10** Reproducibility: run manifests, dataset catalogue, versioned outputs.

### 2.3 User Classes

| User class | Needs |
|---|---|
| Researcher / student (primary) | Full pipeline control, metrics, reproducibility |
| Faculty guide / reviewer | Clear results, honest baselines, exportable figures |
| Urban-planning audience (secondary) | Readable maps, hotspots, plain-language limitations |

### 2.4 Operating Environment

- **Stage A, prototyping: Google Colab** with a GPU runtime. Data is obtained through the Google Earth Engine Python API (one authentication gives access to MODIS, Landsat, WorldCover, ERA5-Land and terrain with server-side clipping) and saved to Google Drive. Colab runtimes are ephemeral: installed packages and local files disappear when a session ends, so every notebook starts with a setup cell that installs pinned dependencies and mounts Drive, and nothing of value lives only on the runtime disk.
- **Stage B, finalisation: offline.** Windows 11 laptop (RTX 4060, 8 GB) using the repository's Python 3.12 `uv` environment and lockfile, reading the frozen GeoTIFF and NumPy files exported in Stage A, with QGIS for final GIS mapping. The optional GPU server mentioned in v1.1 may be used but no requirement depends on it.
- The same code in `src/uhrm/` runs in both stages; only configuration differs (FR-X.2). Colab's preinstalled Python and packages may differ from the lockfile, so compatibility is checked in Week 1 (OI-10).
- QGIS for boundary preparation and visual verification.
- Dashboard: Streamlit run locally; hosting is optional and not required.

### 2.5 Design and Implementation Constraints

- **C1 Few pixels per city.** At 1 km, Greater Mumbai (on the order of 600 km²) holds roughly 600 pixels and Navi Mumbai (on the order of 160 km²) roughly 160, before coastal and water exclusions. Exact counts are reported in Phase 1. A 64 × 64 patch (4,096 km²) is far larger than either city, which is why FR-P1.4 defines a buffered model domain.
- **C2 Coastal and mixed pixels.** Many 1 km cells straddle land and sea or creek; their LST is a mixture. They must be identified and handled consistently (FR-P1.5).
- **C3 Missing observations.** Monsoon cloud (about June-September) leaves large parts of many 8-day composites invalid. Gaps are not silently filled; targets are never filled (FR-P3.4).
- **C4 Strong seasonality.** An 8-day-ahead LST forecast is dominated by the seasonal cycle. Beating persistence is not sufficient evidence of skill; seasonal climatology is a mandatory baseline, and R² on absolute LST is inflated by the cycle (FR-P7.7).
- **C5 Thermal resolution.** Landsat 8/9 thermal bands are 100 m native, distributed on a 30 m grid. Resampling is not extra thermal detail, and no result may claim 30 m thermal detail. MODIS thermal is 1 km.
- **C6 Coarse weather.** ERA5-Land is about 9 km native. Mumbai and Navi Mumbai fall in the same or adjacent cells, so weather is regional context, not neighbourhood weather, and is nearly identical for both cities.
- **C7 Shared weather and nearby geography.** The two cities experience the same synoptic weather in the same 8-day windows. Cross-city results are therefore weaker evidence of generalisation than results for distant cities, and the transfer design must prevent temporal leakage (FR-P8.4).
- **C8 Time-varying inputs have different histories.** MODIS LST starts in 2000 (full years from 2001). Landsat 8/9 starts in 2013. WorldCover exists only for 2020 and 2021 and is a static layer. A model sequence spanning 2001-2025 cannot use Landsat indices or WorldCover as time-varying inputs (FR-P4.7).
- **C9 Static land cover.** WorldCover cannot show urban development change across 2001-2025. Development change is analysed with Landsat-derived indices from 2013 (FR-P5.7).
- **C10 Account-gated data.** MODIS (NASA Earthdata) and ERA5-Land (Copernicus Climate Data Store) require free accounts and API setup. Credentials are never committed (NFR-6.1).
- **C11 Terra overpass time.** MOD11A2 is a mid-morning (Terra) product and does not capture the afternoon peak of surface heating. Findings describe mid-morning LST.
- **C12 Short effective series for urban change.** Annual and seasonal comparisons from Landsat have few years; trends must be reported with their sampling noise.
- **C13 Ephemeral Colab runtime.** Sessions time out and disconnect, GPU availability and type are not guaranteed, and runtime disk is limited and temporary. Long jobs must checkpoint to Drive and be resumable.
- **C14 Earth Engine dependency in the prototype stage.** Earth Engine requires a registered Cloud project under its research or non-commercial terms, imposes request and export-task quotas, and its collections can be reprocessed or re-versioned. Therefore prototype data is exported and frozen as files, and no final result may depend on a live query.
- **C15 Provisional boundaries and data.** Stage A uses boundaries obtained quickly from open sources. Final GIS mapping in Stage B may change which cells belong to each city, so Stage A results are provisional until regenerated.

### 2.6 Assumptions and Dependencies

- MODIS MOD11A2 v6.1, Landsat 8/9 C2 L2, ESA WorldCover and ERA5-Land remain freely available.
- Boundary files for both cities can be obtained from municipal, survey, census or OpenStreetMap sources and their provenance documented.
- At least one GPU environment (laptop or Colab) is available for ConvLSTM training.

---

## 3. External Interface Requirements

### 3.1 Data Sources

| Source | Product | Access |
|---|---|---|
| NASA LP DAAC | MOD11A2 v6.1 (LST_Day_1km, QC_Day; optionally night LST and view time) | NASA Earthdata (free account), or an equivalent mirror if verified (OI-6) |
| USGS | Landsat 8/9 Collection 2 Level-2 (ST, SR, QA_PIXEL) | STAC via Microsoft Planetary Computer (existing pipeline) or USGS |
| ESA | WorldCover 2020 and 2021 | Free download or STAC |
| Copernicus C3S | ERA5-Land hourly | Climate Data Store API (free account) |
| Terrain | Copernicus DEM or SRTM | Free download |
| Vegetation at MODIS cadence | A MODIS vegetation or reflectance product (OI-2) | NASA Earthdata |
| Boundaries | Municipal / survey / census / OSM | Free download; provenance recorded |

**Prototype access path (Stage A, Colab).** Where available, each source is first obtained through Google Earth Engine and exported to Drive as GeoTIFF:

| Dataset | Earth Engine collection (verify identifier at implementation) |
|---|---|
| MOD11A2 v6.1 | `MODIS/061/MOD11A2` |
| Landsat 8 / 9 C2 L2 | `LANDSAT/LC08/C02/T1_L2`, `LANDSAT/LC09/C02/T1_L2` |
| WorldCover 2020 / 2021 | `ESA/WorldCover/v100`, `ESA/WorldCover/v200` |
| ERA5-Land | `ECMWF/ERA5_LAND/HOURLY` (or a daily-aggregated product) |
| Terrain (SRTM; a single image, so it averages cleanly onto the grid) | `USGS/SRTMGL1_003` |
| MODIS 8-day reflectance (NDVI, NDBI, NDWI at MODIS cadence) | `MODIS/061/MOD09A1` |

Planetary Computer (existing code) remains a fallback for Landsat, and NASA Earthdata and the Copernicus CDS remain fallbacks for MODIS and ERA5-Land. Stage B reads only the exported files and does not require Earth Engine. Prototype boundaries come from open sources via GeoPandas; final boundaries are prepared in QGIS.

### 3.2 Software Interfaces

Python libraries for raster, vector, ML and visualisation work (Appendix B); QGIS for inspection; a Colab runtime as an optional compute target.

### 3.3 User Interface (Dashboard, Phase 10)

| Page | Function |
|---|---|
| Overview | Project introduction and city selection |
| Historical Analysis | Past temperature maps and trends |
| Environmental Analysis | NDVI, NDBI, NDWI, land cover |
| Prediction | ConvLSTM predicted temperature maps, ΔLST, hotspots |
| City Comparison | Mumbai versus Navi Mumbai |
| Model Evaluation | MAE, RMSE, R², skill versus baselines, error maps |

---

## 4. Functional Requirements by Phase

### 4.0 Execution Stages and Portability (applies to all phases)

- **FR-X.1** Every phase shall have a Colab prototype deliverable and, where Section 1.8 says so, an offline final deliverable. The completion criteria below are first met on prototype data and re-met on frozen data in Stage B for every phase that is re-run.
- **FR-X.2 One code base, two modes.** All logic shall live in `src/uhrm/` and be installed in Colab from the repository. Notebooks contain orchestration and display only. Paths, data backend (`earthengine`, `planetary_computer`, `local_files`), compute device and storage root shall come from a configuration file with a mode setting (`colab` or `offline`), never hard-coded.
- **FR-X.3 Interchangeable data backends.** Each source shall be read through a small interface (for example a function returning MODIS LST for a region and date range) with interchangeable backends. Moving from Colab to offline changes the backend, not the callers. A test shall confirm two backends return the same shape and units on a sample.
- **FR-X.4 Freeze and export.** Every dataset used in Stage A shall be exported to Drive as GeoTIFF with validity masks, and catalogued (FR-P2.7) with export date, collection identifier and version, region, date range, scale, CRS and checksum. The frozen dataset shall be downloadable in one step and verifiable by checksum, so that Stage B and later reruns never depend on a live Earth Engine query.
- **FR-X.5 Colab session safety.** Notebooks shall begin with a setup cell that installs the pinned requirements, mounts Drive, authenticates, loads the configuration and reports the GPU. Acquisition shall be chunked by date range and skip finished chunks. Training shall checkpoint to Drive (FR-P7.9).
- **FR-X.6 Secrets.** Earth Engine and other credentials shall use Colab Secrets or interactive authentication, never notebook cells or the repository.
- **FR-X.7 Provisional results.** Stage A outputs shall be labelled "prototype" in their manifests and directory names. No prototype number may appear in the final report unless regenerated in Stage B.
- **FR-X.8 Gate G1.** Stage B shall begin only after the Gate G1 criteria of Section 1.8 are met and recorded.
- **FR-X.9 Dependency compatibility.** A Colab requirements file derived from the lockfile shall be maintained and tested on a fresh Colab runtime. Library versions shall be logged in every run manifest so differences between Colab and offline are visible.

### 4.1 Phase 1: Define the Study Areas (Weeks 1-2)

**Objective:** consistent geographical boundaries and grids for both regions.

- **FR-P1.1** The system shall use two separate study-area polygons: Greater Mumbai (municipal region) and Navi Mumbai. Rectangles around the cities shall not be used as study areas.
- **FR-P1.2** The Navi Mumbai boundary shall be documented precisely: the authority (for example the municipal corporation versus the planning authority), source, retrieval date, and the reason for the choice (OI-1).
- **FR-P1.3** Boundary files shall be checked for coordinate reference system, repaired for invalid geometry, and saved as GeoJSON: `mumbai.geojson`, `navi_mumbai.geojson`, and `combined_study_area.geojson`.
- **FR-P1.4** The system shall define a **model domain**: the bounding region of both cities plus a documented buffer, large enough to cut model patches. City polygons remain the masks for analysis and for every reported metric. (Rationale: constraint C1.)
- **FR-P1.5** The system shall define, for the 1 km model grid, a documented rule assigning cells to a city (for example majority overlap with the polygon), and shall flag coastal and water-dominated cells using a land/water fraction.
- **FR-P1.6** One projected CRS (UTM zone 43N, EPSG:32643, covers both cities) and one set of grid definitions (1 km model grid; 30 m Landsat analysis grid) shall be used by all later phases.
- **FR-P1.7** The system shall report the number of valid 1 km cells per city after masking.
- **FR-P1.8** Stage A may use provisional boundaries from open sources (for example OpenStreetMap through GeoPandas). Stage B shall replace them with authoritative boundaries, repaired and verified in QGIS. The pipeline shall take the boundary file as a parameter, so replacing it and re-running regenerates all masks and downstream products.

**Outputs:** `data/boundaries/` files; grid definitions; cell-count report.
**Completion:** both boundaries display correctly in QGIS; CRS and cell counts documented.

### 4.2 Phase 2: Build the Free Dataset Collection (Weeks 1-2)

**Objective:** a locally organised historical dataset for both cities.

- **FR-P2.1** Download MODIS MOD11A2 v6.1 for the model domain: daytime LST and its QC layer (night LST and view time optional). Historical period 2001 to the latest complete year; the end date is verified from actual availability before being fixed (OI-4).
- **FR-P2.2** Retrieve Landsat 8/9 Collection 2 Level-2 scenes (ST, SR, QA_PIXEL) for 2013 to the latest year, restricted to selected years and to March-May for the main seasonal comparison.
- **FR-P2.3** Retrieve ESA WorldCover tiles for 2020 and 2021 covering the model domain.
- **FR-P2.4** Retrieve ERA5-Land hourly variables for the domain for 2001 to the latest year: air temperature, humidity-related variables (for example dewpoint), 10 m wind components, and optionally soil or skin temperature.
- **FR-P2.5** Retrieve terrain (SRTM elevation) and MODIS MOD09A1 8-day surface reflectance with its state-QA band. MOD09A1 gives NDVI, NDBI and NDWI at the MODIS cadence from 2001 onward (OI-2 resolved). It is fetched raw at 500 m and aggregated to 1 km in Phase 3 with a validity mask.
- **FR-P2.6** Raw downloads shall be stored separately from processed files and never modified.
- **FR-P2.7** The system shall maintain a CSV catalogue with dataset name, date, city, source, file path, and checksum. An older catalogue without the city column is upgraded in place.
- **FR-P2.12** Landsat 8/9 is stored as one cloud-masked (QA_PIXEL bits 0-4) median composite of raw digital numbers per selected year over the configured months, on the Phase 1 30 m analysis grid, with a per-pixel count of clear scenes used. WorldCover is stored both as the 10 m class map over the analysis extent and as class fractions per 1 km cell.
- **FR-P2.13** Before bulk download, the number of MOD11A2 composites available for the configured end year shall be reported, to confirm the year is complete (OI-4).
- **FR-P2.8** Downloads shall be resumable, retry up to 3 times with exponential backoff, and log failures without stopping.
- **FR-P2.9** A single-date test download of MOD11A2 shall pass Phase 3 checks before bulk download starts.
- **FR-P2.10** In Stage A, acquisition shall run through the Earth Engine backend for the model-domain extent and be exported to Drive as GeoTIFF per FR-X.4.
- **FR-P2.11** In Stage B, additional data (further variables, more Landsat years, other products) may be added through the same interface. Each addition shall be catalogued, shall trigger regeneration of the affected features, and shall produce a new dataset version.

**Outputs:** `data/raw/` tree and catalogue CSV.
**Completion:** catalogue complete; expected (about 46 per year) versus retrieved MODIS composites reported per year, with gaps listed.

### 4.3 Phase 3: Preprocess All Satellite Data (Weeks 3-4)

**Objective:** clean, aligned rasters for both cities.

- **FR-P3.1 MODIS.** Apply the QC mask (exact retention rule fixed from the product user guide and recorded), remove fill and out-of-range values, apply the product scale factor (0.02), and convert to Celsius:
  LST_K = DN × 0.02, LST_°C = LST_K − 273.15.
- **FR-P3.2 Landsat.** Apply the official Collection 2 scale factors and quality flags:
  ST_K = DN × 0.00341802 + 149; ST_°C = ST_K − 273.15; SR = DN × 0.0000275 − 0.2.
  Reflectance outside its valid range shall be set to missing, never clamped.
- **FR-P3.3** Clip all data to the model domain and city masks, reproject to the common CRS, and resample to the common grids. Continuous bands use averaging or bilinear resampling; categorical and quality layers use nearest-neighbour or mode only.
- **FR-P3.4 Missing data.** Every processed raster shall carry a validity mask. The system shall not silently fill gaps. Any gap-filling method shall be documented and flagged in the mask, and **target values shall never be gap-filled.** A configurable minimum valid-pixel fraction decides whether a date is usable in training sequences, and the chosen threshold is justified from the observed distribution.
- **FR-P3.5** All outputs shall use consistent dates keyed to the MODIS 8-day composite start date; other sources are aligned to these windows (FR-P4.4).
- **FR-P3.6** Processed data shall be saved with a manifest recording sources, scale factors, masks applied and the grid. Implemented as one compressed NumPy cube per year range (`processed/cube_<y0>_<y1>.npz`, variables on the 1 km model grid) plus `.manifest.json`, because model sequences need the whole time series at once; raw inputs remain GeoTIFF (FR-P2.6).
- **FR-P3.7** Unit tests shall check scaling against hand-computed values, mask decoding against constructed bit patterns, and grid alignment between layers.

**Outputs:** `data/processed/` aligned rasters; manifest.
**Completion:** the first milestone maps pass visual and numeric sanity checks (plausible °C range, coastline in the right place, masked pixels where cloud is expected).

### 4.4 Phase 4: Generate Environmental Features (Weeks 3-4)

**Objective:** a feature stack per city and observation period.

- **FR-P4.1** Compute NDVI = (NIR − Red)/(NIR + Red), NDBI = (SWIR1 − NIR)/(SWIR1 + NIR) and NDWI = (Green − NIR)/(Green + NIR) from Landsat reflectance, with a guard for zero denominators.
- **FR-P4.2** Derive land-cover fractions per 1 km cell from WorldCover: built-up, tree cover, grass and cropland, water and wetland (mangroves kept distinguishable where present), and other. The grouping shall be documented.
- **FR-P4.3** Where data quality permits, derive elevation and distance to major water bodies.
- **FR-P4.4** Aggregate ERA5-Land variables to the same 8-day windows as MODIS, resample to the model grid, and document the aggregation (for example mean of daytime hours near the Terra overpass). Coarse weather shall not be presented as fine-scale measurement.
- **FR-P4.5** Encode time of year with cyclical features (for example sine and cosine of day of year).
- **FR-P4.6** Save the feature stack with a channel manifest listing, per channel, name, unit, source, native resolution, temporal cadence, and aggregation method.
- **FR-P4.7** For sequences spanning 2001-2025, vegetation, built-up and water inputs shall come from NDVI, NDBI and NDWI derived from MOD09A1 (OI-2 resolved). Landsat indices (2013 onward) are used for detailed analysis and for any 2013-onward experiment, not for the full-history model. WorldCover fractions are time-invariant channels and shall be labelled as such.

**Outputs:** feature stacks per city and period.
**Completion:** channel manifest complete; a spot check of each channel against its source for one date.

### 4.5 Phase 5: Historical Urban Heat Analysis (Weeks 5-6)

**Objective:** establish what the model is meant to learn; this is the research baseline.

- **FR-P5.1** Generate historical MODIS LST maps for both cities.
- **FR-P5.2** Generate detailed Landsat LST maps for selected years, labelled with the 100 m native thermal resolution (C5).
- **FR-P5.3** Compute annual and March-May (pre-monsoon) LST statistics per city, using observed pixels only and reporting the valid fraction.
- **FR-P5.4** Identify persistent hotspots by a documented method (for example the fraction of years in which a pixel exceeds a percentile threshold).
- **FR-P5.5** Compare vegetation and built-up areas with LST (NDVI and NDBI versus LST), with water masked before interpreting any relationship.
- **FR-P5.6** Compare Mumbai with Navi Mumbai.
- **FR-P5.7** Analyse change over time in the spatial distribution of heat, and in urban development using Landsat-derived index change from 2013 (C8, C9).
- **FR-P5.8** Present trends with their sampling noise; annual composite means can shift because of which dates were cloud-free, not only because of real change.

**Outputs:** historical LST maps, hotspot maps, city comparison charts, trend graphs, NDVI/NDBI-versus-LST analysis.
**Completion:** each map and chart reproducible from a script or notebook; findings written up per city.

### 4.6 Phase 6: Machine-Learning Baseline (Weeks 7-8)

**Objective:** show whether deep learning improves on conventional methods.

- **FR-P6.1** Train a Random Forest regressor on observed MODIS daytime LST in two variants:
  (a) **explanatory** (concurrent NDVI, NDBI, NDWI proxies, land-cover fractions, weather; no lagged LST), used for the factor analysis of objective 2 and not as a forecasting benchmark;
  (b) **forecast** (lagged LST plus features available at the forecast origin), predicting the same targets and horizons as the ConvLSTM, and used as the forecasting benchmark.
- **FR-P6.2** Also compute the **persistence** and **seasonal climatology** baselines on the identical splits.
- **FR-P6.3** Splits shall be chronological. Neighbouring pixels from the same observation shall never be split randomly between training and test.
- **FR-P6.4** Report MAE, RMSE and R², where MAE = (1/n)Σ|y − ŷ| and RMSE = √((1/n)Σ(y − ŷ)²), computed on observed pixels only, per city and per season.
- **FR-P6.5** Report feature importance, noting that importance indicates association, not causation.
- **FR-P6.6** After these results, the numeric accuracy target of NFR-2.1 shall be fixed and recorded as a controlled change (OI-5).

**Outputs:** `models/random_forest/`, evaluation report.
**Completion:** working Random Forest baseline and a written evaluation report.

### 4.7 Phase 7: ConvLSTM Prediction Model (Weeks 9-11)

**Objective:** the central deep-learning component.

- **FR-P7.1 Inputs.** Each sequence has 12 previous 8-day observations. Channels: daytime LST, vegetation (MODIS-cadence), built-up fraction, weather, seasonal encoding, and a validity-mask channel. Initial patch size 64 × 64 cells cut from the model domain.
- **FR-P7.2 Outputs.** The next 1-3 observations. The implementation shall document whether the multi-step prediction is direct (all steps at once) or recursive.
- **FR-P7.3 Sequences.** Build sliding windows over the chronological series. No window may contain an observation later than its target. Windows with too many invalid target pixels are excluded or handled by a masked loss, as documented.
- **FR-P7.4 Splits.** Chronological: earlier years train, later years validate, the most recent complete years test. Default: 2001-2019 train, 2020-2022 validation, 2023-2025 test, adjusted after the availability check of Phase 2. Configuration loading shall refuse overlapping or misordered splits.
- **FR-P7.5 Normalisation.** Statistics are computed on training years only and stored with the model.
- **FR-P7.6 Training.** PyTorch (existing code base). Loss computed only over observed target pixels. Every run records its configuration, seed, dataset manifest, per-epoch metrics, and best checkpoint in a run directory. Each configuration reported as a final result is trained with **at least three seeds**; Stage A prototype runs may use one.
- **FR-P7.7 Evaluation.** Report MAE, RMSE and R² on observed pixels, per city, per season, and per horizon; spatial error maps; R² additionally on **LST anomalies** (observation minus seasonal climatology), because absolute-LST R² is inflated by the seasonal cycle; and a skill score SS = 1 − RMSE_model/RMSE_reference against persistence, seasonal climatology and the Random Forest forecast variant.
- **FR-P7.8** The model shall be compared against Random Forest, persistence and seasonal climatology on identical splits. If it does not beat the simpler methods, that is reported as a finding.
- **FR-P7.9** Training shall be resumable from a per-epoch checkpoint, written to Drive in Colab, so a disconnect does not lose a run.
- **FR-P7.10** Stage A establishes a working configuration and the baseline comparison. Stage B fine-tunes on the frozen (and any extended) dataset: hyperparameters are chosen on the validation split only, the test split is evaluated once per final configuration, and final multi-seed runs are executed offline.

**Outputs:** `models/convlstm/`, evaluation tables and error maps.
**Completion:** a trained model producing future temperature maps, with the full comparison table.

### 4.8 Phase 8: Two-City Experiments (Weeks 12-13)

**Objective:** the comparative research contribution.

- **FR-P8.1 Experiment A (individual).** Train and test on Mumbai and on Navi Mumbai separately.
- **FR-P8.2 Experiment B (combined).** Train one model on both cities.
- **FR-P8.3 Experiment C (transfer).** Train on one city and test on the other, in both directions.
- **FR-P8.4 No temporal leakage across cities.** Because both cities share the same weather and dates (C7), transfer experiments shall use the same chronological split: the source city is trained on training years only, and the target city is evaluated on test years only.
- **FR-P8.5 No spatial contamination.** City-specific models shall compute loss only inside their own city mask, and the other city's cells inside the model domain shall be masked from the input, or the design shall document and justify an alternative. A test shall verify that no cell of the held-out city contributes to training.
- **FR-P8.6** Report each experiment per city and per season, with mean and standard deviation across seeds, and state plainly that the small number of Navi Mumbai cells limits statistical power.
- **FR-P8.7** Transfer results are reported whether positive or negative.

**Outputs:** experiment results table and discussion.
**Completion:** all of A, B and C run with the same metrics as Phase 7.

### 4.9 Phase 9: Future Prediction and Hotspot Analysis (Weeks 12-13)

**Objective:** predicted maps in a form people can use.

- **FR-P9.1** Generate predicted LST maps for the next 8, 16 and 24 days from the latest available forecast origin, saved as GeoTIFF.
- **FR-P9.2** Compute ΔLST = LST_predicted − LST_latest and map expected warming and cooling.
- **FR-P9.3** Classify predicted temperatures into relative heat categories using fixed, documented thresholds derived from the training-period distribution (default: percentile bands for the same calendar composite window; hotspot = at or above the 90th percentile). Thresholds are configuration and are recorded with each product.
- **FR-P9.4** Produce monthly and seasonal summaries only from **rolling backtest predictions over the test period** and from the latest forecast. No summary shall imply a prediction beyond the forecast horizon.
- **FR-P9.5** Every product shall state "model output, not a confirmed future event", the horizon, the issue date, the training period, and an uncertainty indication (for example spread across seeds).
- **FR-P9.6** Refreshing the forecast with newer MODIS data is a manual batch step; the dashboard states the date of the latest forecast origin.

**Outputs:** `outputs/predicted_maps/`, hotspot and ΔLST maps.
**Completion:** maps for all three horizons generated for both cities and checked against held-out observations where available.

### 4.10 Phase 10: Interactive Dashboard (Week 14)

**Objective:** explore historical and predicted results without retraining anything.

- **FR-P10.1** Implement the pages listed in Section 3.3 in Streamlit, using Folium for maps, Plotly for charts, and Rasterio and GeoPandas for data access.
- **FR-P10.2** The dashboard shall load pre-generated rasters and tables. It shall not train or run heavy inference when a page opens.
- **FR-P10.3** The user shall be able to select Mumbai or Navi Mumbai, view historical maps, view predicted maps and hotspots, and compare the two cities.
- **FR-P10.4** Every layer shall carry a plain-language description and its limitations, including the prediction target statement of Section 1.3.
- **FR-P10.5** Colour schemes shall be perceptually uniform and colour-vision-deficiency safe.
- **FR-P10.6** The dashboard is built in Stage B and shall read only frozen, offline outputs; it shall not depend on Colab or Earth Engine.

**Outputs:** `dashboard/app.py`.
**Completion:** all six pages render correct maps, charts and predictions on a clean machine.

### 4.11 Phase 11: Testing and Research Validation (Weeks 15-16)

**Objective:** show that the project works scientifically, not only that it produces maps.

- **FR-P11.1** Run the tests of Section 10.
- **FR-P11.2 Ablation study.** ConvLSTM with (a) temperature only, (b) temperature plus vegetation, (c) temperature plus vegetation plus built-up plus weather, each with at least three seeds, reported with identical metrics. The runs are scheduled in weeks 12-13 alongside Phase 8; reporting is in this phase.
- **FR-P11.3** Compile all baselines, experiments and ablations into one comparison table with the skill scores of FR-P7.7.
- **FR-P11.4** Document limitations: target definition, mid-morning LST, coarse weather, static land cover, shared weather across the two cities.

**Completion:** a signed-off test report and a final comparison table.

### 4.12 Phase 12: Documentation and IP Preparation (continuous; final in Weeks 15-16)

**Objective:** documentation maintained from the start, not written in the last week.

- **FR-P12.1** Maintain: dataset sources and licences; preprocessing methodology; mathematical formulas; model architecture; training configurations; experiment results; prediction examples; model evaluation; system architecture; screenshots and version history.
- **FR-P12.2** Record each dataset's licence and the attribution it requires, and include it in outputs where required.
- **FR-P12.3 IP review.** Copyright protects original code, documentation and diagrams, not the general idea. Patentability, if pursued, shall be assessed separately against novelty, inventive step, industrial applicability, and Section 3(k), with a prior-art search and advice from the college IP cell or a registered patent professional **before any public disclosure** of potentially patentable details. The project shall not be designed on the assumption that a patent will be granted.
- **FR-P12.4** The repository carries the GNU GPL. The IP review shall decide whether public release of code and any patent filing are compatible before either happens (OI-9).

**Completion:** documents complete and current; IP review recorded.

---

## 5. Non-Functional Requirements

### 5.1 Performance

- **NFR-1.1** Each dashboard page shall render within 5 s on a laptop from pre-generated files.
- **NFR-1.2** Preprocessing the full MODIS series for the model domain shall complete in a single working session on a laptop.
- **NFR-1.3** ConvLSTM training for one configuration and seed shall fit on the 8 GB laptop GPU or a free Colab GPU.

### 5.2 Accuracy and Validity

- **NFR-2.1** The ConvLSTM shall be evaluated against persistence, seasonal climatology and the Random Forest forecast variant. The numeric RMSE target at +1 period is fixed after Phase 6 and recorded as a controlled change (OI-5). Meeting or missing it is reported honestly.
- **NFR-2.2** Predictions shall be validated only against later observed MODIS LST on observed (non-gap-filled) pixels.
- **NFR-2.3** Claims that the ConvLSTM outperforms another method shall state the margin relative to the spread across seeds.
- **NFR-2.4** No result shall be described as air temperature, heat exposure, or long-range projection.

### 5.3 Reliability

- **NFR-3.1** Pipeline stages shall be idempotent and resumable. Existing downloads are skipped.
- **NFR-3.2** Failed downloads retry up to 3 times with exponential backoff before being logged and skipped.
- **NFR-3.3** Interrupted training shall lose no more than one epoch.

### 5.4 Usability

- **NFR-4.1** A first-time user shall reach a predicted map for either city within three clicks of the Overview page.
- **NFR-4.2** Every layer shall carry a plain-language description and its limitations.

### 5.5 Maintainability and Portability

- **NFR-5.1** Code shall be modular, pinned in a lockfile, and tested. Notebooks call functions in `src/`; logic is not duplicated in notebooks.
- **NFR-5.2** A new data source or model shall be added by implementing a documented interface, not by editing the pipeline core.
- **NFR-5.3** The same code shall run unchanged in Colab (Stage A) and offline on Windows (Stage B), differing only in configuration (FR-X.2). A Linux GPU server is optional.

### 5.6 Security and Privacy

- **NFR-6.1** Earth Engine, Earthdata, Copernicus and any other credentials shall be supplied through Colab Secrets, environment variables or user configuration files outside the repository, never committed or left in notebook outputs.
- **NFR-6.2** No personal data is collected or stored.

### 5.7 Ethical and Publication Requirements

- **NFR-7.1** Every published map shall state its date range, sensor, spatial resolution, and uncertainty.
- **NFR-7.2** Outputs are research products, not operational warnings.
- **NFR-7.3** Attribution and licences of all datasets shall be honoured (FR-P12.2).

### 5.8 Reproducibility

- **NFR-8.1** Every result shall be reproducible from its configuration, seed, dataset catalogue and run directory. A Stage B result shall be reproducible from the frozen dataset without network access.
- **NFR-8.2** Comparisons across models shall use the same seeds and splits.

---

## 6. Data Requirements

| Dataset | Resolution | Cadence / period | Use | Phase |
|---|---|---|---|---|
| MODIS MOD11A2 v6.1 (day LST, QC) | 1 km | 8-day, 2001-latest | Primary prediction series | 2, 3 |
| MODIS MOD09A1 surface reflectance | 500 m | 8-day, 2001-latest | Time-varying NDVI, NDBI, NDWI channels | 2, 4 |
| Landsat 8/9 C2 L2 (ST, SR) | 30 m grid (thermal 100 m native) | 16-day, 2013-latest, selected years, Mar-May | Detailed LST maps, NDVI/NDBI/NDWI | 2, 4, 5 |
| ESA WorldCover | 10 m | 2020, 2021 (static) | Land-cover fractions | 2, 4 |
| ERA5-Land | about 9 km | hourly, 2001-latest | Weather channels | 2, 4 |
| Copernicus DEM / SRTM | 30 m | static | Elevation, distance features | 4 |
| Boundaries | vector | static | Study areas | 1 |

**Study area:** Mumbai and Navi Mumbai polygons; buffered model domain for model input; EPSG:32643.

**Storage:** the clipped MODIS series is small; Landsat for selected years dominates. A measured estimate replaces this sentence after Phase 2.

---

## 7. System Models

### 7.1 Pipeline

```
[Boundaries + grids] ─► (2) Acquire ─► [Raw data + catalogue]
                                         ↓
                         (3) Preprocess ─► [Aligned rasters + validity masks]
                                         ↓
                         (4) Features ──► [Feature stacks]
                                         ↓
         ┌───────────────────────────────┼────────────────────────────┐
         ↓                               ↓                            ↓
 (5) Historical analysis      (6) Baselines: persistence,   (7) ConvLSTM sequences
                                  climatology, Random Forest         ↓
                                         └──────────────► (8) Two-city experiments
                                                                      ↓
                                                  (9) Predicted maps, ΔLST, hotspots
                                                                      ↓
                                                   (10) Streamlit dashboard
```

### 7.2 Use Cases

| ID | Actor | Use case | Main flow |
|---|---|---|---|
| UC-1 | Researcher | Build the dataset | Configure → download → preprocess → feature stack |
| UC-2 | Researcher | Train and compare models | Select config and seeds → train → metrics and error maps |
| UC-3 | Researcher | Run the two-city experiments | Run A, B, C → results table |
| UC-4 | Researcher | Refresh the forecast | Add new MODIS composites → predict → regenerate dashboard files |
| UC-5 | Reviewer / planner | Explore historical heat | Select city → historical page → maps and trends |
| UC-6 | Reviewer / planner | View predicted hotspots | Select city → Prediction page → horizon → hotspot map |
| UC-7 | Reviewer | Inspect model validity | Model Evaluation page → baselines, errors, ablations |
| UC-8 | Researcher | Move from prototype to offline | Pass Gate G1 → download frozen dataset from Drive → verify checksums → switch configuration to offline → regenerate results |

---

## 8. Repository Layout

The target layout follows the plan. Existing directories (`config/`, `scripts/`, `tests/`, `runs/`, `docs/`) are kept; `src/uhrm/` is extended rather than replaced. Stage A uses a single combined notebook, `notebooks/uhrm_prototype.ipynb` (setup, access check, boundaries, grid, acquisition and first-milestone map in one file, because Colab works on one file at a time), plus `config/modes/colab.yaml` and `offline.yaml`. The numbered notebooks above describe the Stage B split and are optional. In Colab, `data/` lives on Google Drive and is downloaded to the laptop for Stage B.

```
UrbanHeatPrediction/
├── data/
│   ├── boundaries/        mumbai.geojson, navi_mumbai.geojson, combined_study_area.geojson
│   ├── raw/               modis/, landsat/, worldcover/, era5/
│   ├── processed/
│   └── sequences/
├── notebooks/
│   ├── 01_study_area.ipynb          06_random_forest.ipynb
│   ├── 02_data_acquisition.ipynb    07_sequence_preparation.ipynb
│   ├── 03_preprocessing.ipynb       08_convlstm_training.ipynb
│   ├── 04_feature_engineering.ipynb 09_prediction_evaluation.ipynb
│   └── 05_historical_analysis.ipynb 10_city_comparison.ipynb
├── models/                random_forest/, convlstm/
├── outputs/               historical_maps/, predicted_maps/, charts/, evaluation/
├── dashboard/app.py
├── src/uhrm/              reusable code called by notebooks and scripts
├── config/  scripts/  tests/  runs/  docs/
├── requirements.txt
└── README.md
```

---

## 9. Project Plan

16 weeks from the start of Phase 1. Calendar dates are fixed once the deadline is known (OI-7).

| Weeks | Work | Phases |
|---|---|---|
| 1-2 | Study areas; dataset acquisition; first-milestone maps | 1, 2 |
| 3-4 | Preprocessing; feature engineering | 3, 4 |
| 5-6 | Historical analysis: hotspots, city comparison, environmental relationships | 5 |
| 7-8 | Random Forest, persistence, climatology baselines | 6 |
| 9-11 | Sequence preparation, ConvLSTM training and validation | 7 |
| 12-13 | Two-city experiments; forecasting and hotspots; ablation runs | 8, 9, 11 (runs) |
| 14 | Dashboard | 10 |
| 15-16 | Testing, final report, presentation, IP review | 11, 12 |

Stage A (Colab prototype) covers Weeks 1-11 and is closed by Gate G1 at the end of Week 11. Stage B (offline finalisation) covers Weeks 12-16. Phases 8-11 run in Stage B on frozen data; a smoke test of the Phase 8 and 9 code on prototype data is allowed in Stage A.

Phase 12 documentation runs throughout. If data collection or ConvLSTM training overruns, reduce dashboard polish, not validation.

**Milestone gates:** M1 first-milestone maps correct (Week 2) · M2 aligned dataset and feature stack (Week 4) · M3 research baseline analysis (Week 6) · M4 baselines reported (Week 8) · M5 ConvLSTM evaluated against baselines and prototype frozen, Gate G1 (Week 11) · M6 experiments and predictions done (Week 13) · M7 dashboard working (Week 14) · M8 validated and documented (Week 16).

---

## 10. Verification and Validation

| Test | What it verifies | Phase |
|---|---|---|
| Data quality | Missing and invalid pixels are handled; masks carried through | 3 |
| Preprocessing | Scaling matches hand-computed values; layers align spatially | 3 |
| Leakage | No future observation in any training window; no target year in two splits; normalisation uses training years only | 7 |
| Model | Predicted LST is close to observed LST on held-out years | 7 |
| Baseline comparison | ConvLSTM versus Random Forest, persistence and climatology, with skill scores | 7 |
| Cross-city | Transfer experiments obey FR-P8.4 and FR-P8.5; no held-out-city cell reaches training | 8 |
| Ablation | Contribution of each input group | 11 |
| Reproducibility | A published figure regenerates from its run directory on a clean machine | 11 |
| Dashboard | Maps, charts and predictions display correctly on all pages | 10, 11 |

Unit and integration tests run offline. A synthetic series with a known pattern is used to confirm the training code works; a synthetic result demonstrates wiring only and is never cited as model skill.

---

## 11. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Very few 1 km pixels per city (C1) | High | Buffered model domain; metrics masked to city polygons; report cell counts; state limited statistical power |
| Monsoon cloud leaves many composites mostly invalid (C3) | High | Validity masks and masked loss; minimum valid fraction; metrics per season; never fill targets |
| Seasonal cycle makes the task look easy (C4) | High | Seasonal climatology baseline; skill scores; R² on anomalies |
| Time-varying vegetation unavailable for 2001-2012 from Landsat (C8) | High | MODIS-cadence vegetation product (OI-2); Landsat indices only for 2013-onward work |
| Land cover is static and cannot show urban growth (C9) | Medium | Landsat index change for development; label WorldCover as static |
| Cross-city result is optimistic because both cities share weather (C7) | High | Chronological transfer design; no spatial contamination (FR-P8.4, P8.5); state the limit |
| ConvLSTM does not beat the simpler methods | High | Report honestly; ablations; the finding is a result |
| Boundary definition for Navi Mumbai is ambiguous | Medium | Document authority and choice (OI-1) |
| Coastal mixed pixels distort LST | Medium | Flag by water fraction; consistent inclusion rule |
| Earthdata or CDS access setup delays data | Medium | Start account setup in Week 1; test single-date download first |
| Colab session limits interrupt training | Medium | Per-epoch checkpoints and resume (FR-P7.9) |
| Overclaiming future prediction | High | Target statement (Section 1.3); NFR-2.4; no 2030 claim |
| IP route conflicts with open-source release (OI-9) | Medium | IP review before any public disclosure |
| Schedule slips | Medium | Cut dashboard polish before validation |
| Colab sessions disconnect or time out | Medium | Setup cell, Drive-first storage, chunked acquisition, per-epoch checkpoints (FR-X.5) |
| Earth Engine terms, quotas or reprocessed collections | Medium | Export and freeze all data (FR-X.4); record collection identifiers and versions; Planetary Computer, Earthdata and CDS as fallbacks |
| Colab packages conflict with the lockfile | Medium | Tested Colab requirements file; library versions logged per run (FR-X.9, OI-10) |
| Prototype results change after final boundaries or added data | Medium | Label prototype outputs (FR-X.7); boundary file is a parameter; regenerate in Stage B |
| Notebook code diverges from the repository | Medium | Logic only in `src/`; backend interface test (FR-X.2, FR-X.3) |

---

## 12. Future Enhancements

- A scenario-based long-term projection (for example 2030) using future climate and land-use assumptions, as a separate study after near-term forecasting is proven.
- Afternoon-peak analysis using MODIS Aqua (MYD11A2) alongside Terra.
- Additional cities, which would also strengthen the cross-city generalisation test.
- Heat risk index combining hazard with exposure and vulnerability data.
- Higher-resolution thermal products and night-time thermal (ECOSTRESS).
- Mitigation scenario simulation (cool roofs, tree planting, water bodies).

## 13. Open Items

| ID | Item | Resolves with |
|---|---|---|
| OI-1 | Which Navi Mumbai boundary (municipal corporation versus planning authority) and which source | Phase 1 |
| OI-2 | **Resolved:** MODIS MOD09A1 (8-day, 500 m reflectance) supplies NDVI, NDBI and NDWI at MODIS cadence for 2001-2025. Remaining: confirm its cloud screening (StateQA bits) in Phase 3 | Phase 3 |
| OI-3 | Model-domain buffer size and patch size given the cell counts | Phase 1 |
| OI-4 | Final end year of the historical series | Phase 2 |
| OI-5 | Numeric RMSE target for NFR-2.1 | After Phase 6 |
| OI-6 | Confirm Earth Engine access (registered Cloud project, research use) and that every required collection, including the MODIS-cadence vegetation product, is available there; Earthdata and CDS remain fallbacks | Week 1 |
| OI-7 | Project deadline and calendar start date for the 16 weeks | Team |
| OI-8 | Colab tier (free or paid) and whether its GPU limits suffice for Phase 7 | Week 1, before Phase 7 |
| OI-9 | IP route: open-source release (GPL) versus patent filing | Phase 12 |
| OI-10 | Compatibility between Colab's Python and packages and the repository's Python 3.12 lockfile. Observed 16 Nov 2026: Colab runs Python 3.13 and `uhrm` imports and starts up there via `sys.path`; the 3.12 pins in `pyproject.toml` do not apply on Colab. Still to confirm: the full pipeline run (Earth Engine, rasterio) on 3.13 | Week 1 |
| OI-11 | Which additional data Stage B will add (more years, variables, MODIS Aqua, more cities) | Gate G1 |

---

## Appendix A: Traceability

| Objective | Phases | Key requirements | Verification |
|---|---|---|---|
| 1 Analyse the past | 1-5 | FR-P5.1 to P5.8 | Phase 5 reproducibility |
| 2 Understand the present | 4-6 | FR-P5.4, P5.5, P6.1(a), P6.5 | Phase 5, 6 reports |
| 3 Predict the future | 6-9 | FR-P7.x, P8.x, P9.x, NFR-2.x | Leakage, baseline, cross-city, ablation tests |
| 4 Dashboard | 10 | FR-P10.x | Dashboard test |

## Appendix B: Technology Stack

| Component | Technology |
|---|---|
| Language and environment | Python 3.12, `uv`, lockfile |
| Satellite data | MODIS MOD11A2, Landsat 8/9 |
| Land cover | ESA WorldCover |
| Weather | ERA5-Land |
| GIS | QGIS, GeoPandas |
| Raster processing | Rasterio, GDAL, xarray |
| Traditional ML | scikit-learn |
| Deep learning | PyTorch (existing code); ConvLSTM |
| Dashboard | Streamlit, Folium, Plotly |
| Storage | GeoTIFF, CSV, NumPy |
| Data access (Stage A) | Google Earth Engine Python API, exporting GeoTIFF to Google Drive; Planetary Computer as Landsat fallback |
| Compute | Stage A: Google Colab GPU runtime, data on Drive. Stage B: laptop RTX 4060 (8 GB), offline |

React, FastAPI, PostGIS, container orchestration, multiple deep-learning architectures and cloud deployment are not needed for the first working version.

## Appendix C: Changes from Version 1.1

| Change | Reason |
|---|---|
| Six 40 km city boxes replaced by Mumbai and Navi Mumbai polygons | Boundary-accurate study areas; two connected, differently developed regions |
| MODIS MOD11A2 added as the primary prediction dataset (2001-) | Long, regular 8-day series suitable for sequence learning; Landsat alone is too sparse |
| Landsat demoted to detailed analysis and indices | Thermal is 100 m native; 16-day revisit and cloud gaps |
| Sentinel-2 ingestion dropped | Not in the revised data plan; Landsat reflectance supplies the indices |
| ERA5-Land weather added | Regional meteorological context |
| ESA WorldCover used directly as land cover | The supervised classifier, U-Net, TerraMind and Prithvi comparison are removed |
| Heat Risk Index removed (moved to Future Enhancements) | Not among the three revised objectives |
| Future prediction promoted to a core objective, defined as near-term LST forecasting | Prediction target defined precisely; no long-range claim |
| Random Forest, persistence and seasonal climatology baselines | Honest comparison; climatology added because of strong seasonality |
| Two-city experiments A, B, C added | Research contribution |
| Ablation study added | Measures input contributions |
| Streamlit dashboard replaces React, FastAPI, PostGIS | Sufficient for the project; far less to build |
| Remote GPU requirements (FR-8) removed | Dataset is small; laptop or Colab suffices; server remains optional |
| Requirements reorganised into twelve phases and a 16-week plan | Each phase produces a working output |
| Increments I1-I4 retired | Replaced by phases |

## Appendix D: Relationship to the Existing Implementation

The repository contains an Increment 1 pipeline built for the earlier plan. It is not discarded, but its role changes.

| Component | Status |
|---|---|
| Landsat scaling and QA masking (`src/uhrm/sensors.py`) | Reused in Phases 3-5 |
| STAC acquisition and compositing (`src/uhrm/download.py`) | Reused as the Planetary Computer backend for Landsat; Earth Engine backends for MODIS, ERA5-Land, WorldCover and terrain are new work (FR-X.3) |
| ConvLSTM (`src/uhrm/model.py`) | Starting point for Phase 7; changes needed for validity-mask input, multi-step output and masked loss |
| Training loop, splits, scaler, baselines (`src/uhrm/train.py`, `dataset.py`) | Starting point; climatology baseline and mask handling are new |
| Config validation of split ordering (`src/uhrm/config.py`) | Reused |
| Six-city cubes, Sentinel-2 path, ten-layer cube design | Superseded |
| Existing runs and `data/results/` | Archival; not evidence for the new plan |

## Appendix E: Lessons from Increment 1

Annual Landsat composites over Mumbai showed year-to-year swings of up to 4 °C in the city-wide mean that came from which dates were cloud-free, not from real change, and the ConvLSTM only tied the persistence baseline (1.635 °C versus 1.659 °C RMSE on a pooled run). Three consequences carry into this version: validity masks and honest handling of cloud gaps (FR-P3.4), strong simple baselines including climatology (FR-P6.2), and reporting sampling noise with every trend (FR-P5.8).

## Appendix F: Changes from Version 2.0

| Change | Reason |
|---|---|
| Two execution stages: Colab prototype, then offline finalisation, separated by Gate G1 (Section 1.8) | Fast prototyping with minimal setup now; reproducible, offline final results later |
| Earth Engine is the prototype data path (Section 3.1) | One login reaches all four datasets with server-side clipping. v1.1 rejected Earth Engine over account dependency and quotas; in Colab the ease of access outweighs that, provided data is exported and frozen (FR-X.4) |
| Cross-cutting requirements FR-X.1 to FR-X.9 | One code base for both modes, swappable data backends, freeze and export, Colab session safety, secrets, provisional labelling |
| Boundaries are provisional in Stage A and authoritative in Stage B (FR-P1.8) | Quick start now; final GIS mapping in QGIS later, with the boundary a pipeline parameter |
| Stage B may add data and fine-tune (FR-P2.11, FR-P7.10) | Matches the plan to extend the data and tune the model after the prototype works |
| Single-seed runs allowed in Stage A; final results need three seeds (FR-P7.6) | Keeps prototyping quick without weakening final claims |
| New constraints C13-C15, risks, and open items OI-10 and OI-11 | Colab runtime limits, Earth Engine terms, dependency compatibility |
