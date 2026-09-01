# Software Requirements Specification

## Urban Heat Risk Mapping and Future Prediction using Multi-Temporal Satellite Imagery and Deep Learning

**Version:** 1.0
**Date:** 31 August 2026
**Prepared by:** Shubham Yedve (project team)
**Institution:** Ramrao Adik Institute of Technology (D.Y. Patil University), Navi Mumbai
**Document standard:** Adapted from IEEE Std 830-1998 (condensed)

---

## 1. Introduction

### 1.1 Purpose

This document specifies the software requirements for a system that maps present-day urban heat risk and forecasts future heat risk for a metropolitan study area, using multi-temporal optical and thermal satellite imagery combined with deep learning models. It is intended for the project team, the faculty guide, the review panel, and any future contributor extending the work.

### 1.2 Scope

The system, referred to as **UHRM** (Urban Heat Risk Mapper), performs five things:

1. Ingests multi-temporal satellite imagery (Landsat 8/9, Sentinel-2, MODIS) and ancillary geospatial/socio-economic data for a defined study area.
2. Derives Land Surface Temperature (LST), spectral indices (NDVI, NDBI, NDWI, albedo, impervious fraction), and Surface Urban Heat Island (SUHI) intensity per time step.
3. Trains deep learning models on the resulting spatio-temporal stack to (a) predict LST/SUHI for future time steps and (b) classify land cover / land use change driving the heating.
4. Combines the predicted hazard with exposure and vulnerability layers into a composite **Heat Risk Index (HRI)**, mapped at grid and ward level.
5. Serves results through an interactive web map with time-slider, scenario comparison, hotspot ranking, and exportable reports.

**Out of scope:** real-time (sub-daily) nowcasting, air-temperature sensor networks, IoT hardware, mobile native apps, and operational early-warning alerting to the public.

### 1.3 Definitions and Acronyms

| Term | Meaning |
|---|---|
| LST | Land Surface Temperature, derived from thermal infrared bands |
| SUHI | Surface Urban Heat Island — urban LST minus rural reference LST |
| NDVI / NDBI / NDWI | Normalized Difference Vegetation / Built-up / Water Index |
| LULC | Land Use / Land Cover |
| HRI | Heat Risk Index = f(Hazard, Exposure, Vulnerability) |
| GEE | Google Earth Engine |
| ConvLSTM | Convolutional Long Short-Term Memory network |
| COG | Cloud-Optimized GeoTIFF |
| AOI | Area of Interest (study area) |
| RMSE / MAE / R² | Regression accuracy metrics |

### 1.4 References

- IEEE Std 830-1998, Recommended Practice for Software Requirements Specifications.
- USGS Landsat 8/9 Collection 2 Level-2 Science Products (surface temperature).
- ESA Copernicus Sentinel-2 MSI Level-2A.
- NASA MODIS MOD11A2 / MYD11A2 8-day LST.
- IPCC AR6 risk framing: Risk = Hazard × Exposure × Vulnerability.
- WorldPop / Census of India ward-level demographic data.

---

## 2. Overall Description

### 2.1 Product Perspective

UHRM is a new, self-contained research system with three tiers:

```
┌─────────────────────────────────────────────────────────┐
│  Presentation Tier — React + Leaflet/MapLibre web app   │
│  time-slider map · scenario compare · hotspot table      │
└───────────────────────┬─────────────────────────────────┘
                        │ REST / JSON + raster tiles
┌───────────────────────┴─────────────────────────────────┐
│  Application Tier — FastAPI service                     │
│  job orchestration · inference API · tile server         │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────┴─────────────────────────────────┐
│  Processing Tier                                        │
│  GEE ingestion → preprocessing (rasterio/GDAL/xarray)   │
│  → DL models (PyTorch: U-Net, ConvLSTM, Transformer)    │
│  → risk index engine → COG / PostGIS store              │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Product Functions (summary)

- **F1** Data acquisition and cataloguing from GEE/USGS/Copernicus for a user-defined AOI and date range.
- **F2** Preprocessing: cloud masking, atmospheric-corrected product selection, co-registration, resampling to a common grid, gap filling, temporal compositing.
- **F3** Feature derivation: LST, emissivity, spectral indices, SUHI intensity, LULC classification.
- **F4** Model training and evaluation for LST forecasting and LULC change prediction.
- **F5** Future scenario prediction for configurable horizons (e.g. +1, +5, +10 years).
- **F6** Heat Risk Index computation and ward-level aggregation.
- **F7** Visualization, comparison, hotspot ranking, and export (GeoTIFF/CSV/PDF).
- **F8** Reproducibility: experiment tracking, versioned datasets and model checkpoints.

### 2.3 User Classes

| User class | Description | Key needs |
|---|---|---|
| Researcher / student (primary) | Team members running experiments | Full pipeline control, metrics, reproducibility |
| Faculty guide / reviewer | Evaluates methodology and results | Clear results, metric tables, exportable figures |
| Urban planner (secondary) | Domain consumer of outputs | Ward-level risk maps, hotspot lists, plain-language reports |
| General visitor | Public demo viewer | Read-only map exploration |

### 2.4 Operating Environment

- **Development/training:** Linux (Ubuntu 22.04+), Python 3.10+, CUDA-capable GPU (≥ 8 GB VRAM) or Google Colab Pro / Kaggle GPU.
- **Cloud preprocessing:** Google Earth Engine (Python API) with an authenticated academic account.
- **Serving:** Docker containers; PostgreSQL + PostGIS; object storage for COGs.
- **Client:** Any modern desktop or mobile browser (Chrome/Firefox/Safari, last two versions).

### 2.5 Design and Implementation Constraints

- **C1** Landsat thermal data is 100 m resampled to 30 m and revisits every 16 days — cloud cover in the Indian monsoon (Jun–Sep) severely limits usable scenes; the system must tolerate irregular temporal sampling.
- **C2** MODIS gives daily LST but at 1 km; any fusion must handle the scale mismatch explicitly.
- **C3** GEE quotas cap export size and concurrent tasks; large AOIs must be tiled.
- **C4** Ground-truth air temperature stations are sparse; validation relies primarily on held-out satellite LST plus limited IMD station comparison.
- **C5** Free-tier hosting limits inference to pre-computed rasters rather than on-demand training.
- **C6** Socio-economic vulnerability data is available only at ward/census-tract resolution, coarser than the hazard grid.

### 2.6 Assumptions and Dependencies

- Continued free availability of Landsat, Sentinel and MODIS archives.
- GEE academic access remains active for the project duration.
- Census/WorldPop and ward boundary shapefiles are obtainable for the AOI.
- At least one GPU resource is available for the training phase.

---

## 3. External Interface Requirements

### 3.1 User Interfaces

- **UI-1 Map view:** basemap, LST/HRI raster overlay with legend, opacity control, year time-slider, layer toggles.
- **UI-2 Scenario panel:** side-by-side or swipe comparison of observed vs predicted year.
- **UI-3 Hotspot table:** ranked wards/grid cells by HRI, sortable, with drill-down chart of the LST time series.
- **UI-4 Job console (researcher only):** launch ingestion/training runs, view logs, metrics, and checkpoints.
- **UI-5 Export dialog:** GeoTIFF, CSV, PNG, and a one-page PDF summary.

### 3.2 Software Interfaces

| Interface | Purpose | Protocol/Format |
|---|---|---|
| Google Earth Engine Python API | Imagery query, cloud masking, export | REST via `earthengine-api` |
| USGS M2M / Copernicus Data Space | Fallback direct scene download | HTTPS, GeoTIFF/SAFE |
| PostGIS | Vector boundaries, ward statistics, run metadata | SQL |
| Object store (S3-compatible) | COG raster tiles and model checkpoints | S3 API |
| MLflow (or Weights & Biases) | Experiment tracking | HTTP |

### 3.3 Communications Interfaces

HTTPS for all client-server traffic; JSON for API payloads; XYZ/WMTS tile endpoints for raster delivery; JWT bearer tokens for authenticated researcher routes.

---

## 4. Functional Requirements

### 4.1 Data Acquisition (FR-1)

- **FR-1.1** The system shall accept an AOI as a GeoJSON polygon or a named administrative boundary.
- **FR-1.2** The system shall accept a date range and target sensor set (Landsat 8/9 L2, Sentinel-2 L2A, MODIS MOD11A2).
- **FR-1.3** The system shall filter scenes by cloud cover threshold (default ≤ 20 %) and record the count of usable scenes per season.
- **FR-1.4** The system shall persist a data catalogue entry (scene ID, sensor, acquisition date, cloud %, checksum) for every ingested scene.
- **FR-1.5** The system shall ingest ancillary layers: ward boundaries, population density, building footprints, elevation (SRTM/Copernicus DEM), and available socio-economic indicators.

### 4.2 Preprocessing (FR-2)

- **FR-2.1** The system shall apply cloud and cloud-shadow masking using the QA_PIXEL / SCL bands.
- **FR-2.2** The system shall reproject and resample all layers to a common CRS (UTM zone of the AOI) and grid (default 30 m).
- **FR-2.3** The system shall generate temporal composites (monthly, seasonal, annual median) to mitigate missing observations.
- **FR-2.4** The system shall fill residual gaps using a documented method (temporal interpolation or spatio-temporal fusion) and flag filled pixels in a quality mask.
- **FR-2.5** The system shall normalize all model inputs and store the scaling parameters with the dataset version.

### 4.3 Feature Derivation (FR-3)

- **FR-3.1** The system shall compute LST in °C from thermal bands, applying emissivity correction derived from NDVI-based fractional vegetation cover (or use Landsat Collection-2 ST product directly).
- **FR-3.2** The system shall compute NDVI, NDBI, NDWI, albedo, and impervious surface fraction per time step.
- **FR-3.3** The system shall compute SUHI intensity as urban-pixel LST minus the mean LST of a defined rural reference buffer.
- **FR-3.4** The system shall produce an LULC classification (minimum classes: built-up, vegetation, water, bare soil) per year with a reported overall accuracy and kappa.

### 4.4 Modelling and Prediction (FR-4)

- **FR-4.1** The system shall train a spatio-temporal model (baseline: ConvLSTM; alternatives: U-Net + temporal encoder, Vision-Transformer variant) to forecast the LST surface for future time steps from the historical stack.
- **FR-4.2** The system shall train an LULC change-prediction component (e.g. CA-Markov baseline vs deep model) whose output feeds the LST forecast as a covariate.
- **FR-4.3** The system shall support configurable forecast horizons of +1, +5, and +10 years.
- **FR-4.4** The system shall use a temporally disjoint train/validation/test split (no future leakage) and report RMSE, MAE, R², and SSIM against held-out observed LST.
- **FR-4.5** The system shall benchmark every deep model against at least two baselines (linear trend per pixel; classical ML such as Random Forest / XGBoost on the same features).
- **FR-4.6** The system shall record each run's hyperparameters, dataset version, random seed, and metrics to the experiment tracker.
- **FR-4.7** The system shall produce per-pixel uncertainty estimates (MC dropout or ensemble spread) alongside every forecast.

### 4.5 Risk Index (FR-5)

- **FR-5.1** The system shall compute Hazard from predicted LST/SUHI intensity and exceedance frequency above a threshold.
- **FR-5.2** The system shall compute Exposure from population density and built-up density.
- **FR-5.3** The system shall compute Vulnerability from available socio-economic and physical indicators (e.g. dependency ratio, green-space access, housing type), with the chosen indicators documented.
- **FR-5.4** The system shall normalize the three components and combine them into an HRI using a documented weighting scheme (default: equal weights; AHP/entropy weighting configurable).
- **FR-5.5** The system shall classify HRI into five ordinal risk classes and aggregate to ward level with the mean and 90th percentile.
- **FR-5.6** The system shall run a sensitivity analysis showing how ward rankings change under alternative weightings.

### 4.6 Visualization and Export (FR-6)

- **FR-6.1** The system shall render LST, SUHI, LULC, and HRI layers as map overlays with a shared legend and year selector.
- **FR-6.2** The system shall allow comparison of any two years (observed or predicted) side by side.
- **FR-6.3** The system shall display a ranked hotspot list with ward name, HRI, and dominant contributing factor.
- **FR-6.4** The system shall plot the LST/HRI time series for any clicked location, distinguishing observed from predicted with an uncertainty band.
- **FR-6.5** The system shall export layers as GeoTIFF, tabular statistics as CSV, and a one-page PDF summary for a selected ward.

### 4.7 System Management (FR-7)

- **FR-7.1** The system shall authenticate researcher-role users before allowing ingestion or training jobs; public map view requires no login.
- **FR-7.2** The system shall log every pipeline stage with timestamps, inputs, outputs, and failures.
- **FR-7.3** The system shall version datasets and model checkpoints so any published figure can be regenerated from its run ID.

---

## 5. Non-Functional Requirements

### 5.1 Performance

- **NFR-1.1** Map tiles for a pre-computed layer shall load in ≤ 2 s on a 10 Mbps connection.
- **NFR-1.2** Ward-level statistics queries shall return in ≤ 3 s.
- **NFR-1.3** A full preprocessing run for a 1,000 km² AOI over 10 years shall complete in ≤ 6 hours on the target environment.
- **NFR-1.4** Inference for one forecast horizon over the AOI shall complete in ≤ 10 minutes on the training GPU.

### 5.2 Accuracy (acceptance targets)

- **NFR-2.1** LST forecast RMSE ≤ 2.0 °C on the held-out test period; the deep model must beat both baselines.
- **NFR-2.2** LULC classification overall accuracy ≥ 85 %, kappa ≥ 0.80.
- **NFR-2.3** Derived LST shall be validated against IMD/AWS station observations where available, with the correlation and bias reported.

### 5.3 Reliability and Availability

- **NFR-3.1** Pipeline stages shall be idempotent and resumable after failure.
- **NFR-3.2** A failed scene download shall retry up to 3 times with exponential backoff before being logged and skipped.
- **NFR-3.3** Demo deployment availability target: 95 % during the evaluation period.

### 5.4 Usability

- **NFR-4.1** A first-time user shall be able to view a risk map for the default AOI within three clicks of landing.
- **NFR-4.2** All map colour schemes shall be perceptually uniform and colour-vision-deficiency safe (e.g. viridis/magma); risk classes shall be distinguishable in greyscale.
- **NFR-4.3** Every layer shall carry a plain-language description of what it shows and its limitations.

### 5.5 Maintainability and Portability

- **NFR-5.1** Code shall be modular (ingest / preprocess / features / models / risk / api / ui) with dependencies pinned in a lockfile.
- **NFR-5.2** The full stack shall run via `docker compose up` on any Linux or macOS host.
- **NFR-5.3** Adding a new sensor or a new model shall require implementing a documented interface, not editing the pipeline core.

### 5.6 Security and Privacy

- **NFR-6.1** API keys and GEE service-account credentials shall be supplied via environment variables, never committed.
- **NFR-6.2** Only aggregated (ward-level or coarser) socio-economic data shall be stored; no individual-level records.
- **NFR-6.3** Write endpoints shall be authenticated and rate-limited.

### 5.7 Ethical and Publication Requirements

- **NFR-7.1** Every published map shall state its date range, sensor, spatial resolution, and uncertainty.
- **NFR-7.2** Vulnerability indicators and their weights shall be disclosed, since they encode value judgements that affect which neighbourhoods appear at risk.
- **NFR-7.3** Outputs shall be labelled as research products, not operational warnings.
- **NFR-7.4** Data, code, and trained weights shall be released under an open licence on acceptance of the paper, subject to institutional approval.

---

## 6. Data Requirements

| Dataset | Source | Resolution | Temporal | Use |
|---|---|---|---|---|
| Landsat 8/9 C2 L2 (ST, SR) | USGS / GEE | 30 m (ST 100 m resampled) | 16-day, 2013– | Primary LST + indices |
| Sentinel-2 L2A | Copernicus / GEE | 10–20 m | 5-day, 2017– | LULC, indices |
| MODIS MOD11A2/MYD11A2 | NASA / GEE | 1 km | 8-day, 2000– | Long-term trend, temporal densification |
| Copernicus DEM / SRTM | ESA / USGS | 30 m | static | Terrain covariate |
| Ward boundaries | Municipal corporation / Survey of India | vector | static | Aggregation units |
| Population | WorldPop / Census of India | 100 m / ward | decadal | Exposure |
| Socio-economic indicators | Census / municipal open data | ward | decadal | Vulnerability |
| Building footprints | OSM / Microsoft/Google open buildings | vector | static | Built density |
| Station air temperature | IMD / local AWS | point | daily | Validation |

**Storage estimate:** ~50–150 GB for a 10-year, 1,000 km² multi-sensor stack including intermediates.

---

## 7. System Models

### 7.1 Pipeline (DFD Level 1)

```
[AOI + date range]
      ↓
(1) Ingest ──→ [Scene catalogue] ──→ (2) Preprocess ──→ [Analysis-ready stack]
                                                              ↓
                                    (3) Feature derivation → [LST, indices, SUHI, LULC]
                                                              ↓
                        ┌─────────────────────────────────────┴──────────┐
                        ↓                                                ↓
              (4a) LST forecast model                       (4b) LULC change model
                        └─────────────────┬──────────────────────────────┘
                                          ↓
                              [Predicted hazard surfaces]
                                          ↓
      [Exposure + Vulnerability] → (5) Risk index engine → [HRI rasters + ward stats]
                                          ↓
                              (6) API + Web map + Exports
```

### 7.2 Primary Use Cases

| ID | Actor | Use case | Main flow |
|---|---|---|---|
| UC-1 | Researcher | Run ingestion for a new AOI | Define AOI/date → validate → queue GEE export → catalogue scenes |
| UC-2 | Researcher | Train and evaluate a forecast model | Select dataset version + config → train → metrics logged → checkpoint saved |
| UC-3 | Researcher | Generate a future scenario | Select model + horizon → infer → write COGs + uncertainty |
| UC-4 | Planner | Identify high-risk wards for 2035 | Open map → select year 2035 → open hotspot table → export PDF |
| UC-5 | Planner | Compare mitigation framing | Compare 2025 observed vs 2035 predicted side by side |
| UC-6 | Visitor | Explore current heat map | Land on default AOI → toggle layers → click a location for its time series |

---

## 8. Project Plan (8-month timeline)

| Phase | Duration | Deliverables |
|---|---|---|
| P1 Literature review + problem formulation | Weeks 1–4 | Survey, gap statement, finalized AOI |
| P2 Data acquisition + AR-stack build | Weeks 5–10 | Scene catalogue, preprocessed stack, EDA report |
| P3 Feature derivation + baselines | Weeks 11–15 | LST/SUHI/LULC layers, baseline metrics |
| P4 Deep model development | Weeks 16–22 | ConvLSTM/U-Net models, ablations, benchmark table |
| P5 Risk index + validation | Weeks 23–26 | HRI maps, sensitivity analysis, station validation |
| P6 Web application | Weeks 27–30 | Deployed map + API + exports |
| P7 Paper writing + submission | Weeks 28–34 | Manuscript draft, figures, code release |
| P8 Report, demo, viva | Weeks 33–35 | Final report, presentation, defence |

**Milestone gates:** M1 stack ready (end P2) · M2 baseline beaten (end P4) · M3 validated risk maps (end P5) · M4 demo live (end P6) · M5 paper submitted (end P7).

---

## 9. Verification and Validation

- **V1 Unit tests** for index formulas, reprojection, and masking against hand-computed fixtures.
- **V2 Integration test** running the whole pipeline on a small AOI and short date range in CI.
- **V3 Model validation** on a temporally held-out period, with the two baselines run on identical splits.
- **V4 Ground validation** against IMD/AWS station temperatures where co-located observations exist.
- **V5 Spatial cross-validation** — train on some city districts, test on others, to check generalization.
- **V6 Reproducibility check** — regenerate one published figure end-to-end from its run ID on a clean machine.
- **V7 Usability walkthrough** with the faculty guide against UC-4 and UC-6.

---

## 10. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Monsoon cloud cover leaves too few usable scenes | High | Seasonal compositing; use MODIS for temporal density; focus analysis on pre-monsoon heat season (Mar–May) |
| Deep model does not beat simple baselines | High | Treat this as a valid finding; report honestly; strengthen with ablations rather than hiding it |
| Insufficient ground-truth for validation | Medium | Use held-out satellite LST as primary; state station-validation limits explicitly |
| GPU/compute limits with large AOIs | Medium | Patch-based training, mixed precision, tile the AOI |
| Vulnerability data unavailable at useful resolution | Medium | Use proxies (building density, green-space access) and document the substitution |
| Scope creep from adding sensors/models | Medium | Freeze scope at M2; extensions become future work |
| Publication timeline slips past the academic deadline | Medium | Draft paper in parallel from P4; target a venue with rolling submission |

---

## 11. Future Enhancements

- Downscaling MODIS to 30 m via spatio-temporal fusion (STARFM/deep fusion) for daily hazard.
- Physics-informed models coupling LST with urban canopy or WRF-UCM outputs.
- Mitigation scenario simulation — cool roofs, tree planting, added water bodies — with predicted LST deltas.
- Integration of night-time thermal (ECOSTRESS) for nocturnal heat stress.
- Health-outcome coupling with hospital admission data where ethically obtainable.

---

## Appendix A — Requirements Traceability (abridged)

| Objective | Requirements | Verification |
|---|---|---|
| Map current urban heat | FR-1, FR-2, FR-3 | V1, V2, V4 |
| Predict future heat | FR-4 | V3, V5 |
| Quantify risk, not just heat | FR-5 | V4, sensitivity analysis (FR-5.6) |
| Make results usable | FR-6, NFR-4 | V7 |
| Make results publishable | FR-7, NFR-7 | V6 |

## Appendix B — Proposed Technology Stack

| Layer | Choice |
|---|---|
| Data access | Google Earth Engine Python API; `stackstac` / `pystac-client` fallback |
| Raster processing | `rasterio`, `xarray`, `rioxarray`, `GDAL`, `geopandas` |
| Deep learning | PyTorch, `torchgeo`, `segmentation-models-pytorch` |
| Classical baselines | scikit-learn, XGBoost |
| Experiment tracking | MLflow or Weights & Biases |
| Backend | FastAPI, `titiler` for COG tiling |
| Database | PostgreSQL + PostGIS |
| Frontend | React + MapLibre GL (or Leaflet), deck.gl for large overlays, Recharts |
| Deployment | Docker Compose; Hugging Face Spaces / Render / institutional server |
