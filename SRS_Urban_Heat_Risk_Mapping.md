# Software Requirements Specification

## Urban Heat Risk Mapping and Future Prediction using Multi-Temporal Satellite Imagery and Deep Learning

**Version:** 1.1 (Finalized)
**Date:** 9 November 2026
**Prepared by:** Shubham Yedve (project team)
**Institution:** Ramrao Adik Institute of Technology (D.Y. Patil University), Navi Mumbai
**Document standard:** Adapted from IEEE Std 830-1998 (condensed)

### Revision history

| Version | Date | Summary |
|---|---|---|
| 1.0 | 31 Aug 2026 | Initial SRS. |
| 1.1 | 9 Nov 2026 | Aligned with the built pipeline (STAC data access, six cities, multi-sensor cube). Added delivery increments, the LST anomaly target, the LULC classification track with a weak-label plus hand-verified evaluation scheme, the candidate-model register and selection protocol, and remote GPU training requirements (FR-8). Deferred MODIS, LULC change prediction, risk index and web app to later increments. See Appendix D. |

---

## 1. Introduction

### 1.1 Purpose

This document specifies the software requirements for a system that maps present-day urban heat risk and forecasts future heat risk for Indian metropolitan study areas, using multi-temporal optical and thermal satellite imagery combined with deep learning models. It is intended for the project team, the faculty guide, the review panel, and any future contributor extending the work.

### 1.2 Scope

The system, referred to as **UHRM** (Urban Heat Risk Mapper), performs five things:

1. Ingests multi-temporal satellite imagery (Landsat 8/9 and Sentinel-2; MODIS deferred) and, in later increments, ancillary geospatial and socio-economic data for defined study areas.
2. Derives Land Surface Temperature (LST), spectral indices (NDVI, NDBI, NDWI), and Surface Urban Heat Island (SUHI) intensity per time step, while preserving the underlying reflectance bands.
3. Trains deep learning models on the resulting spatio-temporal stack to (a) forecast the spatial pattern of LST for future time steps and (b) classify land cover (LULC) for each year.
4. Combines the predicted hazard with exposure and vulnerability layers into a composite **Heat Risk Index (HRI)**, mapped at grid and ward level. *(Later increment.)*
5. Serves results through an interactive web map with time-slider, scenario comparison, hotspot ranking, and exportable reports. *(Later increment.)*

**Out of scope:** real-time (sub-daily) nowcasting, air-temperature sensor networks, IoT hardware, mobile native apps, and operational early-warning alerting to the public.

### 1.3 Definitions and Acronyms

| Term | Meaning |
|---|---|
| LST | Land Surface Temperature, derived from thermal infrared bands |
| SUHI | Surface Urban Heat Island: urban LST minus rural reference LST |
| NDVI / NDBI / NDWI | Normalized Difference Vegetation / Built-up / Water Index |
| LULC | Land Use / Land Cover |
| HRI | Heat Risk Index = f(Hazard, Exposure, Vulnerability) |
| STAC | SpatioTemporal Asset Catalog, the open API used to search satellite archives |
| ConvLSTM | Convolutional Long Short-Term Memory network |
| SimVP | A purely convolutional video-prediction model (no recurrence) |
| Foundation model | A large model pretrained on satellite imagery (e.g. Prithvi-EO-2.0, TerraMind), fine-tuned for a downstream task |
| HLS | Harmonized Landsat Sentinel-2, the 30 m product family Prithvi-EO-2.0 was pretrained on |
| TerraTorch | Fine-tuning toolkit that wraps geospatial foundation models |
| Scene level | The single reference temperature of a city-year (defined in FR-4.8) |
| LST anomaly | A pixel's LST minus the scene level; the spatial pattern with the city-wide offset removed |
| Weak label | A land-cover label taken from an existing global map rather than annotated by the team; known to contain errors |
| Verification sample | A set of points labelled by the team by eye on high-resolution imagery, used only to measure accuracy |
| Increment | A delivered slice of UHRM (Section 1.5) |
| COG | Cloud-Optimized GeoTIFF |
| AOI | Area of Interest (study area) |
| RMSE / MAE / R² / SSIM | Regression and image-similarity accuracy metrics |

### 1.4 References

- IEEE Std 830-1998, Recommended Practice for Software Requirements Specifications.
- USGS Landsat 8/9 Collection 2 Level-2 Science Products (surface temperature).
- ESA Copernicus Sentinel-2 MSI Level-2A.
- Microsoft Planetary Computer STAC API (data access).
- ESA WorldCover global land-cover product (weak labels).
- Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model for Earth Observation Applications, arXiv:2412.02732.
- TerraMind: IBM/ESA multimodal Earth-observation foundation model (model cards on Hugging Face, `ibm-esa-geospatial`).
- SimVP: Simpler yet Better Video Prediction, arXiv:2211.12509; OpenSTL benchmark, arXiv:2306.11249.
- IPCC AR6 risk framing: Risk = Hazard × Exposure × Vulnerability.
- WorldPop / Census of India ward-level demographic data (later increment).

### 1.5 Delivery Increments

The full product is delivered in increments. Requirements below are tagged with the increment that must satisfy them. Untagged requirements belong to the increment named in their section heading, or to I2 where none is named.

| Increment | Content | Status |
|---|---|---|
| **I1** Data pipeline and baseline model | STAC ingest, cloud masking, six-city cube, ConvLSTM, two baselines | Built. Cubes for five of six cities must be rebuilt (see I2). |
| **I2** Dataset v2 and model selection (current focus) | Cube with preserved reflectance bands; LST anomaly forecasting; candidate-model comparison; LULC classification; remote GPU training | This SRS revision |
| **I3** Risk index | Exposure, vulnerability, HRI, ward aggregation, station validation, LULC change prediction | Not started |
| **I4** Web application | API, map UI, exports, deployment | Not started |

Requirements tagged **[I3]** or **[I4]** are retained as the product definition but are not acceptance criteria for I2.

---

## 2. Overall Description

### 2.1 Product Perspective

UHRM is a new, self-contained research system with three tiers. Only the Processing Tier is built in I1 and I2.

```
┌─────────────────────────────────────────────────────────┐
│  Presentation Tier — React + MapLibre web app   [I4]    │
│  time-slider map · scenario compare · hotspot table      │
└───────────────────────┬─────────────────────────────────┘
                        │ REST / JSON + raster tiles
┌───────────────────────┴─────────────────────────────────┐
│  Application Tier — FastAPI service             [I4]    │
│  job orchestration · inference API · tile server         │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────┴─────────────────────────────────┐
│  Processing Tier                                [I1-I3] │
│  STAC ingest → preprocessing (odc-stac / xarray)         │
│  → analysis-ready cube (reflectance + LST + indices)     │
│  → LST forecast models   → LULC classifiers              │
│  → risk index engine [I3] → COG / PostGIS store [I3]     │
└─────────────────────────────────────────────────────────┘
```

Training runs on either of two compute targets using the same code and configuration: the researcher's laptop, or a remote GPU server (Section 2.4, FR-8).

### 2.2 Product Functions (summary)

- **F1** Data acquisition and cataloguing from STAC archives for a configured set of cities and date range.
- **F2** Preprocessing: cloud masking, per-sensor scaling, co-registration, resampling to a common grid, temporal compositing, gap filling.
- **F3** Feature derivation: LST, spectral indices, SUHI intensity, LULC classification.
- **F4** Model training, comparison, and selection for LST forecasting and LULC classification.
- **F5** Future scenario prediction for configurable horizons (+1 year validated in I2; longer horizons see FR-4.3).
- **F6** Heat Risk Index computation and ward-level aggregation. *[I3]*
- **F7** Visualization, comparison, hotspot ranking, and export (GeoTIFF/CSV/PDF). *[I4]*
- **F8** Reproducibility: run manifests, versioned datasets and model checkpoints.
- **F9** Portable training execution on local and remote GPUs.

### 2.3 User Classes

| User class | Description | Key needs |
|---|---|---|
| Researcher / student (primary) | Team members running experiments | Full pipeline control, metrics, reproducibility, remote training |
| Faculty guide / reviewer | Evaluates methodology and results | Clear results, metric tables, honest baselines, exportable figures |
| Urban planner (secondary) | Domain consumer of outputs *[I3/I4]* | Ward-level risk maps, hotspot lists, plain-language reports |
| General visitor | Public demo viewer *[I4]* | Read-only map exploration |

### 2.4 Operating Environment

- **Language and environment:** Python 3.12, managed with `uv`, dependencies pinned in `uv.lock`. Python 3.13 and 3.14 are not supported because PyTorch, rasterio, and odc-stac publish no wheels for them.
- **Local development and small-model training:** Windows 11 laptop, NVIDIA RTX 4060 Laptop GPU (8 GB VRAM), CUDA 12.6 PyTorch build. The code must also run on CPU.
- **Remote training:** a GPU server available to the team. Its hardware (GPU model, VRAM, RAM, disk), operating system, and access method (SSH, hosted notebook, or batch scheduler) are **not yet known** and are recorded as an open item (Section 12, OI-1). All FR-8 requirements are written to hold under any of the three access methods.
- **Data access:** Microsoft Planetary Computer STAC API; no account or export quota (supersedes Google Earth Engine, Appendix D).
- **Serving [I4]:** Docker containers; PostgreSQL + PostGIS; object storage for COGs.
- **Client [I4]:** Any modern desktop or mobile browser (Chrome/Firefox/Safari, last two versions).

### 2.5 Design and Implementation Constraints

- **C1** Landsat thermal data is 100 m resampled to 30 m and revisits every 16 days. Cloud cover in the Indian monsoon (Jun-Sep) makes those months unusable in every study city, so the analysis is restricted to the pre-monsoon season (Mar-May) and the system must tolerate irregular temporal sampling. Chennai has the fewest clear scenes of the six cities.
- **C2** MODIS gives daily LST but at 1 km; any fusion must handle the scale mismatch explicitly. MODIS is deferred (Appendix D).
- **C3** Planetary Computer asset URLs are signed and expire; reads can fail transiently. Large areas must be processed per city and per year, with retry (NFR-3.2).
- **C4** Ground-truth air temperature stations are sparse; validation relies primarily on held-out satellite LST plus limited IMD station comparison [I3].
- **C5** Free-tier hosting limits inference to pre-computed rasters rather than on-demand training [I4].
- **C6** Socio-economic vulnerability data is available only at ward/census-tract resolution, coarser than the hazard grid [I3].
- **C7 Short time series.** Annual composites over 2013-2025 give 13 time steps per city. With a 3-year input window, forecast targets exist only for 2016-2025 (10 years), of which three are test years. Model capacity and seed variance must be judged against this.
- **C8 Scene-level noise.** The city-wide mean LST of a pre-monsoon composite moves by up to 4 °C between consecutive years because of which dates happened to be cloud-free, not because the city changed. This offset is not predictable from local image patches and dominates absolute-LST forecast error (Appendix E).
- **C9 Label quality.** No ground-truth land-cover labels exist for the study areas. Global land-cover maps used as weak labels are themselves imperfect, so accuracy measured against them alone is not accuracy against the ground (FR-3.4, NFR-2.2).
- **C10 Pretraining mismatch.** Foundation models are pretrained at specific resolutions and band sets (Prithvi-EO-2.0 on 30 m HLS reflectance; TerraMind on 10 m Sentinel-2 L2A). UHRM inputs are 30 m. Sentinel-2 bands are averaged to 30 m, which is a distribution shift for models pretrained at 10 m.
- **C11 Mixed-sensor series.** Sentinel-2 starts in 2017. For earlier years, spectral indices come from Landsat. The sensor used is recorded per year (`index_source`) because a sensor change can create a step in a series that is not a real change.

### 2.6 Assumptions and Dependencies

- Continued free availability of Landsat and Sentinel archives through Planetary Computer.
- Hugging Face Hub remains reachable for pretrained weights; weight licences permit the intended use (checked before release, NFR-7.4).
- The remote GPU server provides enough VRAM for the model sizes in Appendix C. This is unverified until the server probe (FR-8.7) is run.
- ESA WorldCover (or an equivalent global product) covers all six study areas.
- Census/WorldPop and ward boundary shapefiles are obtainable for the AOIs [I3].

---

## 3. External Interface Requirements

### 3.1 User Interfaces *[I4]*

- **UI-1 Map view:** basemap, LST/HRI raster overlay with legend, opacity control, year time-slider, layer toggles.
- **UI-2 Scenario panel:** side-by-side or swipe comparison of observed vs predicted year.
- **UI-3 Hotspot table:** ranked wards/grid cells by HRI, sortable, with drill-down chart of the LST time series.
- **UI-4 Job console (researcher only):** launch ingestion/training runs, view logs, metrics, and checkpoints.
- **UI-5 Export dialog:** GeoTIFF, CSV, PNG, and a one-page PDF summary.

Until I4, the researcher interface is the command line (`scripts/build_dataset.py`, `scripts/train.py`, `scripts/make_maps.py`).

### 3.2 Software Interfaces

| Interface | Purpose | Protocol/Format |
|---|---|---|
| Planetary Computer STAC API | Imagery search and signed asset access | HTTPS, `pystac-client`, `planetary-computer` |
| `odc-stac` | Read STAC items onto a fixed grid | Python API |
| ESA WorldCover (via STAC or direct download) | Weak land-cover labels | GeoTIFF |
| Hugging Face Hub | Pretrained foundation-model weights | HTTPS |
| TerraTorch | Fine-tuning wrapper for Prithvi / TerraMind | Python API |
| Remote GPU server | Training execution | SSH / notebook / batch scheduler (OI-1) |
| PostGIS *[I3]* | Vector boundaries, ward statistics | SQL |
| Object store *[I4]* | COG tiles and checkpoints | S3 API |

### 3.3 Communications Interfaces *[I4]*

HTTPS for all client-server traffic; JSON for API payloads; XYZ/WMTS tile endpoints for raster delivery; JWT bearer tokens for authenticated researcher routes.

---

## 4. Functional Requirements

### 4.1 Data Acquisition (FR-1)

- **FR-1.1** The system shall accept the study areas as a list of named cities, each defined by centre latitude/longitude and a square radius (default 40 × 40 km), from a single configuration file. Arbitrary GeoJSON polygons are a future extension.
- **FR-1.2** The system shall accept a year range and sensor set: Landsat 8/9 Level-2 (default from 2013) and Sentinel-2 L2A (from 2017). MODIS is deferred.
- **FR-1.3** The system shall filter scenes by cloud cover (default ≤ 20 %), restrict to configured months (default Mar-May), cap scenes per year (default 12), and record the count of usable scenes per city-year.
- **FR-1.4** The system shall persist a catalogue entry (scene ID, sensor, acquisition date, cloud %) for every ingested scene.
- **FR-1.5** The system shall ingest ancillary layers: ward boundaries, population density, building footprints, elevation, and socio-economic indicators. **[I3]**
- **FR-1.6** The system shall ingest an external land-cover product for each study area as weak labels for FR-4.9, recording its name, version, year, and licence in the dataset manifest.

### 4.2 Preprocessing (FR-2)

- **FR-2.1** The system shall mask cloud and cloud shadow using `QA_PIXEL` (Landsat) and `SCL` (Sentinel-2). Water shall be kept, because it has a real surface temperature. Quality bands shall be resampled by nearest neighbour only.
- **FR-2.2** The system shall reproject and resample all layers to a common grid. The CRS shall be the UTM zone of each city, computed from its longitude (India spans zones 43N and 44N). Default resolution is 30 m; Sentinel-2 reflectance shall be averaged down to it. Thermal data shall never be upsampled.
- **FR-2.3** The system shall generate one annual composite per city-year from the configured months, using the per-pixel median of cloud-free observations, and shall store the number of clear observations per pixel.
- **FR-2.4** The system shall fill residual gaps using documented temporal interpolation and flag filled values so that filled pixels can be excluded from metrics.
- **FR-2.5** The system shall normalize model inputs using statistics computed on **training-period years only** and shall store the scaling parameters with the dataset version.
- **FR-2.6** The system shall apply each sensor's scaling rules exactly, including the Sentinel-2 processing-baseline offset of -1000 DN for scenes from 25 January 2022 onward, and shall reject reflectance outside its valid range as missing rather than clamp it. Tests shall guard both rules.
- **FR-2.7 (new)** The system shall **preserve the six surface-reflectance bands** (blue, green, red, nir, swir16, swir22) in the cube alongside the derived layers. Spectral indices alone are not sufficient input for pretrained models and cannot be used to derive new indices or albedo later.
- **FR-2.8 (new)** The system shall record, per city-year, which sensor supplied the reflectance bands and indices (`index_source`).

### 4.3 Feature Derivation (FR-3)

- **FR-3.1** The system shall obtain LST in °C from the Landsat Collection-2 surface temperature product directly, treating values outside 0-70 °C as invalid.
- **FR-3.2** The system shall compute NDVI, NDBI, and NDWI per time step. Albedo (Liang 2001 coefficients) and impervious surface fraction are **[I3]** and derive from the preserved bands (FR-2.7) and the LULC product.
- **FR-3.3** The system shall compute SUHI intensity as urban-pixel LST minus the mean LST of a rural reference buffer. SUHI is a diagnostic layer, not a model input, because it is a linear function of LST.
- **FR-3.4** The system shall produce an LULC classification per year with minimum classes built-up, vegetation, water, and bare soil, and shall report overall accuracy, kappa, and per-class precision/recall measured on the **verification sample** (FR-4.9, NFR-2.2).

**Cube specification (Dataset v2).** One file per city with dimensions `(year, layer, y, x)` and ten layers in this order: `lst, ndvi, ndbi, ndwi, blue, green, red, nir, swir16, swir22`. Alongside it: `n_clear(year, y, x)`, `index_source(year)`, the CRS and transform, and a manifest with configuration hash, band order, and content hash. Dataset v1 (four layers) is superseded; changing the layer list changes the dataset version.

### 4.4 Modelling and Prediction (FR-4)

**Job 1: LST forecasting**

- **FR-4.1** The system shall implement and compare the candidate forecasters of Appendix C: a ConvLSTM (reference), SimVP, and Prithvi-EO-2.0 with a frozen encoder and a trained head. All shall consume the same input window and produce the same output (FR-4.8), so that swapping a model is a configuration change (NFR-5.3).
- **FR-4.2** The system shall train an LULC change-prediction component whose output feeds the LST forecast as a covariate. **[I3]**
- **FR-4.3** The system shall validate a **+1 year** forecast horizon. Horizons of +5 and +10 years may be produced by recursive roll-out as demonstrations, but shall be labelled unvalidated and carry no accuracy claim unless a backtest on held-out years supports one.
- **FR-4.4** The system shall use a temporally disjoint split on **target year** with no future leakage, configured as: train targets 2016-2021, validation target 2022, test targets 2023-2025, with a 3-year input window. Configuration loading shall refuse overlapping or misordered splits. The system shall report RMSE, MAE, R², and SSIM against held-out observed LST over valid (non-filled) pixels only. Water shall be handled explicitly and identically for the model and all baselines, and the handling shall be recorded in the run manifest.
- **FR-4.5** The system shall benchmark every deep model against, at minimum, (i) persistence (next year equals last year) and (ii) a per-pixel linear trend, both evaluated on identical splits and in both the anomaly and absolute framings (FR-4.8). A classical machine-learning baseline (Random Forest or XGBoost on the same features) **should** also be reported.
- **FR-4.6** The system shall record each run's resolved configuration, dataset hash, random seed, per-epoch metrics, best checkpoint, and final metrics table in a self-contained run directory. MLflow or Weights & Biases are optional later additions.
- **FR-4.7** The system **should** produce per-pixel uncertainty for each forecast, using the spread across seeds (FR-4.10) or MC dropout. Mandatory from I4.
- **FR-4.8 (new) Target definition.** The forecasting target is the **LST anomaly**, defined as pixel LST minus the scene level, where the scene level is a single reference temperature per city-year computed over valid land pixels. The exact definition (scene mean vs rural-reference mean) is fixed in the I2 design specification and recorded in the manifest (OI-2). Consequently:
  - The model predicts the spatial pattern of next year's anomaly; the city-wide offset, which the model cannot observe from local patches (C8), is removed before training.
  - Absolute LST is reconstructed as predicted anomaly plus a scene level. The scene level shall be handled as a separate, explicitly reported component (forecast by the baselines of FR-4.5, or supplied as an oracle for diagnosis), never silently folded into the model's score.
  - Every run shall report **both** the anomaly-framed metrics (the measure of spatial skill) and absolute-framed metrics using a stated scene-level source. NFR-2.1 is assessed on the absolute °C metric.
  - The raw-LST result of Increment 1 is retained as the reference point for showing whether the anomaly framing helped.
- **FR-4.9 (new) Job 2: LULC classification.** The system shall train a four-class land-cover classifier and compare at least: (a) a U-Net with an ImageNet-initialised ResNet-34 encoder (`segmentation-models-pytorch`) as the baseline, and (b) a foundation-model encoder (TerraMind or Prithvi-EO-2.0) with a segmentation head via TerraTorch.
  - Training uses the weak labels of FR-1.6.
  - **Accuracy is measured only on the verification sample:** at least 300 points, stratified across the four classes and the six cities, labelled by the team on high-resolution imagery and without sight of any model output. Each class shall have at least 30 points; a class that cannot reach this is reported as unevaluated rather than estimated.
  - Agreement with the weak labels is reported separately and labelled as such.
  - Validation uses spatial hold-out (tiles or whole cities not seen in training) so that neighbouring pixels do not leak.
  - Because global products exist only for specific years, the classifier is trained on the labelled years and applied to the others. This temporal transfer shall be stated wherever per-year maps are shown (OI-3).
- **FR-4.10 (new) Model selection protocol.** Candidates shall be compared on identical splits, identical inputs, and the same target. Each candidate shall be trained with **at least three random seeds** and reported as mean ± standard deviation. Hyperparameters and the final candidate are chosen on the validation split only; the test split is evaluated once per final candidate. Results where a large model does not beat a simple one are reported as findings, consistent with Section 10.

### 4.5 Risk Index (FR-5) *[I3]*

- **FR-5.1** The system shall compute Hazard from predicted LST/SUHI intensity and exceedance frequency above a threshold.
- **FR-5.2** The system shall compute Exposure from population density and built-up density.
- **FR-5.3** The system shall compute Vulnerability from available socio-economic and physical indicators (e.g. dependency ratio, green-space access, housing type), with the chosen indicators documented.
- **FR-5.4** The system shall normalize the three components and combine them into an HRI using a documented weighting scheme (default: equal weights; AHP/entropy weighting configurable).
- **FR-5.5** The system shall classify HRI into five ordinal risk classes and aggregate to ward level with the mean and 90th percentile.
- **FR-5.6** The system shall run a sensitivity analysis showing how ward rankings change under alternative weightings.

### 4.6 Visualization and Export (FR-6)

- **FR-6.1 [I2]** The system shall produce static maps and charts per city: LST, indices, SUHI, and year-by-year comparison on a shared colour scale (`scripts/make_maps.py`).
- **FR-6.2 [I4]** The system shall render LST, SUHI, LULC, and HRI layers as map overlays with a shared legend and year selector, and allow side-by-side comparison of any two years.
- **FR-6.3 [I4]** The system shall display a ranked hotspot list with ward name, HRI, and dominant contributing factor.
- **FR-6.4 [I4]** The system shall plot the LST/HRI time series for any clicked location, distinguishing observed from predicted with an uncertainty band.
- **FR-6.5 [I4]** The system shall export layers as GeoTIFF, tabular statistics as CSV, and a one-page PDF summary for a selected ward.

### 4.7 System Management (FR-7)

- **FR-7.1 [I4]** The system shall authenticate researcher-role users before allowing ingestion or training jobs; public map view requires no login.
- **FR-7.2** The system shall log every pipeline stage with timestamps, inputs, outputs, and failures.
- **FR-7.3** The system shall version datasets and model checkpoints so any published figure can be regenerated from its run ID.

### 4.8 Training Execution (FR-8) — new

Training must run unchanged on the laptop and on a remote GPU server whose specification and access method are not yet known.

- **FR-8.1** Training shall be launched by one entry point that takes a configuration file path and an output directory. No path, device, or batch size shall be hard-coded.
- **FR-8.2** The device shall be resolved at run time (CUDA if present, otherwise CPU). Mixed-precision training shall be switchable by configuration, and batch size and patch size shall be configurable so a run can be sized to the available VRAM.
- **FR-8.3** Training shall write a checkpoint at the end of every epoch containing model, optimizer, scheduler, random-number state, and epoch counter, and shall resume from it with a single flag, so a run survives disconnects, queue time limits, and pre-emption.
- **FR-8.4** The dataset shall be portable. The cube files plus their manifest shall be transferable as a unit (e.g. `rsync`), and the training entry point shall verify the content hash on start and refuse to run on a mismatch.
- **FR-8.5** The server environment shall be reproducible from the same lockfile. Where `uv` cannot be installed on the server, an exported `requirements.txt` shall be provided and kept in sync with the lockfile.
- **FR-8.6** Runs shall be headless: all output to a log file, no interactive prompts, safe under `tmux`/`nohup` or a batch scheduler. A scheduler submission template shall be provided once the access method is known.
- **FR-8.7** The system shall include a **server probe** script that reports GPU model and VRAM, CPU count, RAM, free disk, Python version, whether `uv`, `git`, `rsync`, and a scheduler are present, and outbound access to the Hugging Face Hub. Its output closes OI-1.
- **FR-8.8** A run directory shall be self-contained and copyable back to the laptop for analysis, with no reference to absolute paths on the server.

---

## 5. Non-Functional Requirements

### 5.1 Performance

- **NFR-1.1 [I4]** Map tiles for a pre-computed layer shall load in ≤ 2 s on a 10 Mbps connection.
- **NFR-1.2 [I4]** Ward-level statistics queries shall return in ≤ 3 s.
- **NFR-1.3** A full six-city dataset build (Dataset v2, ten layers) shall complete in ≤ 3 hours on a broadband connection. The four-layer build of Increment 1 took about 85 minutes.
- **NFR-1.4** Inference for one forecast horizon over one city shall complete in ≤ 10 minutes on the training GPU.
- **NFR-1.5** The reference ConvLSTM and the U-Net baseline shall train on the 8 GB laptop GPU. Foundation-model runs may require the remote server.

### 5.2 Accuracy (acceptance targets)

- **NFR-2.1** LST forecast RMSE ≤ 2.0 °C (absolute framing) on the held-out test period, and the selected deep model must beat both baselines of FR-4.5. If it does not, the result is reported as a finding with the supporting decomposition (Appendix E).
- **NFR-2.2** LULC overall accuracy ≥ 85 % and kappa ≥ 0.80, **measured on the verification sample**, not against the weak labels.
- **NFR-2.3** Derived LST shall be validated against IMD/AWS station observations where available, with correlation and bias reported. **[I3]**
- **NFR-2.4** Any claim that a foundation model outperforms a simpler model shall be supported by the multi-seed comparison of FR-4.10 and shall state the margin relative to the seed spread.

### 5.3 Reliability and Availability

- **NFR-3.1** Pipeline stages shall be idempotent and resumable after failure. Cities already built are skipped on re-run.
- **NFR-3.2** A failed scene download shall retry up to 3 times with exponential backoff before being logged and skipped.
- **NFR-3.3 [I4]** Demo deployment availability target: 95 % during the evaluation period.
- **NFR-3.4** An interrupted training run shall be resumable without loss of more than one epoch (FR-8.3).

### 5.4 Usability *[I4]*

- **NFR-4.1** A first-time user shall be able to view a risk map for the default AOI within three clicks of landing.
- **NFR-4.2** All map colour schemes shall be perceptually uniform and colour-vision-deficiency safe (e.g. viridis/magma); risk classes shall be distinguishable in greyscale.
- **NFR-4.3** Every layer shall carry a plain-language description of what it shows and its limitations.

### 5.5 Maintainability and Portability

- **NFR-5.1** Code shall be modular (ingest / preprocess / features / models / risk / api / ui) with dependencies pinned in a lockfile. Modules for later increments are created only when those increments begin.
- **NFR-5.2 [I4]** The full stack shall run via `docker compose up` on any Linux or macOS host.
- **NFR-5.3** Adding a new sensor, forecaster, or classifier shall require implementing a documented interface, not editing the pipeline core. Sensors implement the `Sensor` interface; forecasters share one input/output contract (FR-4.1).
- **NFR-5.4** The code shall run on Windows (development) and Linux (server).

### 5.6 Security and Privacy

- **NFR-6.1** API keys, Hugging Face tokens, and server credentials shall be supplied via environment variables or the user's SSH configuration, never committed.
- **NFR-6.2 [I3]** Only aggregated (ward-level or coarser) socio-economic data shall be stored; no individual-level records.
- **NFR-6.3 [I4]** Write endpoints shall be authenticated and rate-limited.

### 5.7 Ethical and Publication Requirements

- **NFR-7.1** Every published map shall state its date range, sensor, spatial resolution, and uncertainty.
- **NFR-7.2 [I3]** Vulnerability indicators and their weights shall be disclosed, since they encode value judgements that affect which neighbourhoods appear at risk.
- **NFR-7.3** Outputs shall be labelled as research products, not operational warnings.
- **NFR-7.4** Data, code, and trained weights shall be released under an open licence on acceptance of the paper, subject to institutional approval. The licences of all pretrained weights and label products used shall be checked before release.
- **NFR-7.5** Land-cover maps shall disclose their label source, the weak-label method, and the verification-sample accuracy, and shall not present weak-label agreement as accuracy.

### 5.8 Reproducibility

- **NFR-8.1** Every run shall be reproducible from its run directory: configuration, seed, and dataset hash. Residual GPU non-determinism shall be documented, and results compared across laptop and server within a stated tolerance (V9).
- **NFR-8.2** Model comparisons shall use the same seeds across candidates.

---

## 6. Data Requirements

| Dataset | Source | Resolution | Temporal | Use | Status |
|---|---|---|---|---|---|
| Landsat 8/9 C2 L2 (ST, SR) | USGS via Planetary Computer | 30 m (ST 100 m resampled) | 16-day, 2013- | LST, reflectance, fallback indices | Built |
| Sentinel-2 L2A | Copernicus via Planetary Computer | 10-20 m, averaged to 30 m | 5-day, 2017- | Reflectance, indices | Built |
| ESA WorldCover (or equivalent) | ESA | 10 m | specific release years | Weak LULC labels | I2 |
| Verification sample | Team-labelled on high-resolution imagery | points | one reference year | LULC accuracy | I2 |
| MODIS MOD11A2/MYD11A2 | NASA | 1 km | 8-day, 2000- | Long-term trend | Deferred |
| Copernicus DEM / SRTM | ESA / USGS | 30 m | static | Terrain covariate | I3 |
| Ward boundaries | Municipal corporation / Survey of India | vector | static | Aggregation units | I3 |
| Population | WorldPop / Census of India | 100 m / ward | decadal | Exposure | I3 |
| Socio-economic indicators | Census / municipal open data | ward | decadal | Vulnerability | I3 |
| Building footprints | OSM / open buildings | vector | static | Built density | I3 |
| Station air temperature | IMD / local AWS | point | daily | Validation | I3 |

**Study areas:** Mumbai, Delhi, Ahmedabad, Nagpur, Hyderabad, Chennai; 40 × 40 km each (1333 × 1333 px at 30 m); per-city UTM zone; years 2013-2025; March-May.

**Storage estimate:** Dataset v1 (four layers) measured about 400 MB per city (`mumbai.nc`, 397 MB). Dataset v2 (ten layers) is estimated at about 1 GB per city, roughly 6 GB for six cities, plus a scene cache. This is an estimate until the rebuild.

---

## 7. System Models

### 7.1 Pipeline (DFD Level 1)

```
[Cities + year range]
      ↓
(1) Ingest ──→ [Scene catalogue] ──→ (2) Preprocess ──→ [Analysis-ready cube v2]
                                                           (LST + indices + 6 bands)
                                                              ↓
                                    (3) Feature derivation → [LST, indices, SUHI]
                                                              ↓
                        ┌─────────────────────────────────────┴──────────┐
                        ↓                                                ↓
        (4a) LST anomaly forecast                          (4b) LULC classification
   ConvLSTM · SimVP · Prithvi-EO-2.0                  U-Net · TerraMind / Prithvi
        + scene-level component                       weak labels → verification sample
                        └─────────────────┬──────────────────────────────┘
                                          ↓
                              [Predicted hazard surfaces + LULC maps]
                                          ↓                     (below: I3 / I4)
      [Exposure + Vulnerability] → (5) Risk index engine → [HRI rasters + ward stats]
                                          ↓
                              (6) API + Web map + Exports
```

### 7.2 Training execution flow (I2)

```
laptop:  build_dataset → cube v2 + manifest ──rsync──► server: verify hash
laptop:  small runs, debugging                          server: train (headless, resumable)
laptop:  ◄── run directory (config, metrics, checkpoint, manifest) ──┘
```

### 7.3 Primary Use Cases

| ID | Actor | Use case | Main flow |
|---|---|---|---|
| UC-1 | Researcher | Build the dataset for the study cities | Edit configuration → run build → catalogue scenes → composites → cube + manifest |
| UC-2 | Researcher | Train and compare candidate models | Select dataset version + config → train ≥ 3 seeds per candidate → metrics logged → comparison table |
| UC-3 | Researcher | Generate a future scenario | Select model + horizon → infer → write rasters + uncertainty |
| UC-4 | Planner | Identify high-risk wards for a future year *[I4]* | Open map → select year → open hotspot table → export PDF |
| UC-5 | Planner | Compare mitigation framing *[I4]* | Compare observed vs predicted year side by side |
| UC-6 | Visitor | Explore current heat map *[I4]* | Land on default AOI → toggle layers → click a location for its time series |
| UC-7 | Researcher | Train on the remote GPU server | Probe server → transfer cube + verify hash → launch headless run → resume if interrupted → copy run directory back |
| UC-8 | Researcher | Measure land-cover accuracy | Draw stratified sample → label blind → compute accuracy and kappa against model output |

---

## 8. Project Plan

The original 35-week plan is retained. Status is as of 9 November 2026.

| Phase | Duration | Deliverables | Status |
|---|---|---|---|
| P1 Literature review + problem formulation | Weeks 1-4 | Survey, gap statement, finalized AOI | AOI fixed (six cities) |
| P2 Data acquisition + analysis-ready stack | Weeks 5-10 | Scene catalogue, preprocessed stack, EDA report | Pipeline built; Dataset v2 rebuild pending |
| P3 Feature derivation + baselines | Weeks 11-15 | LST/SUHI/LULC layers, baseline metrics | LST, indices, baselines done; LULC pending |
| P4 Deep model development | Weeks 16-22 | Candidate models, ablations, benchmark table | **Current: Increment 2** |
| P5 Risk index + validation | Weeks 23-26 | HRI maps, sensitivity analysis, station validation | Not started (I3) |
| P6 Web application | Weeks 27-30 | Deployed map + API + exports | Not started (I4) |
| P7 Paper writing + submission | Weeks 28-34 | Manuscript draft, figures, code release | Not started |
| P8 Report, demo, viva | Weeks 33-35 | Final report, presentation, defence | Not started |

**Increment 2 work order:**

1. Run the server probe (FR-8.7) and close OI-1.
2. Extend the cube to Dataset v2 and rebuild all six cities (the five missing cubes must be rebuilt regardless).
3. Implement the anomaly target and re-baseline persistence and linear trend in both framings.
4. Train ConvLSTM and SimVP (laptop or server), then Prithvi-EO-2.0 (server), three seeds each.
5. Fetch weak labels, draw and label the verification sample, train U-Net, then the foundation-model classifier.
6. Select models per FR-4.10 and record the benchmark table.

**Milestone gates:** M1 stack ready (end P2) · M2 baseline beaten, or its failure explained (end P4) · M3 validated risk maps (end P5) · M4 demo live (end P6) · M5 paper submitted (end P7).

---

## 9. Verification and Validation

- **V1 Unit tests** for index formulas, sensor scaling (including the Sentinel-2 2022 offset), reprojection, and masking against hand-computed fixtures.
- **V2 Integration test** running the full dataset-build-to-training path on a synthetic cube offline, where the model must beat persistence. A synthetic win demonstrates correct wiring only and is never cited as model skill.
- **V3 Model validation** on a temporally held-out period, with all baselines run on identical splits, in anomaly and absolute framings.
- **V4 Ground validation** against IMD/AWS station temperatures where co-located observations exist. **[I3]**
- **V5 Spatial cross-validation:** leave one city out (train on five, test on the sixth) for the forecaster; tile hold-out for the classifier.
- **V6 Reproducibility check:** regenerate one published figure end-to-end from its run ID on a clean machine.
- **V7 Usability walkthrough** with the faculty guide against UC-4 and UC-6. **[I4]**
- **V8 Label accuracy check:** LULC accuracy and kappa computed on the verification sample, with the labelling done blind to model output.
- **V9 Remote parity check:** the same configuration on a small subset run on the laptop and the server produces metrics within a stated tolerance; resume-from-checkpoint is exercised by interrupting a run deliberately.
- **V10 Leakage test:** an explicit test asserts no target year appears in two splits, no training target postdates a test target, and normalization statistics are independent of validation and test years.

---

## 10. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Monsoon cloud cover leaves too few usable scenes | High | Seasonal compositing; focus on pre-monsoon (Mar-May); Chennai flagged as the weakest city |
| Deep model does not beat simple baselines | High | Anomaly target (FR-4.8) removes the unpredictable scene offset; multi-seed comparison (FR-4.10); if still no gain, report honestly with the decomposition of Appendix E |
| Anomaly framing does not close the gap | High | Fall back to the alternatives recorded in Appendix E (scene-level covariate, larger patches, a better-posed target such as SUHI or a multi-year smoothed level) |
| Large pretrained model overfits 13 annual steps × 6 cities | Medium | Frozen encoder with small trained head; compare against the small models; treat a loss as a finding |
| Pretraining resolution/band mismatch (C10) | Medium | Prefer Prithvi (30 m HLS) for forecasting; for TerraMind, test with and without the shift and report it |
| Weak labels contain errors that inflate or deflate accuracy | High | Accuracy measured only on the verification sample (FR-4.9); weak-label agreement reported separately |
| Verification sample too small or biased | Medium | ≥ 300 points, stratified, ≥ 30 per class; unevaluable classes reported as such |
| Remote server specification or access method unknown | High | Server probe (FR-8.7); FR-8 written to hold under SSH, notebook, or scheduler; laptop remains a complete fallback for the baselines |
| Training interrupted on a shared server | Medium | Per-epoch full checkpoints and resume flag (FR-8.3) |
| Only 10 target years makes results seed-sensitive | Medium | ≥ 3 seeds, mean ± std; leave-one-city-out check (V5) |
| Sensor change at 2017 creates an artificial step | Medium | `index_source` recorded per year (FR-2.8); report results with and without pre-2017 years if the effect is visible |
| Cubes for five cities are missing from disk | Medium | Rebuild as part of Dataset v2; builds are resumable and skip finished cities |
| Insufficient ground-truth for validation | Medium | Held-out satellite LST as primary; state station-validation limits explicitly |
| Vulnerability data unavailable at useful resolution | Medium | Use proxies and document the substitution [I3] |
| Scope creep from adding sensors/models | Medium | Scope frozen by Section 1.5; extensions become future work |
| Publication timeline slips | Medium | Draft paper in parallel from P4; target a venue with rolling submission |

---

## 11. Future Enhancements

- LULC change prediction (CA-Markov vs deep) feeding the LST forecast as a covariate (FR-4.2).
- Seasonal rather than annual composites, if non-monsoon seasons can be made usable, to raise the number of time steps.
- Downscaling MODIS to 30 m via spatio-temporal fusion for daily hazard.
- Physics-informed models coupling LST with urban canopy or WRF-UCM outputs.
- Mitigation scenario simulation (cool roofs, tree planting, added water) with predicted LST deltas.
- Night-time thermal (ECOSTRESS) for nocturnal heat stress.
- Health-outcome coupling with hospital admission data where ethically obtainable.
- Longer history from Landsat 5/7 (1988-), accepting the sensor-generation caveat.

## 12. Open Items

| ID | Item | Resolves with |
|---|---|---|
| OI-1 | Remote server GPU, VRAM, RAM, disk, OS, and access method are unknown | Server probe output (FR-8.7) |
| OI-2 | Exact scene-level definition for the anomaly target (scene mean vs rural reference; land-only mask) | I2 design specification |
| OI-3 | Which years the weak-label product covers, and whether a per-year source is available for all six cities | Label source check before FR-4.9 training |
| OI-4 | Licence terms of the chosen pretrained weights and label products | Before release (NFR-7.4) |
| OI-5 | Deadline and review expectations, which determine whether LULC change prediction can enter scope | Team decision |

---

## Appendix A — Requirements Traceability (abridged)

| Objective | Requirements | Verification |
|---|---|---|
| Map current urban heat | FR-1, FR-2, FR-3 | V1, V2, V4 |
| Predict future heat | FR-4.1-4.8, FR-4.10 | V3, V5, V10 |
| Classify land cover | FR-1.6, FR-3.4, FR-4.9 | V8 |
| Train on available hardware | FR-8 | V9 |
| Quantify risk, not just heat | FR-5 | V4, sensitivity analysis (FR-5.6) |
| Make results usable | FR-6, NFR-4 | V7 |
| Make results publishable | FR-7, NFR-7, NFR-8 | V6 |

## Appendix B — Technology Stack

| Layer | Choice |
|---|---|
| Data access | Microsoft Planetary Computer via `pystac-client`, `planetary-computer`, `odc-stac`, `odc-geo` (Google Earth Engine and `stackstac` not used; see Appendix D) |
| Raster processing | `rasterio`, `xarray`, `netcdf4`, `pyproj` |
| Environment | Python 3.12, `uv`, `uv.lock`; PyTorch CUDA 12.6 build |
| Deep learning | PyTorch; `segmentation-models-pytorch` (U-Net); TerraTorch (Prithvi-EO-2.0, TerraMind); SimVP implemented or adapted from OpenSTL |
| Classical baselines | scikit-learn, XGBoost (should) |
| Experiment tracking | Run-directory manifests (MLflow or W&B optional later) |
| Training compute | Laptop RTX 4060 (8 GB) and a remote GPU server (OI-1) |
| Backend *[I4]* | FastAPI, `titiler` for COG tiling |
| Database *[I3]* | PostgreSQL + PostGIS |
| Frontend *[I4]* | React + MapLibre GL, deck.gl, Recharts |
| Deployment *[I4]* | Docker Compose; Hugging Face Spaces / Render / institutional server |

## Appendix C — Candidate Model Register

Selection follows FR-4.10. Parameter counts are omitted on purpose: they size compute, they are not selection criteria.

**Job 1: LST anomaly forecasting**

| ID | Model | Idea | Compute | Role |
|---|---|---|---|---|
| F1 | ConvLSTM (scaled: 2 stacked layers, more memory channels, larger patches) | Recurrent convolution; the Increment 1 model, fixed | Laptop | Reference |
| F2 | SimVP | Convolutional encoder, translator, decoder; no recurrence; light to train | Laptop or server | Main challenger |
| F3 | Prithvi-EO-2.0 (300M) | NASA/IBM model pretrained on HLS (Landsat + Sentinel-2); frozen encoder, trained decoder; LST history injected through a small trainable covariate branch because LST is not a pretraining band | Server | Foundation-model arm |
| B1 | Persistence | Next year equals last year | Any | Baseline |
| B2 | Per-pixel linear trend | OLS per pixel | Any | Baseline |
| B3 | Random Forest / XGBoost | Same features, tabular | Any | Baseline (should) |

Rationale: the closest published comparison (vegetation forecasting on EarthNet2021) found SimVP and PredRNN above ConvLSTM, and dropped the Earthformer transformer because of its cost. With 13 annual steps per city, a 300M-parameter model is at risk of overfitting; F3 is included so that the result, whichever way it falls, is measured rather than assumed.

**Job 2: LULC classification**

| ID | Model | Idea | Compute | Role |
|---|---|---|---|---|
| L1 | U-Net, ResNet-34 encoder (ImageNet init) | Standard supervised segmentation | Laptop | Baseline |
| L2 | TerraMind-1.0 (tiny/small/base) or Prithvi-EO-2.0 + segmentation head | Foundation-model encoder via TerraTorch | Server | Foundation-model arm |

Rationale: the developers' PANGAEA benchmark reports TerraMind ahead of task-specific U-Nets, while an independent analysis puts the margin at about 2 mIoU points. Neither figure was measured on this project's data; L1 is the bar L2 must clear on the verification sample.

## Appendix D — Changes from Version 1.0

| Change | Reason |
|---|---|
| Google Earth Engine replaced by Planetary Computer STAC (`odc-stac`) | No academic-account dependency or export quota; fully local and reproducible. `stackstac` was rejected because it breaks against NumPy 2 and pandas 3. |
| Study area changed from one metropolitan AOI to six cities at 40 × 40 km | Pooling cities multiplies the training examples and tests generalization. |
| MODIS deferred | A 1 km product on a 30 m grid needs a fusion method of its own. |
| Sentinel-2 added; both sensors on one 30 m grid; Landsat supplies LST only | Sentinel-2 has no thermal band. Averaging 10 m reflectance to 30 m avoids inventing thermal detail. |
| Study period 2013-2025, pre-monsoon only | Landsat 8 launch gives a single sensor generation; June-September is unusable in every city. |
| Reflectance bands preserved in the cube (FR-2.7) | Pretrained models and albedo need raw bands; indices alone cannot be inverted. |
| Forecast target changed to LST anomaly (FR-4.8) | The Increment 1 result showed absolute-LST error dominated by an unobservable scene-wide offset (C8, Appendix E). |
| Candidate-model register and selection protocol (FR-4.1, FR-4.10, Appendix C) | Model choice made on evidence and a fixed protocol rather than preference. |
| LULC classification brought into I2, evaluated on a team-labelled verification sample (FR-4.9) | Weak labels alone would make the accuracy figure a measure of agreement with another map. |
| LULC change prediction, risk index, web app, MODIS, station validation moved to later increments or future work | Keeps the current scope to what can be finished and defended. |
| Horizons +5 and +10 years labelled unvalidated (FR-4.3) | A 13-year series cannot validate them. |
| Remote GPU training requirements added (FR-8) | A GPU server is available; its specification is unknown, so the requirements must hold under any access method. |
| Python pinned to 3.12 | PyTorch, rasterio, and odc-stac publish no wheels for 3.13 or 3.14. |
| Run-directory manifests replace MLflow for now (FR-4.6) | No multi-service deployment yet. |

## Appendix E — Increment 1 Findings

Recorded so the reasons for the Increment 2 changes are on file.

**Pooled six-city run** (`data/results/results.json`), RMSE in °C on held-out test years, absolute-LST framing:

| Method | RMSE (°C) | MAE (°C) | R² |
|---|---|---|---|
| ConvLSTM | 1.635 | 1.243 | 0.943 |
| Persistence ("no change") | 1.659 | 1.195 | 0.941 |
| Per-pixel linear trend | 2.964 | 2.249 | 0.812 |

The ConvLSTM is ahead of persistence by about 0.02 °C RMSE and behind it on MAE. This is a near tie and is not accepted as satisfying NFR-2.1 without the multi-seed comparison of FR-4.10. The cubes behind this run are not all on disk, so the run cannot currently be reproduced (V6).

**Diagnosis (Mumbai tile, Increment 1 analysis).** The annual mean LST swings by up to 4 °C between consecutive years (2020: 40.0 °C, 2021: 36.0 °C). For the 2023 to 2024 step, persistence RMSE of 2.41 °C splits into a scene-wide offset of 2.17 °C and a spatial residual of 1.04 °C. Most of the error is therefore a single scalar determined by which dates were cloud-free, which a model that sees only 64 × 64 patches cannot predict. This is the basis of constraint C8 and requirement FR-4.8.

**Alternatives recorded for Increment 2 if the anomaly framing is not sufficient:** supply the scene level as an input channel; use larger patches; forecast SUHI intensity or a multi-year smoothed level instead of the single-year composite.

**A synthetic-cube integration run** (RMSE 0.21 °C) beats both baselines. It confirms the training code is wired correctly and says nothing about performance on real imagery.
