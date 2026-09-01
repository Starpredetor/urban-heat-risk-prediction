# UHRM — Phase 1 Design: Dataset Pipeline and LST Forecast Model

**Date:** 2026-08-31
**Scope:** Phase 1 only — analysis-ready data cube + ConvLSTM LST forecasting.
**Parent document:** `SRS_Urban_Heat_Risk_Mapping.md`
**Status:** Approved for implementation planning

---

## 1. Purpose and Scope

This design covers the first slice of UHRM: getting real Landsat imagery onto
disk as a normalized spatio-temporal cube, and training a ConvLSTM to forecast
the next Land Surface Temperature surface from the historical stack.

It implements SRS requirements FR-1.x (acquisition), FR-2.x (preprocessing),
FR-3.1–3.3 (LST, indices, SUHI), FR-4.1/4.3/4.4/4.5 (modelling, splits,
metrics, baselines), and NFR-5.1 (pinned dependencies).

### Explicitly out of scope for Phase 1

Deferred until this slice works end to end:

| Deferred | SRS ref | Reason |
|---|---|---|
| Web API and map UI | FR-6, UI-1..5 | Nothing to serve until forecasts exist |
| Heat Risk Index engine | FR-5 | Needs hazard surfaces from this phase first |
| LULC classification and change model | FR-3.4, FR-4.2 | Second modelling track; adds labelling work |
| Sentinel-2 and MODIS ingest | FR-1.2 | Landsat alone gives LST + indices on one grid |
| PostGIS, object store, Docker | 3.2, NFR-5.2 | No multi-service deployment yet |
| MLflow experiment tracking | FR-4.6 | Replaced in Phase 1 by run-directory JSON manifests |
| Ancillary socio-economic layers | FR-1.5 | Only needed by the risk engine |

Directories for the deferred modules are **not** created. They arrive with
their own design.

---

## 2. Key Decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | Data via **STAC / Microsoft Planetary Computer**, not Google Earth Engine | No academic-account dependency, no export quotas (SRS C3), fully local and reproducible. SRS Appendix B lists `pystac-client`/`stackstac` as the sanctioned fallback. |
| D2 | **Python 3.12** in a `uv`-managed venv | The machine has only Python 3.14.7, for which PyTorch, rasterio and stackstac publish no wheels. `uv` installs a managed 3.12 without disturbing the system install. |
| D3 | Start on a **~20 × 20 km central Mumbai tile** | Whole pipeline green in minutes. The same YAML config re-runs on the full AOI by changing the bbox. |
| D4 | **ConvLSTM written from scratch** | SRS FR-4.1 names it as the baseline. No pretrained weights to download, trains on the RTX 4060, and every layer is ours to inspect. |
| D5 | **Zarr** as the cube format | Chunked, lazily readable, xarray-native, and it round-trips coordinate/CRS metadata that `.npz` would lose. |
| D6 | **Synthetic cube fixture before real data** | The training loop is proved correct in seconds with no network. Download bugs and model bugs stay separable. |
| D7 | **`odc-stac` for reading, not `stackstac`** | Amended during implementation. stackstac 0.5.0 is unmaintained and breaks against both NumPy 2 (NEP 50 scalar casting in `np.can_cast`) and pandas 3 (removed `infer_datetime_format`). odc-stac is actively maintained and consumes the same STAC items. The SRS names stackstac in Appendix B; this supersedes it. |

---

## 3. Environment

`uv` manages a Python 3.12 virtual environment. `pyproject.toml` declares
direct dependencies; `uv.lock` is committed and pins the full transitive tree,
satisfying NFR-5.1.

**Core dependencies**

| Package | Role |
|---|---|
| `pystac-client`, `planetary-computer` | STAC search and asset URL signing |
| `odc-stac`, `odc-geo` | STAC-items-to-xarray reading on a fixed GeoBox (see D7) |
| `rioxarray`, `rasterio` | Reprojection, resampling, GeoTIFF I/O |
| `xarray`, `zarr`, `dask` | Cube assembly and chunked on-disk storage |
| `numpy`, `torch` | Arrays and model |
| `pyyaml`, `pydantic` | Config loading and validation |
| `pytest` | Tests |

PyTorch is installed from the CUDA 12.x index for GPU support. The code must
run on CPU as well — device is resolved at runtime, never hardcoded.

---

## 4. Repository Layout

```
config/
  aoi.yaml                  AOI bbox, CRS, target grid resolution
  dataset.yaml              sensors, date range, season, cloud threshold, paths
  model.yaml                ConvLSTM hyperparameters, split years, training config

src/uhrm/
  config.py                 pydantic models; load + validate YAML

  ingest/
    stac.py                 STAC search against Planetary Computer
    catalog.py              scene catalogue persistence (FR-1.4)

  preprocess/
    mask.py                 QA_PIXEL cloud / shadow / cirrus masking (FR-2.1)
    grid.py                 reproject + resample to common UTM grid (FR-2.2)
    composite.py            seasonal median compositing (FR-2.3)

  features/
    lst.py                  Landsat C2 ST band -> degrees Celsius (FR-3.1)
    indices.py              NDVI, NDBI, NDWI, albedo (FR-3.2)
    suhi.py                 SUHI intensity vs rural reference (FR-3.3)

  data/
    stack.py                assemble (T, C, H, W) cube -> zarr
    normalize.py            per-channel statistics, fit on train years only
    dataset.py              torch Dataset: temporal windows + spatial patches
    synthetic.py            synthetic cube generator for tests

  models/
    convlstm.py             ConvLSTM cell, encoder, regression head

  train/
    loop.py                 training / validation / test loops
    metrics.py              RMSE, MAE, R2, SSIM (FR-4.4)
    baselines.py            persistence + per-pixel linear trend (FR-4.5)

  utils/
    logging.py              structured stage logging (FR-7.2)
    run.py                  run directory, manifest, seeding

scripts/
  build_dataset.py          config -> zarr cube
  train.py                  config + cube -> checkpoint + metrics

tests/
data/                       raw/ interim/ processed/   (gitignored)
runs/                       per-run checkpoints + manifests (gitignored)
```

---

## 5. Data Pipeline

### 5.1 Configuration

`config/aoi.yaml` — the test tile:

```yaml
name: mumbai_central_test
bbox: [72.78, 18.99, 72.97, 19.17]   # WGS84 lon/lat, ~20 x 20 km
crs: EPSG:32643                       # UTM zone 43N
resolution: 30                        # metres
```

`config/dataset.yaml`:

```yaml
collections: [landsat-c2-l2]
platforms: [landsat-8, landsat-9]
years: [2014, 2025]
season: {name: pre_monsoon, months: [3, 4, 5]}
max_cloud_cover: 20
composite: median
```

Scaling the study area to the full SRS AOI means editing `bbox` alone. Season
and year range are knobs: if 12 annual composites prove too few temporal
samples, switching to four seasonal composites per year raises the step count
roughly fourfold without any code change.

### 5.2 Acquisition (FR-1)

`ingest/stac.py` queries the Planetary Computer STAC endpoint for
`landsat-c2-l2` items intersecting the AOI bbox, within the configured months
of each year, with `eo:cloud_cover <= 20`. Asset hrefs are signed with
`planetary_computer.sign_inplace`.

`ingest/catalog.py` writes one catalogue row per item — scene id, platform,
acquisition datetime, cloud cover, WRS path/row, asset hrefs — to a Parquet
file under `data/raw/`. This is FR-1.4, and it is what makes a run auditable:
a later reader can see exactly which scenes produced a figure.

Failed asset reads retry three times with exponential backoff, then log and
skip (NFR-3.2). A season that ends up with zero usable scenes is recorded as a
gap, not an exception — SRS C1 guarantees this will happen in some years.

### 5.3 Preprocessing (FR-2)

**Bands read** (Landsat Collection 2 Level-2 asset names):
`blue`, `green`, `red`, `nir08`, `swir16`, `swir22`, `lwir11` (surface
temperature), `qa_pixel`.

**Masking** (`preprocess/mask.py`) decodes `qa_pixel` bitflags and discards
pixels flagged as fill (bit 0), dilated cloud (1), cirrus (2), cloud (3), or
cloud shadow (4). Masked pixels become NaN.

**Gridding** (`preprocess/grid.py`) reprojects every scene to EPSG:32643 at
30 m on a fixed grid derived from the AOI bbox, so every time step is pixel
aligned. Bilinear for continuous bands, nearest for QA.

**Compositing** (`preprocess/composite.py`) reduces all masked scenes within
one season-year to a per-pixel median. Median rejects residual cloud edges
that masking misses. Two arrays are recorded per composite: the value, and an
observation count used as the quality mask required by FR-2.4.

Residual gaps — pixels with zero valid observations in a season — are filled
by linear interpolation along the time axis and flagged in the quality mask.
Filled pixels are excluded from metric computation.

### 5.4 Feature Derivation (FR-3)

**LST** (`features/lst.py`). Landsat C2 L2 ships a retrieved surface
temperature product, so per FR-3.1 we use it directly rather than re-deriving
emissivity:

```
LST_kelvin  = ST_B10 * 0.00341802 + 149.0
LST_celsius = LST_kelvin - 273.15
```

Values outside a plausible 0–70 °C range are treated as invalid.

**Surface reflectance** rescaling, applied before any index:

```
rho = SR_band * 0.0000275 - 0.2      valid for DN in [7273, 43636]
```

The valid-range check is not cosmetic. Digital numbers outside the USGS
published range are saturated, fill, or failed atmospheric correction, and they
scale to reflectances near or below zero. That drives the denominator of a
normalized-difference index toward zero: real Mumbai 2020 scenes produced NDVI
values as low as **-12** before the check was added, against a mathematical
range of [-1, 1]. Out-of-range pixels become NaN rather than being clamped,
since a clamped value is a fabricated observation.

**Indices** (`features/indices.py`):

```
NDVI = (nir08  - red)   / (nir08  + red)
NDBI = (swir16 - nir08) / (swir16 + nir08)
NDWI = (green  - nir08) / (green  + nir08)
```

Albedo uses the Liang (2001) narrowband-to-broadband coefficients:

```
albedo = (0.356*blue + 0.130*red + 0.373*nir08
          + 0.085*swir16 + 0.072*swir22 - 0.0018) / 1.016
```

The denominator normalizes the coefficient sum; it is written explicitly
because omitting it is a common and silent error.

**SUHI** (`features/suhi.py`) computes urban-pixel LST minus the mean LST of a
rural reference ring — an annulus outside the built-up core, defined by a
buffer distance and restricted to low-NDBI pixels. Phase 1 computes SUHI as a
derived diagnostic layer; it is not a model input, since it is a linear
function of LST the network already sees.

### 5.5 Cube Assembly

`data/stack.py` writes a zarr store with dimensions `(time, band, y, x)`:

- `time` — one entry per season-year composite (12 for the default config)
- `band` — `lst`, `ndvi`, `ndbi`, `ndwi`, `albedo`
- `y`, `x` — approximately 667 × 667 at 30 m over the test tile

Alongside it: the quality mask, the CRS and transform, and a
`dataset_manifest.json` recording config hash, scene ids used, band order,
and a content hash. That hash is the dataset version referenced by FR-4.6 and
FR-7.3.

---

## 6. Model and Training

### 6.1 Temporal Split (FR-4.4)

Splits are temporally disjoint on the **target** year, so no model ever sees a
target at or after a later split's targets:

| Split | Target years |
|---|---|
| Train | 2018–2021 |
| Validation | 2022–2023 |
| Test | 2024–2025 |

Input windows may reach back across a split boundary — predicting the 2022
target legitimately uses 2018–2021 observations, exactly as deployment would.
What must never happen, and what the split enforces, is a target year
appearing in more than one split or a training target postdating a test
target.

Normalization statistics are computed on **training targets' years only** and
frozen. Fitting them on the whole cube would leak test-period distribution
into training, and it is an easy mistake to make.

### 6.2 Dataset

`data/dataset.py` yields `(input_window, target)` pairs:

- `input_window`: `(T_in, C, P, P)` — default `T_in = 4`, `C = 5`, patch `P = 64`
- `target`: `(1, P, P)` — the LST channel of the following time step

Patches are sampled on a stride-32 grid over the spatial extent. Patches whose
target is more than 30 % gap-filled are dropped — training on interpolated
values teaches the model to reproduce the interpolator.

With 12 time steps and `T_in = 4`, each patch position yields 8 samples; a
667 × 667 tile at stride 32 gives roughly 380 positions, so on the order of
3,000 windows before the split. Thin but sufficient for the test tile, and the
seasonal-composite knob in §5.1 is the documented remedy if it proves too thin.

### 6.3 Architecture

`models/convlstm.py` implements a ConvLSTM cell from scratch: the four LSTM
gates computed by a single convolution over the concatenated input and hidden
state, with the standard sigmoid/tanh gating and a convolutional cell state.

The model stacks two ConvLSTM layers (default 32 and 64 hidden channels,
3 × 3 kernels), takes the final hidden state, and passes it through a small
convolutional head to a single-channel LST prediction. Dropout sits in the
head so that MC-dropout uncertainty (FR-4.7) is available later without an
architecture change; Phase 1 does not yet report uncertainty.

The model predicts the **residual** from the last observed LST frame rather
than absolute temperature. Absolute urban LST is dominated by a large static
spatial pattern that a persistence baseline already captures; making the
network predict only the change is what forces it to learn something
persistence does not.

Loss is masked L1 over valid target pixels. L1 over MSE because LST composites
retain outliers from imperfect cloud masking, and squared error would let them
dominate the gradient.

### 6.4 Baselines (FR-4.5)

`train/baselines.py` computes two reference predictions on the identical test
split, reported in the same table as the model:

1. **Persistence** — next LST equals last observed LST.
2. **Per-pixel linear trend** — ordinary least squares on each pixel's
   training-period time series, extrapolated forward.

NFR-2.1 requires the deep model to beat both. Random Forest and XGBoost
baselines are deferred to the phase that introduces the full feature set.

### 6.5 Metrics (FR-4.4)

RMSE, MAE, R², and SSIM, all computed over valid (non-gap-filled) test pixels
only. RMSE in °C is the headline number against the ≤ 2.0 °C target in
NFR-2.1.

### 6.6 Run Artefacts

Each training run writes `runs/<timestamp>-<config-hash>/` containing the
resolved config, the random seed, the dataset manifest hash, per-epoch
metrics as JSONL, the best checkpoint, and a final metrics table including the
baselines. This is the Phase 1 substitute for MLflow and satisfies FR-4.6 and
FR-7.3 well enough to regenerate any figure from its run id.

---

## 7. Testing

| Level | Coverage |
|---|---|
| Unit (V1) | Index formulas, LST scaling, and albedo against hand-computed fixtures with known inputs and expected outputs. QA_PIXEL bit decoding against constructed bitmasks. Grid alignment: two scenes with different native footprints must land on identical coordinates. |
| Unit | Split correctness — an explicit test asserting no target year appears in two splits and no training target postdates a test target. Normalization statistics must be independent of test-period data. |
| Model | ConvLSTM shape contract; a gradient-flow test; an overfit test driving the model to near-zero loss on a single synthetic batch, which catches most silent wiring bugs. |
| Integration (V2) | `synthetic.py` builds a small cube with a known linear warming trend plus spatial structure. The full `build_dataset -> train` path runs on it in seconds with no network, and the model must beat persistence on it. |

Every test runs offline. Network-touching STAC tests are marked and excluded
from the default run.

---

## 8. Implementation Order

1. Environment: `uv` + Python 3.12 + `pyproject.toml` + lockfile.
2. Skeleton, config schema, run/logging utilities.
3. Synthetic cube generator and its tests.
4. ConvLSTM, metrics, baselines, training loop — green on synthetic data.
5. STAC ingest and scene catalogue against the real tile.
6. Masking, gridding, compositing, gap filling.
7. LST and index derivation, with fixture tests.
8. Cube assembly to zarr.
9. Train on real data; report metrics against both baselines.

Steps 1–4 need no network and no imagery. That ordering is deliberate: it
means a download problem in step 5 can never be confused with a model problem,
because the model was already proven.

---

## 9. Risks Specific to This Phase

| Risk | Mitigation |
|---|---|
| Twelve time steps is very few for a recurrent model | Seasonal compositing knob (§5.1) raises it ~4x with no code change. Patch sampling multiplies spatial samples. If the model still cannot beat persistence, that is a reportable finding, not a failure to hide (SRS §10). |
| Pre-monsoon Landsat scenes still partly cloudy | Median compositing plus per-pixel observation counts; gap-filled pixels excluded from metrics so scores are not flattered by interpolation. |
| Planetary Computer availability or API change | The catalogue records signed asset hrefs and scene ids, and `data/raw/` is cached, so a built cube survives upstream changes. Ingest sits behind one module if a source swap is ever needed. |
| Python 3.14 wheels absent | Pinned 3.12 via `uv`; the lockfile makes this reproducible on any machine. |

---

## 10. Phase 1 Acceptance

Phase 1 is done when:

- `pytest` passes fully offline.
- `scripts/build_dataset.py` produces a zarr cube plus manifest for the test tile from a cold cache.
- `scripts/train.py` trains a ConvLSTM and writes a run directory with metrics for the model and both baselines on the held-out test years.
- The metrics table is recorded honestly, whether or not the model beats the baselines.

---

## 12. Amendment: Multi-Sensor Ingest (2026-09-01)

Agreed pipeline, replacing the Landsat-only ingest of §5:

```
Satellite data (Landsat + Sentinel-2)
   -> preprocess (cloud removal, alignment)
   -> feature extraction (S2: NDVI/NDBI/NDWI | Landsat: LST)
   -> multi-temporal dataset -> ConvLSTM -> risk -> dashboard
```

**This amendment implements the first three boxes only.** Feature extraction
still runs Landsat-only; rewiring it to take indices from Sentinel-2 is the
next slice.

### Why the split of duties

Sentinel-2 has **no thermal infrared band**, so it cannot measure surface
temperature at all. Landsat is the only source of LST here. Sentinel-2 is the
better source for the indices: 10 m rather than 30 m, and a 5-day rather than
16-day revisit. This matches the SRS §6 data table.

### Amended decisions

| # | Decision | Rationale |
|---|---|---|
| D8 | **Both sensors on one 30 m grid** | Sentinel-2 reflectance is averaged from 10 m down to the Landsat analysis grid. Upsampling 100 m-native thermal to 10 m would invent thermal detail the sensor never measured. One grid also keeps SRS FR-2.2 satisfied. |
| D9 | **Study period shortened to 2017-2025** | Sentinel-2 L2A global coverage begins in 2017. Costs three time steps; see the warning below. |
| D10 | **Seasons are a list, not a single season** | The documented remedy for too few time steps becomes a config edit. Four seasons over 2017-2025 gives 36 steps rather than 9. |
| D11 | **`Sensor` interface, one class per satellite** | SRS NFR-5.3: adding a sensor means implementing an interface, not editing the pipeline core. |

**Warning carried forward:** 9 annual steps with `t_in=4` leaves only five
usable target years, two of them training targets. That is below what the model
can learn from, and it worsens the §11 finding rather than fixing it. Seasonal
compositing (D10) is the intended remedy and needs only a config change.

### Cloud removal, by sensor

| | Landsat | Sentinel-2 |
|---|---|---|
| Source | `QA_PIXEL` 16-bit bitfield | `SCL` class codes |
| Rejected | fill, dilated cloud, cirrus, cloud, cloud shadow | no-data, saturated, dark area, cloud shadow, cloud medium/high, thin cirrus |
| Kept | water, snow, clear | water, vegetation, bare soil, unclassified, snow |

Both keep **water**. Water has a real surface temperature, and masking it would
erase the coastline from a coastal study area. Both quality bands are resampled
with nearest neighbour — interpolating a class code or a bitfield would invent
values that mean nothing.

### The Sentinel-2 processing-baseline trap

ESA processing baseline 04.00, effective 25 January 2022, added a **-1000 DN
offset** to every reflectance band. Reading post-2022 scenes with the old
convention shifts reflectance by 0.1 with no error and no obvious symptom,
silently corrupting every index from 2022 onward — which is more than half this
study period.

`Sentinel2Sensor.boa_offset` reads `s2:processing_baseline` per scene and falls
back to the acquisition date when the property is absent. Tests assert that
DN 2000 under baseline 03.01 and DN 3000 under 04.00 both yield reflectance
0.20, and that ignoring the offset would shift the result by exactly 0.10.

This is the same class of defect as the reflectance-range bug in §5.4: a
plausible-looking number that is quietly wrong.

### Output of this stage

```
data/interim/composites_landsat.zarr     (time, band, y, x) + observation count
data/interim/composites_sentinel2.zarr   (time, band, y, x) + observation count
data/interim/composites_manifest.json
data/raw/scene_catalog.parquet           one row per scene, both sensors
```

Bands are canonical names in physical units: `blue, green, red, nir, swir16,
swir22` for both, plus `thermal` (°C) for Landsat only.

### Pipeline stages

`scripts/build_dataset.py --stage {survey,composites,cube}` — search only,
through compositing, or on to the cube.

---

## 11. Phase 1 Results (recorded 2026-08-31)

### Dataset

Built from 74 Landsat 8/9 Collection 2 scenes over the Mumbai central test tile,
pre-monsoon (Mar–May) 2014–2025.

| Property | Value |
|---|---|
| Grid | 674 × 675 px at 30 m, EPSG:32643 (20.2 × 20.2 km) |
| Cube | 12 time steps × 5 bands, 88 MB zarr |
| Usable scenes | 74; every year has at least 3, no empty years |
| Coverage | 99.999 % genuinely observed, 0.001 % gap filled |
| Build time | ~2 minutes from a cold cache |

Physical validation on land pixels (NDWI < 0), 2025 composite:

- `corr(LST, NDVI) = -0.591` — vegetation cools, as expected
- `corr(LST, NDBI) = +0.768` — built-up heats, as expected
- Urban heat island: built-up 42.9 °C vs vegetated 36.7 °C, **+6.2 °C**
- SUHI intensity: mean +2.6 °C, max +18.4 °C

Whole-tile correlations invert these signs, because 23.6 % of the tile is
Arabian Sea and harbour at 29.6 °C against 39.4 °C on land, and that contrast
dominates. Any future analysis must mask water before interpreting an
LST-versus-index relationship.

### Model

**The ConvLSTM did not beat the baselines.** Recorded as run
`20260831T105905Z-2bf7c4f2`, test years 2024–2025:

| Model | RMSE (°C) | MAE (°C) | R² | SSIM |
|---|---|---|---|---|
| ConvLSTM | 2.263 | 1.856 | 0.826 | 0.647 |
| Persistence | 1.915 | 1.580 | 0.876 | 0.566 |
| Per-pixel linear trend | **1.352** | **1.031** | **0.938** | 0.617 |

NFR-2.1 (RMSE ≤ 2.0 °C, deep model beats both baselines) is **not met**.

### Why, and what it means

The annual mean LST series does not warm smoothly. It swings by up to 4 °C
between consecutive years (2020: 40.0 °C, 2021: 36.0 °C), with a standard
deviation of year-to-year change of 1.64 °C. Decomposing the persistence error
for the test steps:

| Step | Persistence RMSE | Scene-wide offset | Spatial residual |
|---|---|---|---|
| 2023 → 2024 | 2.41 °C | **2.17 °C** | 1.04 °C |
| 2024 → 2025 | 1.22 °C | 0.78 °C | 0.95 °C |

Most of what a next-step forecaster must predict is a **single scalar** — the
whole-scene temperature level — not a spatial pattern. That level is set by
which acquisition dates happened to be cloud-free in a given pre-monsoon window,
which is sampling noise rather than urban change. This is SRS constraint C1
showing up quantitatively.

A ConvLSTM trained on 64 × 64 patches cannot see scene-wide context, so it
cannot predict that offset; it learns local structure and is then penalised by a
level shift it had no way to observe. The per-pixel linear trend implicitly
tracks the level and therefore wins. The model is not broken — the same code
beats both baselines on the synthetic cube, where the trend is genuine and
locally predictable.

This is the outcome SRS §10 named in advance ("deep model does not beat simple
baselines — treat this as a valid finding; report honestly"). It is reported,
not hidden.

### Implications for Phase 2

The finding points at specific, testable changes rather than at more tuning:

1. **Remove the nuisance level shift.** Model LST anomaly relative to the
   scene-wide (or rural-reference) mean, so the network predicts spatial
   pattern — which is the thing urban heat mapping actually cares about — and
   the scene offset is handled separately or not at all.
2. **More time steps.** Twelve annual composites give only four training target
   years. Seasonal compositing (the §5.1 knob) raises this roughly fourfold
   without touching the code.
3. **Give the model scene context.** A global-mean covariate channel, or larger
   patches, would let it see what it is currently blind to.
4. **Reconsider the target.** Forecasting the pre-monsoon composite one year
   ahead may simply be dominated by sampling noise. Forecasting SUHI intensity,
   or a multi-year smoothed level, is a better-posed problem.

Item 1 is the highest-value change and should be tried first.
