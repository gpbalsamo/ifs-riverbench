# ifs-riverbench

`ifs-riverbench` benchmarks IFS river discharge simulations against observed streamflow data and builds interactive multi-experiment dashboards.

The workflow retrieves archived IFS river discharge fields from MARS, extracts model hydrographs at river-gauge locations, compares them with observed discharge, and generates HTML dashboards for visual inspection of model skill, experiment winners, and pairwise experiment differences.

## Overview

The workflow is organised around the scripts in `Workflow/`:

```text
Workflow/
├── 00_convert_qobs_to_zarr.py      # optional one-time conversion of observations to Zarr
├── 00_extract_rivers_mars.py       # retrieve monthly river discharge GRIB files from MARS
├── 01_extract_hydrographs.py       # sample model discharge, read observations, compute metrics
├── 02_build_dashboard.py           # build multi-experiment HTML dashboards
├── 03_upload_dashboard.py          # upload dashboards to ECMWF Sites
└── ifs-riverbench.sh               # full workflow driver
```

The main benchmark output is a set of HTML dashboards plus the supporting `dashboard_data/` directory.

## Parameter convention

The river discharge parameter is:

```text
235270 = river discharge
235275 = flood fraction
```

The current workflow focuses on `235270`.

## Input data

### IFS river discharge GRIB files

`00_extract_rivers_mars.py` retrieves monthly GRIB files from MARS. The expected archive convention is:

```text
date = first day of month
time = 00 UTC
step = 24, 48, 72, ... up to the last day of the month
```

For example, January 2018 is retrieved as:

```text
date = 20180101
step = 24/to/744/by/24
```

The output layout is:

```text
/perm/pad/flood_cases/grib/<expver>/<date_start>_<date_end>/
├── Globe_river_discharge_<expver>_201801.grb
├── Globe_river_discharge_<expver>_201802.grb
└── ...
```

### Station metadata

The station metadata CSV is expected to contain station coordinates, observed upstream area, and CaMa-Flood model-grid information at several resolutions:

```text
Id, Name, StatLon, StatLat, ProvArea, River, Country,
Cama1lon, Cama1lat, Cama1area, ...
Cama3lon, Cama3lat, Cama3area, ...
Cama6lon, Cama6lat, Cama6area, ...
Cama15lon, Cama15lat, Cama15area, ...
```

Default path:

```text
/perm/pad/flood_cases/Stations/allstations_V1_2.csv
```

### Observed discharge

Observed discharge can be read either from NetCDF or from a Zarr store.

Default NetCDF:

```text
/perm/pad/flood_cases/Stations/Qobs_24_1980-2025_withcaravan.nc
```

Recommended Zarr store:

```text
/perm/pad/flood_cases/Stations/Qobs_24_1980-2025_withcaravan.zarr
```

Expected dimensions and variables:

```text
dimensions:
  time
  station

variables:
  time(time)
  statid(station)
  discharge(time, station)
```

The observation `statid` is assumed to match the station CSV `Id`.

## Optional one-time step: convert observations to Zarr

For multi-year, multi-experiment workflows, Zarr is much faster than repeated NetCDF fancy indexing. Convert the observation archive once with:

```bash
python3 00_convert_qobs_to_zarr.py
```

The recommended Zarr chunking is:

```text
time    = 365
station = 1024
```

The script writes a Zarr v2 store to avoid warnings about consolidated metadata in Zarr v3.

## Step 1: retrieve river discharge from MARS

Retrieve one experiment:

```bash
python3 00_extract_rivers_mars.py \
  --expver iyp3 \
  --date-start 20180101 \
  --date-end 20221231
```

Retrieve several experiments:

```bash
python3 00_extract_rivers_mars.py \
  --expver j6ft j6fu j6fs j6gq iyp3 \
  --date-start 20180101 \
  --date-end 20221231
```

Main options:

```text
--expver       one or more experiment IDs
--date-start   start date, YYYYMMDD or YYYY-MM-DD
--date-end     end date, YYYYMMDD or YYYY-MM-DD
--out-root     root directory for GRIB output
--req-root     root directory for saved MARS requests
--params       MARS parameter list, default 235270
--overwrite    overwrite existing non-empty GRIB files
--dry-run      write requests but do not run MARS
```

## Step 2: extract station hydrographs and compute metrics

Extract one experiment using the recommended Zarr observations:

```bash
python3 01_extract_hydrographs.py \
  --expver iyp3 \
  --date-start 20180101 \
  --date-end 20221231 \
  --resolution 15 \
  --valid-time-shift-hours -24 \
  --obs-file /perm/pad/flood_cases/Stations/Qobs_24_1980-2025_withcaravan.zarr
```

Use NetCDF observations instead:

```bash
python3 01_extract_hydrographs.py \
  --expver iyp3 \
  --date-start 20180101 \
  --date-end 20221231 \
  --resolution 15 \
  --valid-time-shift-hours -24 \
  --obs-file /perm/pad/flood_cases/Stations/Qobs_24_1980-2025_withcaravan.nc
```

The `--valid-time-shift-hours -24` option is needed for the monthly archive convention where `date=first day of month` and `step=24` represents the first day of the month.

For each station, the script:

1. reads station metadata;
2. uses the selected CaMa-Flood columns, for example `Cama15lon`, `Cama15lat`, `Cama15area`;
3. finds the nearest GRIB grid point once;
4. extracts the model discharge time series;
5. reads the corresponding observed discharge;
6. computes KGE, correlation and RMSE on common daily means;
7. writes dashboard-ready JSON and CSV outputs.

Output layout:

```text
dashboard_data/<expver>/<date_start>_<date_end>_<resolution>arcmin/
├── stations/
│   ├── station_000000.json
│   ├── station_000001.json
│   └── ...
├── stations_catalog.json
└── global_station_metrics.csv
```

## Step 3: build dashboards

`02_build_dashboard.py` supports one or more experiments.

### Best metric dashboard

Stations are coloured by the best KGE or correlation achieved by any experiment:

```bash
python3 02_build_dashboard.py \
  --expver j6ft j6fu j6fs j6gq iyp3 \
  --date-start 20180101 \
  --date-end 20221231 \
  --resolution 15 \
  --metric kge \
  --colour-mode best_metric
```

### Best experiment dashboard

Stations are coloured by the experiment that performs best. A control experiment can be specified, and stations are coloured grey when the winning experiment does not improve over the control by at least the chosen threshold.

```bash
python3 02_build_dashboard.py \
  --expver j6ft j6fu j6fs j6gq iyp3 \
  --control-expver iyp3 \
  --date-start 20180101 \
  --date-end 20221231 \
  --resolution 15 \
  --metric kge \
  --colour-mode best_experiment \
  --best-experiment-min-improvement 0.02
```

### Pairwise difference dashboard

For pairwise differences, the dashboard computes:

```text
second experiment - first experiment
```

For example, to show `j6fu - iyp3`:

```bash
python3 02_build_dashboard.py \
  --expver iyp3 j6fu \
  --date-start 20180101 \
  --date-end 20221231 \
  --resolution 15 \
  --metric kge \
  --colour-mode experiment_difference \
  --difference-threshold 0.02
```

Stations with an absolute difference below the threshold are coloured grey.

### Useful dashboard options

```text
--metric                              kge or correlation
--colour-mode                         best_metric, best_experiment, experiment_difference
--control-expver                      control experiment for best_experiment mode
--best-experiment-min-improvement     threshold above control for best_experiment mode
--difference-threshold                neutral threshold for pairwise differences
--map-height-vh                       map height in viewport-height units
--river-resolution                    Natural Earth river layer resolution: 110m, 50m, 10m
--projection                          equirectangular or natural earth
--no-rivers                           disable river overlay
```

The dashboard station panel shows:

```text
station metadata
observed and model upstream areas
relative upstream-area difference
KGE, correlation and RMSE by experiment
best experiment
obs/model median-ratio diagnostic
```

The obs/model median-ratio diagnostic is intended to flag possible scale, unit, or basin-area mismatches. The dashboard does not apply a blind unit conversion.

## Full workflow driver

The full workflow can be run with:

```bash
./ifs-riverbench.sh
```

The driver currently sets:

```text
DATE_START=20180101
DATE_END=20221231
RESOLUTION=15
VALID_TIME_SHIFT_HOURS=-24
REFERENCE_EXPVER=iyp3
METRIC=kge
```

and runs:

1. MARS extraction;
2. station hydrograph extraction with Zarr observations;
3. best-metric dashboard;
4. best-experiment dashboard;
5. pairwise experiment-difference dashboards against the reference experiment.

Edit the settings at the top of `ifs-riverbench.sh` to change the experiment list, metric, reference experiment, threshold or map height.

## Upload dashboards to ECMWF Sites

Set your ECMWF Sites token:

```bash
export ECMWF_SITES_TOKEN="..."
```

Upload all HTML files and `dashboard_data/`:

```bash
python3 03_upload_dashboard.py
```

Upload only HTML files:

```bash
python3 03_upload_dashboard.py --html-only
```

Dry run:

```bash
python3 03_upload_dashboard.py --dry-run
```

Useful options:

```text
--workflow-dir                 directory containing dashboard HTML and dashboard_data
--html-pattern                 glob pattern for HTML files
--html-only                    upload only HTML files
--skip-dashboard-data          skip dashboard_data upload
--list-before                  list remote files before upload
--list-after                   list remote files after upload
--dry-run                      print actions without uploading
```

The upload script reads the token from the `ECMWF_SITES_TOKEN` environment variable. Do not hard-code API tokens in the repository.

## Viewing dashboards locally

From the `Workflow/` directory:

```bash
python3 -m http.server 8000
```

Then open:

```text
http://localhost:8000/
```

The dashboards fetch station JSON files from `dashboard_data/`, so opening the HTML file directly from the filesystem may not work in all browsers.

## Generated files and Git

Generated dashboard outputs should not normally be committed. The repository should ignore:

```text
dashboard_data/
global_station_dashboard_*.html
*.zarr/
*.npz
__pycache__/
```

Commit the workflow scripts, not the generated benchmark outputs.

## Requirements

The workflow requires access to ECMWF MARS and the `mars` command.

Main Python dependencies:

```text
numpy
pandas
xarray
dask
scipy
eccodes
plotly
cartopy
zarr
```

The upload step additionally requires the ECMWF Sites SDK:

```text
sites.sdk
```

## Notes

- Use Zarr observations for multi-year, multi-experiment benchmarks.
- Use `--valid-time-shift-hours -24` with the current monthly archive convention.
- For `experiment_difference`, remember that the result is always `second experiment - first experiment`.
- For `best_experiment`, use `--control-expver` and a non-zero improvement threshold to avoid over-interpreting negligible differences.
