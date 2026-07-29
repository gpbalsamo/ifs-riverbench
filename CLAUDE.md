# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`ifs-riverbench` is a river-discharge benchmarking pipeline. It retrieves model discharge data (IFS/CaMa-Flood, GloFAS/LISFLOOD, and JRC LSTM), extracts hydrographs at gauge stations, computes metrics (KGE′, correlation, RMSE), and builds interactive Plotly HTML dashboards for multi-experiment comparison — optionally uploaded to ECMWF Sites. All code lives in `Workflow/`.

The **active work** (`AIFL` branch) compares the JRC LSTM experiments **AIFL** (free run) and **AIFL-DA** (data-assimilated) against **EFAS 6.0** and the gauge **observations**, for 2023 at 6 arcmin. The AIFL/AIFL-DA hydrographs carry a per-station **lead-time selector** (`lead_series` in the station JSON): the LSTM is issued every 6 h and each of the 20 forecast leads (+6 h … +120 h) can be stacked across all issue times to give a dense 6-hourly curve, overlaid on the observations. The live deployment is <https://sites.ecmwf.int/ecm7072/riverbench_AIFL/> (login-gated).

> **Hooks / where the detail lives:** the LSTM timestamp + lead handling is documented in [`Workflow/AIFL_TIMESERIES.md`](Workflow/AIFL_TIMESERIES.md); repository-scoped agent notes (verified gotchas, upload recipe, compact-JSON rationale) live in agent **repo memory** at `/memories/repo/jrc-dashboard.md`. Update both when this pipeline changes.

## Environment

This is an ECMWF HPC project. Set up the conda environment before running anything:

```bash
module load conda/26.1.1-3
conda activate riverbench      # Python 3.12
```

Key runtime dependencies: `numpy pandas xarray dask scipy eccodes plotly cartopy zarr`. GloFAS benchmarking (`benchmark_cmf_vs_glofas5.py`) additionally uses `metview`. The upload step needs `sites-toolkit` (`pip install sites-toolkit --user -U -i https://get.ecmwf.int/repository/pypi-all/simple`).

MARS retrieval requires the `mars` command and MARS access.

## Running the pipeline

The end-to-end driver is `Workflow/ifs-riverbench.sh`. **All configuration lives in the settings block at the top of that script** (experiment list, date range, resolution, metrics, reference experiment, and the `EXTRACT_MARS_GRIB` / `EXTRACT_MARS_HYDRO` / `EXTRACT_GLOFAS_HYDRO` / `INCLUDE_GLOFAS_IN_DASHBOARD` switches) — edit there rather than passing flags.

```bash
cd Workflow
bash ifs-riverbench.sh
```

Set `EXTRACT_MARS_GRIB=False` to reuse existing GRIB instead of re-retrieving from MARS.

The individual steps can also be run standalone (see README.md for the full flag reference on each). Ordered stages:

1. `00_extract_rivers_mars.py` — retrieve river-discharge GRIB from MARS
2. `00_convert_qobs_to_zarr.py` — optional one-time NetCDF→Zarr conversion of observations
3. `01_extract_hydrographs.py` — extract station hydrographs from GRIB + observations, compute metrics
4. `02_build_dashboard.py` — build the interactive HTML dashboard
5. `04_prepare_sites_bundle.py` — assemble HTML + `dashboard_data/` into `site_bundle/`
6. `03_upload_dashboard.py` — upload the bundle to ECMWF Sites

### Uploading

```bash
export ECMWF_SITES_TOKEN="<token>"      # env var only, never commit
python3 03_upload_dashboard.py --workflow-dir Workflow/site_bundle
```

For the live AIFL deployment the site name is **`riverbench_AIFL`** (pass `--site-name riverbench_AIFL`; default is `riverbench`). The SDK `cm.upload` does **not** follow symlinks, so upload from the real `dashboard_data/{AIFL,AIFL_DA,EFAS_6.0}` paths, not `site_bundle/`. When the JSON layout changes size, purge the stale remote tree first (`cm.delete(remote_path='dashboard_data/<exp>', recursive=True)`) so the fresh payload has quota headroom, then re-upload and refresh `index.html` (delete + upload). See `/memories/repo/jrc-dashboard.md` for the exact recipe.


### Local preview

```bash
cd Workflow && python3 -m http.server 8000   # then browse http://localhost:8000/site_bundle/
```
Dashboards fetch station JSON via HTTP, so opening the HTML file directly from disk may not work.

## Architecture

The pipeline is a series of standalone scripts connected by a **shared on-disk data contract**, not by imports. Every stage after extraction reads and writes this layout:

```
dashboard_data/<expver>/<date_start>_<date_end>_<resolution>arcmin/
├── stations_catalog.json          # station index + per-station metric summary
├── global_station_metrics.csv     # flat metrics table across all stations
└── stations/station_NNNNNN.json   # per-station time series + metrics (model & obs curves)
```

`02_build_dashboard.py` treats any experiment as just another `<expver>` directory in this layout, regardless of the model that produced it. That is the key extension point: to add a new model source, write a producer that emits this layout, then add the expver to the dashboard experiment list.

There are three independent **data producers**, all writing the layout above:

- **IFS / CaMa-Flood** (`01_extract_hydrographs.py`): reads monthly global GRIB, samples model discharge at each station's CaMa-Flood grid cell (`Cama{1,3,6,15}lon/lat/area` columns in the station CSV select the resolution), reads observed discharge from NetCDF or Zarr, computes metrics.
- **GloFAS / LISFLOOD** (`benchmark_cmf_vs_glofas5.py`, `extract_Glofas_KGE_at_station_points.py`): benchmark engine using `metview` against local GloFAS GRIB (not from MARS).
- **JRC LSTM — AIFL / AIFL-DA / EFAS 6.0** (`05_prepare_jrc_data.py`): converts LSTM NPZ predictions and EFAS CSV into the station-JSON layout. Read `Workflow/AIFL_TIMESERIES.md` before touching this. Each NPZ (`station_{sid}_{free,da}.npz`) has shape `(n_windows, 60)` = 40 assimilation + 20 forecast steps, and now carries an exact **`forecast_times`** key `(n_windows, 20)` — so `05` reads valid times **directly from the NPZ** (the old synthetic-forcing-grid + `+3`-offset heuristic is retired). The LSTM is issued at a **6-hourly cadence** (inference stride = 1); stacking a fixed lead across all windows yields a 6-hourly series.

`02_build_dashboard.py` supports three `--colour-mode` values: `best_metric`, `best_experiment` (needs `--control-expver` + `--best-experiment-min-improvement`), and `experiment_difference` (computes `second_expver − first_expver`; `--difference-threshold` sets the neutral band).

### Compact `lead_series` (station JSON size vs Sites quota)

Storing 20 leads × ~1460 six-hourly points per station is what makes the AIFL/AIFL-DA JSONs large. To stay under the ECMWF Sites storage quota (~2 GB; a naive layout blew it at ~1.1 MB/station, ~2.9 G total), `05_prepare_jrc_data.py` writes a **compact** `lead_series`:

- `base_time` — the lead-1 valid-time axis stored **once** (not per lead).
- `sim` — per-lead model values `[20][n_windows]`, ordered to match `base_time`.
- `kge` — per-lead KGE′; `n_leads`, `step_hours` (6).
- **No** per-lead `time` or `obs`: every lead's valid time is `base_time + lead × step_hours`, reconstructed in the browser by `leadTimeAxis()` in `02_build_dashboard.py`; observations come from the shared obs payload, so per-lead obs copies are redundant.

This cut the payload to ~366 KB/station (~1.0 G total, ~3× smaller) with identical metrics. `leadTimeAxis()` falls back to a legacy per-lead `time[]` array if present, so old JSONs still render.


## Conventions

- GRIB param `235270` = river discharge (the focus), `235275` = flood fraction.
- Resolution is in arcmin: `1`, `3`, `6`, or `15`.
- `--valid-time-shift-hours -24` is required for the monthly MARS archive convention (files stored with `date=first day of month`, but each `step` is a later valid day).
- Prefer **Zarr** observations over NetCDF for multi-year / multi-experiment runs (much faster).
- KGE′ here is the modified KGE: `1 − √[(r−1)² + (γ−1)² + (β−1)²]` with `γ = CV_sim/CV_obs`; components stored per station as `metrics.kge_r`, `metrics.kge_gamma`, `metrics.kge_beta`.

## Git / generated files

- `main` — upstream stable; `AIFL` — active branch (EFAS vs AIFL/AIFL-DA comparison work).
- Do **not** commit generated outputs: `dashboard_data/`, `global_station_dashboard_*.html`, `*.zarr/`, `*.npz`. Commit the workflow scripts, not benchmark outputs.
