#!/usr/bin/env python3
"""
01_extract_hydrographs.py

Extract IFS river discharge hydrographs at station locations and prepare
dashboard-ready benchmarking data.

This script is the second step of the ifs-riverbench workflow. It reads
monthly global river discharge GRIB files produced by `00_extract_rivers_mars.py`,
extracts model river discharge at the selected CaMa-Flood grid resolution,
reads observed discharge from a NetCDF file or Zarr store, computes basic metrics,
and writes dashboard-ready JSON/CSV files.

Expected GRIB structure
-----------------------
/perm/pad/flood_cases/grib/<expver>/<date_start>_<date_end>/
├── Globe_river_discharge_<expver>_201801.grb
├── Globe_river_discharge_<expver>_201802.grb
└── ...

Expected station CSV columns
----------------------------
Id, Provid, Efas, Glofas, Gid, Name, StatLon, StatLat, ProvArea,
River, Country, ..., Cama1lon, Cama1lat, Cama1area, ...,
Cama3lon, Cama3lat, Cama3area, ...,
Cama6lon, Cama6lat, Cama6area, ...,
Cama15lon, Cama15lat, Cama15area, ...

Expected observation NetCDF or Zarr
------------------------------------
The observation file/store may be either:

- a NetCDF file, for example:
  Qobs_24_1980-2025_withcaravan.nc

- a Zarr store, for example:
  Qobs_24_1980-2025_withcaravan.zarr

Expected dimensions:
    time
    station

Expected variables:
    time(time)
    statid(station)
    discharge(time, station)

The observation `statid` is assumed to match the CSV `Id`.

Parameter convention
--------------------
235270 = river discharge
235275 = flood fraction

Main outputs
------------
dashboard_data/<expver>/<date_start>_<date_end>_<resolution>arcmin/
├── stations/
│   ├── station_000000.json
│   ├── station_000001.json
│   └── ...
├── stations_catalog.json
└── global_station_metrics.csv

Examples
--------
python Workflow/01_extract_hydrographs.py \\
  --expver iyp3 \\
  --date-start 20180101 \\
  --date-end 20181231 \\
  --resolution 15

For monthly archives stored as date=first day of month and step=24/to/...,
if step=24 represents the first day of the month, use:

python Workflow/01_extract_hydrographs.py \\
  --expver iyp3 \\
  --date-start 20180101 \\
  --date-end 20181231 \\
  --resolution 15 \\
  --valid-time-shift-hours -24
"""

from pathlib import Path
import argparse
import json
import time

import numpy as np
import pandas as pd
import xarray as xr

try:
    from dask.diagnostics import ProgressBar
except Exception:
    ProgressBar = None

from scipy.spatial import cKDTree

from eccodes import (
    codes_grib_new_from_file,
    codes_get,
    codes_get_array,
    codes_release,
)


# ------------------------------------------------------------
# Defaults
# ------------------------------------------------------------
DEFAULT_GRIB_ROOT = Path(f"/perm/{os.environ['USER']}/flood_cases/grib")
DEFAULT_DASHBOARD_ROOT = Path("dashboard_data")

DEFAULT_STATION_FILE = Path(
    f"/perm/pad/flood_cases/Stations/allstations_V1_2.csv"
)

DEFAULT_OBS_FILE = Path(
    "/perm/moi/disobs/obs_20260219/Qobs_24_1980-2025_withcaravan.nc"
)

# 235270 = river discharge
DISCHARGE_PARAM = 235270

# Station CSV columns.
station_id_col = "Id"
lon_col = "StatLon"
lat_col = "StatLat"
area_col = "ProvArea"
name_col = "Name"
river_col = "River"
country_col = "Country"

# Observation NetCDF variables.
OBS_TIME_COORD = "time"
OBS_STATID_VAR = "statid"
OBS_VAR = "discharge"

PROGRESS_EVERY_N_STATIONS = 250


# ------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract station hydrographs from monthly IFS river discharge "
            "GRIB files and compare them with observed discharge."
        )
    )

    parser.add_argument(
        "--expver",
        required=True,
        help="Experiment ID, for example iyp3.",
    )

    parser.add_argument(
        "--date-start",
        required=True,
        help="Start date, format YYYY-MM-DD or YYYYMMDD.",
    )

    parser.add_argument(
        "--date-end",
        required=True,
        help="End date, format YYYY-MM-DD or YYYYMMDD.",
    )

    parser.add_argument(
        "--resolution",
        type=int,
        default=15,
        choices=[1, 3, 6, 15],
        help=(
            "CaMa-Flood model resolution in arcmin. "
            "Selects Cama1*, Cama3*, Cama6* or Cama15* columns. "
            "Default: 15."
        ),
    )

    parser.add_argument(
        "--valid-time-shift-hours",
        type=int,
        default=0,
        help=(
            "Shift GRIB valid times before matching observations. "
            "For monthly archives stored as date=first day and step=24/to/... "
            "but representing days of the same month, use -24."
        ),
    )

    parser.add_argument(
        "--grib-root",
        type=Path,
        default=DEFAULT_GRIB_ROOT,
        help="Root directory containing GRIB files.",
    )

    parser.add_argument(
        "--dashboard-root",
        type=Path,
        default=DEFAULT_DASHBOARD_ROOT,
        help="Root output directory for dashboard data.",
    )

    parser.add_argument(
        "--station-file",
        type=Path,
        default=DEFAULT_STATION_FILE,
        help="Station metadata CSV file.",
    )

    parser.add_argument(
        "--obs-file",
        type=Path,
        default=DEFAULT_OBS_FILE,
        help=(
            "Observation discharge file/store. Supported formats: "
            ".nc NetCDF file or .zarr Zarr store."
        ),
    )

    parser.add_argument(
        "--min-matched-days",
        type=int,
        default=2,
        help="Minimum matched daily values required to compute metrics.",
    )

    parser.add_argument(
        "--min-matched-days-for-map",
        type=int,
        default=2,
        help="Minimum matched days required for a station to appear on the map.",
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit on number of GRIB files, useful for testing.",
    )

    parser.add_argument(
        "--max-stations",
        type=int,
        default=None,
        help="Optional limit on number of stations, useful for testing.",
    )

    parser.add_argument(
        "--clip-to-date-range",
        action="store_true",
        help=(
            "Keep only GRIB messages whose shifted valid time falls inside "
            "date-start/date-end."
        ),
    )

    return parser.parse_args()


# ------------------------------------------------------------
# Small utilities
# ------------------------------------------------------------
def parse_date(value):
    return pd.Timestamp(value).normalize()


def safe_date_label(ts):
    return pd.Timestamp(ts).strftime("%Y%m%d")


def cama_columns(resolution):
    """
    Return the station CSV column names for the requested CaMa-Flood resolution.
    """
    prefix = f"Cama{resolution}"

    return {
        "lon": f"{prefix}lon",
        "lat": f"{prefix}lat",
        "area": f"{prefix}area",
        "row": f"{prefix}row",
        "col": f"{prefix}col",
    }


def safe_json_value(v):
    if pd.isna(v):
        return ""

    if isinstance(v, (np.integer, np.floating)):
        return v.item()

    return v


def clean_int(value):
    """
    Convert CSV/NetCDF station IDs to comparable integers.

    Handles values read as:
    1, 1.0, "1", "1.0".
    """
    if pd.isna(value):
        return None

    try:
        return int(float(str(value).strip()))

    except Exception:
        return None


def lonlat_to_unit_xyz(lon, lat):
    """
    Convert lon/lat in degrees to 3D unit-sphere Cartesian coordinates.

    This allows fast spherical nearest-neighbour search with cKDTree.
    """
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)

    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)

    return np.column_stack([x, y, z])


def haversine_km(lon1, lat1, lon2, lat2):
    """
    Great-circle distance between two lon/lat points in kilometres.
    """
    r = 6371.0

    lon1 = np.deg2rad(lon1)
    lat1 = np.deg2rad(lat1)
    lon2 = np.deg2rad(lon2)
    lat2 = np.deg2rad(lat2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )

    return float(2.0 * r * np.arcsin(np.sqrt(a)))


def percent_difference(obs_area, model_area):
    """
    Percentage difference relative to observed upstream area.
    """
    if obs_area is None or model_area is None:
        return None

    if obs_area <= 0:
        return None

    return 100.0 * (model_area - obs_area) / obs_area


def valid_time_from_grib(gid):
    """
    Return the valid time of a GRIB message.

    Prefer validityDate/validityTime. If those are missing, fall back to
    dataDate/dataTime plus forecast step.
    """
    try:
        vd = int(codes_get(gid, "validityDate"))
        vt = int(codes_get(gid, "validityTime"))

        return pd.to_datetime(
            f"{vd:08d}{vt:04d}",
            format="%Y%m%d%H%M",
        )

    except Exception:
        data_date = int(codes_get(gid, "dataDate"))
        data_time = int(codes_get(gid, "dataTime"))
        step = int(codes_get(gid, "step"))

        t0 = pd.to_datetime(
            f"{data_date:08d}{data_time:04d}",
            format="%Y%m%d%H%M",
        )

        return t0 + pd.to_timedelta(step, unit="h")


def station_area_km2(row):
    """
    Observed station upstream area in km² from ProvArea.
    """
    if area_col in row.index and pd.notna(row.get(area_col)):
        try:
            a = float(row.get(area_col))

            if np.isfinite(a) and a > 0:
                return a

        except Exception:
            pass

    return None


def model_area_km2(row, model_area_col):
    """
    Model upstream area in km² from selected CaMa area column.
    """
    if model_area_col in row.index and pd.notna(row.get(model_area_col)):
        try:
            a = float(row.get(model_area_col))

            if np.isfinite(a) and a > 0:
                return a

        except Exception:
            pass

    return None


# ------------------------------------------------------------
# Observation helpers
# ------------------------------------------------------------
def infer_obs_storage_kind(obs_file):
    """
    Infer observation storage type from path.

    Supported:
      - NetCDF file: .nc, .nc4, .cdf
      - Zarr store : .zarr directory/store
    """
    name = str(obs_file).lower()

    if name.endswith(".zarr"):
        return "zarr"

    if name.endswith(".nc") or name.endswith(".nc4") or name.endswith(".cdf"):
        return "netcdf"

    # Conservative fallback:
    # directories are often Zarr stores; files are usually NetCDF.
    if obs_file.is_dir():
        return "zarr"

    return "netcdf"


def open_obs_dataset(obs_file):
    """
    Open the observation discharge archive.

    The observation archive can be either:
      - NetCDF: *.nc
      - Zarr  : *.zarr

    The data are not loaded immediately. The bulk discharge block is read later
    by preload_observations_for_stations().
    """
    if not obs_file.exists():
        raise FileNotFoundError(f"Observation file/store not found: {obs_file}")

    kind = infer_obs_storage_kind(obs_file)

    print(f"Observation storage type: {kind}", flush=True)

    if kind == "zarr":
        print("Opening observation Zarr store...", flush=True)

        try:
            ds = xr.open_zarr(obs_file, consolidated=True)

        except Exception as exc:
            print(
                "  Could not open Zarr with consolidated=True. "
                f"Retrying with consolidated=False. Original error: {exc}",
                flush=True,
            )

            ds = xr.open_zarr(obs_file, consolidated=False)

    elif kind == "netcdf":
        print("Opening observation file/store...", flush=True)
        ds = xr.open_dataset(obs_file, decode_times=True)

    else:
        raise ValueError(f"Unsupported observation storage type: {kind}")

    required = [OBS_TIME_COORD, OBS_STATID_VAR, OBS_VAR]

    for v in required:
        if v not in ds:
            ds.close()
            raise RuntimeError(f"Missing variable {v!r} in {obs_file}")

    print(f"Observation dataset dimensions: {dict(ds.sizes)}", flush=True)

    try:
        chunks = ds[OBS_VAR].chunks

        if chunks is not None:
            print(f"Observation discharge chunks: {chunks}", flush=True)

        else:
            print("Observation discharge chunks: not chunked / eager backend", flush=True)

    except Exception:
        pass

    return ds


def build_obs_station_index(ds_obs):
    """
    Build mapping from observed station ID to NetCDF/Zarr station index.
    """
    print("  Reading observation station IDs...", flush=True)
    t0 = time.time()

    statids = np.asarray(ds_obs[OBS_STATID_VAR].values)

    index = {}

    for i, sid in enumerate(statids):
        sid_int = clean_int(sid)

        if sid_int is not None:
            index[sid_int] = i

    print(
        f"Observation station IDs indexed: {len(index):,}; "
        f"elapsed {(time.time() - t0) / 60.0:.2f} min",
        flush=True,
    )

    return index


def decode_obs_time(ds_obs):
    """
    Decode observation time coordinate into pandas timestamps.

    xarray should decode the CF time units automatically. The fallback handles
    the explicit units in the provided file:
      days since 1980-01-02 00:00:00
    """
    print("  Reading/decoding observation time coordinate...", flush=True)
    t0 = time.time()

    try:
        obs_time = pd.to_datetime(ds_obs[OBS_TIME_COORD].values)

    except Exception:
        raw = np.asarray(ds_obs[OBS_TIME_COORD].values)

        obs_time = pd.Timestamp("1980-01-02 00:00:00") + pd.to_timedelta(
            raw,
            unit="D",
        )

    print(
        f"  Decoded observation time in {(time.time() - t0) / 60.0:.2f} min",
        flush=True,
    )

    return obs_time


def _is_contiguous_indices(indices):
    """
    Return True if a 1D integer index array is contiguous.
    """
    if len(indices) <= 1:
        return True

    return bool(np.all(np.diff(indices) == 1))


def preload_observations_for_stations(
    ds_obs,
    obs_station_index,
    obs_time,
    stations,
    model_time,
):
    """
    Preload observed discharge for all selected stations and the model period.

    This avoids reading one time series per station, which is very slow.

    Works with both NetCDF and Zarr observations. For Zarr, the selected block
    is loaded with Dask-aware progress reporting when Dask is available.

    Returns:
      obs_time_sel
      obs_q_matrix
      station_to_obs_col

    where obs_q_matrix has shape:
      n_time_selected x n_stations_with_obs
    """

    print("Preparing bulk observation preload...", flush=True)
    t_prepare = time.time()

    t0 = min(model_time)
    t1 = max(model_time)

    time_mask = (obs_time >= t0) & (obs_time <= t1)
    time_indices = np.where(time_mask)[0]
    obs_time_sel = obs_time[time_mask]

    if len(time_indices) == 0:
        raise RuntimeError(
            f"No observation times found in requested model period: {t0} to {t1}"
        )

    print(
        f"  Observation time slice: {len(obs_time_sel):,} steps "
        f"from {pd.Timestamp(obs_time_sel[0]):%Y-%m-%d} "
        f"to {pd.Timestamp(obs_time_sel[-1]):%Y-%m-%d}",
        flush=True,
    )

    station_indices = []
    station_loop_indices = []

    for i, row in stations.iterrows():
        station_id = clean_int(row.get(station_id_col))

        if station_id is None:
            continue

        obs_idx = obs_station_index.get(station_id)

        if obs_idx is None:
            continue

        station_loop_indices.append(int(i))
        station_indices.append(int(obs_idx))

    print(
        f"  Stations with matching observation IDs: "
        f"{len(station_indices):,}/{len(stations):,}",
        flush=True,
    )

    if len(station_indices) == 0:
        return obs_time_sel, np.empty((len(obs_time_sel), 0), dtype=np.float32), {}

    # Sorting station indices makes NetCDF/HDF5 reads much more predictable and
    # also matches the typical Zarr chunk layout along the station dimension.
    order = np.argsort(station_indices)
    station_indices_sorted = np.asarray(station_indices, dtype=np.int64)[order]
    station_loop_indices_sorted = np.asarray(station_loop_indices, dtype=np.int64)[order]

    station_to_obs_col = {
        int(station_loop_indices_sorted[j]): int(j)
        for j in range(len(station_loop_indices_sorted))
    }

    if _is_contiguous_indices(time_indices):
        time_indexer = slice(int(time_indices[0]), int(time_indices[-1]) + 1)
        print(
            f"  Using contiguous time slice: "
            f"{int(time_indices[0])}:{int(time_indices[-1]) + 1}",
            flush=True,
        )

    else:
        time_indexer = time_indices
        print(
            "  Using non-contiguous time index array "
            f"with {len(time_indices):,} entries",
            flush=True,
        )

    estimated_mb = (
        len(time_indices)
        * len(station_indices_sorted)
        * np.dtype("float32").itemsize
        / 1024.0**2
    )

    print(
        f"  Observation read request: {len(time_indices):,} times × "
        f"{len(station_indices_sorted):,} stations "
        f"≈ {estimated_mb:.1f} MB as float32",
        flush=True,
    )

    print(
        "  Loading observation discharge block into memory...",
        flush=True,
    )

    t0_load = time.time()

    obs_da = ds_obs[OBS_VAR].isel(
        time=time_indexer,
        station=station_indices_sorted,
    )

    try:
        chunks = obs_da.chunks

        if chunks is not None:
            print(f"  Selected observation chunks: {chunks}", flush=True)

    except Exception:
        chunks = None

    # Cast before loading so the loaded matrix is float32.
    obs_da = obs_da.astype("float32")

    data_has_dask_chunks = getattr(obs_da.data, "chunks", None) is not None

    if data_has_dask_chunks and ProgressBar is not None:
        print("  Dask-backed read detected; showing progress:", flush=True)

        with ProgressBar():
            obs_q_matrix = np.asarray(obs_da.compute().values, dtype=np.float32)

    else:
        if data_has_dask_chunks:
            print(
                "  Dask-backed read detected but dask ProgressBar is unavailable.",
                flush=True,
            )

        obs_q_matrix = np.asarray(obs_da.values, dtype=np.float32)

    obs_q_matrix = np.where(
        np.isfinite(obs_q_matrix) & (np.abs(obs_q_matrix) < 1.0e19),
        obs_q_matrix,
        np.nan,
    )

    print(
        f"  Loaded obs matrix shape: {obs_q_matrix.shape}; "
        f"elapsed {(time.time() - t0_load) / 60.0:.2f} min",
        flush=True,
    )

    print(
        f"  Bulk observation preload complete; "
        f"total elapsed {(time.time() - t_prepare) / 60.0:.2f} min",
        flush=True,
    )

    return obs_time_sel, obs_q_matrix, station_to_obs_col


def read_obs_series_from_preloaded(
    station_loop_index,
    row,
    obs_time_sel,
    obs_q_matrix,
    station_to_obs_col,
    obs_file,
):
    """
    Read observed discharge for one station from the preloaded observation block.
    """

    obs_col = station_to_obs_col.get(int(station_loop_index))

    if obs_col is None:
        return None

    obs_q = obs_q_matrix[:, obs_col]

    mask = np.isfinite(obs_q)

    if not np.any(mask):
        return None

    obs_time_station = obs_time_sel[mask]
    obs_q_station = obs_q[mask]

    station_id = clean_int(row.get(station_id_col))

    return {
        "time": [
            pd.Timestamp(t).strftime("%Y-%m-%d %H:%M")
            for t in obs_time_station
        ],
        "values": np.round(obs_q_station, 3).tolist(),
        "file": str(obs_file),
        "statid": None if station_id is None else int(station_id),
        "unit": "m³ s⁻¹",
    }

# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------
def compute_metrics(model_time, model_values, obs, min_matched_days):
    """
    Compute simple model-observation metrics on common daily dates.

    Metrics:
    - number of matched days
    - Kling-Gupta Efficiency, KGE
    - Pearson correlation
    - RMSE in m³/s
    """
    if obs is None:
        return None

    try:
        mt = pd.to_datetime(model_time)
        ot = pd.to_datetime(obs["time"])
        mv = np.asarray(model_values, dtype=float)
        ov = np.asarray(obs["values"], dtype=float)

    except Exception:
        return None

    mod = (
        pd.Series(mv, index=mt)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    obv = (
        pd.Series(ov, index=ot)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if len(mod) == 0 or len(obv) == 0:
        return None

    # Compare daily means over common dates.
    mod_d = mod.resample("D").mean()
    obs_d = obv.resample("D").mean()

    joined = pd.concat(
        [mod_d.rename("model"), obs_d.rename("obs")],
        axis=1,
    ).dropna()

    if len(joined) < min_matched_days:
        return None

    sim = joined["model"].to_numpy()
    obs_values = joined["obs"].to_numpy()

    rmse = float(np.sqrt(np.mean((sim - obs_values) ** 2)))

    if np.std(sim) == 0 or np.std(obs_values) == 0:
        corr = np.nan
        kge = np.nan

    else:
        corr = float(np.corrcoef(sim, obs_values)[0, 1])

        beta = (
            float(np.mean(sim) / np.mean(obs_values))
            if np.mean(obs_values) != 0
            else np.nan
        )

        alpha = (
            float(np.std(sim) / np.std(obs_values))
            if np.std(obs_values) != 0
            else np.nan
        )

        kge = 1.0 - np.sqrt(
            (corr - 1.0) ** 2
            + (alpha - 1.0) ** 2
            + (beta - 1.0) ** 2
        )

    return {
        "n": int(len(joined)),
        "kge": None if not np.isfinite(kge) else round(float(kge), 3),
        "correlation": None if not np.isfinite(corr) else round(float(corr), 3),
        "rmse": round(rmse, 3),
        "rmse_unit": "m³ s⁻¹",
        "comparison": "daily means on common dates",
    }


# ------------------------------------------------------------
# GRIB scanning and extraction
# ------------------------------------------------------------
def collect_grib_times(
    grib_files,
    date_start,
    date_end,
    clip_to_date_range,
    valid_time_shift_hours,
):
    """
    First GRIB pass: collect shifted valid times for river discharge messages.

    keep_flags has one entry for every discharge message encountered. It is
    used in the second pass to extract only the messages retained here.
    """
    print("Scanning GRIB messages...", flush=True)

    times = []
    keep_flags = []

    shift = pd.to_timedelta(valid_time_shift_hours, unit="h")

    for grib_file in grib_files:
        print(f"  scanning {grib_file.name}", flush=True)

        with open(grib_file, "rb") as f:
            while True:
                gid = codes_grib_new_from_file(f)

                if gid is None:
                    break

                try:
                    param_id = int(codes_get(gid, "paramId"))

                    if param_id != DISCHARGE_PARAM:
                        continue

                    vt_raw = valid_time_from_grib(gid)
                    vt = pd.Timestamp(vt_raw) + shift

                    keep = True

                    if clip_to_date_range:
                        keep = (
                            pd.Timestamp(vt) >= date_start
                            and pd.Timestamp(vt) <= date_end
                        )

                    keep_flags.append(keep)

                    if keep:
                        times.append(vt)

                finally:
                    codes_release(gid)

    if len(times) == 0:
        raise RuntimeError(f"No messages found for paramId={DISCHARGE_PARAM}")

    return times, keep_flags


def extract_model_at_stations(
    grib_files,
    nsta,
    nt,
    target_lons,
    target_lats,
    keep_flags,
):
    """
    Second GRIB pass: extract model river discharge at station model locations.
    """
    print(
        "Finding nearest GRIB grid point once, then extracting all times...",
        flush=True,
    )

    model_q = np.full((nsta, nt), np.nan, dtype=np.float32)

    nearest_indices = None
    it_all = 0
    it_keep = 0

    t_start_extract = time.time()

    for grib_file in grib_files:
        print(f"  extracting from {grib_file.name}", flush=True)

        with open(grib_file, "rb") as f:
            while True:
                gid = codes_grib_new_from_file(f)

                if gid is None:
                    break

                try:
                    param_id = int(codes_get(gid, "paramId"))

                    if param_id != DISCHARGE_PARAM:
                        continue

                    if it_all >= len(keep_flags):
                        raise RuntimeError(
                            "Internal mismatch: more discharge messages in "
                            "second GRIB pass than in first GRIB pass."
                        )

                    keep = keep_flags[it_all]
                    it_all += 1

                    if not keep:
                        continue

                    # Build nearest-neighbour index only once.
                    # All monthly files are assumed to be on the same grid.
                    if nearest_indices is None:
                        print(
                            "  Reading grid lat/lon arrays from first "
                            "river discharge message...",
                            flush=True,
                        )

                        grid_lats = np.asarray(
                            codes_get_array(gid, "latitudes"),
                            dtype=np.float64,
                        )

                        grid_lons = np.asarray(
                            codes_get_array(gid, "longitudes"),
                            dtype=np.float64,
                        )

                        target_lons_for_lookup = target_lons.copy()

                        if (
                            np.nanmin(grid_lons) >= 0.0
                            and np.nanmax(grid_lons) > 180.0
                        ):
                            target_lons_for_lookup = np.mod(
                                target_lons_for_lookup,
                                360.0,
                            )

                        print(
                            f"  Grid points: {len(grid_lats):,}; "
                            f"stations: {nsta:,}",
                            flush=True,
                        )

                        tree = cKDTree(
                            lonlat_to_unit_xyz(grid_lons, grid_lats)
                        )

                        _, nearest_indices = tree.query(
                            lonlat_to_unit_xyz(
                                target_lons_for_lookup,
                                target_lats,
                            ),
                            k=1,
                        )

                        nearest_indices = nearest_indices.astype(np.int64)

                        print("  Nearest grid index built.", flush=True)

                    values = np.asarray(
                        codes_get_array(gid, "values"),
                        dtype=np.float32,
                    )

                    model_q[:, it_keep] = values[nearest_indices]

                    it_keep += 1

                    if it_keep % 10 == 0 or it_keep == nt:
                        print(
                            f"  extracted time step {it_keep:,}/{nt:,}; "
                            f"elapsed "
                            f"{(time.time() - t_start_extract) / 60.0:.1f} min",
                            flush=True,
                        )

                finally:
                    codes_release(gid)

    if it_keep != nt:
        print(
            f"WARNING: expected {nt:,} kept river discharge time steps, "
            f"extracted {it_keep:,}",
            flush=True,
        )

        model_q = model_q[:, :it_keep]

    print(
        f"Fast model extraction complete in "
        f"{(time.time() - t_start_extract) / 60.0:.1f} min",
        flush=True,
    )

    return model_q


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    args = parse_args()

    expver = args.expver
    date_start = parse_date(args.date_start)
    date_end = parse_date(args.date_end)

    if date_end < date_start:
        raise ValueError(
            f"date-end must be >= date-start, got {date_start} to {date_end}"
        )

    date_label = f"{safe_date_label(date_start)}_{safe_date_label(date_end)}"

    model_cols = cama_columns(args.resolution)

    model_lon_col = model_cols["lon"]
    model_lat_col = model_cols["lat"]
    model_area_col = model_cols["area"]
    model_row_col = model_cols["row"]
    model_col_col = model_cols["col"]

    # --------------------------------------------------------
    # GRIB files
    # --------------------------------------------------------
    grib_dir = args.grib_root / expver / date_label

    grib_files = sorted(
        grib_dir.glob(f"Globe_river_discharge_{expver}_*.grb")
    )

    if args.max_files is not None:
        grib_files = grib_files[: args.max_files]

    if len(grib_files) == 0:
        raise RuntimeError(
            f"No GRIB files found in {grib_dir} "
            f"with pattern Globe_river_discharge_{expver}_*.grb"
        )

    # --------------------------------------------------------
    # Output files
    # --------------------------------------------------------
    out_label = f"{date_label}_{args.resolution}arcmin"

    outdir = args.dashboard_root / expver / out_label
    station_json_dir = outdir / "stations"
    catalog_json = outdir / "stations_catalog.json"
    metrics_csv = outdir / "global_station_metrics.csv"

    outdir.mkdir(parents=True, exist_ok=True)
    station_json_dir.mkdir(parents=True, exist_ok=True)

    print("")
    print("============================================================")
    print("IFS river discharge hydrograph extraction")
    print("============================================================")
    print(f"Experiment             : {expver}")
    print(f"Date range             : {date_start:%Y-%m-%d} to {date_end:%Y-%m-%d}")
    print(f"CaMa resolution        : {args.resolution} arcmin")
    print(f"Model lon/lat          : {model_lon_col}, {model_lat_col}")
    print(f"Model area             : {model_area_col}")
    print(f"GRIB directory         : {grib_dir}")
    print(f"GRIB files             : {len(grib_files)}")
    print(f"Station CSV            : {args.station_file}")
    print(f"Observation file       : {args.obs_file}")
    print(f"Output directory       : {outdir}")
    print(f"Valid time shift       : {args.valid_time_shift_hours} hours")

    if args.clip_to_date_range:
        print("Time clipping          : enabled")

    else:
        print("Time clipping          : disabled")

    print(f"Max files              : {args.max_files}")
    print(f"Max stations           : {args.max_stations}")
    print("============================================================")
    print("")

    # --------------------------------------------------------
    # Basic file checks
    # --------------------------------------------------------
    if not args.station_file.exists():
        raise FileNotFoundError(
            f"Station CSV not found: {args.station_file}\n"
            f"Check available files with:\n"
            f"  ls -lh {args.station_file.parent}"
        )

    if not args.obs_file.exists():
        raise FileNotFoundError(
            f"Observation file/store not found: {args.obs_file}\n"
            f"Check available files with:\n"
            f"  ls -lh {args.obs_file.parent}"
        )

    # --------------------------------------------------------
    # Read stations
    # --------------------------------------------------------
    print("Reading station CSV...", flush=True)
    stations = pd.read_csv(args.station_file)

    print(f"Loaded stations from CSV: {len(stations):,}", flush=True)

    required_station_cols = [
        station_id_col,
        lon_col,
        lat_col,
        area_col,
        model_lon_col,
        model_lat_col,
        model_area_col,
    ]

    missing_cols = [
        c for c in required_station_cols
        if c not in stations.columns
    ]

    if missing_cols:
        raise RuntimeError(
            "Missing required columns in station CSV: "
            + ", ".join(missing_cols)
        )

    print("Cleaning station coordinates and IDs...", flush=True)

    for c in [lon_col, lat_col, model_lon_col, model_lat_col]:
        stations[c] = pd.to_numeric(stations[c], errors="coerce")

    stations[station_id_col] = stations[station_id_col].apply(clean_int)

    stations = stations.dropna(
        subset=[
            station_id_col,
            lon_col,
            lat_col,
            model_lon_col,
            model_lat_col,
        ]
    ).copy()

    stations = stations[
        stations[lon_col].between(-180, 180)
        & stations[lat_col].between(-90, 90)
        & stations[model_lon_col].between(-180, 360)
        & stations[model_lat_col].between(-90, 90)
    ].copy()

    stations = stations.reset_index(drop=True)

    if args.max_stations is not None:
        stations = stations.iloc[: args.max_stations].copy()
        stations = stations.reset_index(drop=True)

    nsta = len(stations)

    print(f"Valid stations used: {nsta:,}", flush=True)

    # --------------------------------------------------------
    # Open observations
    # --------------------------------------------------------
    print("Opening observation file/store...", flush=True)
    ds_obs = open_obs_dataset(args.obs_file)

    print("Building observation station index...", flush=True)
    obs_station_index = build_obs_station_index(ds_obs)

    print("Decoding observation time axis...", flush=True)
    obs_time = decode_obs_time(ds_obs)

    print(f"Observation time steps: {len(obs_time):,}", flush=True)
    print(
        f"Observation period: "
        f"{pd.Timestamp(obs_time[0]):%Y-%m-%d} to "
        f"{pd.Timestamp(obs_time[-1]):%Y-%m-%d}",
        flush=True,
    )

    # --------------------------------------------------------
    # First GRIB pass: collect valid times
    # --------------------------------------------------------
    times, keep_flags = collect_grib_times(
        grib_files=grib_files,
        date_start=date_start,
        date_end=date_end,
        clip_to_date_range=args.clip_to_date_range,
        valid_time_shift_hours=args.valid_time_shift_hours,
    )

    nt = len(times)

    time_labels = [
        pd.Timestamp(t).strftime("%Y-%m-%d %H:%M")
        for t in times
    ]

    print(f"Found {nt:,} river discharge time steps kept for extraction", flush=True)
    print(
        f"Model period after time shift: "
        f"{pd.Timestamp(times[0]):%Y-%m-%d %H:%M} to "
        f"{pd.Timestamp(times[-1]):%Y-%m-%d %H:%M}",
        flush=True,
    )

    # --------------------------------------------------------
    # Second GRIB pass: extract model values at station locations
    # --------------------------------------------------------
    target_lats = stations[model_lat_col].to_numpy(dtype=float)
    target_lons = stations[model_lon_col].to_numpy(dtype=float)

    model_q = extract_model_at_stations(
        grib_files=grib_files,
        nsta=nsta,
        nt=nt,
        target_lons=target_lons,
        target_lats=target_lats,
        keep_flags=keep_flags,
    )

    if model_q.shape[1] != len(time_labels):
        time_labels = time_labels[: model_q.shape[1]]

    # --------------------------------------------------------
    # Observations, metrics, per-station JSON, and catalogue
    # --------------------------------------------------------
    catalog = []
    metrics_rows = []

    print(
        f"Writing per-station JSON files and computing metrics for "
        f"{nsta:,} stations...",
        flush=True,
    )

    t_start_json = time.time()
    model_time_pd = pd.to_datetime(time_labels)

    n_with_obs = 0
    n_with_metrics = 0
    n_on_map = 0

    obs_time_sel, obs_q_matrix, station_to_obs_col = preload_observations_for_stations(
        ds_obs=ds_obs,
        obs_station_index=obs_station_index,
        obs_time=obs_time,
        stations=stations,
        model_time=model_time_pd,
    )

    for i, row in stations.iterrows():

        if (
            (i + 1) == 1
            or (i + 1) % PROGRESS_EVERY_N_STATIONS == 0
            or (i + 1) == nsta
        ):
            elapsed_min = (time.time() - t_start_json) / 60.0
            rate = (i + 1) / elapsed_min if elapsed_min > 0 else np.nan
            remaining = nsta - (i + 1)
            eta_min = (
                remaining / rate
                if np.isfinite(rate) and rate > 0
                else np.nan
            )

            print(
                f"  station {i + 1:,}/{nsta:,} | "
                f"with obs: {n_with_obs:,} | "
                f"with metrics: {n_with_metrics:,} | "
                f"on map: {n_on_map:,} | "
                f"elapsed: {elapsed_min:.1f} min | "
                f"ETA: {eta_min:.1f} min",
                flush=True,
            )

        obs = read_obs_series_from_preloaded(
            station_loop_index=i,
            row=row,
            obs_time_sel=obs_time_sel,
            obs_q_matrix=obs_q_matrix,
            station_to_obs_col=station_to_obs_col,
            obs_file=args.obs_file,
        )

        if obs is not None:
            n_with_obs += 1

        metrics = compute_metrics(
            model_time=time_labels,
            model_values=model_q[i, :],
            obs=obs,
            min_matched_days=args.min_matched_days,
        )

        if metrics is not None:
            n_with_metrics += 1

        station_id = f"station_{i:06d}"
        json_file = station_json_dir / f"{station_id}.json"

        payload = {
            "station_index": int(i),
            "station_id": station_id,
            "source_station_id": int(row[station_id_col]),
            "expver": expver,
            "resolution_arcmin": int(args.resolution),
            "valid_time_shift_hours": int(args.valid_time_shift_hours),
            "time": time_labels,
            "model_discharge": np.round(model_q[i, :], 3).astype(float).tolist(),
            "obs": obs,
            "metrics": metrics,
        }

        json_file.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )

        obs_area_km2 = station_area_km2(row)
        mod_area_km2 = model_area_km2(row, model_area_col)

        area_diff_pct = percent_difference(
            obs_area_km2,
            mod_area_km2,
        )

        distance_km = haversine_km(
            float(row[lon_col]),
            float(row[lat_col]),
            float(row[model_lon_col]),
            float(row[model_lat_col]),
        )

        cat = {
            "station_index": int(i),
            "station_id": station_id,
            "source_station_id": int(row[station_id_col]),
            "expver": expver,
            "resolution_arcmin": int(args.resolution),
            "valid_time_shift_hours": int(args.valid_time_shift_hours),
            "file": f"stations/{station_id}.json",
            "name": safe_json_value(row.get(name_col, "")),
            "river": safe_json_value(row.get(river_col, "")),
            "country_code": safe_json_value(row.get(country_col, "")),
            "lon": float(row[lon_col]),
            "lat": float(row[lat_col]),
            "model_lon": float(row[model_lon_col]),
            "model_lat": float(row[model_lat_col]),
            "model_row": safe_json_value(row.get(model_row_col, "")),
            "model_col": safe_json_value(row.get(model_col_col, "")),
            "distance_station_to_model_km": round(distance_km, 3),
            "upstream_area_km2": (
                None if obs_area_km2 is None else round(obs_area_km2, 3)
            ),
            "model_upstream_area_km2": (
                None if mod_area_km2 is None else round(mod_area_km2, 3)
            ),
            "upstream_area_difference_pct": (
                None if area_diff_pct is None else round(area_diff_pct, 2)
            ),
            "kge": None if metrics is None else metrics.get("kge"),
            "correlation": None if metrics is None else metrics.get("correlation"),
            "rmse": None if metrics is None else metrics.get("rmse"),
            "matched_days": None if metrics is None else metrics.get("n"),
        }

        include_on_map = (
            metrics is not None
            and metrics.get("n") is not None
            and metrics.get("n") >= args.min_matched_days_for_map
        )

        if include_on_map:
            catalog.append(cat)
            n_on_map += 1

        if metrics is not None:
            metrics_rows.append({
                **cat,
                "KGE": metrics.get("kge"),
                "Correlation": metrics.get("correlation"),
                "RMSE_m3s": metrics.get("rmse"),
                "Matched_days": metrics.get("n"),
                "Obs_file": str(args.obs_file),
            })

    print("Writing final catalogue and metrics CSV...", flush=True)

    catalog_json.write_text(
        json.dumps(catalog, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    pd.DataFrame(metrics_rows).to_csv(metrics_csv, index=False)

    ds_obs.close()

    print("")
    print("============================================================")
    print("Hydrograph extraction summary")
    print("============================================================")
    print(f"Saved catalogue             : {catalog_json}")
    print(f"Saved metrics CSV           : {metrics_csv}")
    print(f"Saved station JSON directory: {station_json_dir}")
    print(f"Stations processed          : {nsta:,}")
    print(f"Stations with observations  : {n_with_obs:,}")
    print(f"Stations with metrics       : {n_with_metrics:,}")
    print(f"Stations in map catalogue   : {len(catalog):,}")
    print(f"Done in {(time.time() - t_start_json) / 60.0:.1f} min")
    print("============================================================")


if __name__ == "__main__":
    main()
