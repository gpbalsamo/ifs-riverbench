#!/usr/bin/env python3
"""
01_extract_hydrographs.py

Extract model river discharge hydrographs at station locations and prepare
dashboard-ready benchmarking data.

This script is the second step of the ifs-riverbench workflow. It reads
monthly global GRIB files produced by `00_extract_rivers_mars.py`, extracts
the model river discharge at the nearest 1 arcmin model grid point for each
station, reads the corresponding observed discharge time series, computes
basic skill metrics, and writes compact JSON/CSV files used by the dashboard.

Workflow position
-----------------
1. 00_extract_rivers_mars.py
   Retrieve monthly GRIB files from MARS.

2. 01_extract_hydrographs.py
   Extract station hydrographs, match observations, compute metrics,
   and prepare dashboard data.

3. 02_build_dashboard.py
   Build the interactive dashboard for model-observation benchmarking.

Main outputs
------------
dashboard_data/
├── stations/
│   ├── station_000000.json
│   ├── station_000001.json
│   └── ...
├── stations_catalog.json
└── global_station_metrics.csv

Parameter convention
--------------------
235270 = river discharge

Notes
-----
- Model extraction is optimized by building the nearest-neighbour grid lookup
  only once from the first GRIB message.
- Observations are converted from mm/day to m³/s using the station upstream area.
- Skill metrics are computed on daily means over common model-observation dates.
- Stations are included in the dashboard catalogue only when they have at least
  MIN_MATCHED_DAYS_FOR_MAP matched days. This avoids clickable stations with
  empty or unusable hydrographs.

Requirements
------------
Python packages:
- numpy
- pandas
- xarray
- scipy
- eccodes

The script expects:
- Monthly GRIB files in GRIB_DIR
- Station metadata CSV in station_file
- Observation NetCDF files under OBS_ROOT
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import xarray as xr

from scipy.spatial import cKDTree

from eccodes import (
    codes_grib_new_from_file,
    codes_get,
    codes_get_array,
    codes_release,
)


# ------------------------------------------------------------
# User settings
# ------------------------------------------------------------
# Station metadata file.
#
# The newer CSV appears to use slightly different column names, so several
# helper functions below are written to support both old and new formats.
#
# Old example:
# station_file = Path("/perm/pad/flood_cases/Stations/All_stations.csv")
#
# Current station file:
station_file = Path("/perm/pad/flood_cases/Stations/allstations_V1.1.csv")

# Minimum number of matched daily model-observation values required
# to compute metrics.
MIN_MATCHED_DAYS = 2

# Minimum number of matched days required for a station to appear
# in the dashboard map catalogue.
MIN_MATCHED_DAYS_FOR_MAP = 2


# ------------------------------------------------------------
# GRIB input files
# ------------------------------------------------------------
# Directory containing the monthly GRIB files produced by script 00.
GRIB_DIR = Path("/perm/pad/flood_cases/grib")

# Current file pattern.
#
# The filename still says "flood", but the extracted parameter below is
# river discharge: paramId 235270.
#
# You can later rename the GRIB files to something like
# Globe_river_discharge_YYYYMM.grb if you want the naming to be clearer.
grib_files = sorted(GRIB_DIR.glob("Globe_flood_*.grb"))

# Optional fast test mode for development.
#
# Set TEST_MONTH to a non-empty string to process only the first few files.
# Leave it as an empty string to process the full available period.
TEST_MONTH = "201801-201803"
TEST_MONTH = ""  # Empty string disables test mode.

if TEST_MONTH != "":
    MIN_MATCHED_DAYS = 0
    MIN_MATCHED_DAYS_FOR_MAP = 0
    grib_files = grib_files[0:3]

    if not grib_files[0].exists():
        raise RuntimeError(
            f"Missing test GRIB file: {grib_files[0]}"
        )

    print("")
    print("========================================")
    print("FAST TEST MODE ENABLED")
    print(f"Using only month: {TEST_MONTH}")
    print("========================================")
    print("")

if len(grib_files) == 0:
    raise RuntimeError(
        f"No GRIB files found in {GRIB_DIR}"
    )

print(f"Found {len(grib_files)} monthly GRIB files")


# ------------------------------------------------------------
# Output files and dashboard structure
# ------------------------------------------------------------
OUTDIR = Path("dashboard_data")
STATION_JSON_DIR = OUTDIR / "stations"
CATALOG_JSON = OUTDIR / "stations_catalog.json"
METRICS_CSV = OUTDIR / "global_station_metrics.csv"

OUTDIR.mkdir(exist_ok=True)
STATION_JSON_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# Model parameter and station column names
# ------------------------------------------------------------
# MARS/GRIB parameter to extract.
#
# 235270 = river discharge
DISCHARGE_PARAM = 235270

# Observed station coordinates.
lon_col = "StationLon"
lat_col = "StationLat"

# Pre-matched model grid-point coordinates from the station metadata file.
model_lon_col = "X_1_arcmin"
model_lat_col = "Y_1_arcmin"

# Country column names differ between versions of the station CSV.
country_code_cols = ["Country co", "Country"]


# ------------------------------------------------------------
# Observation settings
# ------------------------------------------------------------
OBS_ROOT = Path("/perm/pad/flood_cases/Stations/netcdf")

# Observation subdirectories to scan recursively for NetCDF files.
OBS_DIRS = [
    "camels",
    "camelsaus",
    "camelsbr",
    "camelsch",
    "camelscl",
    "camelsde",
    "camelsdk",
    "camelses",
    "camelsgb",
    "grdc",
    "hysets",
    "il",
    "lamah",
    "lamahice",
]

# Observation variable and time coordinate names.
OBS_VAR = "streamflow"
OBS_TIME_COORD = "date"

# If True, observation timestamps are shifted from local solar time to UTC
# using longitude / 15 hours. This is useful when daily observations are
# timestamped using local basin/station time rather than UTC.
CONVERT_OBS_LOCAL_SOLAR_TIME_TO_UTC = True


# ------------------------------------------------------------
# Progress-reporting settings
# ------------------------------------------------------------
PROGRESS_EVERY_N_STATIONS = 1000


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def first_available(row, cols, default=""):
    """Return the first non-missing value found in a list of possible columns."""
    for col in cols:
        if col in row.index and pd.notna(row.get(col)):
            return row.get(col)
    return default


def clean_id(value):
    """Clean station/source IDs read from CSV columns."""
    if pd.isna(value):
        return None

    s = str(value).strip()

    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None

    # CSV numeric IDs are sometimes read as floats, for example 12345.0.
    if s.endswith(".0"):
        s = s[:-2]

    return s


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


def valid_time_from_grib(gid):
    """
    Return the valid time of a GRIB message.

    Prefer validityDate/validityTime when present. If these keys are missing,
    fall back to dataDate/dataTime plus forecast step.
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


def build_obs_file_index():
    """
    Build a lookup dictionary linking possible station IDs to observation files.

    The index includes:
    - full NetCDF file stems
    - lowercase file stems
    - final underscore-separated tokens
    - lowercase tokens

    This increases robustness when different datasets use slightly different
    file naming conventions.
    """
    index = {}

    for d in OBS_DIRS:
        folder = OBS_ROOT / d

        if not folder.exists():
            continue

        for f in folder.rglob("*.nc"):
            stem = f.stem

            index[stem] = f
            index[stem.lower()] = f

            if "_" in stem:
                token = stem.split("_")[-1]
                index[token] = f
                index[token.lower()] = f

    print(f"Observation files indexed: {len(set(index.values())):,}")

    return index


def observation_keys_from_station(row):
    """
    Generate possible observation file keys from one station metadata row.
    """
    keys = []

    cols = [
        "G_ID",
        "ID",
        "EFAS-ID",
        "ProvID_1",
        "ProvID_2",
        "Gid",
        "Id",
        "Efas",
        "Glofas",
        "Provid",
    ]

    source = clean_id(row.get("Source_database"))

    for col in cols:
        v = clean_id(row.get(col))

        if v:
            keys.append(v)
            keys.append(v.lower())

            if source:
                keys.append(f"{source}_{v}")
                keys.append(f"{source}_{v}".lower())

    # Remove duplicates while preserving order.
    out, seen = [], set()

    for k in keys:
        if k not in seen:
            out.append(k)
            seen.add(k)

    return out


def find_obs_file(row, obs_index):
    """Find the observation NetCDF file corresponding to one station."""
    for key in observation_keys_from_station(row):
        if key in obs_index:
            return obs_index[key]

    return None


def haversine_km(lon1, lat1, lon2, lat2):
    """Great-circle distance between two lon/lat points in kilometres."""
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


def station_area_m2(row):
    """
    Observed station upstream area in m².

    Old CSV:
      DrainingAr, assumed km².

    New CSV:
      ProvArea, assumed km².
    """
    for col in ["DrainingAr", "ProvArea"]:
        if col in row.index and pd.notna(row.get(col)):
            try:
                a = float(row.get(col))

                if np.isfinite(a) and a > 0:
                    return a * 1e6

            except Exception:
                pass

    return None


def station_area_km2(row):
    """
    Observed station upstream area in km².

    Old CSV:
      DrainingAr.

    New CSV:
      ProvArea.
    """
    for col in ["DrainingAr", "ProvArea"]:
        if col in row.index and pd.notna(row.get(col)):
            try:
                a = float(row.get(col))

                if np.isfinite(a) and a > 0:
                    return a

            except Exception:
                pass

    return None


def model_area_km2(row):
    """
    Model upstream area in km².

    Old CSV:
      - 1_arcmin_uparea(m2)
      - 1_arcmin_uparea(km2)

    New CSV:
      - 1_arcmin_uparea

    The new generic field is interpreted as m² when very large, otherwise km².
    """
    if (
        "1_arcmin_uparea(km2)" in row.index
        and pd.notna(row.get("1_arcmin_uparea(km2)"))
    ):
        return float(row.get("1_arcmin_uparea(km2)"))

    if (
        "1_arcmin_uparea(m2)" in row.index
        and pd.notna(row.get("1_arcmin_uparea(m2)"))
    ):
        return float(row.get("1_arcmin_uparea(m2)")) / 1e6

    if (
        "1_arcmin_uparea" in row.index
        and pd.notna(row.get("1_arcmin_uparea"))
    ):
        a = float(row.get("1_arcmin_uparea"))

        # Likely m² if very large; km² if moderate.
        if a > 1e6:
            return a / 1e6

        return a

    return None


def percent_difference(obs_area, model_area):
    """
    Percentage difference relative to observed upstream area.
    """
    if obs_area is None or model_area is None:
        return None

    if obs_area <= 0:
        return None

    return 100.0 * (model_area - obs_area) / obs_area


def read_obs_series(row, obs_index, model_time):
    """
    Read the observed discharge series for one station.

    Observations are expected as streamflow in mm/day and are converted to m³/s
    using the observed upstream area.
    """
    obs_file = find_obs_file(row, obs_index)

    if obs_file is None:
        return None

    area_m2 = station_area_m2(row)

    if area_m2 is None:
        return None

    try:
        ds = xr.open_dataset(obs_file, decode_times=True)

    except Exception:
        return None

    try:
        if OBS_VAR not in ds:
            return None

        try:
            obs_time = pd.to_datetime(ds[OBS_TIME_COORD].values)

        except Exception:
            raw = np.asarray(ds[OBS_TIME_COORD].values)
            obs_time = (
                pd.Timestamp("1950-01-01")
                + pd.to_timedelta(raw, unit="D")
            )

        obs_mmday = np.asarray(ds[OBS_VAR].values, dtype=float).reshape(-1)

    finally:
        ds.close()

    # Convert mm/day over the upstream area to volumetric discharge in m³/s.
    obs_m3s = obs_mmday * area_m2 / (1000.0 * 86400.0)

    if CONVERT_OBS_LOCAL_SOLAR_TIME_TO_UTC:
        lon = float(row[lon_col])
        shift_hours = lon / 15.0
        obs_time = obs_time - pd.to_timedelta(shift_hours, unit="h")

    # Keep only observations overlapping the model period.
    t0 = min(model_time)
    t1 = max(model_time)

    mask = (
        (obs_time >= t0)
        & (obs_time <= t1)
        & np.isfinite(obs_m3s)
    )

    obs_time = obs_time[mask]
    obs_m3s = obs_m3s[mask]

    if len(obs_time) == 0:
        return None

    return {
        "time": [
            pd.Timestamp(t).strftime("%Y-%m-%d %H:%M")
            for t in obs_time
        ],
        "values": np.round(obs_m3s, 3).tolist(),
        "file": str(obs_file),
        "area_m2": float(area_m2),
    }


def compute_metrics(model_time, model_values, obs):
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

    # Compare daily means. This is more robust than comparing exact
    # sub-daily timestamps directly.
    mod_d = mod.resample("D").mean()
    obs_d = obv.resample("D").mean()

    joined = pd.concat(
        [mod_d.rename("model"), obs_d.rename("obs")],
        axis=1,
    ).dropna()

    if len(joined) < MIN_MATCHED_DAYS:
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


def safe_json_value(v):
    """Convert missing or NumPy scalar values to JSON-safe Python values."""
    if pd.isna(v):
        return ""

    if isinstance(v, (np.integer, np.floating)):
        return v.item()

    return v


# ------------------------------------------------------------
# 1. Read and clean stations
# ------------------------------------------------------------
stations = pd.read_csv(station_file)

print(f"Loaded stations: {len(stations):,}")

# Convert coordinate columns to numeric values.
for c in [lon_col, lat_col, model_lon_col, model_lat_col]:
    stations[c] = pd.to_numeric(stations[c], errors="coerce")

# Keep only stations with valid observed and model coordinates.
stations = stations.dropna(
    subset=[lon_col, lat_col, model_lon_col, model_lat_col]
).copy()

stations = stations[
    stations[lon_col].between(-180, 180)
    & stations[lat_col].between(-90, 90)
    & stations[model_lon_col].between(-180, 360)
    & stations[model_lat_col].between(-90, 90)
].copy()

stations = stations.reset_index(drop=True)

nsta = len(stations)

print(f"Valid stations: {nsta:,}")


# ------------------------------------------------------------
# 2. First GRIB pass: collect valid times
# ------------------------------------------------------------
# This first pass scans the GRIB messages and records the valid time for each
# message matching DISCHARGE_PARAM. The actual values are extracted in the
# second pass.
print("Scanning GRIB messages...")

times = []

for grib_file in grib_files:
    print(f"  scanning {grib_file.name}")

    with open(grib_file, "rb") as f:
        while True:
            gid = codes_grib_new_from_file(f)

            if gid is None:
                break

            try:
                param_id = int(codes_get(gid, "paramId"))

                if param_id == DISCHARGE_PARAM:
                    times.append(valid_time_from_grib(gid))

            finally:
                codes_release(gid)

nt = len(times)

if nt == 0:
    raise RuntimeError(f"No messages found for paramId={DISCHARGE_PARAM}")

time_labels = [
    pd.Timestamp(t).strftime("%Y-%m-%d %H:%M")
    for t in times
]

print(f"Found {nt:,} river discharge time steps")


# ------------------------------------------------------------
# 3. Second GRIB pass: extract model values at station locations
# ------------------------------------------------------------
print(
    "Finding nearest grid point once, then extracting all times...",
    flush=True,
)

# Model array: one row per station, one column per time step.
model_q = np.full((nsta, nt), np.nan, dtype=np.float32)

target_lats = stations[model_lat_col].to_numpy(dtype=float)
target_lons = stations[model_lon_col].to_numpy(dtype=float)

nearest_indices = None
it = 0
t0 = time.time()

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

                # Build the nearest-neighbour lookup only once.
                # All monthly files are assumed to use the same model grid.
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

                    # If the GRIB grid uses 0..360 longitudes, convert station
                    # longitudes from -180..180 to 0..360 for the lookup.
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

                # Extract one river discharge value per station for this time.
                model_q[:, it] = values[nearest_indices]

                it += 1

                if it % 10 == 0 or it == nt:
                    print(
                        f"  extracted time step {it:,}/{nt:,}; "
                        f"elapsed {(time.time() - t0) / 60.0:.1f} min",
                        flush=True,
                    )

            finally:
                codes_release(gid)

if it != nt:
    print(
        f"WARNING: expected {nt:,} river discharge time steps, "
        f"extracted {it:,}",
        flush=True,
    )

    model_q = model_q[:, :it]
    time_labels = time_labels[:it]

print(
    f"Fast model extraction complete in "
    f"{(time.time() - t0) / 60.0:.1f} min",
    flush=True,
)


# ------------------------------------------------------------
# 4. Observations, metrics, per-station JSON, and catalogue
# ------------------------------------------------------------
obs_index = build_obs_file_index()

catalog = []
metrics_rows = []

print("Writing per-station JSON files...")

t0 = time.time()

for i, row in stations.iterrows():

    if (i + 1) % PROGRESS_EVERY_N_STATIONS == 0:
        print(f"  station JSON {i + 1:,}/{nsta:,}", flush=True)

    # Read observed discharge and compute model-observation metrics.
    obs = read_obs_series(row, obs_index, pd.to_datetime(time_labels))
    metrics = compute_metrics(time_labels, model_q[i, :], obs)

    station_id = f"station_{i:06d}"
    json_file = STATION_JSON_DIR / f"{station_id}.json"

    # Per-station payload used by the dashboard when a station is clicked.
    payload = {
        "station_index": int(i),
        "station_id": station_id,
        "time": time_labels,
        "model_discharge": np.round(model_q[i, :], 3).astype(float).tolist(),
        "obs": obs,
        "metrics": metrics,
    }

    json_file.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    # Ancillary station diagnostics for the dashboard catalogue.
    obs_area_km2 = station_area_km2(row)
    mod_area_km2 = model_area_km2(row)

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
        "file": f"stations/{station_id}.json",
        "name": safe_json_value(row.get("Name", "")),
        "river": safe_json_value(row.get("River", "")),
        "country_code": safe_json_value(
            first_available(row, country_code_cols, "")
        ),
        "lon": float(row[lon_col]),
        "lat": float(row[lat_col]),
        "model_lon": float(row[model_lon_col]),
        "model_lat": float(row[model_lat_col]),
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

    # Only stations with enough matched model-observation days are shown
    # on the dashboard map. This avoids empty or unusable clicked stations.
    include_on_map = (
        metrics is not None
        and metrics.get("n") is not None
        and metrics.get("n") >= MIN_MATCHED_DAYS_FOR_MAP
    )

    if include_on_map:
        catalog.append(cat)

    # The CSV contains only stations for which metrics could be computed.
    if metrics is not None:
        metrics_rows.append({
            **cat,
            "KGE": metrics.get("kge"),
            "Correlation": metrics.get("correlation"),
            "RMSE_m3s": metrics.get("rmse"),
            "Matched_days": metrics.get("n"),
            "Obs_file": "" if obs is None else obs.get("file", ""),
        })


# ------------------------------------------------------------
# 5. Write dashboard data
# ------------------------------------------------------------
CATALOG_JSON.write_text(
    json.dumps(catalog, ensure_ascii=False, separators=(",", ":")),
    encoding="utf-8",
)

pd.DataFrame(metrics_rows).to_csv(METRICS_CSV, index=False)

print(f"Saved catalogue: {CATALOG_JSON}")
print(f"Saved metrics CSV: {METRICS_CSV}")
print(f"Saved station JSON directory: {STATION_JSON_DIR}")
print(f"Done in {(time.time() - t0) / 60:.1f} min")
