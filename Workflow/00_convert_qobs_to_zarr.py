#!/usr/bin/env python3
"""
convert_qobs_nc_to_zarr.py

Convert the river discharge observation NetCDF archive to a chunked Zarr store.

Recommended for repeated ifs-riverbench extraction over many experiments.

Input:
  /perm/moi/disobs/obs_20260219/Qobs_24_1980-2025_withcaravan.nc

Output:
  /perm/{os.environ['USER']}/flood_cases/Stations/Qobs_24_1980-2025_withcaravan.zarr
for example:
  /perm/pad/flood_cases/Stations/Qobs_24_1980-2025_withcaravan.zarr

Suggested chunking:
  discharge(time, station) chunks = (365, 1024)

This script forces Zarr v2 to avoid warnings about consolidated metadata in
Zarr v3.
"""

from pathlib import Path
import shutil
import time
import os
import xarray as xr
from dask.diagnostics import ProgressBar


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------
# FC add: generalise the path for the user passed the test:
#python3 -c "import os; from pathlib import Path; print(Path(f'/perm/{os.environ[\"USER\"]}/flood_cases/Stations/#Qobs_24_1980-2025_withcaravan.zarr'))"

IN_NC = Path("/perm/moi/disobs/obs_20260219/Qobs_24_1980-2025_withcaravan.nc")
OUT_ZARR = Path(f"/perm/{os.environ['USER']}/flood_cases/Stations/Qobs_24_1980-2025_withcaravan.zarr")

TIME_CHUNK = 365
STATION_CHUNK = 1024

OVERWRITE = True


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message):
    print(f"[{now()}] {message}", flush=True)


def elapsed_minutes(t0):
    return (time.time() - t0) / 60.0


def format_bytes(nbytes):
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(nbytes)

    for unit in units:
        if x < 1024.0:
            return f"{x:.1f} {unit}"
        x /= 1024.0

    return f"{x:.1f} PB"


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    t_script = time.time()

    log("============================================================")
    log("Convert Qobs NetCDF to Zarr")
    log("============================================================")
    log(f"Input NetCDF : {IN_NC}")
    log(f"Output Zarr  : {OUT_ZARR}")
    log(f"Chunks       : time={TIME_CHUNK}, station={STATION_CHUNK}")
    log("Zarr format  : v2")
    log("============================================================")

    if not IN_NC.exists():
        raise FileNotFoundError(f"Input NetCDF not found: {IN_NC}")

    if OUT_ZARR.exists():
        if OVERWRITE:
            log(f"Removing existing Zarr store: {OUT_ZARR}")
            t0 = time.time()
            shutil.rmtree(OUT_ZARR)
            log(f"Removed existing store in {elapsed_minutes(t0):.2f} min")
        else:
            raise FileExistsError(
                f"Output Zarr already exists and OVERWRITE=False: {OUT_ZARR}"
            )

    # --------------------------------------------------------
    # Open NetCDF
    # --------------------------------------------------------
    log("Opening NetCDF dataset...")
    t0 = time.time()

    ds = xr.open_dataset(IN_NC)

    log(f"Opened dataset in {elapsed_minutes(t0):.2f} min")
    log(f"Dataset variables: {list(ds.data_vars)}")
    log(f"Dataset dimensions: {dict(ds.sizes)}")

    if "discharge" not in ds:
        raise KeyError("Variable 'discharge' not found in dataset.")

    if "time" not in ds.dims or "station" not in ds.dims:
        raise KeyError("Expected dimensions 'time' and 'station' not found.")

    n_time = int(ds.sizes["time"])
    n_station = int(ds.sizes["station"])

    log(f"time steps : {n_time:,}")
    log(f"stations   : {n_station:,}")

    approx_size = n_time * n_station * 4
    log(f"Approx discharge array size as float32: {format_bytes(approx_size)}")

    # --------------------------------------------------------
    # Decode useful metadata for sanity check
    # --------------------------------------------------------
    try:
        tmin = str(ds["time"].values[0])
        tmax = str(ds["time"].values[-1])
        log(f"Time range : {tmin} to {tmax}")
    except Exception as exc:
        log(f"Could not print time range: {exc}")

    try:
        statid_min = int(ds["statid"].values.min())
        statid_max = int(ds["statid"].values.max())
        log(f"statid range: {statid_min} to {statid_max}")
    except Exception as exc:
        log(f"Could not print statid range: {exc}")

    # --------------------------------------------------------
    # Chunk dataset
    # --------------------------------------------------------
    log("Applying Dask chunks...")
    t0 = time.time()

    ds = ds.chunk(
        {
            "time": TIME_CHUNK,
            "station": STATION_CHUNK,
        }
    )

    log(f"Chunking prepared in {elapsed_minutes(t0):.2f} min")

    log("Chunk structure:")
    for name in ds.variables:
        try:
            chunks = ds[name].chunks
            if chunks is not None:
                log(f"  {name}: {chunks}")
        except Exception:
            pass

    # --------------------------------------------------------
    # Encoding
    # --------------------------------------------------------
    encoding = {
        "discharge": {
            "chunks": (TIME_CHUNK, STATION_CHUNK),
            "dtype": "float32",
        },
        "time": {
            "chunks": (TIME_CHUNK,),
        },
        "statid": {
            "chunks": (STATION_CHUNK,),
        },
    }

    log("Encoding:")
    for key, value in encoding.items():
        log(f"  {key}: {value}")

    # --------------------------------------------------------
    # Write Zarr
    # --------------------------------------------------------
    log("Starting Zarr write...")
    log("This is the long step. Dask progress will be printed below.")
    t0 = time.time()

    with ProgressBar():
        ds.to_zarr(
            OUT_ZARR,
            mode="w",
            encoding=encoding,
            consolidated=True,
            zarr_format=2,
        )

    log(f"Finished Zarr write in {elapsed_minutes(t0):.2f} min")

    # --------------------------------------------------------
    # Reopen and verify
    # --------------------------------------------------------
    log("Reopening Zarr store for verification...")
    t0 = time.time()

    ds_z = xr.open_zarr(
        OUT_ZARR,
        consolidated=True,
    )

    log(f"Reopened Zarr in {elapsed_minutes(t0):.2f} min")
    log(f"Zarr dimensions: {dict(ds_z.sizes)}")

    if "discharge" in ds_z:
        log(f"Zarr discharge chunks: {ds_z['discharge'].chunks}")

    try:
        z_tmin = str(ds_z["time"].values[0])
        z_tmax = str(ds_z["time"].values[-1])
        log(f"Zarr time range: {z_tmin} to {z_tmax}")
    except Exception as exc:
        log(f"Could not verify Zarr time range: {exc}")

    log("============================================================")
    log(f"Done. Total elapsed time: {elapsed_minutes(t_script):.2f} min")
    log(f"Written: {OUT_ZARR}")
    log("============================================================")


if __name__ == "__main__":
    main()
