#!/usr/bin/env python3
"""
00_extract_rivers_mars.py

Retrieve monthly global IFS river/flood-related fields from MARS and save them
as GRIB files.

This script is the first step of the ifs-riverbench workflow. It retrieves
monthly GRIB files from MARS over a user-defined date range. Each monthly file
contains one or more IFS surface parameters needed for the river discharge
benchmarking workflow.

Current configuration
---------------------
- Domain: global
- Period: 2018-01-01 to 2019-12-31
- Forecast stream: oper
- Forecast type: fc
- Forecast step: 24 h
- Surface parameter(s): PARAMS
- Output format: GRIB

Workflow position
-----------------
1. 00_extract_rivers_mars.py
   Retrieve monthly GRIB files from MARS.

2. 01_extract_hydrographs.py
   Extract station hydrographs from the retrieved GRIB files.

3. 02_build_dashboard.py
   Build the interactive dashboard for model-observation benchmarking.

Requirements
------------
This script must be run in an environment where the ECMWF `mars` command is
available and correctly configured.

Example
-------
python 00_extract_rivers_mars.py
"""

from pathlib import Path
import subprocess
import os

import pandas as pd

# ------------------------------------------------------------
# User settings
# ------------------------------------------------------------
# Date range to retrieve.
# The script retrieves complete calendar months between START and END.
START = pd.Timestamp("2018-01-01")
END = pd.Timestamp("2022-12-31")

# Main output directory where monthly GRIB files will be written.
OUTDIR = Path("/perm/pad/flood_cases/grib")

# Directory where the generated MARS request files will be saved.
# Keeping the request files is useful for reproducibility and debugging.
REQDIR = Path("/perm/pad/flood_cases/mars_requests")

# Temporary directory used by MARS during retrieval.
# This can become large, so it is better to place it on a filesystem
# with sufficient space rather than relying on the system default /tmp.
TMPDIR = Path("/perm/pad/flood_cases/tmp_mars")

# MARS parameter:
# 235270 = river discharge
PARAMS = "235270"

# ------------------------------------------------------------
# Create required directories
# ------------------------------------------------------------
# The script creates the output, request, and temporary directories
# if they do not already exist.
OUTDIR.mkdir(parents=True, exist_ok=True)
REQDIR.mkdir(parents=True, exist_ok=True)
TMPDIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# MARS environment
# ------------------------------------------------------------
# Copy the current shell environment and modify only the variables needed
# for this retrieval.
env = dict(os.environ)

# Force MARS temporary files to be written to TMPDIR.
env["TMPDIR"] = str(TMPDIR)
env["TEMP"] = str(TMPDIR)
env["TMP"] = str(TMPDIR)

# Increase MARS read buffer size.
# This can help with large retrievals.
env["MARS_READANY_BUFFER_SIZE"] = "2147483648"

# Disable automatic splitting by dates.
# Here we explicitly control the retrieval month by month.
env["MARS_AUTO_SPLIT_BY_DATES"] = "0"

# ------------------------------------------------------------
# Monthly retrieval loop
# ------------------------------------------------------------
# Build a list of monthly periods between START and END.
# Each period is retrieved independently and written to one GRIB file.
months = pd.period_range(start=START, end=END, freq="M")

print(f"Retrieving {len(months)} monthly GRIB files")
print(f"Output directory: {OUTDIR}")

for im, month in enumerate(months, start=1):

    # First and last day of the current month.
    d0 = month.start_time
    d1 = min(month.end_time, END)

    # String used for file naming, for example 201801.
    yyyymm = month.strftime("%Y%m")

    # Output GRIB file for the current month.
    out_grib = OUTDIR / f"Globe_flood_{yyyymm}.grb"

    # MARS request file for the current month.
    req_file = REQDIR / f"mars_extract_globe_flood_{yyyymm}.req"

    # Avoid re-retrieving files that already exist and are non-empty.
    # This makes the script restartable after interruptions.
    if out_grib.exists() and out_grib.stat().st_size > 0:
        print(f"[{im}/{len(months)}] Skipping {yyyymm}: file already exists")
        continue

    # Build the MARS request.
    #
    # Notes:
    # - class=rd and expver=j1ee are experiment-specific settings.
    # - stream=oper and type=fc retrieve operational forecasts.
    # - levtype=sfc retrieves surface fields.
    # - time=0 and step=24 retrieve the +24 h forecast from the 00 UTC run.
    # - target is the output GRIB file.
    request_text = f"""
retrieve,
  class=rd,
  expver=j1ee,
  stream=oper,
  type=fc,
  levtype=sfc,
  date={d0.strftime('%Y%m%d')}/to/{d1.strftime('%Y%m%d')},
  time=0,
  step=24,
  param={PARAMS},
  target="{out_grib}"
"""

    # Save the request file for traceability.
    req_file.write_text(request_text.strip() + "\n")

    print("")
    print(f"[{im}/{len(months)}] Retrieving {yyyymm}")
    print(f"Date range: {d0.strftime('%Y%m%d')} to {d1.strftime('%Y%m%d')}")
    print(f"Output: {out_grib}")

    try:
        # Execute the MARS request.
        # check=True makes Python raise an exception if MARS returns
        # a non-zero exit status.
        subprocess.run(
            ["mars", str(req_file)],
            env=env,
            check=True,
        )

    except subprocess.CalledProcessError:
        # If MARS fails, keep any partial GRIB file but rename it.
        # This avoids mistaking an incomplete file for a successful retrieval
        # when the script is restarted.
        print(f"FAILED: {yyyymm}")

        if out_grib.exists():
            failed_file = out_grib.with_suffix(".failed.grb")
            out_grib.rename(failed_file)
            print(f"Partial file renamed to: {failed_file}")

        raise

print("")
print("Done.")
print(f"Monthly GRIB files written to: {OUTDIR}")
