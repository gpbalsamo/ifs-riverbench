#!/usr/bin/env python3
"""
00_extract_rivers_mars.py

Retrieve monthly archived IFS river discharge fields from MARS and save them
as GRIB files.

This script is the first step of the ifs-riverbench workflow. It retrieves
global river discharge fields from one or more IFS experiments and stores them
in experiment-specific output directories.

Monthly archive convention
--------------------------
This version assumes the fields are archived as monthly forecast chunks:

- one forecast run per month;
- archive date = first day of the month;
- time = 00 UTC;
- steps = 24, 48, 72, ..., up to the last day of the month.

For example:

January 2018:
  date = 20180101
  step = 24/to/744/by/24

February 2018:
  date = 20180201
  step = 24/to/672/by/24

If the requested date range starts or ends inside a month, only the required
subset of forecast steps is retrieved.

Parameter convention
--------------------
235270 = river discharge
235275 = flood fraction

Workflow position
-----------------
1. 00_extract_rivers_mars.py
   Retrieve monthly GRIB files from MARS.

2. 01_extract_hydrographs.py
   Extract station hydrographs from the retrieved GRIB files.

3. 02_build_dashboard.py
   Build the interactive dashboard for model-observation benchmarking.

Examples
--------
Retrieve one experiment:

python Workflow/00_extract_rivers_mars.py \\
  --expver iyp3 \\
  --date-start 2018-01-01 \\
  --date-end 2022-12-31

Retrieve several experiments:

python Workflow/00_extract_rivers_mars.py \\
  --expver iyp3 j1ee \\
  --date-start 2018-01-01 \\
  --date-end 2022-12-31

Custom output root:

python Workflow/00_extract_rivers_mars.py \\
  --expver iyp3 j1ee \\
  --date-start 2018-01-01 \\
  --date-end 2022-12-31 \\
  --out-root /perm/USER/flood_cases/riverbench_grib
"""

from pathlib import Path
import argparse
import os
import subprocess
import pandas as pd


# ------------------------------------------------------------
# Defaults
# ------------------------------------------------------------
DEFAULT_OUT_ROOT = Path(f"/perm/{os.environ['USER']}/flood_cases/grib")
DEFAULT_REQ_ROOT = Path(f"/perm/{os.environ['USER']}/flood_cases/mars_requests")
DEFAULT_TMPDIR = Path(f"/perm/{os.environ['USER']}/flood_cases/tmp_mars")

# MARS parameter:
# 235270 = river discharge
# 235275 = flood fraction
DEFAULT_PARAMS = "235270"

DEFAULT_CLASS = "rd"
DEFAULT_STREAM = "oper"
DEFAULT_TYPE = "fc"
DEFAULT_LEVTYPE = "sfc"
DEFAULT_TIME = "0"


# ------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve monthly archived IFS river discharge fields from MARS "
            "for one or more experiments."
        )
    )

    parser.add_argument(
        "--expver",
        nargs="+",
        required=True,
        help=(
            "One or more experiment IDs, for example: "
            "--expver iyp3 or --expver iyp3 j1ee"
        ),
    )

    parser.add_argument(
        "--date-start",
        required=True,
        help="Start date to retrieve, format YYYY-MM-DD or YYYYMMDD.",
    )

    parser.add_argument(
        "--date-end",
        required=True,
        help="End date to retrieve, format YYYY-MM-DD or YYYYMMDD.",
    )

    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help=(
            "Root output directory. Files are written under "
            "<out-root>/<expver>/<date_start>_<date_end>/"
        ),
    )

    parser.add_argument(
        "--req-root",
        type=Path,
        default=DEFAULT_REQ_ROOT,
        help=(
            "Root directory for generated MARS request files. Requests are "
            "written under <req-root>/<expver>/<date_start>_<date_end>/"
        ),
    )

    parser.add_argument(
        "--tmpdir",
        type=Path,
        default=DEFAULT_TMPDIR,
        help="Temporary directory used by MARS.",
    )

    parser.add_argument(
        "--params",
        default=DEFAULT_PARAMS,
        help=(
            "MARS parameter list. Default is 235270, river discharge. "
            "Use slash-separated values for multiple parameters."
        ),
    )

    parser.add_argument(
        "--mars-class",
        default=DEFAULT_CLASS,
        help=f"MARS class. Default: {DEFAULT_CLASS}",
    )

    parser.add_argument(
        "--stream",
        default=DEFAULT_STREAM,
        help=f"MARS stream. Default: {DEFAULT_STREAM}",
    )

    parser.add_argument(
        "--type",
        default=DEFAULT_TYPE,
        help=f"MARS type. Default: {DEFAULT_TYPE}",
    )

    parser.add_argument(
        "--levtype",
        default=DEFAULT_LEVTYPE,
        help=f"MARS levtype. Default: {DEFAULT_LEVTYPE}",
    )

    parser.add_argument(
        "--time",
        default=DEFAULT_TIME,
        help=f"MARS forecast base time. Default: {DEFAULT_TIME}",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing non-empty GRIB files.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write request files and print commands, but do not run MARS.",
    )

    return parser.parse_args()


# ------------------------------------------------------------
# Date and step utilities
# ------------------------------------------------------------
def parse_date(value):
    """
    Parse YYYY-MM-DD or YYYYMMDD into a pandas Timestamp.
    """
    return pd.Timestamp(value)


def month_step_range(month, date_start, date_end):
    """
    Return the archive date and required MARS step range for one month.

    The archive convention is:

      date = first day of the month
      step = 24, 48, 72, ..., N*24

    where each step represents the valid date:

      valid date = archive date + step hours

    but because step=24 corresponds to the first valid day of the month,
    the day offset is:

      step_hours = (valid_date - first_day_of_month + 1 day) * 24

    Examples
    --------
    January 2018 full month:
      archive date = 20180101
      steps = 24/to/744/by/24

    January 2018 from 2018-01-15:
      archive date = 20180101
      steps = 360/to/744/by/24
    """

    first_day = month.start_time.normalize()
    last_day = month.end_time.normalize()

    valid_start = max(first_day, date_start)
    valid_end = min(last_day, date_end)

    if valid_start > valid_end:
        return None

    start_day_index = (valid_start - first_day).days + 1
    end_day_index = (valid_end - first_day).days + 1

    start_step = start_day_index * 24
    end_step = end_day_index * 24

    if start_step == end_step:
        step_text = str(start_step)
    else:
        step_text = f"{start_step}/to/{end_step}/by/24"

    return {
        "archive_date": first_day,
        "valid_start": valid_start,
        "valid_end": valid_end,
        "start_step": start_step,
        "end_step": end_step,
        "step_text": step_text,
    }


def safe_date_label(ts):
    """
    Return compact YYYYMMDD string for directory/file naming.
    """
    return pd.Timestamp(ts).strftime("%Y%m%d")


# ------------------------------------------------------------
# MARS environment
# ------------------------------------------------------------
def build_mars_env(tmpdir):
    """
    Prepare the environment used by the MARS command.
    """
    env = dict(os.environ)

    env["TMPDIR"] = str(tmpdir)
    env["TEMP"] = str(tmpdir)
    env["TMP"] = str(tmpdir)

    # Increase MARS read buffer size for large retrievals.
    env["MARS_READANY_BUFFER_SIZE"] = "2147483648"

    # The script explicitly controls monthly splitting.
    env["MARS_AUTO_SPLIT_BY_DATES"] = "0"

    return env


# ------------------------------------------------------------
# Request builder
# ------------------------------------------------------------
def build_request_text(
    expver,
    mars_class,
    stream,
    mars_type,
    levtype,
    archive_date,
    time,
    step_text,
    params,
    target,
):
    """
    Build one MARS request for one experiment and one monthly archive chunk.
    """
    return f"""
retrieve,
  class={mars_class},
  expver={expver},
  stream={stream},
  type={mars_type},
  levtype={levtype},
  date={archive_date.strftime('%Y%m%d')},
  time={time},
  step={step_text},
  param={params},
  target="{target}"
"""


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    args = parse_args()

    date_start = parse_date(args.date_start).normalize()
    date_end = parse_date(args.date_end).normalize()

    if date_end < date_start:
        raise ValueError(
            f"date_end must be >= date_start, got {date_start} to {date_end}"
        )

    date_label = f"{safe_date_label(date_start)}_{safe_date_label(date_end)}"

    args.out_root.mkdir(parents=True, exist_ok=True)
    args.req_root.mkdir(parents=True, exist_ok=True)
    args.tmpdir.mkdir(parents=True, exist_ok=True)

    env = build_mars_env(args.tmpdir)

    months = pd.period_range(start=date_start, end=date_end, freq="M")

    print("")
    print("============================================================")
    print("IFS river discharge MARS retrieval")
    print("============================================================")
    print(f"Experiments : {', '.join(args.expver)}")
    print(f"Date range  : {date_start:%Y-%m-%d} to {date_end:%Y-%m-%d}")
    print(f"Months      : {len(months)}")
    print(f"Parameters  : {args.params}")
    print(f"Output root : {args.out_root}")
    print(f"Request root: {args.req_root}")
    print(f"Temporary   : {args.tmpdir}")

    if args.dry_run:
        print("Mode        : dry-run, MARS will not be executed")

    print("============================================================")
    print("")

    n_done = 0
    n_skipped = 0
    n_failed = 0

    for expver in args.expver:

        outdir = args.out_root / expver / date_label
        reqdir = args.req_root / expver / date_label

        outdir.mkdir(parents=True, exist_ok=True)
        reqdir.mkdir(parents=True, exist_ok=True)

        print("")
        print("------------------------------------------------------------")
        print(f"Experiment: {expver}")
        print(f"Output    : {outdir}")
        print(f"Requests  : {reqdir}")
        print("------------------------------------------------------------")

        for im, month in enumerate(months, start=1):

            info = month_step_range(month, date_start, date_end)

            if info is None:
                continue

            yyyymm = month.strftime("%Y%m")

            out_grib = outdir / f"Globe_river_discharge_{expver}_{yyyymm}.grb"
            req_file = reqdir / f"mars_extract_river_discharge_{expver}_{yyyymm}.req"

            if (
                out_grib.exists()
                and out_grib.stat().st_size > 0
                and not args.overwrite
            ):
                print(
                    f"[{im}/{len(months)}] {expver} {yyyymm}: "
                    f"skipping existing file"
                )
                n_skipped += 1
                continue

            request_text = build_request_text(
                expver=expver,
                mars_class=args.mars_class,
                stream=args.stream,
                mars_type=args.type,
                levtype=args.levtype,
                archive_date=info["archive_date"],
                time=args.time,
                step_text=info["step_text"],
                params=args.params,
                target=out_grib,
            )

            req_file.write_text(request_text.strip() + "\n")

            print("")
            print(f"[{im}/{len(months)}] Retrieving {expver} {yyyymm}")
            print(f"Archive date : {info['archive_date']:%Y%m%d}")
            print(f"Valid range  : {info['valid_start']:%Y-%m-%d} to {info['valid_end']:%Y-%m-%d}")
            print(f"Steps        : {info['step_text']}")
            print(f"Request      : {req_file}")
            print(f"Output       : {out_grib}")

            if args.dry_run:
                print("Dry-run: MARS not executed")
                continue

            try:
                subprocess.run(
                    ["mars", str(req_file)],
                    env=env,
                    check=True,
                )
                n_done += 1

            except subprocess.CalledProcessError:
                n_failed += 1

                print(f"FAILED: {expver} {yyyymm}")

                if out_grib.exists():
                    failed_file = out_grib.with_suffix(".failed.grb")
                    out_grib.rename(failed_file)
                    print(f"Partial file renamed to: {failed_file}")

                raise

    print("")
    print("============================================================")
    print("Retrieval summary")
    print("============================================================")
    print(f"Completed : {n_done}")
    print(f"Skipped   : {n_skipped}")
    print(f"Failed    : {n_failed}")
    print("Done.")
    print("")


if __name__ == "__main__":
    main()
