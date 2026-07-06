#!/usr/bin/env python3
"""
00_extract_rivers_mars.py

Retrieve archived IFS river discharge fields from MARS and save them as GRIB
files for the ifs-riverbench workflow.

This script supports two common archive layouts.

1. monthly_steps

   One forecast run per month:
     date = first day of month
     step = 24, 48, 72, ..., up to the last day of the month

   Example for January 2018:
     date = 20180101
     step = 24/to/744/by/24

   This is the original ifs-riverbench convention.

2. daily_steps

   One archive date per day, usually with one or more fixed forecast steps.

   Example for daily step-24 archives:
     date = 20160601/to/20160630
     step = 24

   Example for 12-hourly offline LDAS-style archives:
     date = 20160601/to/20160630
     step = 12/24

   The script still writes one GRIB file per month by default, but each file
   can contain many archive dates and one or more steps.

Parameter convention
--------------------
235270 = river discharge
235275 = flood fraction

Workflow position
-----------------
1. 00_extract_rivers_mars.py
   Retrieve GRIB files from MARS.

2. 01_extract_hydrographs.py
   Extract station hydrographs from the retrieved GRIB files.

3. 02_build_dashboard.py
   Build the interactive dashboard for model-observation benchmarking.

Examples
--------
Monthly-step archive, original convention:

python3 Workflow/00_extract_rivers_mars.py \
  --expver iyp3 \
  --date-start 20180101 \
  --date-end 20221231 \
  --archive-layout monthly_steps

Daily archive with step=24 every day:

python3 Workflow/00_extract_rivers_mars.py \
  --expver j6n9 izay \
  --date-start 20160601 \
  --date-end 20161231 \
  --archive-layout daily_steps \
  --step-text 24

Daily archive with 12-hourly steps:

python3 Workflow/00_extract_rivers_mars.py \
  --expver ldas_exp \
  --date-start 20160601 \
  --date-end 20161231 \
  --archive-layout daily_steps \
  --step-text 12/24
"""

from pathlib import Path
import argparse
import os
import subprocess

import pandas as pd


# ------------------------------------------------------------
# Defaults
# ------------------------------------------------------------
DEFAULT_OUT_ROOT = Path("/perm/pad/flood_cases/grib")
DEFAULT_REQ_ROOT = Path("/perm/pad/flood_cases/mars_requests")
DEFAULT_TMPDIR = Path("/perm/pad/flood_cases/tmp_mars")

# MARS parameter:
# 235270 = river discharge
# 235275 = flood fraction
DEFAULT_PARAMS = "235270"

DEFAULT_CLASS = "rd"
DEFAULT_STREAM = "oper"
DEFAULT_TYPE = "fc"
DEFAULT_LEVTYPE = "sfc"
DEFAULT_TIME = "0"

VALID_ARCHIVE_LAYOUTS = [
    "monthly_steps",
    "daily_steps",
]


# ------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve archived IFS river discharge fields from MARS for one "
            "or more experiments."
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
        "--archive-layout",
        choices=VALID_ARCHIVE_LAYOUTS,
        default="monthly_steps",
        help=(
            "Archive layout. "
            "monthly_steps: date is first day of month and step spans the month. "
            "daily_steps: date spans every day in the month and step is fixed "
            "by --step-text. Default: monthly_steps."
        ),
    )

    parser.add_argument(
        "--step-hours",
        type=int,
        default=24,
        help=(
            "Step interval in hours for monthly_steps layout. "
            "Default: 24."
        ),
    )

    parser.add_argument(
        "--step-text",
        default=None,
        help=(
            "Explicit MARS step syntax for daily_steps layout, for example "
            "'24', '12', '12/24', or '0/to/24/by/12'. "
            "If omitted, --step-hours is used."
        ),
    )

    parser.add_argument(
        "--daily-date-offset-days",
        type=int,
        default=0,
        help=(
            "Offset applied to MARS archive dates for daily_steps layout. "
            "Use -1 if the required valid day is stored as previous-day date "
            "with step=24 and you do not want to rely on a later valid-time shift. "
            "Default: 0."
        ),
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
        help=(
            "MARS forecast base time. Use slash-separated values if needed, "
            "for example 0/12. Default: 0."
        ),
    )

    parser.add_argument(
        "--expect-empty-ok",
        action="store_true",
        help=(
            "Do not fail the full workflow when MARS returns no fields for a "
            "request. The failed partial file is still renamed if present. "
            "Use with care."
        ),
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
    return pd.Timestamp(value).normalize()


def safe_date_label(ts):
    """
    Return compact YYYYMMDD string for directory/file naming.
    """
    return pd.Timestamp(ts).strftime("%Y%m%d")


def mars_date_range_text(start, end):
    """
    Return compact MARS date syntax.
    """
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()

    if start == end:
        return start.strftime("%Y%m%d")

    return f"{start:%Y%m%d}/to/{end:%Y%m%d}"


def normalise_step_text(step_text, step_hours):
    """
    Return MARS step syntax.
    """
    if step_text is not None and str(step_text).strip() != "":
        return str(step_text).strip()

    return str(int(step_hours))


def monthly_step_range(month, date_start, date_end, step_hours):
    """
    Return one monthly_steps retrieval chunk.

    Archive convention:

      date = first day of month
      step = step_hours, 2*step_hours, ..., N*step_hours

    where step=24 corresponds to the first valid day of the month in the
    original ifs-riverbench monthly archive convention.

    For daily river-discharge fields, use step_hours=24.
    """
    first_day = month.start_time.normalize()
    last_day = month.end_time.normalize()

    valid_start = max(first_day, date_start)
    valid_end = min(last_day, date_end)

    if valid_start > valid_end:
        return None

    start_day_index = (valid_start - first_day).days + 1
    end_day_index = (valid_end - first_day).days + 1

    start_step = start_day_index * int(step_hours)
    end_step = end_day_index * int(step_hours)

    if start_step == end_step:
        step_text = str(start_step)
    else:
        step_text = f"{start_step}/to/{end_step}/by/{int(step_hours)}"

    yyyymm = month.strftime("%Y%m")

    return {
        "chunk_label": yyyymm,
        "date_text": first_day.strftime("%Y%m%d"),
        "archive_date_label": first_day.strftime("%Y%m%d"),
        "valid_start": valid_start,
        "valid_end": valid_end,
        "step_text": step_text,
        "description": f"monthly_steps {yyyymm}",
        "expected_dates": 1,
    }


def daily_steps_range(month, date_start, date_end, step_text, date_offset_days):
    """
    Return one daily_steps retrieval chunk.

    Archive convention:

      date = YYYYMMDD/to/YYYYMMDD
      step = fixed MARS step syntax, e.g. 24 or 12/24

    The chunk is still monthly to keep output files reasonably sized.
    """
    first_day = month.start_time.normalize()
    last_day = month.end_time.normalize()

    valid_start = max(first_day, date_start)
    valid_end = min(last_day, date_end)

    if valid_start > valid_end:
        return None

    archive_start = valid_start + pd.to_timedelta(int(date_offset_days), unit="D")
    archive_end = valid_end + pd.to_timedelta(int(date_offset_days), unit="D")

    yyyymm = month.strftime("%Y%m")

    return {
        "chunk_label": yyyymm,
        "date_text": mars_date_range_text(archive_start, archive_end),
        "archive_date_label": (
            archive_start.strftime("%Y%m%d")
            if archive_start == archive_end
            else f"{archive_start:%Y%m%d}_{archive_end:%Y%m%d}"
        ),
        "valid_start": valid_start,
        "valid_end": valid_end,
        "archive_start": archive_start,
        "archive_end": archive_end,
        "step_text": step_text,
        "description": f"daily_steps {yyyymm}",
        "expected_dates": int((archive_end - archive_start).days) + 1,
    }


def build_retrieval_chunks(args, date_start, date_end):
    """
    Build retrieval chunks according to archive layout.

    The output is a list of dictionaries. Each dictionary maps to one MARS
    request and one output GRIB file.
    """
    months = pd.period_range(start=date_start, end=date_end, freq="M")
    chunks = []

    if args.archive_layout == "monthly_steps":
        for month in months:
            info = monthly_step_range(
                month=month,
                date_start=date_start,
                date_end=date_end,
                step_hours=args.step_hours,
            )

            if info is not None:
                chunks.append(info)

    elif args.archive_layout == "daily_steps":
        step_text = normalise_step_text(args.step_text, args.step_hours)

        for month in months:
            info = daily_steps_range(
                month=month,
                date_start=date_start,
                date_end=date_end,
                step_text=step_text,
                date_offset_days=args.daily_date_offset_days,
            )

            if info is not None:
                chunks.append(info)

    else:
        raise ValueError(f"Unsupported archive layout: {args.archive_layout}")

    return chunks


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

    # The script explicitly controls request splitting.
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
    date_text,
    time,
    step_text,
    params,
    target,
):
    """
    Build one MARS request for one archive chunk.
    """
    return f"""
retrieve,
  class={mars_class},
  expver={expver},
  stream={stream},
  type={mars_type},
  levtype={levtype},
  date={date_text},
  time={time},
  step={step_text},
  param={params},
  target="{target}"
"""


def remove_or_rename_partial(out_grib):
    """
    Rename a partial MARS output file after failure.
    """
    if not out_grib.exists():
        return None

    failed_file = out_grib.with_suffix(".failed.grb")

    if failed_file.exists():
        failed_file.unlink()

    out_grib.rename(failed_file)

    return failed_file


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    args = parse_args()

    date_start = parse_date(args.date_start)
    date_end = parse_date(args.date_end)

    if date_end < date_start:
        raise ValueError(
            f"date_end must be >= date_start, got {date_start} to {date_end}"
        )

    if args.step_hours <= 0:
        raise ValueError(f"--step-hours must be positive, got {args.step_hours}")

    date_label = f"{safe_date_label(date_start)}_{safe_date_label(date_end)}"

    args.out_root.mkdir(parents=True, exist_ok=True)
    args.req_root.mkdir(parents=True, exist_ok=True)
    args.tmpdir.mkdir(parents=True, exist_ok=True)

    env = build_mars_env(args.tmpdir)
    chunks = build_retrieval_chunks(args, date_start, date_end)

    print("")
    print("============================================================")
    print("IFS river discharge MARS retrieval")
    print("============================================================")
    print(f"Experiments    : {', '.join(args.expver)}")
    print(f"Date range     : {date_start:%Y-%m-%d} to {date_end:%Y-%m-%d}")
    print(f"Archive layout : {args.archive_layout}")
    print(f"Chunks         : {len(chunks)}")
    print(f"Parameters     : {args.params}")
    print(f"MARS class     : {args.mars_class}")
    print(f"Stream/type    : {args.stream}/{args.type}")
    print(f"Levtype        : {args.levtype}")
    print(f"Time           : {args.time}")
    print(f"Step hours     : {args.step_hours}")
    print(f"Step text      : {normalise_step_text(args.step_text, args.step_hours)}")
    print(f"Daily date off.: {args.daily_date_offset_days} days")
    print(f"Output root    : {args.out_root}")
    print(f"Request root   : {args.req_root}")
    print(f"Temporary      : {args.tmpdir}")

    if args.dry_run:
        print("Mode           : dry-run, MARS will not be executed")

    print("============================================================")
    print("")

    n_done = 0
    n_skipped = 0
    n_failed = 0
    n_ignored_failed = 0

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

        for ic, info in enumerate(chunks, start=1):

            chunk_label = info["chunk_label"]

            out_grib = (
                outdir
                / f"Globe_river_discharge_{expver}_{chunk_label}.grb"
            )

            req_file = (
                reqdir
                / f"mars_extract_river_discharge_{expver}_{chunk_label}.req"
            )

            if (
                out_grib.exists()
                and out_grib.stat().st_size > 0
                and not args.overwrite
            ):
                print(
                    f"[{ic}/{len(chunks)}] {expver} {chunk_label}: "
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
                date_text=info["date_text"],
                time=args.time,
                step_text=info["step_text"],
                params=args.params,
                target=out_grib,
            )

            req_file.write_text(request_text.strip() + "\n")

            print("")
            print(f"[{ic}/{len(chunks)}] Retrieving {expver} {chunk_label}")
            print(f"Layout      : {args.archive_layout}")
            print(f"Date        : {info['date_text']}")
            if "archive_start" in info:
                print(
                    f"Archive date: "
                    f"{info['archive_start']:%Y-%m-%d} to "
                    f"{info['archive_end']:%Y-%m-%d}"
                )
            else:
                print(f"Archive date: {info['archive_date_label']}")
            print(
                f"Valid range : "
                f"{info['valid_start']:%Y-%m-%d} to "
                f"{info['valid_end']:%Y-%m-%d}"
            )
            print(f"Steps       : {info['step_text']}")
            print(f"Request     : {req_file}")
            print(f"Output      : {out_grib}")

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

            except subprocess.CalledProcessError as exc:
                n_failed += 1

                print(f"FAILED: {expver} {chunk_label}")

                failed_file = remove_or_rename_partial(out_grib)

                if failed_file is not None:
                    print(f"Partial file renamed to: {failed_file}")

                if args.expect_empty_ok:
                    print(
                        "WARNING: MARS request failed but "
                        "--expect-empty-ok is set, continuing.",
                        flush=True,
                    )
                    n_ignored_failed += 1
                    continue

                raise exc

    print("")
    print("============================================================")
    print("Retrieval summary")
    print("============================================================")
    print(f"Completed       : {n_done}")
    print(f"Skipped         : {n_skipped}")
    print(f"Failed          : {n_failed}")
    print(f"Ignored failed  : {n_ignored_failed}")
    print("Done.")
    print("")


if __name__ == "__main__":
    main()
