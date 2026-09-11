#!/usr/bin/env python3
"""
prepare_public_bundle.py

Build a public-safe copy of a dashboard bundle (as produced by
03_prepare_sites_bundle.py) for open publication on gpbalsamo.github.io.

Observed discharge is only redistributed for stations sourced from GRDC
(Global Runoff Data Centre) via the open-access GRDC-Caravan extension of
the Caravan dataset. Other stations (national/EFAS "Hydro" network, or
non-GRDC Caravan sub-datasets such as CAMELS/HYSETS/LamaH, which carry
their own separate licences) keep their model discharge but have their
"obs" payload replaced with a lightweight {"restricted": true} marker, so
the dashboard shows model results and a "not available in this demo" note
instead of the observed hydrograph.

GRDC stations are identified from the station metadata CSV (the same file
passed to 01_extract_hydrographs.py via --station-file): rows where
Source == "Caravan" and Provid starts with "GRDC_" (case-insensitive).
"""

from pathlib import Path
import argparse
import csv
import json
import shutil
import time


def parse_args():
    p = argparse.ArgumentParser(
        description="Strip non-GRDC observations from a dashboard bundle for public release."
    )
    p.add_argument(
        "--workflow-dir",
        type=Path,
        default=Path("."),
        help="Workflow directory containing the source bundle.",
    )
    p.add_argument(
        "--source-bundle-dirname",
        default="site_bundle",
        help="Bundle directory to read from (built by 03_prepare_sites_bundle.py). Default: site_bundle",
    )
    p.add_argument(
        "--public-bundle-dirname",
        default="public_site_bundle",
        help="Output bundle directory to (re)write. Default: public_site_bundle",
    )
    p.add_argument(
        "--station-file",
        type=Path,
        required=True,
        help="Station metadata CSV with Id/Source/Provid columns, e.g. allstations_v1.3.csv.",
    )
    return p.parse_args()


def load_grdc_station_ids(station_file: Path) -> set:
    grdc_ids = set()
    with station_file.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source = (row.get("Source") or "").strip().lower()
            provid = (row.get("Provid") or "").strip().lower()
            if source == "caravan" and provid.startswith("grdc_"):
                try:
                    grdc_ids.add(int(row["Id"]))
                except (KeyError, ValueError):
                    continue
    return grdc_ids


def strip_station_json(path: Path, grdc_ids: set) -> bool:
    """Rewrite one station JSON in place. Returns True if obs was restricted."""
    data = json.loads(path.read_text(encoding="utf-8"))

    station_id = data.get("source_station_id")
    is_grdc = station_id in grdc_ids

    if not is_grdc and data.get("obs"):
        data["obs"] = {"restricted": True}
        path.write_text(json.dumps(data), encoding="utf-8")
        return True

    return False


def main():
    t0 = time.time()
    args = parse_args()

    workflow_dir = args.workflow_dir.resolve()
    source_bundle = workflow_dir / args.source_bundle_dirname
    public_bundle = workflow_dir / args.public_bundle_dirname

    if not source_bundle.exists():
        raise FileNotFoundError(f"Source bundle not found: {source_bundle}")

    grdc_ids = load_grdc_station_ids(args.station_file.resolve())
    print(f"GRDC-open stations found in {args.station_file}: {len(grdc_ids)}")

    if public_bundle.exists():
        shutil.rmtree(public_bundle)
    shutil.copytree(source_bundle, public_bundle)

    station_files = sorted(public_bundle.glob("dashboard_data/*/*/stations/station_*.json"))
    if not station_files:
        raise RuntimeError(f"No station JSON files found under {public_bundle}/dashboard_data")

    restricted = 0
    for station_file in station_files:
        if strip_station_json(station_file, grdc_ids):
            restricted += 1

    print(f"Station files processed: {len(station_files)}")
    print(f"Observations restricted (non-GRDC): {restricted}")
    print(f"Observations kept (GRDC-open): {len(station_files) - restricted}")
    print(f"Public bundle written to: {public_bundle}")
    print(f"Done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
