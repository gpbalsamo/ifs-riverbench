#!/usr/bin/env python3
"""
Plot empirical cumulative distribution functions of RiverBench station scores
for multiple experiments, globally and by continent.

Metrics:
    - Kling-Gupta Efficiency (KGE)
    - Correlation coefficient (r)
    - Root Mean Square Error (RMSE)

Default input structure:

    /perm/pad/ifs-riverbench/Workflow/dashboard_data/<experiment>/
        20180101_20221231_15arcmin/global_station_metrics.csv

Default experiments:
    iyp3, j6fu, j6gq, j7xs

Regional selection uses station latitude and longitude.
Each regional bounding box is evaluated independently. Consequently, stations
inside overlapping bounding boxes may appear in more than one regional figure.
"""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------

DEFAULT_BASEDIR = Path(
    "/perm/pad/ifs-riverbench/Workflow/dashboard_data"
)

DEFAULT_PERIOD = "20180101_20221231_15arcmin"

DEFAULT_EXPERIMENTS = [
    "iyp3",
    "j6fu",
    "j6gq",
    "j7xs",
]

# Format:
# region_name: [minimum latitude, minimum longitude,
#               maximum latitude, maximum longitude]
REGIONS = {
    "Africa": [-35.0, -20.0, 38.0, 55.0],
    "Europe": [34.0, -25.0, 72.0, 45.0],
    "Asia": [-10.0, 25.0, 82.0, 180.0],
    "NorthAmerica": [5.0, -170.0, 83.0, -50.0],
    "SouthAmerica": [-60.0, -85.0, 15.0, -30.0],
    "Oceania": [-50.0, 110.0, 10.0, 180.0],
    "Globe": [-90.0, -180.0, 90.0, 180.0],
}

DEFAULT_REGIONS = [
    "Globe",
    "Africa",
    "Europe",
    "Asia",
    "NorthAmerica",
    "SouthAmerica",
    "Oceania",
]

METRIC_COLUMNS = {
    "kge": ["kge", "KGE"],
    "r": ["correlation", "Correlation"],
    "rmse": ["rmse", "RMSE_m3s"],
}

MATCHED_DAYS_COLUMNS = [
    "matched_days",
    "Matched_days",
]

STATION_ID_COLUMNS = [
    "station_id",
    "source_station_id",
    "station_index",
]

LATITUDE_COLUMNS = [
    "lat",
    "station_lat",
    "latitude",
    "Latitude",
]

LONGITUDE_COLUMNS = [
    "lon",
    "station_lon",
    "longitude",
    "Longitude",
]

METRIC_LABELS = {
    "kge": "Kling-Gupta efficiency (KGE)",
    "r": "Correlation coefficient (r)",
    "rmse": r"RMSE (m$^3$ s$^{-1}$)",
}


# ---------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Plot global and continental empirical CDFs of KGE, "
            "correlation and RMSE for IFS-RiverBench experiments."
        )
    )

    parser.add_argument(
        "--basedir",
        type=Path,
        default=DEFAULT_BASEDIR,
        help=f"Dashboard-data directory. Default: {DEFAULT_BASEDIR}",
    )

    parser.add_argument(
        "--period",
        default=DEFAULT_PERIOD,
        help=(
            "Period/resolution directory below each experiment. "
            f"Default: {DEFAULT_PERIOD}"
        ),
    )

    parser.add_argument(
        "--experiments",
        nargs="+",
        default=DEFAULT_EXPERIMENTS,
        help="Experiments to compare.",
    )

    parser.add_argument(
        "--regions",
        nargs="+",
        default=DEFAULT_REGIONS,
        choices=list(REGIONS),
        help="Regions for which figures will be generated.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("riverbench_score_cdfs"),
        help=(
            "Directory in which regional figures are written. "
            "Default: riverbench_score_cdfs"
        ),
    )

    parser.add_argument(
        "--output-prefix",
        default="riverbench_score_cdfs",
        help=(
            "Prefix used for output image filenames. "
            "Default: riverbench_score_cdfs"
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output image resolution. Default: 200",
    )

    parser.add_argument(
        "--min-matched-days",
        type=int,
        default=0,
        help=(
            "Retain stations with at least this number of matched days. "
            "Default: 0"
        ),
    )

    parser.add_argument(
        "--common-stations",
        action="store_true",
        help=(
            "Restrict all experiments to stations present in every "
            "experiment."
        ),
    )

    parser.add_argument(
        "--kge-xmin",
        type=float,
        default=-1.0,
        help=(
            "Lower displayed KGE limit. Values below the limit remain "
            "included in the CDF. Default: -1"
        ),
    )

    parser.add_argument(
        "--kge-xmax",
        type=float,
        default=1.0,
        help="Upper displayed KGE limit. Default: 1",
    )

    parser.add_argument(
        "--rmse-xmin",
        type=float,
        default=0.1,
        help="Lower limit for the logarithmic RMSE axis. Default: 0.1",
    )

    parser.add_argument(
        "--rmse-xmax",
        type=float,
        default=None,
        help="Optional upper displayed RMSE limit.",
    )

    parser.add_argument(
        "--linear-rmse",
        action="store_true",
        help="Use a linear rather than logarithmic RMSE axis.",
    )

    parser.add_argument(
        "--no-region-tables",
        action="store_true",
        help="Do not print metric tables separately for every region.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display each figure interactively after saving.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------

def find_first_column(dataframe, candidates, description, required=True):
    """
    Return the first candidate column present in a dataframe.
    """
    for column in candidates:
        if column in dataframe.columns:
            return column

    if required:
        raise KeyError(
            f"Could not find a column for {description}. "
            f"Tried: {candidates}. Available columns are: "
            f"{list(dataframe.columns)}"
        )

    return None


def normalise_longitudes(longitudes):
    """
    Convert longitudes to the -180 to 180 convention.
    """
    longitudes = np.asarray(longitudes, dtype=float)

    return ((longitudes + 180.0) % 360.0) - 180.0


def read_experiment_csv(
    csv_file,
    experiment,
    min_matched_days=0,
):
    """
    Read one experiment CSV and standardise the required columns.
    """
    if not csv_file.is_file():
        raise FileNotFoundError(
            f"CSV file not found for experiment {experiment}: {csv_file}"
        )

    dataframe = pd.read_csv(
        csv_file,
        low_memory=False,
    )

    station_id_column = find_first_column(
        dataframe,
        STATION_ID_COLUMNS,
        "station identifier",
    )

    latitude_column = find_first_column(
        dataframe,
        LATITUDE_COLUMNS,
        "station latitude",
    )

    longitude_column = find_first_column(
        dataframe,
        LONGITUDE_COLUMNS,
        "station longitude",
    )

    kge_column = find_first_column(
        dataframe,
        METRIC_COLUMNS["kge"],
        "KGE",
    )

    correlation_column = find_first_column(
        dataframe,
        METRIC_COLUMNS["r"],
        "correlation",
    )

    rmse_column = find_first_column(
        dataframe,
        METRIC_COLUMNS["rmse"],
        "RMSE",
    )

    matched_days_column = find_first_column(
        dataframe,
        MATCHED_DAYS_COLUMNS,
        "matched days",
        required=False,
    )

    output = pd.DataFrame(
        {
            "station_id": dataframe[station_id_column].astype(str),
            "lat": pd.to_numeric(
                dataframe[latitude_column],
                errors="coerce",
            ),
            "lon": pd.to_numeric(
                dataframe[longitude_column],
                errors="coerce",
            ),
            "kge": pd.to_numeric(
                dataframe[kge_column],
                errors="coerce",
            ),
            "r": pd.to_numeric(
                dataframe[correlation_column],
                errors="coerce",
            ),
            "rmse": pd.to_numeric(
                dataframe[rmse_column],
                errors="coerce",
            ),
        }
    )

    if matched_days_column is not None:
        output["matched_days"] = pd.to_numeric(
            dataframe[matched_days_column],
            errors="coerce",
        )
    else:
        output["matched_days"] = np.nan

    output["lon"] = normalise_longitudes(output["lon"])

    invalid_station_id = output["station_id"].isin(
        ["", "nan", "None"]
    )

    invalid_coordinates = (
        ~np.isfinite(output["lat"])
        | ~np.isfinite(output["lon"])
        | (output["lat"] < -90.0)
        | (output["lat"] > 90.0)
    )

    invalid_records = invalid_station_id | invalid_coordinates

    if invalid_records.any():
        print(
            f"[{experiment}] Removing "
            f"{int(invalid_records.sum()):,} records with invalid IDs "
            "or coordinates"
        )

        output = output.loc[~invalid_records].copy()

    duplicated = output["station_id"].duplicated(
        keep="first"
    )

    if duplicated.any():
        print(
            f"[{experiment}] Removing "
            f"{int(duplicated.sum()):,} duplicated station records"
        )

        output = output.loc[~duplicated].copy()

    if min_matched_days > 0:
        if matched_days_column is None:
            print(
                f"[{experiment}] Warning: matched-days column not found; "
                "the matched-days filter was not applied."
            )
        else:
            before = len(output)

            output = output.loc[
                output["matched_days"] >= min_matched_days
            ].copy()

            print(
                f"[{experiment}] Minimum matched-days filter: "
                f"{before:,} -> {len(output):,} stations"
            )

    return output


# ---------------------------------------------------------------------
# Common-station handling
# ---------------------------------------------------------------------

def restrict_to_common_stations(experiment_data):
    """
    Restrict all experiments to their common station IDs.
    """
    station_sets = [
        set(dataframe["station_id"])
        for dataframe in experiment_data.values()
    ]

    if not station_sets:
        raise ValueError("No experiment datasets were supplied.")

    common_station_ids = set.intersection(*station_sets)

    if not common_station_ids:
        raise ValueError(
            "No common station IDs were found across the experiments."
        )

    ordered_ids = sorted(common_station_ids)

    print(
        f"Stations present in every experiment: "
        f"{len(ordered_ids):,}"
    )

    restricted = {}

    for experiment, dataframe in experiment_data.items():
        selected = (
            dataframe
            .loc[dataframe["station_id"].isin(common_station_ids)]
            .set_index("station_id")
            .loc[ordered_ids]
            .reset_index()
        )

        restricted[experiment] = selected

        print(
            f"[{experiment}] Common-station records: "
            f"{len(selected):,}"
        )

    return restricted


def check_common_station_coordinates(experiment_data):
    """
    Check whether common station coordinates are consistent between files.
    """
    experiments = list(experiment_data)

    if len(experiments) < 2:
        return

    reference_experiment = experiments[0]

    reference = (
        experiment_data[reference_experiment]
        .set_index("station_id")[["lat", "lon"]]
        .sort_index()
    )

    for experiment in experiments[1:]:
        comparison = (
            experiment_data[experiment]
            .set_index("station_id")[["lat", "lon"]]
            .sort_index()
        )

        common_ids = reference.index.intersection(comparison.index)

        lat_difference = np.abs(
            reference.loc[common_ids, "lat"].to_numpy()
            - comparison.loc[common_ids, "lat"].to_numpy()
        )

        lon_difference = np.abs(
            reference.loc[common_ids, "lon"].to_numpy()
            - comparison.loc[common_ids, "lon"].to_numpy()
        )

        inconsistent = (
            (lat_difference > 1.0e-6)
            | (lon_difference > 1.0e-6)
        )

        if inconsistent.any():
            print(
                f"Warning: {int(inconsistent.sum()):,} common stations "
                f"have coordinate differences between "
                f"{reference_experiment} and {experiment}."
            )


# ---------------------------------------------------------------------
# Regional selection
# ---------------------------------------------------------------------

def region_mask(dataframe, region_name):
    """
    Return a boolean mask for one geographic bounding box.
    """
    if region_name not in REGIONS:
        raise KeyError(f"Unknown region: {region_name}")

    lat_min, lon_min, lat_max, lon_max = REGIONS[region_name]

    latitude = dataframe["lat"].to_numpy(dtype=float)
    longitude = dataframe["lon"].to_numpy(dtype=float)

    latitude_mask = (
        (latitude >= lat_min)
        & (latitude <= lat_max)
    )

    if lon_min <= lon_max:
        longitude_mask = (
            (longitude >= lon_min)
            & (longitude <= lon_max)
        )
    else:
        # Supports boxes crossing the dateline.
        longitude_mask = (
            (longitude >= lon_min)
            | (longitude <= lon_max)
        )

    return latitude_mask & longitude_mask


def select_region(experiment_data, region_name):
    """
    Select stations inside one region for every experiment.
    """
    regional_data = {}

    for experiment, dataframe in experiment_data.items():
        mask = region_mask(
            dataframe,
            region_name,
        )

        regional_data[experiment] = dataframe.loc[mask].copy()

    return regional_data


def print_region_station_counts(region_name, regional_data):
    """
    Print station counts for one region.
    """
    counts = {
        experiment: len(dataframe)
        for experiment, dataframe in regional_data.items()
    }

    count_text = ", ".join(
        f"{experiment}={count:,}"
        for experiment, count in counts.items()
    )

    print(f"[{region_name}] {count_text}")


# ---------------------------------------------------------------------
# Metric preparation
# ---------------------------------------------------------------------

def get_metric_values(dataframe, metric):
    """
    Return finite, physically valid values for one metric.
    """
    values = pd.to_numeric(
        dataframe[metric],
        errors="coerce",
    ).to_numpy(dtype=float)

    values = values[np.isfinite(values)]

    if metric == "r":
        values = values[
            (values >= -1.0)
            & (values <= 1.0)
        ]

    elif metric == "rmse":
        values = values[values >= 0.0]

    return values


def empirical_cdf(values):
    """
    Calculate an empirical cumulative distribution function.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values.sort()

    if values.size == 0:
        return np.array([]), np.array([])

    cumulative_fraction = (
        np.arange(
            1,
            values.size + 1,
            dtype=float,
        )
        / values.size
    )

    return values, cumulative_fraction


# ---------------------------------------------------------------------
# Printed statistics
# ---------------------------------------------------------------------

def percentage(condition):
    """
    Return the percentage of True values.
    """
    condition = np.asarray(condition, dtype=bool)

    if condition.size == 0:
        return np.nan

    return 100.0 * np.mean(condition)


def print_region_summary(region_name, regional_data):
    """
    Print metric summaries for one region.
    """
    print()
    print(f"Region: {region_name}")
    print("=" * 102)
    print(
        f"{'Experiment':<12}"
        f"{'Metric':<10}"
        f"{'N':>10}"
        f"{'Mean':>16}"
        f"{'Median':>16}"
        f"{'P10':>16}"
        f"{'P90':>16}"
    )
    print("-" * 102)

    for experiment, dataframe in regional_data.items():
        for metric in ["kge", "r", "rmse"]:
            values = get_metric_values(
                dataframe,
                metric,
            )

            if values.size:
                mean = np.mean(values)
                median = np.median(values)
                p10 = np.percentile(values, 10)
                p90 = np.percentile(values, 90)
            else:
                mean = median = p10 = p90 = np.nan

            print(
                f"{experiment:<12}"
                f"{metric:<10}"
                f"{values.size:>10,d}"
                f"{mean:>16.4f}"
                f"{median:>16.4f}"
                f"{p10:>16.4f}"
                f"{p90:>16.4f}"
            )

    print("=" * 102)

    print()
    print(
        f"{'Experiment':<12}"
        f"{'KGE > 0':>13}"
        f"{'KGE > 0.5':>15}"
        f"{'r > 0.5':>13}"
        f"{'r > 0.7':>13}"
        f"{'RMSE < 10':>15}"
        f"{'RMSE < 100':>16}"
    )
    print("-" * 97)

    for experiment, dataframe in regional_data.items():
        kge = get_metric_values(dataframe, "kge")
        correlation = get_metric_values(dataframe, "r")
        rmse = get_metric_values(dataframe, "rmse")

        print(
            f"{experiment:<12}"
            f"{percentage(kge > 0.0):>12.1f}%"
            f"{percentage(kge > 0.5):>14.1f}%"
            f"{percentage(correlation > 0.5):>12.1f}%"
            f"{percentage(correlation > 0.7):>12.1f}%"
            f"{percentage(rmse < 10.0):>14.1f}%"
            f"{percentage(rmse < 100.0):>15.1f}%"
        )

    print("=" * 97)


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_region_cdfs(
    regional_data,
    region_name,
    output_file,
    dpi=200,
    kge_xmin=-1.0,
    kge_xmax=1.0,
    rmse_xmin=0.1,
    rmse_xmax=None,
    logarithmic_rmse=True,
):
    """
    Plot KGE, correlation and RMSE CDFs for one region.
    """
    metrics = [
        "kge",
        "r",
        "rmse",
    ]

    figure, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(17, 5.8),
        constrained_layout=True,
    )

    for axis, metric in zip(axes, metrics):
        plotted_anything = False

        for experiment, dataframe in regional_data.items():
            values = get_metric_values(
                dataframe,
                metric,
            )

            if metric == "rmse" and logarithmic_rmse:
                values = values[values > 0.0]

            x, y = empirical_cdf(values)

            if x.size == 0:
                print(
                    f"Warning: no valid {metric} values for "
                    f"{experiment} in {region_name}"
                )
                continue

            axis.step(
                x,
                y,
                where="post",
                linewidth=2.0,
                label=f"{experiment} (n={x.size:,})",
            )

            plotted_anything = True

        axis.set_xlabel(
            METRIC_LABELS[metric],
            fontsize=11,
        )

        axis.set_ylabel(
            "Cumulative fraction",
            fontsize=11,
        )

        axis.set_ylim(0.0, 1.0)

        axis.set_yticks(
            np.arange(0.0, 1.01, 0.1)
        )

        axis.grid(
            visible=True,
            which="major",
            linestyle="--",
            linewidth=0.6,
            alpha=0.6,
        )

        if plotted_anything:
            axis.legend(
                loc="best",
                frameon=True,
                fontsize=9,
            )

        if metric == "kge":
            axis.set_xlim(
                kge_xmin,
                kge_xmax,
            )

            axis.axvline(
                0.0,
                linestyle=":",
                linewidth=1.2,
            )

            axis.axvline(
                0.5,
                linestyle=":",
                linewidth=1.2,
            )

            axis.set_title(
                "Higher KGE is better",
                fontsize=12,
            )

            axis.text(
                0.02,
                0.03,
                "Values outside the displayed range\nremain included",
                transform=axis.transAxes,
                fontsize=8,
                verticalalignment="bottom",
            )

        elif metric == "r":
            axis.set_xlim(-1.0, 1.0)

            for threshold in [0.0, 0.5, 0.7]:
                axis.axvline(
                    threshold,
                    linestyle=":",
                    linewidth=1.2,
                )

            axis.set_title(
                "Higher correlation is better",
                fontsize=12,
            )

        elif metric == "rmse":
            if logarithmic_rmse:
                axis.set_xscale("log")
                axis.set_xlim(left=rmse_xmin)

                axis.grid(
                    visible=True,
                    which="minor",
                    linestyle=":",
                    linewidth=0.4,
                    alpha=0.4,
                )
            else:
                axis.set_xlim(left=0.0)

            if rmse_xmax is not None:
                current_left, _ = axis.get_xlim()

                axis.set_xlim(
                    current_left,
                    rmse_xmax,
                )

            axis.set_title(
                "Lower RMSE is better",
                fontsize=12,
            )

    bounds = REGIONS[region_name]

    figure.suptitle(
        (
            f"IFS-RiverBench station-score distributions: {region_name}\n"
            f"lat {bounds[0]:g} to {bounds[2]:g}, "
            f"lon {bounds[1]:g} to {bounds[3]:g}"
        ),
        fontsize=15,
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_file,
        dpi=dpi,
        bbox_inches="tight",
    )

    print(f"Figure written to: {output_file}")

    return figure


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_arguments()

    experiment_data = {}

    for experiment in args.experiments:
        csv_file = (
            args.basedir
            / experiment
            / args.period
            / "global_station_metrics.csv"
        )

        print(f"Reading {experiment}: {csv_file}")

        dataframe = read_experiment_csv(
            csv_file=csv_file,
            experiment=experiment,
            min_matched_days=args.min_matched_days,
        )

        experiment_data[experiment] = dataframe

        print(
            f"[{experiment}] Loaded "
            f"{len(dataframe):,} station records"
        )

    if args.common_stations:
        experiment_data = restrict_to_common_stations(
            experiment_data
        )

        check_common_station_coordinates(
            experiment_data
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("Regional station counts")
    print("-" * 80)

    for region_name in args.regions:
        regional_data = select_region(
            experiment_data,
            region_name,
        )

        print_region_station_counts(
            region_name,
            regional_data,
        )

        if not args.no_region_tables:
            print_region_summary(
                region_name,
                regional_data,
            )

        output_file = (
            args.output_dir
            / f"{args.output_prefix}_{region_name}.png"
        )

        figure = plot_region_cdfs(
            regional_data=regional_data,
            region_name=region_name,
            output_file=output_file,
            dpi=args.dpi,
            kge_xmin=args.kge_xmin,
            kge_xmax=args.kge_xmax,
            rmse_xmin=args.rmse_xmin,
            rmse_xmax=args.rmse_xmax,
            logarithmic_rmse=not args.linear_rmse,
        )

        if args.show:
            plt.show()
        else:
            plt.close(figure)


if __name__ == "__main__":
    main()
