#!/usr/bin/env python3
"""
Compare station-observation and GloFAS-grid KGE distributions for j7xs.

For each RiverBench station, the script:

1. Reads the gauge-based KGE from global_station_metrics.csv.
2. Extracts the nearest KGE value from kge_map_j7xs.nc.
3. Applies optional matched-days filtering.
4. Groups stations using the same regional bounding boxes used by plot_scores.py.
5. Produces one KGE CDF plot for the Globe and each continent.
6. Writes a CSV containing both KGE estimates at every station.

The two KGE quantities use different references:

    kge_station_observations
        j7xs benchmarked against river-gauge observations.

    kge_glofas_grid
        j7xs benchmarked against GloFAS at the nearest 0.25-degree grid point.
"""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------

DEFAULT_STATION_FILE = Path(
    "/perm/pad/ifs-riverbench/Workflow/dashboard_data/j7xs/"
    "20180101_20221231_15arcmin/global_station_metrics.csv"
)

DEFAULT_KGE_FILE = Path(
    "/perm/pad/benchmark_cmf_gp4hydro_vs_glofas_discharge_2018_2022/"
    "kge_map_j7xs.nc"
)

DEFAULT_OUTPUT_DIR = Path(
    "riverbench_glofas_kge_cdfs"
)

DEFAULT_EXTRACTED_CSV = Path(
    "j7xs_station_and_glofas_kge.csv"
)


# Bounds are:
# [minimum latitude, minimum longitude,
#  maximum latitude, maximum longitude]
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


# ---------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Plot regional CDFs of gauge-based and GloFAS-based "
            "KGE for j7xs."
        )
    )

    parser.add_argument(
        "--station-file",
        type=Path,
        default=DEFAULT_STATION_FILE,
        help=f"RiverBench station CSV. Default: {DEFAULT_STATION_FILE}",
    )

    parser.add_argument(
        "--kge-file",
        type=Path,
        default=DEFAULT_KGE_FILE,
        help=f"Gridded KGE NetCDF. Default: {DEFAULT_KGE_FILE}",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Figure output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )

    parser.add_argument(
        "--output-prefix",
        default="j7xs_kge_cdf",
        help="Output figure prefix. Default: j7xs_kge_cdf",
    )

    parser.add_argument(
        "--extracted-csv",
        type=Path,
        default=DEFAULT_EXTRACTED_CSV,
        help=(
            "CSV containing gauge and extracted GloFAS KGE values. "
            f"Default: {DEFAULT_EXTRACTED_CSV}"
        ),
    )

    parser.add_argument(
        "--regions",
        nargs="+",
        choices=list(REGIONS),
        default=DEFAULT_REGIONS,
        help="Regions for which figures are generated.",
    )

    parser.add_argument(
        "--min-matched-days",
        type=int,
        default=0,
        help=(
            "Minimum number of matched gauge-observation days. "
            "Default: 0"
        ),
    )

    parser.add_argument(
        "--kge-xmin",
        type=float,
        default=-1.0,
        help=(
            "Lower displayed KGE limit. Values below this remain "
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
        "--max-distance-km",
        type=float,
        default=None,
        help=(
            "Optional maximum station-to-grid-cell distance. "
            "More distant GloFAS-grid values are set to NaN."
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Figure resolution. Default: 200",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Longitude and distance utilities
# ---------------------------------------------------------------------

def normalize_lon_minus180_180(values):
    """
    Normalize longitude to [-180, 180).
    """
    values = np.asarray(values, dtype=float)

    return ((values + 180.0) % 360.0) - 180.0


def normalize_lon_0_360(values):
    """
    Normalize longitude to [0, 360).
    """
    values = np.asarray(values, dtype=float)

    return values % 360.0


def haversine_distance_km(lat1, lon1, lat2, lon2):
    """
    Calculate paired great-circle distances in kilometres.
    """
    earth_radius_km = 6371.0088

    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))

    delta_latitude = lat2 - lat1
    delta_longitude = lon2 - lon1

    delta_longitude = (
        (delta_longitude + np.pi)
        % (2.0 * np.pi)
        - np.pi
    )

    a = (
        np.sin(delta_latitude / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(delta_longitude / 2.0) ** 2
    )

    a = np.clip(a, 0.0, 1.0)

    return (
        2.0
        * earth_radius_km
        * np.arcsin(np.sqrt(a))
    )


# ---------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------

def find_first_column(
    dataframe,
    candidates,
    description,
    required=True,
):
    """
    Find the first candidate column present in a dataframe.
    """
    for column in candidates:
        if column in dataframe.columns:
            return column

    if required:
        raise KeyError(
            f"Could not find {description}. "
            f"Tried columns: {candidates}. "
            f"Available columns: {list(dataframe.columns)}"
        )

    return None


def read_station_data(
    station_file,
    min_matched_days=0,
):
    """
    Read station coordinates and gauge-based KGE.
    """
    if not station_file.is_file():
        raise FileNotFoundError(
            f"Station CSV not found: {station_file}"
        )

    dataframe = pd.read_csv(
        station_file,
        low_memory=False,
    )

    station_id_column = find_first_column(
        dataframe,
        [
            "station_id",
            "source_station_id",
            "station_index",
        ],
        "station identifier",
    )

    latitude_column = find_first_column(
        dataframe,
        [
            "lat",
            "station_lat",
            "latitude",
            "Latitude",
        ],
        "station latitude",
    )

    longitude_column = find_first_column(
        dataframe,
        [
            "lon",
            "station_lon",
            "longitude",
            "Longitude",
        ],
        "station longitude",
    )

    kge_column = find_first_column(
        dataframe,
        [
            "kge",
            "KGE",
        ],
        "station-observation KGE",
    )

    matched_days_column = find_first_column(
        dataframe,
        [
            "matched_days",
            "Matched_days",
        ],
        "matched-days column",
        required=False,
    )

    stations = pd.DataFrame(
        {
            "station_id": dataframe[
                station_id_column
            ].astype(str),
            "lat": pd.to_numeric(
                dataframe[latitude_column],
                errors="coerce",
            ),
            "lon": pd.to_numeric(
                dataframe[longitude_column],
                errors="coerce",
            ),
            "kge_station_observations": pd.to_numeric(
                dataframe[kge_column],
                errors="coerce",
            ),
        }
    )

    if matched_days_column is not None:
        stations["matched_days"] = pd.to_numeric(
            dataframe[matched_days_column],
            errors="coerce",
        )
    else:
        stations["matched_days"] = np.nan

    invalid_id = (
        stations["station_id"]
        .str.strip()
        .isin(["", "nan", "None", "<NA>"])
    )

    invalid_coordinates = (
        ~np.isfinite(stations["lat"])
        | ~np.isfinite(stations["lon"])
        | (stations["lat"] < -90.0)
        | (stations["lat"] > 90.0)
    )

    invalid = invalid_id | invalid_coordinates

    if invalid.any():
        print(
            f"Removing {int(invalid.sum()):,} records with "
            "invalid station identifiers or coordinates"
        )

        stations = stations.loc[
            ~invalid
        ].copy()

    stations["lon"] = normalize_lon_minus180_180(
        stations["lon"].to_numpy(dtype=float)
    )

    duplicated = stations["station_id"].duplicated(
        keep="first"
    )

    if duplicated.any():
        print(
            f"Removing {int(duplicated.sum()):,} duplicated "
            "station records"
        )

        stations = stations.loc[
            ~duplicated
        ].copy()

    if min_matched_days > 0:
        if matched_days_column is None:
            print(
                "Warning: no matched-days column was found; "
                "the filter was not applied."
            )
        else:
            before = len(stations)

            stations = stations.loc[
                stations["matched_days"]
                >= min_matched_days
            ].copy()

            print(
                f"Minimum matched-days filter: "
                f"{before:,} -> {len(stations):,} stations"
            )

    return stations


# ---------------------------------------------------------------------
# Grid handling
# ---------------------------------------------------------------------

def nearest_indices(
    sorted_coordinates,
    targets,
):
    """
    Find nearest indices in an ascending one-dimensional array.
    """
    coordinates = np.asarray(
        sorted_coordinates,
        dtype=float,
    )

    targets = np.asarray(
        targets,
        dtype=float,
    )

    if coordinates.ndim != 1:
        raise ValueError(
            "Grid coordinates must be one-dimensional."
        )

    if coordinates.size == 0:
        raise ValueError(
            "Grid coordinate array is empty."
        )

    if coordinates.size == 1:
        return np.zeros(
            targets.shape,
            dtype=int,
        )

    if np.any(np.diff(coordinates) < 0.0):
        raise ValueError(
            "Grid coordinates must be ascending."
        )

    insertion = np.searchsorted(
        coordinates,
        targets,
        side="left",
    )

    insertion = np.clip(
        insertion,
        1,
        coordinates.size - 1,
    )

    left_index = insertion - 1
    right_index = insertion

    left_distance = np.abs(
        targets - coordinates[left_index]
    )

    right_distance = np.abs(
        targets - coordinates[right_index]
    )

    return np.where(
        right_distance < left_distance,
        right_index,
        left_index,
    ).astype(int)


def nearest_periodic_longitude_indices(
    grid_longitudes,
    station_longitudes,
):
    """
    Find nearest longitude indices with periodic wrap-around.
    """
    grid_longitudes = np.asarray(
        grid_longitudes,
        dtype=float,
    )

    station_longitudes = np.asarray(
        station_longitudes,
        dtype=float,
    )

    extended_longitudes = np.concatenate(
        [
            grid_longitudes[-1:] - 360.0,
            grid_longitudes,
            grid_longitudes[:1] + 360.0,
        ]
    )

    extended_indices = np.concatenate(
        [
            np.array(
                [grid_longitudes.size - 1],
                dtype=int,
            ),
            np.arange(
                grid_longitudes.size,
                dtype=int,
            ),
            np.array([0], dtype=int),
        ]
    )

    selected_extended_indices = nearest_indices(
        extended_longitudes,
        station_longitudes,
    )

    return extended_indices[
        selected_extended_indices
    ]


def prepare_grid(
    dataset,
    lat_name="lat",
    lon_name="lon",
    kge_name="kge",
):
    """
    Load and sort the gridded KGE field.
    """
    for name in [lat_name, lon_name, kge_name]:
        if (
            name not in dataset.variables
            and name not in dataset.coords
        ):
            raise KeyError(
                f"NetCDF variable or coordinate not found: {name}"
            )

    kge_dataarray = dataset[kge_name]

    if lat_name not in kge_dataarray.dims:
        raise ValueError(
            f"{kge_name} does not contain dimension {lat_name}. "
            f"Dimensions: {kge_dataarray.dims}"
        )

    if lon_name not in kge_dataarray.dims:
        raise ValueError(
            f"{kge_name} does not contain dimension {lon_name}. "
            f"Dimensions: {kge_dataarray.dims}"
        )

    kge_dataarray = kge_dataarray.transpose(
        lat_name,
        lon_name,
    )

    latitudes = np.asarray(
        dataset[lat_name].values,
        dtype=float,
    )

    original_longitudes = np.asarray(
        dataset[lon_name].values,
        dtype=float,
    )

    latitude_order = np.argsort(latitudes)

    latitudes = latitudes[
        latitude_order
    ]

    kge_dataarray = kge_dataarray.isel(
        {
            lat_name: latitude_order,
        }
    )

    finite_longitudes = original_longitudes[
        np.isfinite(original_longitudes)
    ]

    if finite_longitudes.size == 0:
        raise ValueError(
            "Longitude coordinate contains no finite values."
        )

    grid_uses_0_360 = (
        np.nanmin(finite_longitudes) >= 0.0
        and np.nanmax(finite_longitudes) > 180.0
    )

    if grid_uses_0_360:
        longitudes = normalize_lon_0_360(
            original_longitudes
        )
    else:
        longitudes = normalize_lon_minus180_180(
            original_longitudes
        )

    longitude_order = np.argsort(longitudes)

    longitudes = longitudes[
        longitude_order
    ]

    kge_dataarray = kge_dataarray.isel(
        {
            lon_name: longitude_order,
        }
    )

    kge_array = np.asarray(
        kge_dataarray.values,
        dtype=float,
    )

    return (
        kge_array,
        latitudes,
        longitudes,
        grid_uses_0_360,
    )


def extract_glofas_kge(
    stations,
    kge_array,
    grid_latitudes,
    grid_longitudes,
    grid_uses_0_360,
):
    """
    Extract nearest gridded KGE values for all stations.
    """
    station_latitudes = stations[
        "lat"
    ].to_numpy(dtype=float)

    station_longitudes = stations[
        "lon"
    ].to_numpy(dtype=float)

    latitude_indices = nearest_indices(
        grid_latitudes,
        station_latitudes,
    )

    if grid_uses_0_360:
        station_grid_longitudes = (
            normalize_lon_0_360(
                station_longitudes
            )
        )
    else:
        station_grid_longitudes = (
            normalize_lon_minus180_180(
                station_longitudes
            )
        )

    longitude_indices = (
        nearest_periodic_longitude_indices(
            grid_longitudes,
            station_grid_longitudes,
        )
    )

    extracted_kge = kge_array[
        latitude_indices,
        longitude_indices,
    ]

    selected_latitudes = grid_latitudes[
        latitude_indices
    ]

    selected_longitudes = grid_longitudes[
        longitude_indices
    ]

    selected_longitudes_output = (
        normalize_lon_minus180_180(
            selected_longitudes
        )
    )

    distances = haversine_distance_km(
        station_latitudes,
        station_longitudes,
        selected_latitudes,
        selected_longitudes_output,
    )

    result = stations.copy()

    result["kge_glofas_grid"] = extracted_kge
    result["glofas_grid_lat"] = selected_latitudes
    result["glofas_grid_lon"] = (
        selected_longitudes_output
    )
    result["glofas_grid_row"] = latitude_indices
    result["glofas_grid_col"] = longitude_indices
    result[
        "distance_station_to_glofas_grid_km"
    ] = distances

    return result


# ---------------------------------------------------------------------
# Regional selection and CDF
# ---------------------------------------------------------------------

def region_mask(
    dataframe,
    region_name,
):
    """
    Select station coordinates inside a regional bounding box.
    """
    lat_min, lon_min, lat_max, lon_max = REGIONS[
        region_name
    ]

    latitude = dataframe[
        "lat"
    ].to_numpy(dtype=float)

    longitude = normalize_lon_minus180_180(
        dataframe["lon"].to_numpy(dtype=float)
    )

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
        longitude_mask = (
            (longitude >= lon_min)
            | (longitude <= lon_max)
        )

    return latitude_mask & longitude_mask


def empirical_cdf(values):
    """
    Calculate a standard empirical cumulative distribution function.
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


def percentage(condition):
    """
    Calculate the percentage of True values.
    """
    condition = np.asarray(condition, dtype=bool)

    if condition.size == 0:
        return np.nan

    return 100.0 * np.mean(condition)


# ---------------------------------------------------------------------
# Regional statistics
# ---------------------------------------------------------------------

def print_region_statistics(
    region_name,
    regional_data,
):
    """
    Print statistics equivalent to the earlier KGE CDF workflow.
    """
    station_kge = regional_data[
        "kge_station_observations"
    ].to_numpy(dtype=float)

    glofas_kge = regional_data[
        "kge_glofas_grid"
    ].to_numpy(dtype=float)

    station_kge = station_kge[
        np.isfinite(station_kge)
    ]

    glofas_kge = glofas_kge[
        np.isfinite(glofas_kge)
    ]

    print()
    print(f"Region: {region_name}")
    print("=" * 102)
    print(
        f"{'Reference':<28}"
        f"{'N':>10}"
        f"{'Mean':>16}"
        f"{'Median':>16}"
        f"{'P10':>16}"
        f"{'P90':>16}"
    )
    print("-" * 102)

    distributions = {
        "Gauge observations": station_kge,
        "GloFAS grid": glofas_kge,
    }

    for label, values in distributions.items():
        if values.size:
            mean = np.mean(values)
            median = np.median(values)
            p10 = np.percentile(values, 10)
            p90 = np.percentile(values, 90)
        else:
            mean = median = p10 = p90 = np.nan

        print(
            f"{label:<28}"
            f"{values.size:>10,d}"
            f"{mean:>16.4f}"
            f"{median:>16.4f}"
            f"{p10:>16.4f}"
            f"{p90:>16.4f}"
        )

    print("=" * 102)

    print(
        f"{'Reference':<28}"
        f"{'KGE > 0':>16}"
        f"{'KGE > 0.5':>18}"
        f"{'KGE < -1':>18}"
        f"{'KGE < -10':>18}"
    )
    print("-" * 98)

    for label, values in distributions.items():
        print(
            f"{label:<28}"
            f"{percentage(values > 0.0):>15.1f}%"
            f"{percentage(values > 0.5):>17.1f}%"
            f"{percentage(values < -1.0):>17.1f}%"
            f"{percentage(values < -10.0):>17.1f}%"
        )

    print("=" * 98)


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_region_cdf(
    regional_data,
    region_name,
    output_file,
    kge_xmin=-1.0,
    kge_xmax=1.0,
    dpi=200,
):
    """
    Plot gauge-based and GloFAS-based KGE CDFs for one region.
    """
    station_kge = regional_data[
        "kge_station_observations"
    ].to_numpy(dtype=float)

    glofas_kge = regional_data[
        "kge_glofas_grid"
    ].to_numpy(dtype=float)

    distributions = {
        "j7xs vs gauge observations": station_kge,
        "j7xs vs GloFAS": glofas_kge,
    }

    figure, axis = plt.subplots(
        figsize=(8.5, 6.2),
        constrained_layout=True,
    )

    for label, values in distributions.items():
        x, y = empirical_cdf(values)

        if x.size == 0:
            print(
                f"Warning: no valid values for "
                f"{label} in {region_name}"
            )
            continue

        axis.step(
            x,
            y,
            where="post",
            linewidth=2.2,
            label=f"{label} (n={x.size:,})",
        )

    axis.set_xlim(
        kge_xmin,
        kge_xmax,
    )

    axis.set_ylim(
        0.0,
        1.0,
    )

    axis.set_yticks(
        np.arange(0.0, 1.01, 0.1)
    )

    axis.set_xlabel(
        "Kling-Gupta efficiency (KGE)",
        fontsize=12,
    )

    axis.set_ylabel(
        "Cumulative fraction",
        fontsize=12,
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

    axis.grid(
        visible=True,
        which="major",
        linestyle="--",
        linewidth=0.6,
        alpha=0.6,
    )

    axis.legend(
        loc="best",
        frameon=True,
        fontsize=10,
    )

    bounds = REGIONS[region_name]

    axis.set_title(
        (
            f"j7xs KGE cumulative distributions: {region_name}\n"
            f"lat {bounds[0]:g} to {bounds[2]:g}, "
            f"lon {bounds[1]:g} to {bounds[3]:g}"
        ),
        fontsize=14,
    )

    axis.text(
        0.02,
        0.03,
        (
            "Values outside the displayed range remain "
            "included in the CDF"
        ),
        transform=axis.transAxes,
        fontsize=8,
        verticalalignment="bottom",
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


def plot_all_regions_panel(
    extracted_data,
    regions,
    output_file,
    kge_xmin=-1.0,
    kge_xmax=1.0,
    dpi=200,
):
    """
    Create one multipanel summary figure containing all regional KGE CDFs.
    """
    number_of_regions = len(regions)

    number_of_columns = 2
    number_of_rows = int(
        np.ceil(number_of_regions / number_of_columns)
    )

    figure, axes = plt.subplots(
        nrows=number_of_rows,
        ncols=number_of_columns,
        figsize=(15, 4.8 * number_of_rows),
        constrained_layout=True,
        squeeze=False,
    )

    axes = axes.ravel()

    for axis, region_name in zip(axes, regions):
        regional_data = extracted_data.loc[
            region_mask(
                extracted_data,
                region_name,
            )
        ]

        distributions = {
            "vs gauges": regional_data[
                "kge_station_observations"
            ].to_numpy(dtype=float),
            "vs GloFAS": regional_data[
                "kge_glofas_grid"
            ].to_numpy(dtype=float),
        }

        for label, values in distributions.items():
            x, y = empirical_cdf(values)

            if x.size == 0:
                continue

            axis.step(
                x,
                y,
                where="post",
                linewidth=2.0,
                label=f"{label} (n={x.size:,})",
            )

        axis.set_xlim(
            kge_xmin,
            kge_xmax,
        )

        axis.set_ylim(
            0.0,
            1.0,
        )

        axis.axvline(
            0.0,
            linestyle=":",
            linewidth=1.0,
        )

        axis.axvline(
            0.5,
            linestyle=":",
            linewidth=1.0,
        )

        axis.grid(
            visible=True,
            linestyle="--",
            linewidth=0.5,
            alpha=0.6,
        )

        axis.set_xlabel("KGE")
        axis.set_ylabel("Cumulative fraction")
        axis.set_title(region_name)
        axis.legend(
            loc="best",
            fontsize=9,
        )

    for unused_axis in axes[number_of_regions:]:
        unused_axis.set_visible(False)

    figure.suptitle(
        "j7xs KGE distributions: gauges versus GloFAS",
        fontsize=16,
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

    print(f"Combined figure written to: {output_file}")

    return figure


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_arguments()

    print(
        f"Reading station file: "
        f"{args.station_file}"
    )

    stations = read_station_data(
        station_file=args.station_file,
        min_matched_days=args.min_matched_days,
    )

    print(
        f"Stations retained: "
        f"{len(stations):,}"
    )

    if not args.kge_file.is_file():
        raise FileNotFoundError(
            f"KGE NetCDF not found: {args.kge_file}"
        )

    print(
        f"Reading KGE grid: "
        f"{args.kge_file}"
    )

    with xr.open_dataset(
        args.kge_file
    ) as dataset:
        (
            kge_array,
            grid_latitudes,
            grid_longitudes,
            grid_uses_0_360,
        ) = prepare_grid(dataset)

    print(
        f"KGE grid: "
        f"{grid_latitudes.size} lat x "
        f"{grid_longitudes.size} lon"
    )

    extracted_data = extract_glofas_kge(
        stations=stations,
        kge_array=kge_array,
        grid_latitudes=grid_latitudes,
        grid_longitudes=grid_longitudes,
        grid_uses_0_360=grid_uses_0_360,
    )

    if args.max_distance_km is not None:
        too_far = (
            extracted_data[
                "distance_station_to_glofas_grid_km"
            ]
            > args.max_distance_km
        )

        print(
            f"Stations farther than "
            f"{args.max_distance_km:g} km: "
            f"{int(too_far.sum()):,}"
        )

        extracted_data.loc[
            too_far,
            "kge_glofas_grid",
        ] = np.nan

    finite_glofas = np.isfinite(
        extracted_data["kge_glofas_grid"]
    )

    print(
        f"Finite GloFAS KGE values: "
        f"{int(finite_glofas.sum()):,} / "
        f"{len(extracted_data):,}"
    )

    distance = extracted_data[
        "distance_station_to_glofas_grid_km"
    ]

    print(
        f"Median station-to-grid distance: "
        f"{distance.median():.3f} km"
    )

    print(
        f"Maximum station-to-grid distance: "
        f"{distance.max():.3f} km"
    )

    args.extracted_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    extracted_data.to_csv(
        args.extracted_csv,
        index=False,
    )

    print(
        f"Extracted station CSV written to: "
        f"{args.extracted_csv}"
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for region_name in args.regions:
        regional_data = extracted_data.loc[
            region_mask(
                extracted_data,
                region_name,
            )
        ].copy()

        print(
            f"[{region_name}] Stations: "
            f"{len(regional_data):,}"
        )

        print_region_statistics(
            region_name,
            regional_data,
        )

        output_file = (
            args.output_dir
            / (
                f"{args.output_prefix}_"
                f"{region_name}.png"
            )
        )

        figure = plot_region_cdf(
            regional_data=regional_data,
            region_name=region_name,
            output_file=output_file,
            kge_xmin=args.kge_xmin,
            kge_xmax=args.kge_xmax,
            dpi=args.dpi,
        )

        if args.show:
            plt.show()
        else:
            plt.close(figure)

    combined_output = (
        args.output_dir
        / (
            f"{args.output_prefix}_"
            "all_regions.png"
        )
    )

    combined_figure = plot_all_regions_panel(
        extracted_data=extracted_data,
        regions=args.regions,
        output_file=combined_output,
        kge_xmin=args.kge_xmin,
        kge_xmax=args.kge_xmax,
        dpi=args.dpi,
    )

    if args.show:
        plt.show()
    else:
        plt.close(combined_figure)


if __name__ == "__main__":
    main()
