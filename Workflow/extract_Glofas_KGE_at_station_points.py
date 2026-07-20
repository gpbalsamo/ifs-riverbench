#!/usr/bin/env python3
"""
Extract gridded GloFAS-benchmark KGE values at RiverBench station locations.

The script:

1. Reads station coordinates from a RiverBench global_station_metrics.csv file.
2. Reads a regular latitude-longitude KGE NetCDF field.
3. Selects stations globally or within a specified regional bounding box.
4. Extracts the nearest grid-cell KGE value for each station.
5. Calculates the station-to-grid-cell distance.
6. Writes the original station information plus the extracted KGE to CSV.

Default station file:
    /perm/pad/ifs-riverbench/Workflow/dashboard_data/j7xs/
        20180101_20221231_15arcmin/global_station_metrics.csv

Default gridded KGE file:
    /perm/pad/benchmark_cmf_gp4hydro_vs_glofas_discharge_2018_2022/
        kge_map_j7xs.nc
"""

from pathlib import Path
import argparse

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

DEFAULT_OUTPUT = Path(
    "station_kge_j7xs_glofas_grid.csv"
)


# Bounding boxes:
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


# ---------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Extract nearest gridded GloFAS-benchmark KGE values "
            "at RiverBench station coordinates."
        )
    )

    parser.add_argument(
        "--station-file",
        type=Path,
        default=DEFAULT_STATION_FILE,
        help=(
            "RiverBench station metrics CSV. "
            f"Default: {DEFAULT_STATION_FILE}"
        ),
    )

    parser.add_argument(
        "--kge-file",
        type=Path,
        default=DEFAULT_KGE_FILE,
        help=(
            "NetCDF file containing lat, lon and kge. "
            f"Default: {DEFAULT_KGE_FILE}"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Output CSV filename. "
            f"Default: {DEFAULT_OUTPUT}"
        ),
    )

    parser.add_argument(
        "--region",
        choices=list(REGIONS),
        default="Globe",
        help="Regional bounding-box selection. Default: Globe",
    )

    parser.add_argument(
        "--max-distance-km",
        type=float,
        default=None,
        help=(
            "Optional maximum permitted station-to-grid-cell distance. "
            "Extracted KGE values beyond this distance are set to NaN."
        ),
    )

    parser.add_argument(
        "--station-id-column",
        default="station_id",
        help="Station identifier column. Default: station_id",
    )

    parser.add_argument(
        "--station-lat-column",
        default="lat",
        help="Station latitude column. Default: lat",
    )

    parser.add_argument(
        "--station-lon-column",
        default="lon",
        help="Station longitude column. Default: lon",
    )

    parser.add_argument(
        "--grid-lat-name",
        default="lat",
        help="NetCDF latitude coordinate name. Default: lat",
    )

    parser.add_argument(
        "--grid-lon-name",
        default="lon",
        help="NetCDF longitude coordinate name. Default: lon",
    )

    parser.add_argument(
        "--grid-kge-name",
        default="kge",
        help="NetCDF KGE variable name. Default: kge",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Longitude handling
# ---------------------------------------------------------------------

def normalize_lon_minus180_180(values):
    """
    Convert longitude values to the interval [-180, 180).
    """
    values = np.asarray(values, dtype=float)

    return ((values + 180.0) % 360.0) - 180.0


def normalize_lon_0_360(values):
    """
    Convert longitude values to the interval [0, 360).
    """
    values = np.asarray(values, dtype=float)

    return values % 360.0


# ---------------------------------------------------------------------
# Geographic utilities
# ---------------------------------------------------------------------

def haversine_distance_km(lat1, lon1, lat2, lon2):
    """
    Calculate great-circle distance between paired coordinates.

    All arguments may be scalars or equally shaped arrays.
    """
    earth_radius_km = 6371.0088

    lat1 = np.radians(
        np.asarray(lat1, dtype=float)
    )

    lon1 = np.radians(
        np.asarray(lon1, dtype=float)
    )

    lat2 = np.radians(
        np.asarray(lat2, dtype=float)
    )

    lon2 = np.radians(
        np.asarray(lon2, dtype=float)
    )

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    # Wrap longitude difference to [-pi, pi].
    delta_lon = (
        delta_lon + np.pi
    ) % (2.0 * np.pi) - np.pi

    a = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(delta_lon / 2.0) ** 2
    )

    a = np.clip(
        a,
        0.0,
        1.0,
    )

    return (
        2.0
        * earth_radius_km
        * np.arcsin(np.sqrt(a))
    )


def region_mask(dataframe, region_name):
    """
    Return a boolean mask for one regional bounding box.
    """
    if region_name not in REGIONS:
        raise KeyError(
            f"Unknown region: {region_name}"
        )

    lat_min, lon_min, lat_max, lon_max = REGIONS[
        region_name
    ]

    latitude = dataframe["station_lat"].to_numpy(
        dtype=float
    )

    longitude = normalize_lon_minus180_180(
        dataframe["station_lon"].to_numpy(
            dtype=float
        )
    )

    latitude_ok = (
        (latitude >= lat_min)
        & (latitude <= lat_max)
    )

    if lon_min <= lon_max:
        longitude_ok = (
            (longitude >= lon_min)
            & (longitude <= lon_max)
        )
    else:
        # Support bounding boxes crossing the dateline.
        longitude_ok = (
            (longitude >= lon_min)
            | (longitude <= lon_max)
        )

    return latitude_ok & longitude_ok


# ---------------------------------------------------------------------
# Grid utilities
# ---------------------------------------------------------------------

def nearest_indices(sorted_coordinates, targets):
    """
    Return indices of nearest values in an ascending 1D coordinate array.

    Parameters
    ----------
    sorted_coordinates : array-like
        Strictly ascending or non-decreasing one-dimensional coordinates.

    targets : array-like
        Coordinates to locate.

    Returns
    -------
    ndarray
        Integer nearest-neighbour indices.
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
            "nearest_indices requires ascending coordinates."
        )

    insertion_indices = np.searchsorted(
        coordinates,
        targets,
        side="left",
    )

    insertion_indices = np.clip(
        insertion_indices,
        1,
        coordinates.size - 1,
    )

    left_indices = insertion_indices - 1
    right_indices = insertion_indices

    left_distance = np.abs(
        targets - coordinates[left_indices]
    )

    right_distance = np.abs(
        targets - coordinates[right_indices]
    )

    use_right = right_distance < left_distance

    return np.where(
        use_right,
        right_indices,
        left_indices,
    ).astype(int)


def nearest_periodic_longitude_indices(
    grid_longitudes,
    station_longitudes,
):
    """
    Find nearest longitude indices while accounting for periodicity.

    Both arrays must use the same longitude convention.
    """
    grid_longitudes = np.asarray(
        grid_longitudes,
        dtype=float,
    )

    station_longitudes = np.asarray(
        station_longitudes,
        dtype=float,
    )

    if grid_longitudes.ndim != 1:
        raise ValueError(
            "Longitude coordinates must be one-dimensional."
        )

    if grid_longitudes.size == 0:
        raise ValueError(
            "Longitude coordinate array is empty."
        )

    if grid_longitudes.size == 1:
        return np.zeros(
            station_longitudes.shape,
            dtype=int,
        )

    # Extend the longitude array on both sides to handle the seam.
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
            np.array(
                [0],
                dtype=int,
            ),
        ]
    )

    nearest_extended = nearest_indices(
        extended_longitudes,
        station_longitudes,
    )

    return extended_indices[
        nearest_extended
    ]


def prepare_grid(
    dataset,
    lat_name="lat",
    lon_name="lon",
    kge_name="kge",
):
    """
    Prepare the KGE field and coordinates for nearest-neighbour extraction.

    Returns
    -------
    kge_array : ndarray
        KGE field with dimensions latitude x longitude.

    grid_latitudes : ndarray
        Ascending latitude coordinates.

    grid_longitudes : ndarray
        Ascending normalized longitude coordinates.

    grid_uses_0_360 : bool
        True when the prepared grid uses [0, 360).
    """
    required_names = {
        lat_name,
        lon_name,
        kge_name,
    }

    available_names = (
        set(dataset.variables)
        | set(dataset.coords)
    )

    missing_names = (
        required_names - available_names
    )

    if missing_names:
        raise KeyError(
            "Missing required NetCDF variables or coordinates: "
            f"{sorted(missing_names)}"
        )

    kge_dataarray = dataset[kge_name]

    if lat_name not in kge_dataarray.dims:
        raise ValueError(
            f"KGE variable '{kge_name}' does not contain "
            f"latitude dimension '{lat_name}'. "
            f"Dimensions are {kge_dataarray.dims}"
        )

    if lon_name not in kge_dataarray.dims:
        raise ValueError(
            f"KGE variable '{kge_name}' does not contain "
            f"longitude dimension '{lon_name}'. "
            f"Dimensions are {kge_dataarray.dims}"
        )

    kge_dataarray = kge_dataarray.transpose(
        lat_name,
        lon_name,
    )

    # xarray.DataArray.to_numpy() does not accept dtype=.
    grid_latitudes = np.asarray(
        dataset[lat_name].values,
        dtype=float,
    )

    original_longitudes = np.asarray(
        dataset[lon_name].values,
        dtype=float,
    )

    if grid_latitudes.ndim != 1:
        raise ValueError(
            f"Latitude coordinate '{lat_name}' must be 1D."
        )

    if original_longitudes.ndim != 1:
        raise ValueError(
            f"Longitude coordinate '{lon_name}' must be 1D."
        )

    if kge_dataarray.shape != (
        grid_latitudes.size,
        original_longitudes.size,
    ):
        raise ValueError(
            "KGE array shape does not match coordinate lengths: "
            f"KGE shape={kge_dataarray.shape}, "
            f"lat={grid_latitudes.size}, "
            f"lon={original_longitudes.size}"
        )

    # Sort latitude into ascending order.
    latitude_order = np.argsort(
        grid_latitudes
    )

    grid_latitudes = grid_latitudes[
        latitude_order
    ]

    kge_dataarray = kge_dataarray.isel(
        {
            lat_name: latitude_order,
        }
    )

    # Detect longitude convention from the original coordinate.
    finite_longitudes = original_longitudes[
        np.isfinite(original_longitudes)
    ]

    if finite_longitudes.size == 0:
        raise ValueError(
            "The NetCDF longitude coordinate contains no finite values."
        )

    grid_uses_0_360 = (
        np.nanmin(finite_longitudes) >= 0.0
        and np.nanmax(finite_longitudes) > 180.0
    )

    if grid_uses_0_360:
        grid_longitudes = normalize_lon_0_360(
            original_longitudes
        )
    else:
        grid_longitudes = normalize_lon_minus180_180(
            original_longitudes
        )

    # Sort normalized longitude into ascending order.
    longitude_order = np.argsort(
        grid_longitudes
    )

    grid_longitudes = grid_longitudes[
        longitude_order
    ]

    kge_dataarray = kge_dataarray.isel(
        {
            lon_name: longitude_order,
        }
    )

    # Load the KGE array while the NetCDF dataset is still open.
    kge_array = np.asarray(
        kge_dataarray.values,
        dtype=float,
    )

    return (
        kge_array,
        grid_latitudes,
        grid_longitudes,
        grid_uses_0_360,
    )


def extract_nearest_grid_values(
    station_latitudes,
    station_longitudes,
    kge_array,
    grid_latitudes,
    grid_longitudes,
    grid_uses_0_360,
):
    """
    Extract nearest KGE grid values for all stations.
    """
    station_latitudes = np.asarray(
        station_latitudes,
        dtype=float,
    )

    station_longitudes = np.asarray(
        station_longitudes,
        dtype=float,
    )

    if station_latitudes.shape != station_longitudes.shape:
        raise ValueError(
            "Station latitude and longitude arrays must have "
            "the same shape."
        )

    latitude_indices = nearest_indices(
        grid_latitudes,
        station_latitudes,
    )

    if grid_uses_0_360:
        station_longitudes_grid = (
            normalize_lon_0_360(
                station_longitudes
            )
        )
    else:
        station_longitudes_grid = (
            normalize_lon_minus180_180(
                station_longitudes
            )
        )

    longitude_indices = (
        nearest_periodic_longitude_indices(
            grid_longitudes,
            station_longitudes_grid,
        )
    )

    extracted_kge = kge_array[
        latitude_indices,
        longitude_indices,
    ]

    selected_latitudes = grid_latitudes[
        latitude_indices
    ]

    selected_longitudes_grid = grid_longitudes[
        longitude_indices
    ]

    selected_longitudes_output = (
        normalize_lon_minus180_180(
            selected_longitudes_grid
        )
    )

    station_longitudes_output = (
        normalize_lon_minus180_180(
            station_longitudes
        )
    )

    distances_km = haversine_distance_km(
        station_latitudes,
        station_longitudes_output,
        selected_latitudes,
        selected_longitudes_output,
    )

    return {
        "kge_glofas_grid": extracted_kge,
        "glofas_grid_lat": selected_latitudes,
        "glofas_grid_lon": selected_longitudes_output,
        "glofas_grid_row": latitude_indices,
        "glofas_grid_col": longitude_indices,
        "distance_station_to_glofas_grid_km": distances_km,
    }


# ---------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------

def read_stations(
    station_file,
    station_id_column,
    station_lat_column,
    station_lon_column,
):
    """
    Read and validate the station CSV.
    """
    if not station_file.is_file():
        raise FileNotFoundError(
            f"Station CSV not found: {station_file}"
        )

    stations = pd.read_csv(
        station_file,
        low_memory=False,
    )

    required_columns = {
        station_id_column,
        station_lat_column,
        station_lon_column,
    }

    missing_columns = (
        required_columns - set(stations.columns)
    )

    if missing_columns:
        raise KeyError(
            "Missing station columns: "
            f"{sorted(missing_columns)}"
        )

    stations = stations.copy()

    stations["station_id_extraction"] = (
        stations[station_id_column].astype(str)
    )

    stations["station_lat"] = pd.to_numeric(
        stations[station_lat_column],
        errors="coerce",
    )

    stations["station_lon"] = pd.to_numeric(
        stations[station_lon_column],
        errors="coerce",
    )

    invalid_identifier = (
        stations["station_id_extraction"]
        .str.strip()
        .isin(
            [
                "",
                "nan",
                "None",
                "<NA>",
            ]
        )
    )

    invalid_coordinates = (
        ~np.isfinite(stations["station_lat"])
        | ~np.isfinite(stations["station_lon"])
        | (stations["station_lat"] < -90.0)
        | (stations["station_lat"] > 90.0)
    )

    invalid_records = (
        invalid_identifier
        | invalid_coordinates
    )

    if invalid_records.any():
        print(
            f"Removing {int(invalid_records.sum()):,} records "
            "with invalid station IDs or coordinates"
        )

        stations = stations.loc[
            ~invalid_records
        ].copy()

    stations["station_lon"] = (
        normalize_lon_minus180_180(
            stations["station_lon"].to_numpy(
                dtype=float
            )
        )
    )

    return stations


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def print_grid_summary(
    kge_array,
    grid_latitudes,
    grid_longitudes,
    grid_uses_0_360,
):
    """
    Print basic information about the NetCDF KGE grid.
    """
    finite_kge = np.isfinite(kge_array)

    print()
    print("KGE grid summary")
    print("-" * 65)
    print(
        f"Dimensions:             "
        f"{grid_latitudes.size} lat x "
        f"{grid_longitudes.size} lon"
    )
    print(
        f"Latitude range:         "
        f"{grid_latitudes.min():.6f} to "
        f"{grid_latitudes.max():.6f}"
    )
    print(
        f"Longitude range:        "
        f"{grid_longitudes.min():.6f} to "
        f"{grid_longitudes.max():.6f}"
    )
    print(
        "Longitude convention:   "
        + (
            "0 to 360"
            if grid_uses_0_360
            else "-180 to 180"
        )
    )
    print(
        f"Finite KGE grid cells:  "
        f"{int(finite_kge.sum()):,} / "
        f"{kge_array.size:,}"
    )

    if finite_kge.any():
        print(
            f"KGE minimum:            "
            f"{np.nanmin(kge_array):.6f}"
        )
        print(
            f"KGE median:             "
            f"{np.nanmedian(kge_array):.6f}"
        )
        print(
            f"KGE maximum:            "
            f"{np.nanmax(kge_array):.6f}"
        )

    print("-" * 65)


def print_extraction_summary(stations):
    """
    Print extraction and distance statistics.
    """
    kge_values = stations[
        "kge_glofas_grid"
    ].to_numpy(dtype=float)

    distances = stations[
        "distance_station_to_glofas_grid_km"
    ].to_numpy(dtype=float)

    valid_kge = np.isfinite(kge_values)

    print()
    print("Extraction summary")
    print("-" * 65)
    print(
        f"Stations:               "
        f"{len(stations):,}"
    )
    print(
        f"Finite gridded KGE:     "
        f"{int(valid_kge.sum()):,}"
    )
    print(
        f"Missing gridded KGE:    "
        f"{int((~valid_kge).sum()):,}"
    )

    if distances.size:
        print(
            f"Minimum distance:       "
            f"{np.nanmin(distances):.3f} km"
        )
        print(
            f"Median distance:        "
            f"{np.nanmedian(distances):.3f} km"
        )
        print(
            f"95th-percentile dist.:  "
            f"{np.nanpercentile(distances, 95):.3f} km"
        )
        print(
            f"Maximum distance:       "
            f"{np.nanmax(distances):.3f} km"
        )

    if valid_kge.any():
        print(
            f"Extracted KGE minimum:  "
            f"{np.nanmin(kge_values):.6f}"
        )
        print(
            f"Extracted KGE median:   "
            f"{np.nanmedian(kge_values):.6f}"
        )
        print(
            f"Extracted KGE maximum:  "
            f"{np.nanmax(kge_values):.6f}"
        )

    print("-" * 65)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_arguments()

    if not args.kge_file.is_file():
        raise FileNotFoundError(
            f"KGE NetCDF not found: {args.kge_file}"
        )

    print(
        f"Reading station file: "
        f"{args.station_file}"
    )

    stations = read_stations(
        station_file=args.station_file,
        station_id_column=args.station_id_column,
        station_lat_column=args.station_lat_column,
        station_lon_column=args.station_lon_column,
    )

    regional_mask = region_mask(
        stations,
        args.region,
    )

    stations = stations.loc[
        regional_mask
    ].copy()

    print(
        f"Stations selected for {args.region}: "
        f"{len(stations):,}"
    )

    if stations.empty:
        raise ValueError(
            f"No valid stations were selected for region "
            f"'{args.region}'."
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
        ) = prepare_grid(
            dataset=dataset,
            lat_name=args.grid_lat_name,
            lon_name=args.grid_lon_name,
            kge_name=args.grid_kge_name,
        )

    print_grid_summary(
        kge_array=kge_array,
        grid_latitudes=grid_latitudes,
        grid_longitudes=grid_longitudes,
        grid_uses_0_360=grid_uses_0_360,
    )

    extracted = extract_nearest_grid_values(
        station_latitudes=(
            stations["station_lat"].to_numpy(
                dtype=float
            )
        ),
        station_longitudes=(
            stations["station_lon"].to_numpy(
                dtype=float
            )
        ),
        kge_array=kge_array,
        grid_latitudes=grid_latitudes,
        grid_longitudes=grid_longitudes,
        grid_uses_0_360=grid_uses_0_360,
    )

    for column_name, values in extracted.items():
        stations[column_name] = values

    if args.max_distance_km is not None:
        too_far = (
            stations[
                "distance_station_to_glofas_grid_km"
            ]
            > args.max_distance_km
        )

        print(
            f"Stations farther than "
            f"{args.max_distance_km:g} km: "
            f"{int(too_far.sum()):,}"
        )

        stations.loc[
            too_far,
            "kge_glofas_grid",
        ] = np.nan

    # Rename the station-observation KGE column to distinguish it from
    # the extracted GloFAS-grid KGE.
    if "kge" in stations.columns:
        stations = stations.rename(
            columns={
                "kge": "kge_station_observations",
            }
        )

    # Also preserve the standardized coordinate columns in the output.
    stations = stations.rename(
        columns={
            "station_id_extraction": (
                "extraction_station_id"
            ),
        }
    )

    print_extraction_summary(
        stations
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    stations.to_csv(
        args.output,
        index=False,
    )

    print(
        f"Output written to: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()
