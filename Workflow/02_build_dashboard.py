#!/usr/bin/env python3
"""
02_build_dashboard.py

Build an interactive multi-experiment river-discharge benchmarking dashboard.

This version addresses the following dashboard issues:

1. Does not apply a blind observation conversion factor. Instead it adds an
   obs/model scale diagnostic and warning in the station panel.
2. Uses a larger responsive map panel and calls Plotly resize on browser resize.
3. Moves the hydrograph legend below the plot and increases margins so labels,
   title and legend do not overlap.
4. Uses one stable experiment colour palette for both the best-experiment map
   and the hydrograph lines.
5. Shows station upstream area, model upstream area, and area difference.
6. Sorts the station performance table by the selected metric, best first.
7. Keeps the current Plotly-geo map engine. Satellite/orography tile backgrounds
   are not implemented here because they require a tile-map backend such as
   Mapbox/MapLibre rather than Plotly geo.

Expected input structure
------------------------
dashboard_data/<expver>/<date_start>_<date_end>_<resolution>arcmin/
├── stations_catalog.json
├── global_station_metrics.csv
└── stations/
    ├── station_000000.json
    ├── station_000001.json
    └── ...

Examples
--------
python3 02_build_dashboard.py \
  --expver j6fu j6gq iyp3 iwya \
  --date-start 20180101 \
  --date-end 20221231 \
  --resolution 15 \
  --metric kge \
  --colour-mode best_experiment

python3 02_build_dashboard.py \
  --expver iwya j6gq \
  --date-start 20180101 \
  --date-end 20221231 \
  --resolution 15 \
  --metric kge \
  --colour-mode experiment_difference

View the output through a web server, for example:
  python3 -m http.server 8000
"""

from pathlib import Path
import argparse
import json

import numpy as np
import plotly.graph_objects as go

import cartopy.io.shapereader as shpreader


# ------------------------------------------------------------
# Defaults
# ------------------------------------------------------------
DEFAULT_DATA_ROOT = Path("dashboard_data")

VALID_METRICS = ["kge", "correlation"]
VALID_COLOUR_MODES = [
    "best_metric",
    "best_experiment",
    "experiment_difference",
]

# Stable palette used both on the map and in hydrograph curves.
EXPERIMENT_COLOURS = [
  "rgb(0,114,178)",      # blue
  "rgb(230,159,0)",      # orange
  "rgb(0,158,115)",      # bluish green
  "rgb(204,121,167)",    # reddish purple
  "rgb(86,180,233)",     # sky blue
  "rgb(213,94,0)",       # vermillion
  "rgb(240,228,66)",     # yellow
  "rgb(0,0,0)",          # black
  "rgb(127,127,127)",    # grey
  "rgb(51,34,136)",      # deep indigo
]


# ------------------------------------------------------------
# Zoom presets
# ------------------------------------------------------------
ZOOM_PRESETS = [
    {"group": "Global", "name": "Global", "lon": [-180, 180], "lat": [-60, 85]},

    {"group": "Continents", "name": "Europe", "lon": [-12, 45], "lat": [34, 72]},
    {"group": "Continents", "name": "North America", "lon": [-170, -50], "lat": [15, 75]},
    {"group": "Continents", "name": "South America", "lon": [-85, -30], "lat": [-60, 15]},
    {"group": "Continents", "name": "Africa", "lon": [-20, 55], "lat": [-35, 38]},
    {"group": "Continents", "name": "Asia", "lon": [25, 150], "lat": [-10, 75]},
    {"group": "Continents", "name": "Australia", "lon": [110, 155], "lat": [-45, -10]},

    {"group": "Europe", "name": "Alps", "lon": [4, 17], "lat": [43, 49]},
    {"group": "Europe", "name": "Danube", "lon": [8, 30], "lat": [42, 51]},
    {"group": "Europe", "name": "Rhine", "lon": [5, 11], "lat": [46, 53]},
    {"group": "Europe", "name": "Po", "lon": [6, 14], "lat": [43.5, 46.8]},
    {"group": "Europe", "name": "Iberia", "lon": [-10, 4], "lat": [35, 44]},
    {"group": "Europe", "name": "UK & Ireland", "lon": [-11, 3], "lat": [49, 61]},
    {"group": "Europe", "name": "Scandinavia", "lon": [4, 32], "lat": [54, 72]},

    {"group": "Major basins", "name": "Amazon", "lon": [-80, -45], "lat": [-20, 8]},
    {"group": "Major basins", "name": "Mississippi", "lon": [-105, -80], "lat": [28, 50]},
    {"group": "Major basins", "name": "Nile", "lon": [24, 40], "lat": [-5, 32]},
    {"group": "Major basins", "name": "Ganges-Brahmaputra", "lon": [72, 96], "lat": [20, 32]},
    {"group": "Major basins", "name": "Mekong", "lon": [95, 110], "lat": [8, 34]},
]


# ------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Build an interactive river discharge dashboard."
    )

    parser.add_argument("--expver", nargs="+", required=True)
    parser.add_argument("--date-start", required=True)
    parser.add_argument("--date-end", required=True)

    parser.add_argument(
        "--resolution",
        type=int,
        default=15,
        choices=[1, 3, 6, 15],
    )

    parser.add_argument(
        "--metric",
        choices=VALID_METRICS,
        default="kge",
    )

    parser.add_argument(
        "--colour-mode",
        choices=VALID_COLOUR_MODES,
        default="best_metric",
    )

    parser.add_argument(
        "--control-expver",
        default=None,
        help=(
            "Control/reference experiment used by --colour-mode best_experiment "
            "to decide whether the winning experiment is meaningfully better. "
            "Default: first experiment in --expver."
        ),
    )

    parser.add_argument(
        "--best-experiment-min-improvement",
        type=float,
        default=0.01,
        help=(
            "Minimum absolute improvement in the selected metric over the control "
            "needed to colour a station by the winning experiment. Stations below "
            "this threshold are coloured grey. Default: 0.01."
        ),
    )

    parser.add_argument(
        "--difference-threshold",
        type=float,
        default=0.01,
        help=(
            "Minimum absolute metric difference needed to classify a pairwise "
            "experiment_difference station as improved or degraded. Values with "
            "absolute difference below this threshold are coloured grey. Default: 0.01."
        ),
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
    )

    parser.add_argument(
        "--output-html",
        type=Path,
        default=None,
    )

    parser.add_argument(
      "--output-dir",
      type=Path,
      default=Path("."),
      help=(
        "Directory where output HTML will be written when --output-html is not "
        "explicitly provided. Default: current directory."
      ),
    )

    parser.add_argument(
        "--river-resolution",
        default="110m",
        choices=["110m", "50m", "10m"],
    )

    parser.add_argument(
        "--max-river-scalerank",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--projection",
        default="equirectangular",
        choices=["equirectangular", "natural earth"],
        help="Use equirectangular for predictable lon/lat preset zooms.",
    )

    parser.add_argument(
        "--map-height-vh",
        type=float,
        default=68.0,
        help="Map panel height in viewport-height units. Default: 68.",
    )

    parser.add_argument(
        "--no-rivers",
        action="store_true",
    )

    parser.add_argument(
        "--warn-scale-ratio-low",
        type=float,
        default=0.2,
        help="Warn if obs/model median ratio is below this value.",
    )

    parser.add_argument(
        "--warn-scale-ratio-high",
        type=float,
        default=5.0,
        help="Warn if obs/model median ratio is above this value.",
    )

    parser.add_argument(
        "--background",
        default="standard",
        choices=["standard"],
        help=(
            "Currently only 'standard' is supported with Plotly geo. "
            "Satellite/terrain tiles require a map-tile based dashboard engine."
        ),
    )

    return parser.parse_args()


def parse_date(value):
    return str(value).replace("-", "")


def date_label(date_start, date_end):
    return f"{parse_date(date_start)}_{parse_date(date_end)}"


def run_label(date_start, date_end, resolution):
    return f"{date_label(date_start, date_end)}_{resolution}arcmin"


def default_output_name(expvers, date_start, date_end, resolution, metric, colour_mode):
    return Path(
        f"global_station_dashboard_{'_'.join(expvers)}_"
        f"{date_label(date_start, date_end)}_"
        f"{resolution}arcmin_{metric}_{colour_mode}.html"
    )


# ------------------------------------------------------------
# Catalogue loading and merging
# ------------------------------------------------------------
def station_key(station):
    if station.get("source_station_id") is not None:
        return str(station.get("source_station_id"))

    if station.get("station_id") is not None:
        return str(station.get("station_id"))

    return f"{station.get('lon')}_{station.get('lat')}_{station.get('name')}"


def safe_float(value):
    if value is None:
        return None
    try:
        x = float(value)
        if np.isfinite(x):
            return x
    except Exception:
        pass
    return None


def load_experiment_catalogue(data_root, expver, label):
    data_dir = data_root / expver / label
    catalog_json = data_dir / "stations_catalog.json"

    if not catalog_json.exists():
        raise FileNotFoundError(
            f"Catalogue not found for experiment {expver}: {catalog_json}"
        )

    catalog = json.loads(catalog_json.read_text(encoding="utf-8"))

    print(
        f"Loaded {expver}: {len(catalog):,} stations from {catalog_json}",
        flush=True,
    )

    return data_dir, catalog


def experiment_palette(expvers):
    palette = {}
    for i, expver in enumerate(expvers):
        palette[expver] = EXPERIMENT_COLOURS[i % len(EXPERIMENT_COLOURS)]
    return palette


def merge_catalogues(data_root, expvers, label, metric, control_expver=None):
    merged = {}
    experiment_counts = {}

    for expver in expvers:
        data_dir, catalog = load_experiment_catalogue(
            data_root=data_root,
            expver=expver,
            label=label,
        )

        experiment_counts[expver] = len(catalog)

        for station in catalog:
            key = station_key(station)

            if key not in merged:
                merged[key] = {
                    "merge_key": key,
                    "source_station_id": station.get("source_station_id"),
                    "station_id": station.get("station_id"),
                    "name": station.get("name", ""),
                    "river": station.get("river", ""),
                    "country_code": station.get("country_code", ""),
                    "lon": station.get("lon"),
                    "lat": station.get("lat"),
                    "model_lon": station.get("model_lon"),
                    "model_lat": station.get("model_lat"),
                    "distance_station_to_model_km": station.get(
                        "distance_station_to_model_km"
                    ),
                    "upstream_area_km2": station.get("upstream_area_km2"),
                    "runs": {},
                }

            json_path = str(Path(expver) / label / station.get("file", ""))

            merged[key]["runs"][expver] = {
                "expver": expver,
                "file": station.get("file"),
                "json_path": json_path,
                "kge": station.get("kge"),
                "correlation": station.get("correlation"),
                "rmse": station.get("rmse"),
                "matched_days": station.get("matched_days"),
                "model_lon": station.get("model_lon"),
                "model_lat": station.get("model_lat"),
                "model_upstream_area_km2": station.get("model_upstream_area_km2"),
                "upstream_area_difference_pct": station.get(
                    "upstream_area_difference_pct"
                ),
            }

    records = list(merged.values())

    for record in records:
        best_value = None
        best_expver = None

        for expver in expvers:
            run = record["runs"].get(expver, {})
            value = safe_float(run.get(metric))

            if value is None:
                continue

            if best_value is None or value > best_value:
                best_value = value
                best_expver = expver

        record["best_metric"] = best_value
        record["best_expver"] = best_expver
        record["n_experiments"] = len(record["runs"])

        control_value = None
        if control_expver is not None:
            control_value = safe_float(
                record["runs"].get(control_expver, {}).get(metric)
            )

        record["control_expver"] = control_expver
        record["control_metric"] = control_value
        record["best_improvement_over_control"] = (
            None
            if best_value is None or control_value is None
            else best_value - control_value
        )

        if len(expvers) >= 2:
            v0 = safe_float(record["runs"].get(expvers[0], {}).get(metric))
            v1 = safe_float(record["runs"].get(expvers[1], {}).get(metric))
            record["metric_difference"] = None if v0 is None or v1 is None else v1 - v0
            record["difference_reference_expver"] = expvers[0]
            record["difference_target_expver"] = expvers[1]
        else:
            record["metric_difference"] = None
            record["difference_reference_expver"] = None
            record["difference_target_expver"] = None

    print("")
    print("Merged catalogue summary")
    print("------------------------")
    print(f"Experiments: {', '.join(expvers)}")

    for expver in expvers:
        print(f"  {expver}: {experiment_counts[expver]:,} stations")

    print(f"Union of stations: {len(records):,}")
    print(
        f"Stations with at least one valid {metric}: "
        f"{sum(r['best_metric'] is not None for r in records):,}"
    )

    if control_expver is not None:
        n_control = sum(r.get("control_metric") is not None for r in records)
        n_improvement = sum(
            r.get("best_improvement_over_control") is not None for r in records
        )
        print(f"Control experiment for best_experiment: {control_expver}")
        print(f"Stations with valid control {metric}: {n_control:,}")
        print(f"Stations with valid best-control improvement: {n_improvement:,}")

    if len(expvers) >= 2:
        n_diff = sum(r["metric_difference"] is not None for r in records)
        print(
            f"Stations with valid {metric} difference "
            f"({expvers[1]} - {expvers[0]}): {n_diff:,}"
        )

    print("")

    return records


# ------------------------------------------------------------
# Colouring helpers
# ------------------------------------------------------------
def metric_class(metric, value):
    if value is None:
        return "missing"

    if metric == "kge":
        if value < -0.41:
            return "very_poor"
        if -0.41 <= value <= 0.0:
            return "poor"
        if 0.0 < value <= 0.5:
            return "moderate"
        return "good"

    if metric == "correlation":
        if value < 0.3:
            return "very_poor"
        if 0.3 <= value <= 0.5:
            return "poor"
        if 0.5 < value <= 0.7:
            return "moderate"
        return "good"

    raise ValueError(f"Unsupported metric: {metric}")


def best_metric_class_config(metric):
    if metric == "kge":
        return [
            {"class": "missing", "label": "No KGE", "color": "lightgrey", "size": 5, "opacity": 0.35},
            {"class": "very_poor", "label": "Best KGE < -0.41", "color": "red", "size": 7, "opacity": 0.85},
            {"class": "poor", "label": "-0.41 <= best KGE <= 0", "color": "gold", "size": 7, "opacity": 0.85},
            {"class": "moderate", "label": "0 < best KGE <= 0.5", "color": "limegreen", "size": 7, "opacity": 0.85},
            {"class": "good", "label": "Best KGE > 0.5", "color": "darkgreen", "size": 8, "opacity": 0.9},
        ]

    if metric == "correlation":
        return [
            {"class": "missing", "label": "No correlation", "color": "lightgrey", "size": 5, "opacity": 0.35},
            {"class": "very_poor", "label": "Best correlation < 0.3", "color": "red", "size": 7, "opacity": 0.85},
            {"class": "poor", "label": "0.3 <= best correlation <= 0.5", "color": "gold", "size": 7, "opacity": 0.85},
            {"class": "moderate", "label": "0.5 < best correlation <= 0.7", "color": "limegreen", "size": 7, "opacity": 0.85},
            {"class": "good", "label": "Best correlation > 0.7", "color": "darkgreen", "size": 8, "opacity": 0.9},
        ]

    raise ValueError(f"Unsupported metric: {metric}")


def difference_class(value, threshold):
    """
    Classify pairwise metric difference.

    value = target experiment - reference experiment.
    Values within +/- threshold are considered not meaningfully different
    and are coloured grey.
    """
    if value is None:
        return "missing"

    threshold = abs(float(threshold))

    if value <= -0.20:
        return "large_degradation"
    if -0.20 < value <= -threshold:
        return "moderate_degradation"
    if -threshold < value < threshold:
        return "neutral"
    if threshold <= value < 0.20:
        return "moderate_improvement"
    return "large_improvement"


def difference_class_config(metric, ref_expver, target_expver, threshold):
    threshold = abs(float(threshold))
    label = f"{target_expver} - {ref_expver}"

    return [
        {"class": "missing", "label": f"No {metric} difference", "color": "lightgrey", "size": 5, "opacity": 0.35},
        {"class": "large_degradation", "label": f"{label} <= -0.20", "color": "darkred", "size": 8, "opacity": 0.9},
        {"class": "moderate_degradation", "label": f"-0.20 < {label} <= -{threshold:g}", "color": "red", "size": 7, "opacity": 0.85},
        {"class": "neutral", "label": f"|{label}| < {threshold:g} (no meaningful difference)", "color": "lightgrey", "size": 6, "opacity": 0.7},
        {"class": "moderate_improvement", "label": f"{threshold:g} <= {label} < 0.20", "color": "limegreen", "size": 7, "opacity": 0.85},
        {"class": "large_improvement", "label": f"{label} >= 0.20", "color": "darkgreen", "size": 8, "opacity": 0.9},
    ]


# ------------------------------------------------------------
# Plotly map helpers
# ------------------------------------------------------------
def add_river_layer(fig, resolution="50m", max_scalerank=6):
    shp = shpreader.natural_earth(
        resolution=resolution,
        category="physical",
        name="rivers_lake_centerlines",
    )

    reader = shpreader.BasicReader(shp)

    lon_all = []
    lat_all = []
    n_rivers = 0

    for rec in reader.records():
        scalerank = rec.attributes.get("scalerank", 99)

        if scalerank is not None and scalerank > max_scalerank:
            continue

        geom = rec.geometry
        if geom is None:
            continue

        if geom.geom_type == "LineString":
            lines = [geom]
        elif geom.geom_type == "MultiLineString":
            lines = list(geom.geoms)
        else:
            continue

        for line in lines:
            coords = np.asarray(line.coords)
            if coords.size == 0:
                continue

            lon_all.extend(coords[:, 0].tolist())
            lat_all.extend(coords[:, 1].tolist())
            lon_all.append(None)
            lat_all.append(None)
            n_rivers += 1

    fig.add_trace(
        go.Scattergeo(
            lon=lon_all,
            lat=lat_all,
            mode="lines",
            line=dict(color="rgb(60,120,180)", width=0.7),
            opacity=0.55,
            name=f"Rivers ({resolution}, rank <= {max_scalerank})",
            hoverinfo="skip",
            showlegend=True,
        )
    )

    print(f"Added river layer: {n_rivers:,} river segments", flush=True)


def hover_text(record, args):
    metric = args.metric

    station_name = record.get("name", "Station") or "Station"
    country = record.get("country_code", "") or ""
    river = record.get("river", "") or ""

    values_by_exp = []

    for expver in args.expver:
        run = record["runs"].get(expver)

        if not run:
            continue

        value = safe_float(run.get(metric))

        if value is not None:
            values_by_exp.append((expver, value))

    if values_by_exp:
        best_expver, best_value = max(values_by_exp, key=lambda x: x[1])
        worst_expver, worst_value = min(values_by_exp, key=lambda x: x[1])

        best_line = f"Best: {best_expver}, {metric}: {best_value:.3f}"
        worst_line = f"Worst: {worst_expver}, {metric}: {worst_value:.3f}"

    else:
        best_line = f"Best: NA, {metric}: NA"
        worst_line = f"Worst: NA, {metric}: NA"

    return (
        f"{station_name} ({country})"
        f"<br>{river}"
        f"<br>{best_line}"
        f"<br>{worst_line}"
    )


def add_marker_trace(fig, records, indices, name, color, size, opacity, args):
    if len(indices) == 0:
        return

    fig.add_trace(
        go.Scattergeo(
            lon=[records[i]["lon"] for i in indices],
            lat=[records[i]["lat"] for i in indices],
            mode="markers",
            text=[hover_text(records[i], args) for i in indices],
            customdata=indices,
            hovertemplate="%{text}<extra></extra>",
            marker=dict(
                size=size,
                color=color,
                opacity=opacity,
                line=dict(width=0.8, color="black"),
            ),
            name=f"{name} ({len(indices):,})",
        )
    )


def add_best_metric_layers(fig, records, args):
    for cfg in best_metric_class_config(args.metric):
        indices = [
            i for i, record in enumerate(records)
            if metric_class(args.metric, record.get("best_metric")) == cfg["class"]
        ]

        add_marker_trace(
            fig=fig,
            records=records,
            indices=indices,
            name=cfg["label"],
            color=cfg["color"],
            size=cfg["size"],
            opacity=cfg["opacity"],
            args=args,
        )


def add_best_experiment_layers(fig, records, args):
    """
    Colour stations by winning experiment, but colour them grey when the best
    experiment is not meaningfully better than the control experiment.

    Meaningful improvement is defined as:
        best_metric - control_metric >= args.best_experiment_min_improvement

    If the control metric is unavailable for a station, the station is coloured
    by the best available experiment rather than being forced to grey.
    """
    palette = experiment_palette(args.expver)
    threshold = float(args.best_experiment_min_improvement)
    control_expver = args.control_expver

    missing_indices = [
        i for i, record in enumerate(records)
        if record.get("best_expver") is None
    ]

    add_marker_trace(
        fig, records, missing_indices,
        f"No valid {args.metric}", "lightgrey", 5, 0.35, args,
    )

    no_clear_winner_indices = []

    for i, record in enumerate(records):
        if record.get("best_expver") is None:
            continue

        # Only apply the grey no-clear-winner class if a valid control value is
        # present. If the control is missing, keep the best experiment colour.
        improvement = record.get("best_improvement_over_control")

        if improvement is not None and improvement < threshold:
            no_clear_winner_indices.append(i)

    add_marker_trace(
        fig, records, no_clear_winner_indices,
        (
            f"No clear improvement over {control_expver} "
            f"(< {threshold:g} {args.metric})"
        ),
        "rgb(150,150,150)",
        7,
        0.78,
        args,
    )

    no_clear_set = set(no_clear_winner_indices)

    for expver in args.expver:
        indices = [
            i for i, record in enumerate(records)
            if record.get("best_expver") == expver and i not in no_clear_set
        ]

        add_marker_trace(
            fig, records, indices,
            f"Best and meaningful: {expver}", palette[expver], 8, 0.88, args,
        )

def add_difference_layers(fig, records, args):
    if len(args.expver) != 2:
        raise ValueError("--colour-mode experiment_difference requires exactly two experiments.")

    ref_expver = args.expver[0]
    target_expver = args.expver[1]

    for cfg in difference_class_config(
        args.metric,
        ref_expver,
        target_expver,
        args.difference_threshold,
    ):
        indices = [
            i for i, record in enumerate(records)
            if difference_class(
                record.get("metric_difference"),
                args.difference_threshold,
            ) == cfg["class"]
        ]

        add_marker_trace(
            fig, records, indices,
            cfg["label"], cfg["color"], cfg["size"], cfg["opacity"], args,
        )


def colour_title(args):
    if args.colour_mode == "best_metric":
        return f"Stations coloured by best {args.metric}"
    if args.colour_mode == "best_experiment":
        return f"Stations coloured by winning experiment for {args.metric}"
    if args.colour_mode == "experiment_difference":
        return f"Stations coloured by {args.metric} difference ({args.expver[1]} - {args.expver[0]})"
    raise ValueError(f"Unsupported colour mode: {args.colour_mode}")


def build_map(records, args):
    fig = go.Figure()

    if not args.no_rivers:
        add_river_layer(
            fig=fig,
            resolution=args.river_resolution,
            max_scalerank=args.max_river_scalerank,
        )

    if args.colour_mode == "best_metric":
        add_best_metric_layers(fig, records, args)
    elif args.colour_mode == "best_experiment":
        add_best_experiment_layers(fig, records, args)
    elif args.colour_mode == "experiment_difference":
        add_difference_layers(fig, records, args)
    else:
        raise ValueError(f"Unsupported colour mode: {args.colour_mode}")

    fig.update_geos(
        projection_type=args.projection,
        showland=True,
        landcolor="rgb(240,240,240)",
        showcountries=True,
        countrycolor="rgb(120,120,120)",
        showcoastlines=True,
        coastlinecolor="rgb(80,80,80)",
        showocean=True,
        oceancolor="rgb(225,235,245)",
        lataxis_showgrid=True,
        lonaxis_showgrid=True,
    )

    exp_label = ", ".join(args.expver)

    fig.update_layout(
        title=(
            "Global station hydrograph dashboard"
            f"<br><sup>Experiments: {exp_label}. "
            f"{colour_title(args)}. "
            f"Click a station to compare hydrographs.</sup>"
        ),
        autosize=True,
        margin=dict(l=0, r=0, t=78, b=0),
        showlegend=True,
        legend=dict(
            x=0.01,
            y=0.98,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(120,120,120,0.6)",
            borderwidth=1,
        ),
    )

    return fig


# ------------------------------------------------------------
# HTML generation
# ------------------------------------------------------------
def build_html(records, fig, args, output_html):
    map_div = fig.to_html(
    include_plotlyjs=False,
        full_html=False,
        div_id="map",
        default_width="100%",
        default_height=f"{float(args.map_height_vh)}vh",
        config={
            "scrollZoom": True,
            "displaylogo": False,
            "responsive": True,
            "modeBarButtonsToRemove": ["select2d", "lasso2d"],
        },
    )

    records_json = json.dumps(records, ensure_ascii=True, separators=(",", ":"))
    expvers_json = json.dumps(args.expver, ensure_ascii=True, separators=(",", ":"))
    zoom_presets_json = json.dumps(ZOOM_PRESETS, ensure_ascii=True, separators=(",", ":"))
    palette_json = json.dumps(experiment_palette(args.expver), ensure_ascii=True, separators=(",", ":"))

    data_root_js = str(args.data_root).replace("\\", "/")

    html_template = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Global station hydrograph dashboard</title>
    <script src="https://cdn.plot.ly/plotly-3.5.0.min.js"></script>
<style>
html, body {
  margin: 0;
  height: 100%;
  font-family: Arial, sans-serif;
}

#container {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  overflow: visible;
}

#main-top {
  display: block;
}

#top-toolbar {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  padding: 6px 10px;
  border-bottom: 1px solid #ccc;
  background: #f7f7f7;
  font-size: 12px;
  z-index: 50;
}

.toolbar-title {
  font-weight: bold;
  color: #333;
}

#zoom-select {
  min-width: 200px;
  max-width: 460px;
  width: min(460px, 48vw);
  font-size: 12px;
  padding: 3px 5px;
}

#zoom-apply,
#zoom-global {
  font-size: 12px;
  padding: 3px 8px;
  cursor: pointer;
}

#zoom-status {
  color: #555;
  font-size: 11px;
  margin-left: 6px;
}

#station-selector-wrap {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}

#station-search-input {
  min-width: 240px;
  max-width: 560px;
  width: min(560px, 56vw);
  font-size: 12px;
  padding: 3px 5px;
}

#station-search-go {
  font-size: 12px;
  padding: 3px 8px;
  cursor: pointer;
}

#transport-warning {
  display: none;
  margin: 6px 10px;
}

#map-wrap {
  flex: none;
  height: __MAP_HEIGHT_VH__vh;
  min-height: 420px;
  overflow: hidden;
  position: relative;
}

#map {
  width: 100% !important;
  height: 100% !important;
}

#map .plot-container,
#map .svg-container {
  width: 100% !important;
  height: 100% !important;
}

#ecdf-wrap {
  border-top: 1px solid #ccc;
  border-bottom: 1px solid #ccc;
  background: #fafafa;
  padding: 8px 10px 6px 10px;
}

#ecdf-toolbar {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  font-size: 12px;
  margin-bottom: 6px;
}

#ecdf-exp-select {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
}

.ecdf-exp-chip {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 1px 5px;
  border: 1px solid #ccc;
  border-radius: 4px;
  background: #fff;
}

#ecdf-diagnostics {
  font-size: 11px;
  color: #444;
  margin-bottom: 4px;
  line-height: 1.45;
}

#ecdf-plot {
  width: 100%;
  height: 245px;
  min-height: 220px;
}

@media (min-width: 1200px) and (orientation: landscape) {
  #main-top {
    display: grid;
    grid-template-columns: minmax(0, 2.35fr) minmax(340px, 1fr);
    align-items: stretch;
    border-bottom: 1px solid #ccc;
  }

  #map-wrap {
    min-height: 420px;
  }

  #ecdf-wrap {
    display: flex;
    flex-direction: column;
    border-top: none;
    border-bottom: none;
    border-left: 1px solid #ccc;
    height: __MAP_HEIGHT_VH__vh;
    min-height: 420px;
    padding-bottom: 10px;
  }

  #ecdf-plot {
    flex: 1 1 auto;
    height: auto;
    min-height: 240px;
  }
}

#bottom {
  flex: 1 0 300px;
  display: grid;
  grid-template-columns: minmax(0, 1fr) 520px;
  border-top: 1px solid #ccc;
  min-height: 300px;
  background: white;
  overflow: hidden;
}

#hydrograph {
  min-height: 260px;
  overflow: hidden;
}

#info {
  overflow-y: auto;
  border-left: 1px solid #ccc;
  padding: 10px;
  font-size: 12px;
  background: white;
}

table {
  border-collapse: collapse;
  width: 100%;
  margin-top: 6px;
  margin-bottom: 10px;
}

th, td {
  border-bottom: 1px solid #ddd;
  padding: 4px 6px;
  text-align: left;
  vertical-align: top;
}

.badge {
  display: inline-block;
  padding: 2px 6px;
  border-radius: 4px;
  background: #eee;
  margin: 1px 2px 1px 0;
}

.warning {
  display: inline-block;
  padding: 4px 6px;
  border-radius: 4px;
  background: #fff0c2;
  border: 1px solid #d6a900;
  margin: 4px 0;
}

.small {
  font-size: 11px;
  color: #555;
}

.code {
  font-family: monospace;
  font-size: 11px;
}

@media (max-width: 1100px) {
  #bottom {
    grid-template-columns: minmax(0, 1fr) 430px;
  }
}

@media (max-width: 900px) {
  #map-wrap {
    min-height: 320px;
  }

  #bottom {
    grid-template-columns: 1fr;
    grid-auto-rows: minmax(240px, auto);
  }

  #info {
    border-left: none;
    border-top: 1px solid #ccc;
    max-width: 100%;
  }

  #zoom-status {
    margin-left: 0;
    width: 100%;
  }

  #station-selector-wrap {
    width: 100%;
  }

  #station-search-input {
    width: 100%;
    max-width: 100%;
    min-width: 0;
  }

  #main-top {
    display: block;
  }
}
</style>
</head>
<body>
<div id="container">

  <div id="top-toolbar">
    <div class="toolbar-title">Zoom preset</div>
    <select id="zoom-select"></select>
    <button id="zoom-apply" type="button">Go</button>
    <button id="zoom-global" type="button">Global</button>

    <div class="toolbar-title">Station selector</div>
    <div id="station-selector-wrap">
      <input id="station-search-input" list="station-search-list" type="text" placeholder="Type station Id or river" autocomplete="off">
      <datalist id="station-search-list"></datalist>
      <button id="station-search-go" type="button">Load</button>
    </div>

    <span id="zoom-status"></span>
  </div>

  <div id="transport-warning" class="warning"></div>

  <div id="main-top">
    <div id="map-wrap">
      __MAP_DIV__
    </div>

    <div id="ecdf-wrap">
      <div id="ecdf-toolbar">
        <span class="toolbar-title">Visible-extent empirical CDF</span>
        <span id="ecdf-exp-select"></span>
        <label>
          <input id="ecdf-common-checkbox" type="checkbox">
          Use common stations
        </label>
      </div>
      <div id="ecdf-diagnostics" class="small">Initialising eCDF panel...</div>
      <div id="ecdf-plot"></div>
    </div>
  </div>

  <div id="bottom">
    <div id="hydrograph">
      <p style="padding:12px;">Click a station to load hydrographs.</p>
    </div>
    <div id="info">
      <b>Station information</b><br>
      Click a station on the map.
    </div>
  </div>
</div>

<script>
const STATIONS = __RECORDS_JSON__;
const EXPVERS = __EXPVERS_JSON__;
const ZOOM_PRESETS = __ZOOM_PRESETS_JSON__;
const EXPERIMENT_COLOURS = __PALETTE_JSON__;
const DATA_ROOT = "__DATA_ROOT__";
const SELECTED_METRIC = "__METRIC__";
const COLOUR_MODE = "__COLOUR_MODE__";
const CONTROL_EXPVER = "__CONTROL_EXPVER__";
const BEST_EXPERIMENT_MIN_IMPROVEMENT = __BEST_EXPERIMENT_MIN_IMPROVEMENT__;
const WARN_SCALE_LOW = __WARN_SCALE_LOW__;
const WARN_SCALE_HIGH = __WARN_SCALE_HIGH__;
const ECDF_DEBOUNCE_MS = 150;

const mapDiv = document.getElementById("map");
const globalPreset = ZOOM_PRESETS.find(p => p.name === "Global") || {
  lon: [-180, 180],
  lat: [-60, 85]
};

const DEFAULT_EXTENT = {
  south: Number(globalPreset.lat[0]),
  west: Number(globalPreset.lon[0]),
  north: Number(globalPreset.lat[1]),
  east: Number(globalPreset.lon[1])
};

const DEFAULT_CENTER = {
  lon: 0.5 * (DEFAULT_EXTENT.west + DEFAULT_EXTENT.east),
  lat: 0.5 * (DEFAULT_EXTENT.south + DEFAULT_EXTENT.north)
};

let currentGeoCenter = {lon: DEFAULT_CENTER.lon, lat: DEFAULT_CENTER.lat};
let currentGeoScale = 1.0;
let currentVisibleExtent = {
  south: DEFAULT_EXTENT.south,
  west: DEFAULT_EXTENT.west,
  north: DEFAULT_EXTENT.north,
  east: DEFAULT_EXTENT.east
};
let ecdfDebounceTimer = null;
let ecdfInitialised = false;
let currentStationIndex = null;
let currentStationRequestId = 0;
let activeStationAbortController = null;
let resizeDebounceTimer = null;
let selectedStationTraceIndices = [];
const stationPayloadCache = new Map();
const STATION_FETCH_TIMEOUT_MS = 12000;
const STATION_SUGGESTION_LIMIT = 50;
let stationSearchMatches = [];

function initEnvironmentWarning() {
  const warning = document.getElementById("transport-warning");
  if (!warning) return;

  if (window.location.protocol === "file:") {
    warning.style.display = "inline-block";
    warning.innerHTML =
      "This dashboard is opened with file://. Some browsers block local JSON fetch calls. " +
      "Serve this folder with a local web server (for example: <span class='code'>python3 -m http.server</span>) " +
      "or upload it to Sites.";
  }
}

function sanitizeUserPath(pathValue) {
  if (pathValue === null || pathValue === undefined) return "";

  const text = String(pathValue);

  return text
    .replace(/^\\/perm\\/[^/]+\\//, "/perm/$USER/")
    .replace(/^\\/home\\/[^/]+\\//, "/home/$USER/");
}

function esc(value) {
  if (value === null || value === undefined) return "";
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function fmt(value, ndigits=3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "NA";
  }
  return Number(value).toFixed(ndigits);
}

function fmtSigned(value, ndigits=3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "NA";
  }
  const x = Number(value);
  return (x >= 0 ? "+" : "") + x.toFixed(ndigits);
}

function median(values) {
  const v = values
    .map(Number)
    .filter(x => Number.isFinite(x))
    .sort((a, b) => a - b);

  if (v.length === 0) return null;

  const mid = Math.floor(v.length / 2);
  if (v.length % 2 === 1) return v[mid];
  return 0.5 * (v[mid - 1] + v[mid]);
}

function normalizeLongitude(lon) {
  const x = Number(lon);
  if (!Number.isFinite(x)) return null;

  const y = ((x + 180) % 360 + 360) % 360 - 180;

  if (y === -180 && x > 0) return 180;
  return y;
}

function clampLatitude(lat) {
  const x = Number(lat);
  if (!Number.isFinite(x)) return null;
  return Math.max(-90, Math.min(90, x));
}

function finiteMetricValue(station, expver) {
  if (!station || !station.runs || !station.runs[expver]) return null;
  const v = Number(station.runs[expver][SELECTED_METRIC]);
  return Number.isFinite(v) ? v : null;
}

function longitudeSpanDegrees(extent) {
  const west = normalizeLongitude(extent.west);
  const east = normalizeLongitude(extent.east);

  if (west === null || east === null) return 360;

  let span = (east - west + 360) % 360;
  if (span === 0) span = 360;
  return span;
}

function stationInsideExtent(station, extent) {
  if (!station) return false;

  const lat = Number(station.lat);
  const lon = normalizeLongitude(station.lon);

  if (!Number.isFinite(lat) || lon === null) return false;

  const south = Number(extent.south);
  const north = Number(extent.north);
  if (!Number.isFinite(south) || !Number.isFinite(north)) return false;
  if (lat < south || lat > north) return false;

  const west = normalizeLongitude(extent.west);
  const east = normalizeLongitude(extent.east);
  if (west === null || east === null) return false;

  const span = longitudeSpanDegrees(extent);
  if (span >= 359.999) return true;

  if (west <= east) {
    return lon >= west && lon <= east;
  }

  return lon >= west || lon <= east;
}

function deriveExtentFromCenterScale(centerLon, centerLat, scale) {
  const s = Number.isFinite(scale) && scale > 0 ? scale : 1.0;

  const defaultLonSpan = longitudeSpanDegrees(DEFAULT_EXTENT);
  const defaultLatSpan = Number(DEFAULT_EXTENT.north) - Number(DEFAULT_EXTENT.south);

  const lonSpan = Math.max(0.5, Math.min(360, defaultLonSpan / s));
  const latSpan = Math.max(0.5, Math.min(180, defaultLatSpan / s));

  const cLon = normalizeLongitude(centerLon);
  const cLat = clampLatitude(centerLat);

  const centerLonSafe = cLon === null ? DEFAULT_CENTER.lon : cLon;
  const centerLatSafe = cLat === null ? DEFAULT_CENTER.lat : cLat;

  return {
    south: clampLatitude(centerLatSafe - 0.5 * latSpan),
    west: normalizeLongitude(centerLonSafe - 0.5 * lonSpan),
    north: clampLatitude(centerLatSafe + 0.5 * latSpan),
    east: normalizeLongitude(centerLonSafe + 0.5 * lonSpan)
  };
}

function readRangeFromPayload(payload, keyBase) {
  if (!payload) return null;

  if (Array.isArray(payload[keyBase]) && payload[keyBase].length >= 2) {
    return [Number(payload[keyBase][0]), Number(payload[keyBase][1])];
  }

  const k0 = keyBase + "[0]";
  const k1 = keyBase + "[1]";

  if (payload[k0] !== undefined && payload[k1] !== undefined) {
    return [Number(payload[k0]), Number(payload[k1])];
  }

  return null;
}

function deriveInitialExtentFromMap() {
  const fallback = {
    south: DEFAULT_EXTENT.south,
    west: DEFAULT_EXTENT.west,
    north: DEFAULT_EXTENT.north,
    east: DEFAULT_EXTENT.east
  };

  if (!mapDiv || !mapDiv.layout || !mapDiv.layout.geo) {
    return fallback;
  }

  const geo = mapDiv.layout.geo;
  const lonRange = geo.lonaxis && Array.isArray(geo.lonaxis.range) ? geo.lonaxis.range : null;
  const latRange = geo.lataxis && Array.isArray(geo.lataxis.range) ? geo.lataxis.range : null;

  if (lonRange && latRange) {
    const west = normalizeLongitude(lonRange[0]);
    const east = normalizeLongitude(lonRange[1]);
    const south = clampLatitude(Math.min(latRange[0], latRange[1]));
    const north = clampLatitude(Math.max(latRange[0], latRange[1]));

    if (west !== null && east !== null && south !== null && north !== null) {
      return {south: south, west: west, north: north, east: east};
    }
  }

  const centerLon = geo.center && geo.center.lon !== undefined ? geo.center.lon : DEFAULT_CENTER.lon;
  const centerLat = geo.center && geo.center.lat !== undefined ? geo.center.lat : DEFAULT_CENTER.lat;
  const scale = geo.projection && geo.projection.scale !== undefined ? Number(geo.projection.scale) : 1.0;

  const normalizedCenterLon = normalizeLongitude(centerLon);
  const clampedCenterLat = clampLatitude(centerLat);

  currentGeoCenter = {
    lon: normalizedCenterLon !== null ? normalizedCenterLon : DEFAULT_CENTER.lon,
    lat: clampedCenterLat !== null ? clampedCenterLat : DEFAULT_CENTER.lat
  };
  currentGeoScale = Number.isFinite(scale) && scale > 0 ? scale : 1.0;

  return deriveExtentFromCenterScale(currentGeoCenter.lon, currentGeoCenter.lat, currentGeoScale);
}

function updateExtentCacheFromRelayout(payload) {
  let changed = false;
  let hasExplicitRanges = false;
  let hasCenterLon = false;
  let hasCenterLat = false;
  let hasScale = false;

  const lonRange = readRangeFromPayload(payload, "geo.lonaxis.range");
  const latRange = readRangeFromPayload(payload, "geo.lataxis.range");

  if (lonRange) {
    const west = normalizeLongitude(lonRange[0]);
    const east = normalizeLongitude(lonRange[1]);
    if (west !== null && east !== null) {
      currentVisibleExtent.west = west;
      currentVisibleExtent.east = east;
      hasExplicitRanges = true;
      changed = true;
    }
  }

  if (latRange) {
    const south = clampLatitude(Math.min(latRange[0], latRange[1]));
    const north = clampLatitude(Math.max(latRange[0], latRange[1]));
    if (south !== null && north !== null) {
      currentVisibleExtent.south = south;
      currentVisibleExtent.north = north;
      hasExplicitRanges = true;
      changed = true;
    }
  }

  if (payload && payload["geo.center.lon"] !== undefined) {
    const lon = normalizeLongitude(payload["geo.center.lon"]);
    if (lon !== null) {
      currentGeoCenter.lon = lon;
      hasCenterLon = true;
      changed = true;
    }
  }

  if (payload && payload["geo.center.lat"] !== undefined) {
    const lat = clampLatitude(payload["geo.center.lat"]);
    if (lat !== null) {
      currentGeoCenter.lat = lat;
      hasCenterLat = true;
      changed = true;
    }
  }

  if (payload && payload["geo.projection.scale"] !== undefined) {
    const scale = Number(payload["geo.projection.scale"]);
    if (Number.isFinite(scale) && scale > 0) {
      const previousScale = currentGeoScale;
      currentGeoScale = scale;
      hasScale = true;

      if (!hasExplicitRanges) {
        const oldLonSpan = longitudeSpanDegrees(currentVisibleExtent);
        const oldLatSpan = Math.max(0.5, Number(currentVisibleExtent.north) - Number(currentVisibleExtent.south));
        const ratio = previousScale > 0 ? (previousScale / currentGeoScale) : 1.0;

        const newLonSpan = Math.max(0.5, Math.min(360, oldLonSpan * ratio));
        const newLatSpan = Math.max(0.5, Math.min(180, oldLatSpan * ratio));

        currentVisibleExtent.west = normalizeLongitude(currentGeoCenter.lon - 0.5 * newLonSpan);
        currentVisibleExtent.east = normalizeLongitude(currentGeoCenter.lon + 0.5 * newLonSpan);
        currentVisibleExtent.south = clampLatitude(currentGeoCenter.lat - 0.5 * newLatSpan);
        currentVisibleExtent.north = clampLatitude(currentGeoCenter.lat + 0.5 * newLatSpan);
      }

      changed = true;
    }
  }

  if (!hasExplicitRanges && !hasScale && (hasCenterLon || hasCenterLat)) {
    const lonSpan = longitudeSpanDegrees(currentVisibleExtent);
    const latSpan = Math.max(0.5, Number(currentVisibleExtent.north) - Number(currentVisibleExtent.south));

    currentVisibleExtent.west = normalizeLongitude(currentGeoCenter.lon - 0.5 * lonSpan);
    currentVisibleExtent.east = normalizeLongitude(currentGeoCenter.lon + 0.5 * lonSpan);
    currentVisibleExtent.south = clampLatitude(currentGeoCenter.lat - 0.5 * latSpan);
    currentVisibleExtent.north = clampLatitude(currentGeoCenter.lat + 0.5 * latSpan);
    changed = true;
  }

  return changed;
}

function ecdfArrays(values) {
  const sorted = values
    .map(Number)
    .filter(x => Number.isFinite(x))
    .sort((a, b) => a - b);

  const n = sorted.length;
  const y = [];

  for (let i = 0; i < n; i++) {
    y.push((i + 1) / n);
  }

  return {x: sorted, y: y};
}

function initEcdfPanel() {
  const expSelect = document.getElementById("ecdf-exp-select");
  if (expSelect) {
    expSelect.innerHTML = "";

    const label = document.createElement("span");
    label.textContent = "Experiments:";
    expSelect.appendChild(label);

    for (const expver of EXPVERS) {
      const chip = document.createElement("label");
      chip.className = "ecdf-exp-chip";

      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.value = expver;
      cb.className = "ecdf-exp-checkbox";
      cb.checked = true;
      cb.addEventListener("change", function() {
        updateEcdfPanel();
        if (currentStationIndex !== null) {
          loadAndPlotStation(currentStationIndex);
        }
      });

      const marker = document.createElement("span");
      marker.style.color = EXPERIMENT_COLOURS[expver] || "black";
      marker.style.fontWeight = "bold";
      marker.textContent = "●";

      const name = document.createElement("span");
      name.textContent = expver;

      chip.appendChild(cb);
      chip.appendChild(marker);
      chip.appendChild(name);

      expSelect.appendChild(chip);
    }
  }

  const checkbox = document.getElementById("ecdf-common-checkbox");
  if (checkbox) {
    checkbox.checked = EXPVERS.length > 1;
    checkbox.disabled = EXPVERS.length <= 1;
    checkbox.addEventListener("change", function() {
      updateEcdfPanel();
    });
  }

  Plotly.newPlot(
    "ecdf-plot",
    [],
    {
      title: {text: "Empirical CDF in visible extent", x: 0.01, xanchor: "left"},
      margin: {l: 58, r: 15, t: 40, b: 50},
      xaxis: {title: SELECTED_METRIC, range: [-1, 1], automargin: true},
      yaxis: {title: "Empirical CDF", range: [0, 1], automargin: true},
      showlegend: true,
      legend: {orientation: "h", x: 0, y: -0.24, yanchor: "top"},
      paper_bgcolor: "white",
      plot_bgcolor: "white"
    },
    {
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ["select2d", "lasso2d"]
    }
  );

  ecdfInitialised = true;
}

function selectedEcdfExperiments() {
  const boxes = Array.from(document.querySelectorAll(".ecdf-exp-checkbox"));

  if (boxes.length === 0) {
    return EXPVERS.slice();
  }

  return boxes.filter(cb => cb.checked).map(cb => cb.value);
}

function updateEcdfPanel() {
  if (!ecdfInitialised) return;

  const diagnostics = document.getElementById("ecdf-diagnostics");
  const checkbox = document.getElementById("ecdf-common-checkbox");
  const activeExpvers = selectedEcdfExperiments();
  const useCommon = !!(checkbox && checkbox.checked && activeExpvers.length > 1);

  const stationsInExtent = STATIONS.filter(station => stationInsideExtent(station, currentVisibleExtent));

  const commonStations = useCommon
    ? stationsInExtent.filter(station => activeExpvers.every(expver => finiteMetricValue(station, expver) !== null))
    : [];

  const valuesByExp = {};
  const validCounts = {};
  const medians = {};
  const traces = [];

  for (const expver of activeExpvers) {
    const sample = useCommon
      ? commonStations.map(station => finiteMetricValue(station, expver))
      : stationsInExtent.map(station => finiteMetricValue(station, expver));

    const valid = sample.filter(v => Number.isFinite(v));
    valuesByExp[expver] = valid;
    validCounts[expver] = valid.length;
    medians[expver] = median(valid);

    if (valid.length > 0) {
      const ecdf = ecdfArrays(valid);
      traces.push({
        x: ecdf.x,
        y: ecdf.y,
        type: "scatter",
        mode: "lines",
        name: expver + " (n=" + valid.length + ")",
        line: {
          color: EXPERIMENT_COLOURS[expver] || undefined,
          width: 2.0
        },
        hovertemplate:
          expver +
          "<br>" + SELECTED_METRIC + "=%{x:.4f}" +
          "<br>F(x)=%{y:.3f}" +
          "<extra></extra>"
      });
    }
  }

  const commonCount = commonStations.length;
  const anyValid = traces.length > 0;

  let smallSampleWarning = false;
  if (useCommon) {
    smallSampleWarning = commonCount > 0 && commonCount < 10;
  } else {
    smallSampleWarning = activeExpvers.some(expver => validCounts[expver] > 0 && validCounts[expver] < 10);
  }

  const boundsText =
    "Visible bounds: " +
    "south=" + fmt(currentVisibleExtent.south, 3) + ", " +
    "west=" + fmt(currentVisibleExtent.west, 3) + ", " +
    "north=" + fmt(currentVisibleExtent.north, 3) + ", " +
    "east=" + fmt(currentVisibleExtent.east, 3);

  const countBits = [];
  for (const expver of activeExpvers) {
    countBits.push(esc(expver) + ": n=" + esc(validCounts[expver]));
  }

  const medianBits = [];
  for (const expver of activeExpvers) {
    medianBits.push(esc(expver) + ": " + esc(fmt(medians[expver], 4)));
  }

  const noExperimentSelected = activeExpvers.length === 0;

  let diagnosticsHtml =
    "<div><b>Stations inside extent:</b> " + esc(stationsInExtent.length) + "</div>" +
    "<div><b>Experiments shown:</b> " + esc(activeExpvers.join(", ")) + "</div>" +
    (useCommon
      ? "<div><b>Common-station count:</b> " + esc(commonCount) + "</div>"
      : "") +
    "<div><b>Valid counts by experiment:</b> " + (countBits.length ? countBits.join("; ") : "none") + "</div>" +
    "<div><b>Medians by experiment:</b> " + (medianBits.length ? medianBits.join("; ") : "none") + "</div>" +
    "<div><b>" + esc(boundsText) + "</b></div>";

  if (noExperimentSelected) {
    diagnosticsHtml += "<div class='warning'>Select at least one experiment to draw the eCDF.</div>";
  }

  if (!anyValid && !noExperimentSelected) {
    diagnosticsHtml += "<div class='warning'>No valid stations in the visible map extent.</div>";
  }

  if (smallSampleWarning) {
    diagnosticsHtml += "<div class='warning'>Small sample warning: fewer than 10 stations in the selected sample.</div>";
  }

  if (diagnostics) diagnostics.innerHTML = diagnosticsHtml;

  const layout = {
    title: {
      text: "Empirical CDF in visible extent" + (useCommon ? " (common stations)" : ""),
      x: 0.01,
      xanchor: "left"
    },
    margin: {l: 58, r: 15, t: 40, b: 50},
    xaxis: {title: SELECTED_METRIC, range: [-1, 1], automargin: true},
    yaxis: {title: "Empirical CDF", range: [0, 1], automargin: true},
    showlegend: true,
    legend: {orientation: "h", x: 0, y: -0.24, yanchor: "top"},
    paper_bgcolor: "white",
    plot_bgcolor: "white"
  };

  if (!anyValid) {
    layout.annotations = [{
      text: noExperimentSelected
        ? "Select at least one experiment to draw the eCDF."
        : "No valid stations in the visible map extent.",
      x: 0.5,
      y: 0.55,
      xref: "paper",
      yref: "paper",
      showarrow: false,
      font: {size: 13, color: "#555"}
    }];
  }

  Plotly.react("ecdf-plot", traces, layout, {
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ["select2d", "lasso2d"]
  });
}

function scheduleEcdfUpdate() {
  if (ecdfDebounceTimer !== null) {
    window.clearTimeout(ecdfDebounceTimer);
  }

  ecdfDebounceTimer = window.setTimeout(function() {
    ecdfDebounceTimer = null;
    updateEcdfPanel();
  }, ECDF_DEBOUNCE_MS);
}

function computeObsModelScaleDiagnostic(payloadsByExpver) {
  const obsPayload = chooseObsPayload(payloadsByExpver);
  if (!obsPayload) return null;

  let bestExpver = null;
  let bestPayload = null;

  for (const expver of EXPVERS) {
    if (payloadsByExpver[expver]) {
      bestExpver = expver;
      bestPayload = payloadsByExpver[expver];
      break;
    }
  }

  if (!bestPayload) return null;

  const modelByTime = new Map();
  for (let i = 0; i < bestPayload.time.length; i++) {
    const q = Number(bestPayload.model_discharge[i]);
    if (Number.isFinite(q) && q > 0) {
      modelByTime.set(bestPayload.time[i].slice(0, 10), q);
    }
  }

  const ratios = [];
  for (let i = 0; i < obsPayload.time.length; i++) {
    const day = obsPayload.time[i].slice(0, 10);
    const obs = Number(obsPayload.values[i]);
    const model = modelByTime.get(day);

    if (Number.isFinite(obs) && Number.isFinite(model) && obs > 0 && model > 0) {
      ratios.push(obs / model);
    }
  }

  if (ratios.length < 10) return null;

  return {
    expver: bestExpver,
    n: ratios.length,
    medianRatio: median(ratios)
  };
}

function zoomToPreset(preset) {
  if (!preset || !preset.lon || !preset.lat) {
    console.error("Invalid zoom preset:", preset);
    return;
  }

  const lon0 = Number(preset.lon[0]);
  const lon1 = Number(preset.lon[1]);
  const lat0 = Number(preset.lat[0]);
  const lat1 = Number(preset.lat[1]);

  const lonCenter = 0.5 * (lon0 + lon1);
  const latCenter = 0.5 * (lat0 + lat1);

  const update = {
    "geo.center.lon": lonCenter,
    "geo.center.lat": latCenter,
    "geo.lonaxis.autorange": false,
    "geo.lataxis.autorange": false,
    "geo.lonaxis.range": [lon0, lon1],
    "geo.lataxis.range": [lat0, lat1]
  };

  Plotly.relayout("map", update);

  const status = document.getElementById("zoom-status");
  if (status) {
    status.textContent =
      preset.name +
      "  lon [" + lon0 + ", " + lon1 + "]" +
      " lat [" + lat0 + ", " + lat1 + "]";
  }

  resizePlots();
  scheduleEcdfUpdate();
}

function zoomToStationBox(station, halfSpanDegrees = 5) {
  if (!station) return;

  const lat = clampLatitude(station.lat);
  const lon = normalizeLongitude(station.lon);
  if (lat === null || lon === null) return;

  const span = Number(halfSpanDegrees);
  if (!Number.isFinite(span) || span <= 0) return;

  const lat0 = clampLatitude(lat - span);
  const lat1 = clampLatitude(lat + span);
  const lon0 = Math.max(-180, lon - span);
  const lon1 = Math.min(180, lon + span);

  const update = {
    "geo.center.lon": lon,
    "geo.center.lat": lat,
    "geo.lonaxis.autorange": false,
    "geo.lataxis.autorange": false,
    "geo.lonaxis.range": [lon0, lon1],
    "geo.lataxis.range": [lat0, lat1]
  };

  Plotly.relayout("map", update);

  const status = document.getElementById("zoom-status");
  if (status) {
    status.textContent =
      "Station " + stationSearchId(station, 0) +
      "  lon [" + lon0.toFixed(2) + ", " + lon1.toFixed(2) + "]" +
      " lat [" + lat0.toFixed(2) + ", " + lat1.toFixed(2) + "]";
  }

  resizePlots();
  scheduleEcdfUpdate();
}

function updateSelectedStationMarker(station) {
  if (!mapDiv || !window.Plotly || !station) return;

  const lat = Number(station.lat);
  const lon = normalizeLongitude(station.lon);
  if (!Number.isFinite(lat) || lon === null) return;

  const targetTraces = [
    {
      type: "scattergeo",
      lon: [lon],
      lat: [lat],
      mode: "markers",
      hoverinfo: "skip",
      showlegend: false,
      marker: {
        symbol: "circle-open",
        size: 22,
        color: "rgb(255,0,0)",
        line: {color: "rgb(255,0,0)", width: 3}
      }
    },
    {
      type: "scattergeo",
      lon: [lon],
      lat: [lat],
      mode: "markers",
      hoverinfo: "skip",
      showlegend: false,
      marker: {
        symbol: "circle-open",
        size: 14,
        color: "rgb(255,0,0)",
        line: {color: "rgb(255,0,0)", width: 3}
      }
    },
    {
      type: "scattergeo",
      lon: [lon],
      lat: [lat],
      mode: "markers",
      hoverinfo: "skip",
      showlegend: false,
      marker: {
        symbol: "cross-open",
        size: 28,
        color: "rgb(255,0,0)",
        line: {color: "rgb(255,0,0)", width: 3}
      }
    }
  ];

  const haveValidIndices =
    Array.isArray(mapDiv.data) &&
    selectedStationTraceIndices.length === targetTraces.length &&
    selectedStationTraceIndices.every(function(idx) {
      return Number.isInteger(idx) && idx >= 0 && idx < mapDiv.data.length;
    });

  if (!haveValidIndices) {
    Plotly.addTraces("map", targetTraces)
      .then(function() {
        if (Array.isArray(mapDiv.data) && mapDiv.data.length >= targetTraces.length) {
          const end = mapDiv.data.length - 1;
          const start = end - targetTraces.length + 1;
          selectedStationTraceIndices = targetTraces.map(function(_, i) {
            return start + i;
          });
        }
      })
      .catch(function(err) {
        console.warn("Could not add selected-station marker", err);
      });
    return;
  }

  for (let j = 0; j < selectedStationTraceIndices.length; j++) {
    const idx = selectedStationTraceIndices[j];
    const trace = targetTraces[j];
    Plotly.restyle(
      "map",
      {
        lon: [trace.lon],
        lat: [trace.lat]
      },
      [idx]
    );
  }
}

function resetGlobalZoom() {
  const globalPreset = ZOOM_PRESETS.find(p => p.name === "Global");
  if (globalPreset) zoomToPreset(globalPreset);
}

function buildZoomSelect() {
  const select = document.getElementById("zoom-select");
  const applyButton = document.getElementById("zoom-apply");
  const globalButton = document.getElementById("zoom-global");

  if (!select) return;

  select.innerHTML = "";

  let currentGroup = null;
  let optgroup = null;

  for (let i = 0; i < ZOOM_PRESETS.length; i++) {
    const preset = ZOOM_PRESETS[i];

    if (preset.group !== currentGroup) {
      currentGroup = preset.group;
      optgroup = document.createElement("optgroup");
      optgroup.label = currentGroup;
      select.appendChild(optgroup);
    }

    const option = document.createElement("option");
    option.value = String(i);
    option.textContent = preset.name;
    option.title =
      "Lon " + preset.lon[0] + " to " + preset.lon[1] +
      ", Lat " + preset.lat[0] + " to " + preset.lat[1];

    optgroup.appendChild(option);
  }

  select.addEventListener("change", function(event) {
    event.preventDefault();
    const idx = Number(this.value);
    zoomToPreset(ZOOM_PRESETS[idx]);
  });

  if (applyButton) {
    applyButton.addEventListener("click", function(event) {
      event.preventDefault();
      const idx = Number(select.value);
      zoomToPreset(ZOOM_PRESETS[idx]);
    });
  }

  if (globalButton) {
    globalButton.addEventListener("click", function(event) {
      event.preventDefault();
      resetGlobalZoom();
    });
  }
}

function stationSearchLabel(station, index) {
  const sid = station.source_station_id !== undefined && station.source_station_id !== null
    ? String(station.source_station_id)
    : (station.station_id !== undefined && station.station_id !== null
      ? String(station.station_id)
      : String(index));

  const river = String(station.river || "").trim();
  const name = String(station.name || "").trim();

  return sid + " | " + river + " | " + name;
}

function stationSearchId(station, index) {
  if (station.source_station_id !== undefined && station.source_station_id !== null) {
    return String(station.source_station_id);
  }
  if (station.station_id !== undefined && station.station_id !== null) {
    return String(station.station_id);
  }
  return String(index);
}

function buildStationSelector() {
  const input = document.getElementById("station-search-input");
  const list = document.getElementById("station-search-list");
  const button = document.getElementById("station-search-go");

  if (!input || !list || !button) return;

  function computeStationMatches(term) {
    const scored = [];

    for (let i = 0; i < STATIONS.length; i++) {
      const station = STATIONS[i];
      const sid = stationSearchId(station, i).toLowerCase();
      const river = String(station.river || "").toLowerCase();
      const name = String(station.name || "").toLowerCase();

      let score = null;
      if (sid === term) {
        score = 0;
      } else if (sid.indexOf(term) >= 0) {
        score = 1;
      } else if (river.indexOf(term) >= 0) {
        score = 2;
      } else if (name.indexOf(term) >= 0) {
        score = 3;
      }

      if (score !== null) {
        scored.push({
          index: i,
          score: score,
          label: stationSearchLabel(station, i)
        });
      }
    }

    scored.sort(function(a, b) {
      if (a.score !== b.score) return a.score - b.score;
      return a.label.localeCompare(b.label);
    });

    return scored.slice(0, STATION_SUGGESTION_LIMIT);
  }

  function updateSuggestions() {
    const term = String(input.value || "").trim().toLowerCase();
    stationSearchMatches = [];
    list.innerHTML = "";

    if (!term) return;

    stationSearchMatches = computeStationMatches(term);

    for (const row of stationSearchMatches) {
      const option = document.createElement("option");
      option.value = row.label;
      list.appendChild(option);
    }
  }

  function loadFromSearchInput() {
    const raw = String(input.value || "").trim();
    const term = raw.toLowerCase();
    if (!term) return;

    // Recompute from current input so the Load button is not dependent on stale suggestions.
    const matches = computeStationMatches(term);

    let match = null;

    for (const row of matches) {
      const station = STATIONS[row.index];
      const sid = stationSearchId(station, row.index).toLowerCase();
      const label = row.label.toLowerCase();

      if (sid === term || label === term) {
        match = row;
        break;
      }
    }

    // Also support selecting just the ID part of a datalist label: "<id> | ...".
    if (!match) {
      const idOnly = term.split("|")[0].trim();
      for (let i = 0; i < STATIONS.length; i++) {
        const sid = stationSearchId(STATIONS[i], i).toLowerCase();
        if (sid === idOnly) {
          match = {index: i, label: stationSearchLabel(STATIONS[i], i)};
          break;
        }
      }
    }

    if (!match && matches.length > 0) {
      match = matches[0];
    }

    if (match) {
      input.value = match.label;
      stationSearchMatches = matches;
      loadAndPlotStation(match.index, {zoomOnLoad: true});
    }
  }

  input.addEventListener("input", updateSuggestions);
  input.addEventListener("change", loadFromSearchInput);
  input.addEventListener("keydown", function(event) {
    if (event.key === "Enter") {
      event.preventDefault();
      loadFromSearchInput();
    }
  });

  button.addEventListener("click", function(event) {
    event.preventDefault();
    loadFromSearchInput();
  });
}

function sortedRuns(station) {
  const rows = [];

  for (const expver of EXPVERS) {
    const run = station.runs[expver];

    if (!run) {
      rows.push({expver: expver, run: null, value: null});
      continue;
    }

    const value = run[SELECTED_METRIC];
    rows.push({
      expver: expver,
      run: run,
      value: value === null || value === undefined ? null : Number(value)
    });
  }

  rows.sort(function(a, b) {
    if (a.value === null && b.value === null) return 0;
    if (a.value === null) return 1;
    if (b.value === null) return -1;
    return b.value - a.value;
  });

  return rows;
}

function stationInfo(station, payloadsByExpver) {
  let s = "<b>" + esc(station.name || "Station") + "</b><br>";

  if (station.best_expver) {
    s += "<span class='badge'>Best " + esc(SELECTED_METRIC) + ": " +
         esc(station.best_expver) + " = " + esc(fmt(station.best_metric)) +
         "</span>";
  }

  if (station.control_expver && station.control_metric !== null && station.control_metric !== undefined) {
    s += "<span class='badge'>Control " + esc(station.control_expver) + ": " +
         esc(fmt(station.control_metric)) + "</span>";

    if (station.best_improvement_over_control !== null && station.best_improvement_over_control !== undefined) {
      const imp = Number(station.best_improvement_over_control);
      s += "<span class='badge'>Best-control Δ" + esc(SELECTED_METRIC) + ": " +
           esc(fmtSigned(imp)) + "</span>";

      if (imp < BEST_EXPERIMENT_MIN_IMPROVEMENT) {
        s += "<br><span class='warning'>No experiment is meaningfully better than control " +
             esc(station.control_expver) + " at threshold " +
             esc(BEST_EXPERIMENT_MIN_IMPROVEMENT) + " " + esc(SELECTED_METRIC) + ".</span>";
      }
    }
  }

  if (station.metric_difference !== null && station.metric_difference !== undefined) {
    s += "<span class='badge'>Difference ";
    s += esc(station.difference_target_expver) + " - ";
    s += esc(station.difference_reference_expver) + ": ";
    s += esc(fmtSigned(station.metric_difference)) + "</span>";
  }

  const diag = computeObsModelScaleDiagnostic(payloadsByExpver);
  if (diag) {
    s += "<br><span class='badge'>Obs/model median ratio: " +
         esc(fmt(diag.medianRatio, 3)) +
         " using " + esc(diag.expver) +
         ", n=" + esc(diag.n) + "</span>";

    if (diag.medianRatio < WARN_SCALE_LOW || diag.medianRatio > WARN_SCALE_HIGH) {
      s += "<br><span class='warning'>Possible scale, unit, or basin-area mismatch: " +
           "obs/model median ratio is " + esc(fmt(diag.medianRatio, 3)) +
           ". Check model upstream area and station matching before applying any global conversion.</span>";
    }
  }

  s += "<table>";
  s += "<tr><th>Country</th><td>" + esc(station.country_code) + "</td></tr>";
  s += "<tr><th>River</th><td>" + esc(station.river) + "</td></tr>";
  s += "<tr><th>Obs lon/lat</th><td>" + esc(station.lon) + ", " + esc(station.lat) + "</td></tr>";
  s += "<tr><th>Reference model lon/lat</th><td>" + esc(station.model_lon) + ", " + esc(station.model_lat) + "</td></tr>";
  s += "<tr><th>Station-model distance</th><td>" + esc(station.distance_station_to_model_km) + " km</td></tr>";
  s += "<tr><th>Station upstream area</th><td>" + esc(station.upstream_area_km2) + " km²</td></tr>";
  s += "</table>";

  s += "<b>Experiment performance, sorted by " + esc(SELECTED_METRIC) + "</b>";
  s += "<table>";
  s += "<tr><th>Experiment</th><th>KGE</th><th>r</th><th>RMSE</th><th>Days</th><th>Model area km²</th><th>Area diff %</th></tr>";

  const rows = sortedRuns(station);

  for (const item of rows) {
    const expver = item.expver;
    const run = item.run;

    if (!run) {
      s += "<tr><td>" + esc(expver) + "</td><td colspan='6'>No station</td></tr>";
      continue;
    }

    const colour = EXPERIMENT_COLOURS[expver] || "black";
    const bestMark = expver === station.best_expver ? " ★" : "";

    s += "<tr>";
    s += "<td><span style='color:" + esc(colour) + ";font-weight:bold;'>●</span> " +
         esc(expver) + esc(bestMark) + "</td>";
    s += "<td>" + esc(fmt(run.kge)) + "</td>";
    s += "<td>" + esc(fmt(run.correlation)) + "</td>";
    s += "<td>" + esc(fmt(run.rmse)) + "</td>";
    s += "<td>" + esc(run.matched_days) + "</td>";
    s += "<td>" + esc(run.model_upstream_area_km2) + "</td>";
    s += "<td>" + esc(run.upstream_area_difference_pct) + "</td>";
    s += "</tr>";
  }

  s += "</table>";

  for (const expver of EXPVERS) {
    const payload = payloadsByExpver[expver];

    if (payload && payload.obs) {
      const obsPath = sanitizeUserPath(payload.obs.file);
      s += "<div class='small'><b>Obs file:</b><br><span class='code'>" +
           esc(obsPath) + "</span></div>";
      break;
    }
  }

  return s;
}

function combineAbortSignals(primarySignal, timeoutSignal) {
  if (!primarySignal) return timeoutSignal;
  if (!timeoutSignal) return primarySignal;

  const controller = new AbortController();

  const onAbort = function() {
    if (!controller.signal.aborted) {
      controller.abort();
    }
  };

  if (primarySignal.aborted || timeoutSignal.aborted) {
    controller.abort();
  } else {
    primarySignal.addEventListener("abort", onAbort, {once: true});
    timeoutSignal.addEventListener("abort", onAbort, {once: true});
  }

  return controller.signal;
}

async function fetchStationPayload(expver, run, signal) {
  if (!run || !run.json_path) return null;

  const cacheKey = String(run.json_path);
  if (stationPayloadCache.has(cacheKey)) {
    return stationPayloadCache.get(cacheKey);
  }

  const url = DATA_ROOT + "/" + run.json_path;
  const timeoutController = new AbortController();
  const timeoutId = window.setTimeout(function() {
    timeoutController.abort();
  }, STATION_FETCH_TIMEOUT_MS);

  const requestSignal = combineAbortSignals(signal, timeoutController.signal);

  let response;
  try {
    response = await fetch(url, {signal: requestSignal});
  } catch (err) {
    if (timeoutController.signal.aborted) {
      throw new Error("Timed out loading " + url);
    }
    throw err;
  } finally {
    window.clearTimeout(timeoutId);
  }

  if (!response.ok) {
    throw new Error("Could not load " + url);
  }

  const payload = await response.json();
  stationPayloadCache.set(cacheKey, payload);
  return payload;
}

function chooseObsPayload(payloadsByExpver, expvers = EXPVERS) {
  for (const expver of expvers) {
    const payload = payloadsByExpver[expver];
    if (payload && payload.obs) return payload.obs;
  }
  return null;
}

function normaliseSeries(timeValues, dischargeValues) {
  if (!Array.isArray(timeValues) || !Array.isArray(dischargeValues)) {
    return {time: [], values: []};
  }

  // Some payloads can contain repeated cycles of timestamps; keep the last
  // value per timestamp and then sort chronologically for stable plotting.
  const byTimestamp = new Map();

  const n = Math.min(timeValues.length, dischargeValues.length);
  for (let i = 0; i < n; i++) {
    const ts = String(timeValues[i]);
    const value = Number(dischargeValues[i]);

    if (!Number.isFinite(value)) continue;
    byTimestamp.set(ts, value);
  }

  const sortedTime = Array.from(byTimestamp.keys()).sort();
  const sortedValues = sortedTime.map(ts => byTimestamp.get(ts));

  return {time: sortedTime, values: sortedValues};
}

async function loadAndPlotStation(i, options = {}) {
  currentStationRequestId += 1;
  const requestId = currentStationRequestId;

  if (activeStationAbortController) {
    activeStationAbortController.abort();
  }

  const abortController = new AbortController();
  activeStationAbortController = abortController;

  currentStationIndex = i;

  const station = STATIONS[i];

  if (!station) {
    document.getElementById("info").innerHTML = "<b>Error</b><br>Station not found.";
    return;
  }

  updateSelectedStationMarker(station);

  if (options.zoomOnLoad) {
    zoomToStationBox(station, 5);
  }

  const hydroDiv = document.getElementById("hydrograph");
  const hadExistingPlot = !!(hydroDiv && hydroDiv.data && hydroDiv.data.length > 0);

  if (!hadExistingPlot) {
    hydroDiv.innerHTML =
      "<p style='padding:12px;'>Loading " + esc(station.name) + "...</p>";
  }

  const payloadsByExpver = {};
  const errors = [];

  const fetchResults = await Promise.all(EXPVERS.map(async function(expver) {
    const run = station.runs[expver];

    if (!run) {
      return {expver: expver, payload: null, error: null};
    }

    try {
      const payload = await fetchStationPayload(expver, run, abortController.signal);
      return {expver: expver, payload: payload, error: null};
    } catch (err) {
      if (err && err.name === "AbortError") {
        return {expver: expver, payload: null, error: null};
      }
      return {expver: expver, payload: null, error: err};
    }
  }));

  if (requestId !== currentStationRequestId) {
    return;
  }

  for (const item of fetchResults) {
    payloadsByExpver[item.expver] = item.payload;
    if (item.error) {
      console.error(item.error);
      errors.push(String(item.error));
    }
  }

  const traces = [];
  const activeExpvers = selectedEcdfExperiments();

  for (const expver of activeExpvers) {
    const payload = payloadsByExpver[expver];
    if (!payload) continue;

    const run = station.runs[expver];
    const series = normaliseSeries(payload.time, payload.model_discharge);
    if (series.time.length === 0) continue;

    traces.push({
      x: series.time,
      y: series.values,
      type: "scatter",
      mode: "lines",
      name: "Model " + expver,
      yaxis: "y",
      line: {
        color: EXPERIMENT_COLOURS[expver] || undefined,
        width: expver === station.best_expver ? 3 : 1.8
      },
      hovertemplate:
        "%{x}<br>" + expver + " Q=%{y} m³/s" +
        "<br>KGE=" + fmt(run.kge) +
        "<br>r=" + fmt(run.correlation) +
        "<extra></extra>"
    });
  }

  const obs = activeExpvers.length > 0
    ? chooseObsPayload(payloadsByExpver, activeExpvers)
    : null;

  if (obs) {
    const obsSeries = normaliseSeries(obs.time, obs.values);
    traces.push({
      x: obsSeries.time,
      y: obsSeries.values,
      type: "scatter",
      mode: "lines+markers",
      name: "Observed streamflow",
      yaxis: "y",
      line: {dash: "dot", color: "black", width: 2.2},
      marker: {size: 4, color: "black"},
      hovertemplate: "%{x}<br>Obs Q=%{y} m³/s<extra></extra>"
    });
  }

  if (traces.length === 0) {
    if (activeExpvers.length === 0) {
      hydroDiv.innerHTML =
        "<p style='padding:12px;color:#444;'>Select at least one experiment to draw hydrographs.</p>";
    } else {
      hydroDiv.innerHTML =
        "<p style='padding:12px;color:red;'>No hydrograph could be loaded.</p>";
    }
  } else {
    // Remove stale loading paragraph before drawing updated traces.
    const layout = {
      title: {
        text: "Hydrograph comparison: " + station.name,
        x: 0.02,
        xanchor: "left",
        y: 0.96,
        yanchor: "top"
      },
      margin: {l: 78, r: 35, t: 70, b: 82},
      xaxis: {
        title: {text: "Valid time", standoff: 10},
        type: "date",
        automargin: true
      },
      yaxis: {
        title: {text: "River discharge, m³/s", standoff: 14},
        rangemode: "tozero",
        automargin: true
      },
      legend: {
        orientation: "h",
        x: 0,
        y: -0.24,
        yanchor: "top"
      }
    };

    const config = {
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ["select2d", "lasso2d"]
    };

    try {
      if (!hadExistingPlot) {
        hydroDiv.innerHTML = "";
        Plotly.newPlot("hydrograph", traces, layout, config);
      } else {
        Plotly.react("hydrograph", traces, layout, config);
      }
    } catch (err) {
      console.warn("Plotly.react failed for hydrograph; retrying with newPlot", err);
      hydroDiv.innerHTML = "";
      Plotly.newPlot("hydrograph", traces, layout, config);
    }
  }

  let infoHtml = stationInfo(station, payloadsByExpver);

  if (errors.length > 0) {
    infoHtml += "<br><b>Loading warnings</b><br><span class='small'>" +
                esc(errors.join("\\n")) + "</span>";
  }

  document.getElementById("info").innerHTML = infoHtml;
  resizePlots();
}

function resizePlots() {
  const map = document.getElementById("map");
  const mapWrap = document.getElementById("map-wrap");
  const hydro = document.getElementById("hydrograph");
  const ecdf = document.getElementById("ecdf-plot");

  if (map && mapWrap && window.Plotly) {
    const w = Math.max(300, Math.floor(mapWrap.clientWidth));
    const h = Math.max(300, Math.floor(mapWrap.clientHeight));

    map.style.width = w + "px";
    map.style.height = h + "px";

    Plotly.relayout(map, {width: w, height: h});
    Plotly.Plots.resize(map);
  }

  if (hydro && hydro.data && window.Plotly) {
    Plotly.Plots.resize(hydro);
  }

  if (ecdf && ecdf.data && window.Plotly) {
    Plotly.Plots.resize(ecdf);
  }
}

function scheduleResize() {
  if (resizeDebounceTimer !== null) {
    window.clearTimeout(resizeDebounceTimer);
  }

  resizeDebounceTimer = window.setTimeout(function() {
    resizeDebounceTimer = null;
    resizePlots();
  }, 120);
}

buildZoomSelect();
buildStationSelector();
initEcdfPanel();
initEnvironmentWarning();

currentVisibleExtent = deriveInitialExtentFromMap();
scheduleEcdfUpdate();

window.addEventListener("resize", function() {
  scheduleResize();
});

window.setTimeout(scheduleResize, 200);

mapDiv.on("plotly_relayout", function(eventData) {
  updateExtentCacheFromRelayout(eventData || {});
  scheduleEcdfUpdate();
});

mapDiv.on("plotly_click", function(data) {
  if (!data || !data.points || data.points.length === 0) return;

  const pt = data.points[0];

  if (pt.customdata === undefined || pt.customdata === null) return;

  const i = Number(pt.customdata);
  if (!Number.isInteger(i)) return;

  loadAndPlotStation(i);
});
</script>
</body>
</html>
"""

    html = (
        html_template
        .replace("__MAP_DIV__", map_div)
        .replace("__RECORDS_JSON__", records_json)
        .replace("__EXPVERS_JSON__", expvers_json)
        .replace("__ZOOM_PRESETS_JSON__", zoom_presets_json)
        .replace("__PALETTE_JSON__", palette_json)
        .replace("__DATA_ROOT__", data_root_js)
        .replace("__METRIC__", args.metric)
        .replace("__COLOUR_MODE__", args.colour_mode)
        .replace("__CONTROL_EXPVER__", str(args.control_expver))
        .replace("__BEST_EXPERIMENT_MIN_IMPROVEMENT__", str(float(args.best_experiment_min_improvement)))
        .replace("__MAP_HEIGHT_VH__", str(float(args.map_height_vh)))
        .replace("__WARN_SCALE_LOW__", str(float(args.warn_scale_ratio_low)))
        .replace("__WARN_SCALE_HIGH__", str(float(args.warn_scale_ratio_high)))
    )

    output_html.write_text(html, encoding="utf-8")

    print(f"Saved: {output_html}", flush=True)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    args = parse_args()

    if args.control_expver is None:
        args.control_expver = args.expver[0]

    if args.control_expver not in args.expver:
        raise ValueError(
            f"--control-expver {args.control_expver!r} is not present in --expver: "
            + ", ".join(args.expver)
        )

    if args.colour_mode == "experiment_difference" and len(args.expver) != 2:
        raise ValueError("--colour-mode experiment_difference requires exactly two experiments.")

    if args.colour_mode == "best_experiment" and len(args.expver) < 2:
        print("WARNING: --colour-mode best_experiment is more useful with two or more experiments.", flush=True)

    if args.background != "standard":
        raise ValueError("Only --background standard is currently supported in this Plotly geo dashboard.")

    label = run_label(args.date_start, args.date_end, args.resolution)

    if args.output_html is None:
      output_html = args.output_dir / default_output_name(
            expvers=args.expver,
            date_start=args.date_start,
            date_end=args.date_end,
            resolution=args.resolution,
            metric=args.metric,
            colour_mode=args.colour_mode,
        )
    else:
        output_html = args.output_html

    output_html.parent.mkdir(parents=True, exist_ok=True)

    print("")
    print("============================================================")
    print("Build multi-experiment river discharge dashboard")
    print("============================================================")
    print(f"Experiments        : {', '.join(args.expver)}")
    print(f"Date range         : {args.date_start} to {args.date_end}")
    print(f"Resolution         : {args.resolution} arcmin")
    print(f"Metric             : {args.metric}")
    print(f"Colour mode        : {args.colour_mode}")
    print(f"Control experiment : {args.control_expver}")
    print(f"Best-exp threshold : {args.best_experiment_min_improvement:g} {args.metric}")
    print(f"Projection         : {args.projection}")
    print(f"Map height         : {args.map_height_vh} vh")
    print(f"Best exp threshold : {args.best_experiment_min_improvement}")
    print(f"Difference threshold: {args.difference_threshold}")
    print(f"Data root          : {args.data_root}")
    print(f"Input label        : {label}")
    print(f"Output HTML        : {output_html}")
    print("============================================================")
    print("")

    records = merge_catalogues(
        data_root=args.data_root,
        expvers=args.expver,
        label=label,
        metric=args.metric,
        control_expver=args.control_expver,
    )

    fig = build_map(records=records, args=args)

    build_html(
        records=records,
        fig=fig,
        args=args,
        output_html=output_html,
    )


if __name__ == "__main__":
    main()
