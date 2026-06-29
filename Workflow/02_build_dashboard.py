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
    "rgb(0,100,0)",        # darkgreen
    "rgb(65,105,225)",     # royalblue
    "rgb(255,140,0)",      # darkorange
    "rgb(128,0,128)",      # purple
    "rgb(0,128,128)",      # teal
    "rgb(165,42,42)",      # brown
    "rgb(255,20,147)",     # deeppink
    "rgb(128,128,0)",      # olive
    "rgb(0,0,128)",        # navy
    "rgb(220,20,60)",      # crimson
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

    best_metric = record.get("best_metric")
    best_expver = record.get("best_expver")
    metric_difference = record.get("metric_difference")

    metric_text = "NA" if best_metric is None else f"{best_metric:.3f}"
    diff_text = "NA" if metric_difference is None else f"{metric_difference:+.3f}"

    run_bits = []
    for expver in args.expver:
        run = record["runs"].get(expver)
        if not run:
            run_bits.append(f"{expver}: no station")
            continue

        v = run.get(metric)
        run_bits.append(f"{expver}: NA" if v is None else f"{expver}: {v}")

    text = (
        f"{record.get('name', 'Station')}"
        f"<br>Country: {record.get('country_code', '')}"
        f"<br>River: {record.get('river', '')}"
        f"<br>Best {metric}: {metric_text}"
        f"<br>Best experiment: {best_expver}"
        f"<br>{metric} by experiment: {'; '.join(run_bits)}"
    )

    if len(args.expver) >= 2:
        text += f"<br>Difference ({args.expver[1]} - {args.expver[0]}): {diff_text}"

    return text


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
        include_plotlyjs=True,
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

    records_json = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    expvers_json = json.dumps(args.expver, ensure_ascii=False, separators=(",", ":"))
    zoom_presets_json = json.dumps(ZOOM_PRESETS, ensure_ascii=False, separators=(",", ":"))
    palette_json = json.dumps(experiment_palette(args.expver), ensure_ascii=False, separators=(",", ":"))

    data_root_js = str(args.data_root).replace("\\", "/")

    html_template = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Global station hydrograph dashboard</title>
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

#top-toolbar {
  display: flex;
  align-items: center;
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
  min-width: 280px;
  max-width: 460px;
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

#map-wrap {
  flex: 0 0 __MAP_HEIGHT_VH__vh;
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
</style>
</head>
<body>
<div id="container">

  <div id="top-toolbar">
    <div class="toolbar-title">Zoom preset</div>
    <select id="zoom-select"></select>
    <button id="zoom-apply" type="button">Go</button>
    <button id="zoom-global" type="button">Global</button>
    <span id="zoom-status"></span>
  </div>

  <div id="map-wrap">
    __MAP_DIV__
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
      s += "<div class='small'><b>Obs file:</b><br><span class='code'>" +
           esc(payload.obs.file) + "</span></div>";
      break;
    }
  }

  return s;
}

async function fetchStationPayload(expver, run) {
  if (!run || !run.json_path) return null;

  const url = DATA_ROOT + "/" + run.json_path;
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error("Could not load " + url);
  }

  return await response.json();
}

function chooseObsPayload(payloadsByExpver) {
  for (const expver of EXPVERS) {
    const payload = payloadsByExpver[expver];
    if (payload && payload.obs) return payload.obs;
  }
  return null;
}

async function loadAndPlotStation(i) {
  const station = STATIONS[i];

  if (!station) {
    document.getElementById("info").innerHTML = "<b>Error</b><br>Station not found.";
    return;
  }

  document.getElementById("hydrograph").innerHTML =
    "<p style='padding:12px;'>Loading " + esc(station.name) + "...</p>";

  const payloadsByExpver = {};
  const errors = [];

  for (const expver of EXPVERS) {
    const run = station.runs[expver];

    if (!run) {
      payloadsByExpver[expver] = null;
      continue;
    }

    try {
      payloadsByExpver[expver] = await fetchStationPayload(expver, run);
    } catch (err) {
      console.error(err);
      errors.push(String(err));
      payloadsByExpver[expver] = null;
    }
  }

  const traces = [];

  for (const expver of EXPVERS) {
    const payload = payloadsByExpver[expver];
    if (!payload) continue;

    const run = station.runs[expver];

    traces.push({
      x: payload.time,
      y: payload.model_discharge,
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

  const obs = chooseObsPayload(payloadsByExpver);

  if (obs) {
    traces.push({
      x: obs.time,
      y: obs.values,
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
    document.getElementById("hydrograph").innerHTML =
      "<p style='padding:12px;color:red;'>No hydrograph could be loaded.</p>";
  } else {
    Plotly.newPlot(
      "hydrograph",
      traces,
      {
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
      },
      {
        responsive: true,
        displaylogo: false,
        modeBarButtonsToRemove: ["select2d", "lasso2d"]
      }
    );
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
}

buildZoomSelect();

window.addEventListener("resize", function() {
  window.setTimeout(resizePlots, 80);
});

window.setTimeout(resizePlots, 200);

const mapDiv = document.getElementById("map");

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
        output_html = default_output_name(
            expvers=args.expver,
            date_start=args.date_start,
            date_end=args.date_end,
            resolution=args.resolution,
            metric=args.metric,
            colour_mode=args.colour_mode,
        )
    else:
        output_html = args.output_html

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
