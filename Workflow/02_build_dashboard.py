#!/usr/bin/env python3
"""
02_build_dashboard.py

Build an interactive global river-discharge benchmarking dashboard.

This script is the third step of the ifs-riverbench workflow. It reads the
station catalogue and per-station JSON files produced by
`01_extract_hydrographs.py`, creates an interactive global Plotly map, and
embeds JavaScript logic to load and display hydrographs when stations are
clicked.

Workflow position
-----------------
1. 00_extract_rivers_mars.py
   Retrieve monthly global river discharge GRIB files from MARS.

2. 01_extract_hydrographs.py
   Extract station hydrographs, match observations, compute metrics,
   and write dashboard-ready JSON/CSV files.

3. 02_build_dashboard.py
   Build the standalone interactive HTML dashboard.

Main inputs
-----------
dashboard_data/
├── stations_catalog.json
└── stations/
    ├── station_000000.json
    ├── station_000001.json
    └── ...

Main output
-----------
global_station_dashboard_<metric>.html

where <metric> is defined by C_MODE, for example:
- global_station_dashboard_kge.html
- global_station_dashboard_correlation.html

Dashboard behaviour
-------------------
- Stations are coloured by either KGE or correlation.
- Natural Earth rivers are plotted as a contextual background layer.
- Clicking a station loads the corresponding station JSON file.
- The lower panel displays the model and observed hydrograph.
- The side panel displays station metadata and skill metrics.

Requirements
------------
Python packages:
- numpy
- plotly
- cartopy

The produced HTML file should be viewed through a local web server, for example:

python -m http.server 8000

Then open:

http://127.0.0.1:8000/global_station_dashboard_correlation.html

This is needed because most browsers block local JavaScript fetch() calls when
opening HTML files directly with file://.
"""

from pathlib import Path
import json

import numpy as np
import plotly.graph_objects as go

import cartopy.io.shapereader as shpreader


# ------------------------------------------------------------
# Station colouring / clustering mode
# ------------------------------------------------------------
# Choose which skill metric controls the station colours.
#
# Available options:
#   C_MODE = "kge"
#   C_MODE = "correlation"
#
# Keep only one active assignment.
# C_MODE = "kge"
C_MODE = "correlation"


# ------------------------------------------------------------
# Input and output paths
# ------------------------------------------------------------
# Directory produced by 01_extract_hydrographs.py.
DATA_DIR = Path("dashboard_data")

# Catalogue containing one compact metadata record per station shown on the map.
CATALOG_JSON = DATA_DIR / "stations_catalog.json"

# Standalone HTML dashboard output.
OUT_HTML = Path("global_station_dashboard_" + C_MODE + ".html")


# ------------------------------------------------------------
# Load station catalogue
# ------------------------------------------------------------
catalog = json.loads(CATALOG_JSON.read_text(encoding="utf-8"))

print(f"Loaded station catalogue: {len(catalog):,}")

n_total = len(catalog)

n_with_obs = sum(
    s.get("matched_days") is not None and s.get("matched_days", 0) > 0
    for s in catalog
)

n_with_kge = sum(s.get("kge") is not None for s in catalog)
n_with_corr = sum(s.get("correlation") is not None for s in catalog)

print(f"Stations in dashboard: {n_total:,}")
print(f"Stations with valid observations/metrics: {n_with_obs:,}")
print(f"Stations with KGE: {n_with_kge:,}")
print(f"Stations with correlation: {n_with_corr:,}")


# ------------------------------------------------------------
# Create Plotly map figure
# ------------------------------------------------------------
fig = go.Figure()


def add_river_layer(
    resolution="50m",
    max_scalerank=6,
    color="rgb(60,120,180)",
    width=0.7,
    opacity=0.55,
):
    """
    Add global rivers from Natural Earth.

    Parameters
    ----------
    resolution : str
        Natural Earth resolution:
        - "110m": very light
        - "50m": good compromise
        - "10m": detailed but heavier

    max_scalerank : int
        Maximum Natural Earth scalerank to include.
        Lower values correspond to larger / more important rivers.

    color, width, opacity
        Plotly line styling for the river layer.
    """

    shp = shpreader.natural_earth(
        resolution=resolution,
        category="physical",
        name="rivers_lake_centerlines",
    )

    # BasicReader avoids FionaRecord._shape issues on some installations.
    reader = shpreader.BasicReader(shp)

    lon_all = []
    lat_all = []
    n_rivers = 0

    for rec in reader.records():
        scalerank = rec.attributes.get("scalerank", 99)

        # Skip small rivers when scalerank is above the requested threshold.
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

            # None separates independent line segments in Plotly.
            lon_all.append(None)
            lat_all.append(None)

            n_rivers += 1

    fig.add_trace(
        go.Scattergeo(
            lon=lon_all,
            lat=lat_all,
            mode="lines",
            line=dict(color=color, width=width),
            opacity=opacity,
            name=f"Rivers ({resolution}, rank ≤ {max_scalerank})",
            hoverinfo="skip",
            showlegend=True,
        )
    )

    print(f"Added river layer: {n_rivers:,} river segments")


def add_station_layer(mask_func, color, name, size=5, opacity=0.85):
    """
    Add one station marker layer to the map.

    Stations are filtered using mask_func. This allows each metric class
    to be added as a separate Plotly layer and legend entry.
    """

    subset = [
        (idx, s)
        for idx, s in enumerate(catalog)
        if mask_func(s)
    ]

    if len(subset) == 0:
        return

    lon = [s["lon"] for idx, s in subset]
    lat = [s["lat"] for idx, s in subset]

    # Important:
    # customdata stores the station position in the catalogue list, not the
    # original station_index. This is what the JavaScript click handler uses.
    customdata = [idx for idx, s in subset]

    text = []

    for idx, s in subset:
        text.append(
            f"{s.get('name', 'Station')}"
            f"<br>Country: {s.get('country_code', '')}"
            f"<br>River: {s.get('river', '')}"
            f"<br>KGE: {s.get('kge')}"
            f"<br>r: {s.get('correlation')}"
            f"<br>RMSE: {s.get('rmse')} m³/s"
        )

    fig.add_trace(
        go.Scattergeo(
            lon=lon,
            lat=lat,
            mode="markers",
            text=text,
            customdata=customdata,
            hovertemplate="%{text}<extra></extra>",
            marker=dict(
                size=size,
                color=color,
                opacity=opacity,
                line=dict(width=0.8, color="black"),
            ),
            name=f"{name} ({len(subset):,})",
        )
    )


# ------------------------------------------------------------
# KGE classes
# ------------------------------------------------------------
def no_kge(s):
    return s.get("kge") is None


def kge_lt_minus041(s):
    return s.get("kge") is not None and s["kge"] < -0.41


def kge_minus041_to_0(s):
    return s.get("kge") is not None and -0.41 <= s["kge"] <= 0.0


def kge_0_to_05(s):
    return s.get("kge") is not None and 0.0 < s["kge"] <= 0.5


def kge_gt_05(s):
    return s.get("kge") is not None and s["kge"] > 0.5


# ------------------------------------------------------------
# Correlation classes
# ------------------------------------------------------------
def no_corr(s):
    return s.get("correlation") is None


def corr_lt_03(s):
    return s.get("correlation") is not None and s["correlation"] < 0.3


def corr_03_to_05(s):
    return s.get("correlation") is not None and 0.3 <= s["correlation"] <= 0.5


def corr_05_to_07(s):
    return s.get("correlation") is not None and 0.5 < s["correlation"] <= 0.7


def corr_gt_07(s):
    return s.get("correlation") is not None and s["correlation"] > 0.7


# ------------------------------------------------------------
# Add map layers
# ------------------------------------------------------------
# Add rivers first so that station markers are plotted on top.
add_river_layer(
    resolution="50m",
    max_scalerank=6,
    color="rgb(60,120,180)",
    width=0.7,
    opacity=0.55,
)


# ------------------------------------------------------------
# Station clustering and colouring
# ------------------------------------------------------------
if C_MODE.lower() == "kge":

    add_station_layer(no_kge, "lightgrey", "No KGE / no obs", size=5, opacity=0.35)
    add_station_layer(kge_lt_minus041, "red", "KGE < -0.41", size=7)
    add_station_layer(kge_minus041_to_0, "gold", "-0.41 ≤ KGE ≤ 0", size=7)
    add_station_layer(kge_0_to_05, "limegreen", "0 < KGE ≤ 0.5", size=7)
    add_station_layer(kge_gt_05, "darkgreen", "KGE > 0.5", size=8)

    colour_title = "Stations coloured by KGE"

elif C_MODE.lower() == "correlation":

    add_station_layer(no_corr, "lightgrey", "No correlation / no obs", size=5, opacity=0.35)
    add_station_layer(corr_lt_03, "red", "Correlation < 0.3", size=7)
    add_station_layer(corr_03_to_05, "gold", "0.3 ≤ Correlation ≤ 0.5", size=7)
    add_station_layer(corr_05_to_07, "limegreen", "0.5 < Correlation ≤ 0.7", size=7)
    add_station_layer(corr_gt_07, "darkgreen", "Correlation > 0.7", size=8)

    colour_title = "Stations coloured by Correlation"

else:
    raise ValueError(f"Unsupported C_MODE={C_MODE}")


# ------------------------------------------------------------
# Map appearance
# ------------------------------------------------------------
fig.update_geos(
    projection_type="natural earth",
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

fig.update_layout(
    title=(
        "Global station hydrograph dashboard"
        f"<br><sup>{colour_title}. Click a station to load its hydrograph.</sup>"
    ),
    autosize=True,
    margin=dict(l=0, r=0, t=70, b=0),
    showlegend=True,
)


# ------------------------------------------------------------
# Convert Plotly map to embeddable HTML div
# ------------------------------------------------------------
map_div = fig.to_html(
    include_plotlyjs=True,
    full_html=False,
    div_id="map",
    config={
        "scrollZoom": True,
        "displaylogo": False,
        "responsive": True,
        "modeBarButtonsToRemove": [
            "select2d",
            "lasso2d",
        ],
    },
)

catalog_json = json.dumps(catalog, ensure_ascii=False, separators=(",", ":"))


# ------------------------------------------------------------
# Full HTML dashboard template
# ------------------------------------------------------------
# The template embeds:
# - the Plotly map
# - the station catalogue
# - JavaScript to fetch per-station JSON files
# - a lower hydrograph panel
# - a right-hand station information panel
html_template = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Global station hydrograph dashboard</title>
<style>
body {
  margin: 0;
  font-family: Arial, sans-serif;
}

#container {
  display: flex;
  flex-direction: column;
  height: 100vh;
  overflow: hidden;
}

#map-wrap {
  flex: 0 0 62vh;
  height: 62vh;
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
  flex: 1;
  display: grid;
  grid-template-columns: minmax(0, 1fr) 420px;
  border-top: 1px solid #ccc;
  min-height: 280px;
  position: relative;
  z-index: 10;
  background: white;
  overflow: hidden;
}

#hydrograph {
  min-height: 280px;
  overflow: hidden;
}

#info {
  overflow-y: auto;
  border-left: 1px solid #ccc;
  padding: 10px;
  font-size: 12px;
  background: white;
  z-index: 20;
}

table {
  border-collapse: collapse;
  width: 100%;
}

th, td {
  border-bottom: 1px solid #ddd;
  padding: 4px 6px;
  text-align: left;
}
</style>
</head>
<body>
<div id="container">
  <div id="map-wrap">
    __MAP_DIV__
  </div>
  <div id="bottom">
    <div id="hydrograph">
      <p style="padding:12px;">Click a station to load its hydrograph.</p>
    </div>
    <div id="info">
      <b>Station information</b><br>
      Click a station on the map.
    </div>
  </div>
</div>

<script>
const CATALOG = __CATALOG_JSON__;
const DATA_DIR = "dashboard_data";

function esc(value) {
  if (value === null || value === undefined) return "";
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\\\"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function stationInfo(station, payload) {
  let s = "<b>" + esc(station.name) + "</b>";
  s += "<table>";
  s += "<tr><th>Country</th><td>" + esc(station.country_code) + "</td></tr>";
  s += "<tr><th>River</th><td>" + esc(station.river) + "</td></tr>";
  s += "<tr><th>Obs lon/lat</th><td>" + esc(station.lon) + ", " + esc(station.lat) + "</td></tr>";
  s += "<tr><th>Model lon/lat</th><td>" + esc(station.model_lon) + ", " + esc(station.model_lat) + "</td></tr>";

  s += "<tr><th>Station-model distance</th><td>" +
       esc(station.distance_station_to_model_km) + " km</td></tr>";

  s += "<tr><th>Upstream area</th><td>" +
       esc(station.upstream_area_km2) + " km²</td></tr>";

  s += "<tr><th>Model upstream area</th><td>" +
       esc(station.model_upstream_area_km2) + " km²</td></tr>";

  s += "<tr><th>Area difference</th><td>" +
       esc(station.upstream_area_difference_pct) + " %</td></tr>";

  if (payload.metrics) {
    s += "<tr><th>KGE</th><td>" + esc(payload.metrics.kge) + "</td></tr>";
    s += "<tr><th>Correlation</th><td>" + esc(payload.metrics.correlation) + "</td></tr>";
    s += "<tr><th>RMSE</th><td>" + esc(payload.metrics.rmse) + " m³/s</td></tr>";
    s += "<tr><th>Matched days</th><td>" + esc(payload.metrics.n) + "</td></tr>";
  }

  if (payload.obs) {
    s += "<tr><th>Obs file</th><td>" + esc(payload.obs.file) + "</td></tr>";
  }

  s += "</table>";
  return s;
}

async function loadAndPlotStation(i) {

  const station = CATALOG[i];

  if (!station) {
    document.getElementById("info").innerHTML =
      "<b>Error</b><br>Station not found.";
    return;
  }

  const url = DATA_DIR + "/" + station.file;

  document.getElementById("hydrograph").innerHTML =
    "<p style='padding:12px;'>Loading " + esc(station.name) + "...</p>";

  try {

    const response = await fetch(url);

    if (!response.ok) {
      throw new Error("Could not load " + url);
    }

    const payload = await response.json();

    const traces = [];

    traces.push({
      x: payload.time,
      y: payload.model_discharge,
      type: "scatter",
      mode: "lines",
      name: "Model river discharge",
      yaxis: "y",
      hovertemplate: "%{x}<br>Model Q=%{y} m³/s<extra></extra>"
    });

    if (payload.obs) {
      traces.push({
        x: payload.obs.time,
        y: payload.obs.values,
        type: "scatter",
        mode: "lines+markers",
        name: "Observed streamflow",
        yaxis: "y",
        line: {dash: "dot"},
        hovertemplate: "%{x}<br>Obs Q=%{y} m³/s<extra></extra>"
      });
    }

    Plotly.newPlot(
      "hydrograph",
      traces,
      {
        title: {
          text: "Hydrograph: " + station.name,
          x: 0.02,
          xanchor: "left"
        },
        margin: {l: 65, r: 35, t: 55, b: 55},
        xaxis: {title: "Valid time", type: "date"},
        yaxis: {title: "River discharge, m³/s", rangemode: "tozero"},
        legend: {orientation: "h", x: 0, y: 1.12}
      },
      {
        responsive: true,
        displaylogo: false,
        modeBarButtonsToRemove: ["select2d", "lasso2d"]
      }
    );

    document.getElementById("info").innerHTML =
      stationInfo(station, payload);

  } catch (err) {

    console.error(err);

    document.getElementById("hydrograph").innerHTML =
      "<p style='padding:12px;color:red;'>Could not load station JSON.</p>";

    document.getElementById("info").innerHTML =
      "<b>" + esc(station.name) + "</b><br>" +
      "Could not load file:<br><code>" + esc(url) + "</code>";
  }
}

const mapDiv = document.getElementById("map");

mapDiv.on("plotly_click", function(data) {

  if (!data || !data.points || data.points.length === 0) {
    return;
  }

  const pt = data.points[0];

  // Ignore river clicks. River traces do not carry station customdata.
  if (pt.customdata === undefined || pt.customdata === null) {
    return;
  }

  const i = Number(pt.customdata);

  if (!Number.isInteger(i)) {
    return;
  }

  loadAndPlotStation(i);
});
</script>
</body>
</html>
"""


# ------------------------------------------------------------
# Write final standalone dashboard
# ------------------------------------------------------------
html = (
    html_template
    .replace("__MAP_DIV__", map_div)
    .replace("__CATALOG_JSON__", catalog_json)
)

OUT_HTML.write_text(html, encoding="utf-8")

print(f"Saved: {OUT_HTML}")
