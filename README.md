# ifs-riverbench

`ifs-riverbench` contains scripts and tools to benchmark the new river discharge component of the IFS against observed streamflow data.

The workflow extracts global river discharge forecasts from MARS, samples the model discharge at river station locations, compares model and observed hydrographs, and builds an interactive dashboard for visual inspection.

## Workflow

The workflow is composed of three Python scripts, located in the `Workflow/` directory, and should be run in sequence:

```bash
python Workflow/00_extract_rivers_mars.py
python Workflow/01_extract_hydrographs.py
python Workflow/02_build_dashboard.py
```

### Script 00: Extracts the IFS river discharge
`00_extract_rivers_mars.py`
Retrieves monthly global GRIB files from MARS.
The script currently extracts:
235270 = river discharge

#### Main outputs:
The script writes monthly GRIB files and stores the corresponding MARS request files for reproducibility.

```bash
grib/
├── Globe_river_201801.grb
├── Globe_river_201801.grb
└── Globe_river_201801.grb

```

### Script 01: Extract hydrographs and compute metrics
`01_extract_hydrographs.py`
Extracts model river discharge hydrographs at station locations.
For each station, the script:
reads station metadata;
finds the nearest 1 arcmin model grid point;
extracts the model discharge time series;
reads the corresponding observed streamflow data;
converts observations to m³/s;
computes basic metrics such as KGE, correlation and RMSE;
writes dashboard-ready JSON and CSV files.

#### Main outputs:

```bash
dashboard_data/
├── stations/
│   ├── station_000000.json
│   ├── station_000001.json
│   └── ...
├── stations_catalog.json
└── global_station_metrics.csv
```

### Script 02: Build the interactive dashboard
`02_build_dashboard.py`
Builds an interactive HTML dashboard from the files produced by 01_extract_hydrographs.py.
The dashboard shows:
a global map of river stations;
station colours based on KGE or correlation;
model and observed hydrographs when a station is clicked;
station metadata and skill metrics.
dashboard files are named according to the metric choosen to cluster the results.

#### Main outputs:

```bash
global_station_dashboard_correlation.html
global_station_dashboard_kge.html
```

### Requirements
The workflow requires access to ECMWF MARS and the mars command.
Main Python dependencies include:

```bash
numpy
pandas
xarray
scipy
eccodes
plotly
cartopy
```
