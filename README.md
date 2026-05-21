# ifs-riverbench

`ifs-riverbench` contains scripts and tools to benchmark the new river discharge component of the IFS against observed streamflow data.

The workflow extracts global river discharge forecasts from MARS, samples the model discharge at river station locations, compares model and observed hydrographs, and builds an interactive dashboard for visual inspection.

## Workflow

The workflow is composed of three Python scripts, located in the `Workflow/` directory, and should be run in sequence:

```bash
python Workflow/00_extract_rivers_mars.py
python Workflow/01_extract_hydrographs.py
python Workflow/02_build_dashboard.py

## Scripts
`00_extract_rivers_mars.py`
Retrieves monthly global GRIB files from MARS.
The script currently extracts:
235270 = river discharge

The script writes monthly GRIB files and stores the corresponding MARS request files for reproducibility.

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

### Main outputs:

```bash
dashboard_data/
├── stations/
├── stations_catalog.json
└── global_station_metrics.csv

`02_build_dashboard.py`
Builds an interactive HTML dashboard from the files produced by 01_extract_hydrographs.py.
The dashboard shows:
a global map of river stations;
station colours based on KGE or correlation;
model and observed hydrographs when a station is clicked;
station metadata and skill metrics.

The dashboard files are named according to the selected colouring mode, for example:

```bash
global_station_dashboard_correlation.html
global_station_dashboard_kge.html

