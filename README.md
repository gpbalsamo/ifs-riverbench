# ifs-riverbench

`ifs-riverbench` is a lightweight benchmark workflow for the new IFS river discharge component.

The benchmark is based on three scripts run in sequence:

1. `00_extract_rivers_mars.py`  
   Extracts river-related model data from MARS.
2. `01_extract_hydrographs.py`  
   Builds hydrographs from the extracted river discharge data.
3. `02_build_dashboard.py`  
   Produces a dashboard to visualize and compare benchmark results.

In short: extract data, derive hydrographs, then build the dashboard for analysis.
