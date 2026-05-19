# ifs-riverbench
Contains scripts and tools to benchmark the new river discharge component of the IFS

## Workflow
There are 3 python scripts that needs to be executed in order to obtain a global 1 arcmin dashboard that permit to navigate a multi-year river discharge model-observation benchmarking.

The benchmark is based on three scripts run in sequence:

1. `00_extract_rivers_mars.py`
2. `01_extract_hydrographs.py`
3. `02_build_dashboard.py`
