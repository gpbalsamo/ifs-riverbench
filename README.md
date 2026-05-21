# ifs-riverbench

`ifs-riverbench` contains scripts and tools to benchmark the new river discharge component of the IFS against observed streamflow data.

The workflow extracts global river discharge forecasts from MARS, samples the model discharge at river station locations, compares model and observed hydrographs, and builds an interactive dashboard for visual inspection.

## Workflow

The workflow is composed of three Python scripts, located in the `Workflow/` directory, and should be run in sequence:

```bash
python Workflow/00_extract_rivers_mars.py
python Workflow/01_extract_hydrographs.py
python Workflow/02_build_dashboard.py
