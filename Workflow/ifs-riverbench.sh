#!/usr/bin/env bash
set -euo pipefail

# ======================================================================
# ifs-riverbench.sh
#
# Full 2018 river-discharge benchmarking workflow for six experiments.
#
# Experiments:
#   j6ft : MSWEP3 hourly precipitation
#   j6fu : MSWEP3 daily precipitation
#   j6fs : MSWEP3 monthly precipitation
#   j6gq : EFAS 6-hourly precipitation
#   iyp3 : 50r1 bugfix ERA5 control
#   iwya : 50r1 ERA5 control
#
# Workflow:
#   00_extract_rivers_mars.py
#   01_extract_hydrographs.py
#   02_build_dashboard.py
#
# Period:
#   2018-01-01 to 2018-12-31
# ======================================================================

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
DATE_START="20180101"
DATE_END="20221231"
RESOLUTION=15
THRESHOLD=0.02
MAP_HEIGHT_VH=65
RIVER_RESOL=50m
#EXT_HYDRO=False
EXT_HYDRO=True

# For monthly MARS archive:
# date=first day of month, step=24/to/...,
# but fields represent days of the same month.
VALID_TIME_SHIFT_HOURS=-24

EXPERIMENTS=(
  "j6ft"   # MSWEP3 hourly precipitation
  "j6fu"   # MSWEP3 daily precipitation
  "j6fs"   # MSWEP3 monthly precipitation
  "j6gq"   # EFAS 6-hourly precipitation
  "iyp3"   # 50r1 bugfix ERA5 control
#  "iwya"   # 50r1 ERA5 control
)

# Reference experiment for pairwise difference dashboards.
# REFERENCE_EXPVER="iwya"
REFERENCE_EXPVER="iyp3"

# Metric for dashboard colouring.
# METRIC="correlation"
METRIC="kge"

# ----------------------------------------------------------------------
# Make sure we are in the workflow directory or adjust this path.
# ----------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo
echo "======================================================================"
echo "IFS riverbench workflow"
echo "======================================================================"
echo "Date range : ${DATE_START} to ${DATE_END}"
echo "Resolution : ${RESOLUTION} arcmin"
echo "Experiments: ${EXPERIMENTS[*]}"
echo "Reference  : ${REFERENCE_EXPVER}"
echo "Metric     : ${METRIC}"
echo "======================================================================"
echo


# ======================================================================
# 1. Extract river discharge GRIB files from MARS
# ======================================================================
echo
echo "======================================================================"
echo "[1/3] Extracting river discharge from MARS"
echo "======================================================================"

if [[ ${EXT_HYDRO} == True ]] ; then
python3 00_extract_rivers_mars.py \
  --expver "${EXPERIMENTS[@]}" \
  --date-start "${DATE_START}" \
  --date-end "${DATE_END}"

echo "[1/3] MARS extraction complete."
fi


# ======================================================================
# 2. Extract hydrographs and compute station metrics
# ======================================================================
echo
echo "======================================================================"
echo "[2/3] Extracting station hydrographs"
echo "======================================================================"

if [[ ${EXT_HYDRO} == True ]] ; then
for EXPVER in "${EXPERIMENTS[@]}"; do
  echo
  echo "------------------------------------------------------------------"
  echo "Hydrograph extraction for ${EXPVER}"
  echo "------------------------------------------------------------------"

  python3 01_extract_hydrographs.py \
    --expver "${EXPVER}" \
    --date-start "${DATE_START}" \
    --date-end "${DATE_END}" \
    --resolution "${RESOLUTION}" \
    --valid-time-shift-hours "${VALID_TIME_SHIFT_HOURS}" \
    --obs-file /perm/pad/flood_cases/Stations/Qobs_24_1980-2025_withcaravan.zarr
done
fi

echo "[2/3] Hydrograph extraction complete."


# ======================================================================
# 3. Build dashboards
# ======================================================================
echo
echo "======================================================================"
echo "[3/3] Building dashboards"
echo "======================================================================"


# ----------------------------------------------------------------------
# 3a. Best metric class across all experiments
# ----------------------------------------------------------------------
echo
echo "------------------------------------------------------------------"
echo "Dashboard: best_metric"
echo "------------------------------------------------------------------"

python3 02_build_dashboard.py \
  --expver "${EXPERIMENTS[@]}" \
  --date-start "${DATE_START}" \
  --date-end "${DATE_END}" \
  --resolution "${RESOLUTION}" \
  --metric "${METRIC}" \
  --colour-mode best_metric \
  --river-resolution ${RIVER_RESOL} \
  --map-height-vh "${MAP_HEIGHT_VH}"


# ----------------------------------------------------------------------
# 3b. Winning experiment at each station
# ----------------------------------------------------------------------
echo ""
echo "------------------------------------------------------------------"
echo "Dashboard: best_experiment"
echo "------------------------------------------------------------------"

python3 02_build_dashboard.py \
  --expver "${EXPERIMENTS[@]}" \
  --control-expver "${REFERENCE_EXPVER}" \
  --date-start "${DATE_START}" \
  --date-end "${DATE_END}" \
  --resolution "${RESOLUTION}" \
  --metric "${METRIC}" \
  --colour-mode best_experiment \
  --river-resolution ${RIVER_RESOL} \
  --best-experiment-min-improvement "${THRESHOLD}" \
  --map-height-vh "${MAP_HEIGHT_VH}"

# ----------------------------------------------------------------------
# 3c. Pairwise difference dashboards against reference experiment
#
# experiment_difference requires exactly two experiments and computes:
#   second experiment - first experiment
#
# Therefore we run:
#   candidate - iwya
# ----------------------------------------------------------------------
for EXPVER in "${EXPERIMENTS[@]}"; do

  if [[ "${EXPVER}" == "${REFERENCE_EXPVER}" ]]; then
    continue
  fi

  echo
  echo "------------------------------------------------------------------"
  echo "Dashboard: experiment_difference ${EXPVER} - ${REFERENCE_EXPVER}"
  echo "------------------------------------------------------------------"

  python3 02_build_dashboard.py \
    --expver "${REFERENCE_EXPVER}" "${EXPVER}" \
    --date-start "${DATE_START}" \
    --date-end "${DATE_END}" \
    --resolution "${RESOLUTION}" \
    --metric "${METRIC}" \
    --colour-mode experiment_difference \
    --river-resolution ${RIVER_RESOL} \
    --difference-threshold "${THRESHOLD}" \
    --map-height-vh "${MAP_HEIGHT_VH}"
done

echo "[3/3] Dashboard generation complete."


# ======================================================================
# Final message
# ======================================================================
echo
echo "======================================================================"
echo "Workflow complete"
echo "======================================================================"
echo
echo "Generated dashboard data under:"
echo "  dashboard_data/<expver>/${DATE_START}_${DATE_END}_${RESOLUTION}arcmin/"
echo
echo "Generated HTML dashboards in:"
echo "  ${SCRIPT_DIR}"
echo
echo "To view dashboards on sites, run:"
echo "  export ECMWF_SITES_TOKEN='here your token to access sites.ecmwf.int'"

echo "  python3 03_upload_dashboard_all_html_patched.py \
   --workflow-dir /perm/${USER}/ifs-riverbench/Workflow \
   --html-pattern '*.html' \
   --html-only "

echo
echo "Then open the generated HTML files through:"
echo "  https://sites.ecmwf.int/${USER}/riverbench/"
echo "======================================================================"
