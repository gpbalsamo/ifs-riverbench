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
#   j7xs : 50r1 GP4HYDRO
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
SITE_BUNDLE_DIRNAME="site_bundle"

# Switches
EXTRACT_MARS_GRIB=True
EXTRACT_MARS_HYDRO=True
EXTRACT_GLOFAS_HYDRO=True
INCLUDE_GLOFAS_IN_DASHBOARD=True
# Set to False to skip GloFAS v4 entirely: no v4 files are required, no v4
# hydrograph is extracted, and v4 is excluded from the dashboards. v5 is
# always included. Overridable from the environment: INCLUDE_GLOFAS_V4=True bash ...
INCLUDE_GLOFAS_V4=${INCLUDE_GLOFAS_V4:-False}

# MARS extraction output/scratch roots. The script defaults in
# 00_extract_rivers_mars.py point at /perm/pad (not writable by other users),
# so we override them to the current user's space. This also matches the GRIB
# root that 01_extract_hydrographs.py reads from by default
# (/perm/${USER}/flood_cases/grib).
MARS_OUT_ROOT="/perm/${USER}/flood_cases/grib"
MARS_REQ_ROOT="/perm/${USER}/flood_cases/mars_requests"
MARS_TMPDIR="/perm/${USER}/flood_cases/tmp_mars"

# Station metadata CSV. Overrides the script default in 01_extract_hydrographs.py
# (/perm/pad/flood_cases/Stations/allstations_v1.3.csv).
STATION_FILE="/perm/${USER}/flood_cases/Stations/allstations_v1.3.csv"

# ARCHIVE_LAYOUT="daily_steps"
# MARS_STEP_TEXT="24"
ARCHIVE_LAYOUT="monthly_steps"
MARS_STEP_TEXT=""

# For monthly MARS archive:
# date=first day of month, step=24/to/...,
# but fields represent days of the same month.
VALID_TIME_SHIFT_HOURS=-24

# Experiments produced from MARS/CaMa workflow
MARS_EXPERIMENTS=(
#  "j6ft"   # MSWEP3 hourly precipitation
#  "j6fu"   # MSWEP3 daily precipitation
#  "j6fs"   # MSWEP3 monthly precipitation
#  "j6gq"   # EFAS 6-hourly precipitation
  "iyp3"   # 50r1 ERA5 new runoff
#  "j7xs"   # 50r1 GP4HYDRO
  "iwya"   # 50r1 ERA5 control
)

# GloFAS experiments from local GRIB files (not retrieved from MARS here)
GLOFAS_GRIB_DIR="/perm/${USER}/benchmark_cmf_gp4hydro_vs_glofas_discharge_2018_2022"
GLOFAS_V4_EXPVER="glofas_v4"
GLOFAS_V5_EXPVER="glofas_v5"
GLOFAS_V4_PATTERN="glofas_v4.0_ecmf-era5_*.grib"
GLOFAS_V5_PATTERN="glofas_v5.0_ecmf-era5_*.grib"
GLOFAS_SHORTNAMES=("dis24" "avg_dis")

# Reference experiment for pairwise difference dashboards.
REFERENCE_EXPVER="iwya"
# REFERENCE_EXPVER="izay"
# REFERENCE_EXPVER="iyp3"

METRICS=("kge" "correlation")

# Build final dashboard experiment list.
DASHBOARD_EXPERIMENTS=("${MARS_EXPERIMENTS[@]}")
if [[ ${INCLUDE_GLOFAS_IN_DASHBOARD} == True ]]; then
  if [[ ${INCLUDE_GLOFAS_V4} == True ]]; then
    DASHBOARD_EXPERIMENTS+=("${GLOFAS_V4_EXPVER}")
  fi
  DASHBOARD_EXPERIMENTS+=("${GLOFAS_V5_EXPVER}")
fi

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
echo "MARS experiments      : ${MARS_EXPERIMENTS[*]}"
echo "Dashboard experiments : ${DASHBOARD_EXPERIMENTS[*]}"
echo "Extract MARS GRIB     : ${EXTRACT_MARS_GRIB}"
echo "Extract MARS hydro    : ${EXTRACT_MARS_HYDRO}"
echo "Extract GloFAS hydro  : ${EXTRACT_GLOFAS_HYDRO}"
echo "Reference  : ${REFERENCE_EXPVER}"
echo "Metrics    : ${METRICS[*]}"
echo "======================================================================"
echo

if [[ ! " ${DASHBOARD_EXPERIMENTS[*]} " =~ " ${REFERENCE_EXPVER} " ]]; then
  echo "ERROR: REFERENCE_EXPVER=${REFERENCE_EXPVER} is not in DASHBOARD_EXPERIMENTS." >&2
  exit 1
fi

RUN_LABEL="${DATE_START}_${DATE_END}_${RESOLUTION}arcmin"

if [[ ${EXTRACT_GLOFAS_HYDRO} == True ]]; then
  if [[ ! -d "${GLOFAS_GRIB_DIR}" ]]; then
    echo "ERROR: GLOFAS_GRIB_DIR not found: ${GLOFAS_GRIB_DIR}" >&2
    exit 1
  fi

  if [[ ${INCLUDE_GLOFAS_V4} == True ]]; then
  if ! compgen -G "${GLOFAS_GRIB_DIR}/${GLOFAS_V4_PATTERN}" > /dev/null; then
    echo "ERROR: No files found for GloFAS v4 pattern:" >&2
    echo "  ${GLOFAS_GRIB_DIR}/${GLOFAS_V4_PATTERN}" >&2
    exit 1
  fi
  fi

  if ! compgen -G "${GLOFAS_GRIB_DIR}/${GLOFAS_V5_PATTERN}" > /dev/null; then
    echo "ERROR: No files found for GloFAS v5 pattern:" >&2
    echo "  ${GLOFAS_GRIB_DIR}/${GLOFAS_V5_PATTERN}" >&2
    exit 1
  fi
fi

if [[ ${INCLUDE_GLOFAS_IN_DASHBOARD} == True && ${EXTRACT_GLOFAS_HYDRO} == False ]]; then
  GLOFAS_CATALOG_EXPVERS=()
  if [[ ${INCLUDE_GLOFAS_V4} == True ]]; then
    GLOFAS_CATALOG_EXPVERS+=("${GLOFAS_V4_EXPVER}")
  fi
  GLOFAS_CATALOG_EXPVERS+=("${GLOFAS_V5_EXPVER}")
  for GLOFAS_EXPVER in "${GLOFAS_CATALOG_EXPVERS[@]}"; do
    CATALOG_FILE="dashboard_data/${GLOFAS_EXPVER}/${RUN_LABEL}/stations_catalog.json"
    if [[ ! -f "${CATALOG_FILE}" ]]; then
      echo "ERROR: Missing precomputed GloFAS dashboard data: ${CATALOG_FILE}" >&2
      echo "Set EXTRACT_GLOFAS_HYDRO=True to generate it, or set INCLUDE_GLOFAS_IN_DASHBOARD=False." >&2
      exit 1
    fi
  done
fi

# ======================================================================
# 1. Extract river discharge GRIB files from MARS
# ======================================================================
echo
echo "======================================================================"
echo "[1/3] Extracting river discharge from MARS"
echo "======================================================================"

if [[ ${EXTRACT_MARS_GRIB} == True ]] ; then
python3 00_extract_rivers_mars.py \
  --expver "${MARS_EXPERIMENTS[@]}" \
  --date-start "${DATE_START}" \
  --date-end "${DATE_END}" \
  --archive-layout "${ARCHIVE_LAYOUT}" \
  --step-text "${MARS_STEP_TEXT}" \
  --out-root "${MARS_OUT_ROOT}" \
  --req-root "${MARS_REQ_ROOT}" \
  --tmpdir "${MARS_TMPDIR}"

echo "[1/3] MARS extraction complete."
else
echo "[1/3] Skipped MARS extraction (EXTRACT_MARS_GRIB=False)."
fi

# ======================================================================
# 2. Extract hydrographs and compute station metrics
# ======================================================================
echo
echo "======================================================================"
echo "[2/3] Extracting station hydrographs"
echo "======================================================================"

if [[ ${EXTRACT_MARS_HYDRO} == True ]] ; then
for EXPVER in "${MARS_EXPERIMENTS[@]}"; do
  echo
  echo "------------------------------------------------------------------"
  echo "Hydrograph extraction (MARS) for ${EXPVER}"
  echo "------------------------------------------------------------------"

  python3 01_extract_hydrographs.py \
    --expver "${EXPVER}" \
    --date-start "${DATE_START}" \
    --date-end "${DATE_END}" \
    --resolution "${RESOLUTION}" \
    --valid-time-shift-hours "${VALID_TIME_SHIFT_HOURS}" \
    --station-file "${STATION_FILE}" \
    --obs-file /perm/${USER}/flood_cases/Stations/Qobs_24_1980-2025_withcaravan.zarr
done
else
echo "MARS hydrograph extraction skipped (EXTRACT_MARS_HYDRO=False)."
fi

if [[ ${EXTRACT_GLOFAS_HYDRO} == True ]] ; then
  if [[ ${INCLUDE_GLOFAS_V4} == True ]]; then
  echo
  echo "------------------------------------------------------------------"
  echo "Hydrograph extraction (GloFAS v4)"
  echo "------------------------------------------------------------------"

  python3 01_extract_hydrographs.py \
    --expver "${GLOFAS_V4_EXPVER}" \
    --date-start "${DATE_START}" \
    --date-end "${DATE_END}" \
    --resolution "${RESOLUTION}" \
    --model-grid-source glofas \
    --grib-dir "${GLOFAS_GRIB_DIR}" \
    --grib-file-pattern "${GLOFAS_V4_PATTERN}" \
    --discharge-shortnames "${GLOFAS_SHORTNAMES[@]}" \
    --valid-time-shift-hours "${VALID_TIME_SHIFT_HOURS}" \
    --station-file "${STATION_FILE}" \
    --obs-file /perm/${USER}/flood_cases/Stations/Qobs_24_1980-2025_withcaravan.zarr
  else
  echo "GloFAS v4 hydrograph extraction skipped (INCLUDE_GLOFAS_V4=False)."
  fi

  echo
  echo "------------------------------------------------------------------"
  echo "Hydrograph extraction (GloFAS v5)"
  echo "------------------------------------------------------------------"

  python3 01_extract_hydrographs.py \
    --expver "${GLOFAS_V5_EXPVER}" \
    --date-start "${DATE_START}" \
    --date-end "${DATE_END}" \
    --resolution "${RESOLUTION}" \
    --model-grid-source glofas \
    --grib-dir "${GLOFAS_GRIB_DIR}" \
    --grib-file-pattern "${GLOFAS_V5_PATTERN}" \
    --discharge-shortnames "${GLOFAS_SHORTNAMES[@]}" \
    --valid-time-shift-hours "${VALID_TIME_SHIFT_HOURS}" \
    --station-file "${STATION_FILE}" \
    --obs-file /perm/${USER}/flood_cases/Stations/Qobs_24_1980-2025_withcaravan.zarr
else
  echo "GloFAS hydrograph extraction skipped (EXTRACT_GLOFAS_HYDRO=False)."
fi

echo "[2/3] Hydrograph extraction complete."

# ======================================================================
# 3. Build dashboards
# ======================================================================
echo
echo "======================================================================"
echo "[3/3] Building dashboards"
echo "======================================================================"

for METRIC in "${METRICS[@]}"; do

  # ----------------------------------------------------------------------
  # 3a. Best metric class across all experiments
  # ----------------------------------------------------------------------
  echo
  echo "------------------------------------------------------------------"
  echo "Dashboard: best_metric (${METRIC})"
  echo "------------------------------------------------------------------"

  python3 02_build_dashboard.py \
    --expver "${DASHBOARD_EXPERIMENTS[@]}" \
    --date-start "${DATE_START}" \
    --date-end "${DATE_END}" \
    --resolution "${RESOLUTION}" \
    --metric "${METRIC}" \
    --colour-mode best_metric \
    --river-resolution ${RIVER_RESOL} \
    --map-height-vh "${MAP_HEIGHT_VH}" \
    --output-dir "${SITE_BUNDLE_DIRNAME}" \
    --data-root "dashboard_data"

  # ----------------------------------------------------------------------
  # 3b. Winning experiment at each station
  # ----------------------------------------------------------------------
  echo ""
  echo "------------------------------------------------------------------"
  echo "Dashboard: best_experiment (${METRIC})"
  echo "------------------------------------------------------------------"

  python3 02_build_dashboard.py \
    --expver "${DASHBOARD_EXPERIMENTS[@]}" \
    --control-expver "${REFERENCE_EXPVER}" \
    --date-start "${DATE_START}" \
    --date-end "${DATE_END}" \
    --resolution "${RESOLUTION}" \
    --metric "${METRIC}" \
    --colour-mode best_experiment \
    --river-resolution ${RIVER_RESOL} \
    --best-experiment-min-improvement "${THRESHOLD}" \
    --map-height-vh "${MAP_HEIGHT_VH}" \
    --output-dir "${SITE_BUNDLE_DIRNAME}" \
    --data-root "dashboard_data"

  # ----------------------------------------------------------------------
  # 3c. Pairwise difference dashboards against reference experiment
  #
  # experiment_difference requires exactly two experiments and computes:
  #   second experiment - first experiment
  # ----------------------------------------------------------------------
  for EXPVER in "${DASHBOARD_EXPERIMENTS[@]}"; do

    if [[ "${EXPVER}" == "${REFERENCE_EXPVER}" ]]; then
      continue
    fi

    echo
    echo "------------------------------------------------------------------"
    echo "Dashboard: experiment_difference ${EXPVER} - ${REFERENCE_EXPVER} (${METRIC})"
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
      --map-height-vh "${MAP_HEIGHT_VH}" \
      --output-dir "${SITE_BUNDLE_DIRNAME}" \
      --data-root "dashboard_data"
  done
done

echo
echo "------------------------------------------------------------------"
echo "Preparing single-site bundle directory"
echo "------------------------------------------------------------------"

python3 03_prepare_sites_bundle.py \
  --workflow-dir "${SCRIPT_DIR}" \
  --bundle-dirname "${SITE_BUNDLE_DIRNAME}" \
  --html-pattern "global_station_dashboard_*.html"

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
echo "Generated HTML dashboards and index in single upload directory:"
echo "  ${SCRIPT_DIR}/${SITE_BUNDLE_DIRNAME}"
echo
echo "To view dashboards on sites, run:"
echo '  export ECMWF_RIVERBENCH_TOKEN="<set securely outside Git>"'

echo "  python3 04_upload_dashboard.py \
  --workflow-dir /perm/${USER}/ifs-riverbench/Workflow \
  --dashboard-dirname ${SITE_BUNDLE_DIRNAME} \
  --remote-dashboard-dir discharge-dashboard"

echo
echo "Then open the generated HTML files through:"
echo "  https://sites.ecmwf.int/${USER}/riverbench/discharge-dashboard/"
echo "======================================================================"
