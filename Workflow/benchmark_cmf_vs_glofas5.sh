#!/usr/bin/env bash
# Benchmark CaMa-Flood river discharge against GloFAS over a multi-year period.
#
# Example use case:
#   2018-2022 benchmark for precipitation-forcing sensitivity.
#
# Experiments:
#   j65z  : control / reference experiment
#   j6fu  : MSWEPv3 monthly rescaling
#   j6fs  : MSWEPv3 daily rescaling
#   j6gq  : MSWEPv3 hourly replacement
#   iyp3  : EFAS precipitation forcing
#
# Notes:
#   --force-retrieve
#       Optional. Forces retrieval/re-reading of data even if cached/intermediate files
#       already exist. Use this if you changed experiments, dates, masks, MARS/GloFAS
#       data, or suspect stale cached files.
#
# Run normally:
#   bash benchmark_cmf_vs_glofas_long_period.sh
#
# Run forcing re-retrieval:
#   FORCE_RETRIEVE=1 bash benchmark_cmf_vs_glofas_long_period.sh

set -euo pipefail

PYTHON_SCRIPT="benchmark_cmf_vs_glofas5.py"

EXPVERS=(j7xs j6fu iyp3)

BASE_OUTDIR="/perm/${USER}"
BASE_BENCH="benchmark_cmf_gp4hydro_vs_glofas_discharge"

# Long-period benchmark.
START_DATE=${START_DATE:-20180101}
END_DATE=${END_DATE:-20221231}
LABEL=${LABEL:-2018_2022}

OBS_MIN_DISCHARGE=${OBS_MIN_DISCHARGE:-5}   # m3/s

UPAREA_TOLERANCE=${UPAREA_TOLERANCE:-0.10}  # symmetric relative difference; 0.10 = 10%
UPAREA_15MIN_FILE=${UPAREA_15MIN_FILE:-/home/rdx/data/50r1/camaflood/static_network_nc_v2.1/glb_15min/ncdata.nc}
UPAREA_03MIN_FILE=${UPAREA_03MIN_FILE:-/home/rdx/data/50r1/camaflood/static_network_nc_v2.1/glb_03min/ncdata.nc}

# Set to 1 to redo the plots only.
NO_PLOT=${NO_PLOT:-0}

# Set to 1 to redo the plots only.
PLOT_ONLY=${PLOT_ONLY:-1}

# Set to 1 to force re-retrieval/reprocessing of cached inputs.
FORCE_RETRIEVE=${FORCE_RETRIEVE:-0}

OUTDIR="${BASE_OUTDIR}/${BASE_BENCH}_${LABEL}"

COMMON_ARGS=(
  --expvers "${EXPVERS[@]}"
  --cama-archive-mode auto
  --obs-min-discharge "${OBS_MIN_DISCHARGE}"
  --uparea-tolerance "${UPAREA_TOLERANCE}"
  --uparea-15min-file "${UPAREA_15MIN_FILE}"
  --uparea-03min-file "${UPAREA_03MIN_FILE}"
)

if [[ "${NO_PLOT}" == "1" ]]; then
  COMMON_ARGS+=(--no-map-png)
fi

if [[ "${PLOT_ONLY}" == "1" ]]; then
  COMMON_ARGS+=(--plots-only)
fi

if [[ "${FORCE_RETRIEVE}" == "1" ]]; then
  COMMON_ARGS+=(--force-retrieve)
fi

echo
echo "============================================================"
echo "Running long-period CaMa-Flood vs GloFAS benchmark"
echo "Label          : ${LABEL}"
echo "Start date     : ${START_DATE}"
echo "End date       : ${END_DATE}"
echo "Outdir         : ${OUTDIR}"
echo "Expvers        : ${EXPVERS[*]}"
echo "Obs min Q      : ${OBS_MIN_DISCHARGE} m3/s"
echo "Uparea tol.    : ${UPAREA_TOLERANCE}"
echo "Uparea 15 min : ${UPAREA_15MIN_FILE}"
echo "Uparea 03 min : ${UPAREA_03MIN_FILE}"
echo "Force retrieve : ${FORCE_RETRIEVE}"
echo "Stats only     : ${NO_PLOT}"
echo "Plots only     : ${PLOT_ONLY}"
echo "============================================================"
echo

mkdir -p "${OUTDIR}"

python3 -u "${PYTHON_SCRIPT}" \
  --start-date "${START_DATE}" \
  --end-date "${END_DATE}" \
  --outdir "${OUTDIR}" \
  "${COMMON_ARGS[@]}"

if [[ "${NO_PLOT}" == "1" ]]; then
  echo "============================================================"
  echo "Plotting now"
  echo "============================================================"
  python3 -u "${PYTHON_SCRIPT}" \
  --start-date "${START_DATE}" \
  --end-date "${END_DATE}" \
  --outdir "${OUTDIR}" \
  --expvers "${EXPVERS[@]}" \
  --cama-archive-mode auto \
  --obs-min-discharge "${OBS_MIN_DISCHARGE}" \
  --continent-maps --plots-only
fi

echo "Long-period benchmark completed successfully."
echo "Output:"
echo "  ${OUTDIR}"

python3 build_benchmark_html_index.py \
  --outdir "${OUTDIR}" \
  --title "CaMa-Flood vs GloFAS benchmark ${LABEL}"
