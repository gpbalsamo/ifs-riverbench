#!/usr/bin/env python3
"""
05_prepare_jrc_data.py

Prepare JRC LSTM (AIFL / AIFL-DA) and EFAS 6.0 station data for the
ifs-riverbench interactive dashboard.

Reads:
  - AIFL free-run test metrics CSV
  - AIFL-DA test metrics CSV
  - Station metadata (discharge_metadata_6h.csv) for lat/lon
  - 6-hourly observed discharge (Parquet)
  - EFAS 6.0 simulated discharge (CSV)
  - AIFL/AIFL-DA predictions (NPZ, produced by plot_hydrographs_best.py)

Writes:
  dashboard_data/{expver}/20230101_20231231_6h/
    stations_catalog.json
    global_station_metrics.csv
    stations/station_NNNNNN.json

Then use 02_build_dashboard.py to visualise.
"""

import argparse
import json
import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd


# ── Paths ────────────────────────────────────────────────────────────
# Data-source roots are read from the environment so the pipeline is not tied
# to one user's account. Override JRC_ROOT / EFAS_CSV before running, e.g.:
#   export JRC_ROOT=/path/to/jrc
#   export EFAS_CSV=/path/to/Qsim_efas6_full_calib_v2.csv
JRC_ROOT = Path(os.environ.get("JRC_ROOT", "/hpcperm/ecm7072/jrc"))
DATA = JRC_ROOT / "data"

BEST_TAG = "eu_15day_ar_da_climstat_doy_delta_assim10d_arid_pertype_cal2023"

METRICS_FREE = DATA / "lstm_results" / f"lstm_phase1_station_metrics_{BEST_TAG}_test.csv"
METRICS_DA   = DATA / "lstm_results" / f"lstm_phase1_station_metrics_{BEST_TAG}_da_test.csv"

OBS_META     = DATA / "ml_ready" / "discharge_metadata_6h.csv"
OBS_TS       = DATA / "ml_ready" / "discharge_timeseries_6h.parquet"

EFAS_CSV     = Path(os.environ.get(
    "EFAS_CSV", "/ec/vol/efas/calibration/6.0/Qsim_efas6_full_calib_v2.csv"))

# Predictions dir produced by plot_hydrographs_best.py
PRED_DIR = JRC_ROOT / "analysis" / "model" / "final_figures" / "predictions"

# LSTM config constants (must match training)
ASSIMILATION_STEPS = 40   # first 40 steps are assimilation window
OUTPUT_STEPS = 60         # total sequence length per window
FORECAST_STEPS = OUTPUT_STEPS - ASSIMILATION_STEPS  # 20 forecast steps

DEFAULT_OUTPUT_ROOT = Path("dashboard_data")
DATE_START = "20230101"
DATE_END   = "20231231"
RESOLUTION = 6  # use 6 arcmin slot for 6-hourly JRC data
RUN_LABEL  = f"{DATE_START}_{DATE_END}_{RESOLUTION}arcmin"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--date-start", default=DATE_START)
    p.add_argument("--date-end", default=DATE_END)
    return p.parse_args()


def kge_prime(obs, sim):
    """Kling-Gupta Efficiency (modified, 2012).

    Returns (kge, r, gamma, beta, rmse) where:
      r     = Pearson correlation
      gamma = variability ratio = CV_sim / CV_obs
      beta  = bias ratio = mean_sim / mean_obs
    """
    valid = ~(np.isnan(obs) | np.isnan(sim))
    if valid.sum() < 10:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    o, s = obs[valid], sim[valid]
    if np.std(o) == 0 or np.mean(o) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    r = float(np.corrcoef(o, s)[0, 1])
    gamma = float((np.std(s) / np.mean(s)) / (np.std(o) / np.mean(o)))
    beta = float(np.mean(s) / np.mean(o))
    kge = 1.0 - np.sqrt((r - 1)**2 + (gamma - 1)**2 + (beta - 1)**2)
    rmse = float(np.sqrt(np.mean((o - s)**2)))
    return float(kge), r, gamma, beta, rmse


# ── Forecast valid times ─────────────────────────────────────────────
# The NPZ files store the exact 6-hourly valid time of every forecast step
# (key ``forecast_times``, shape (n_windows, FORECAST_STEPS)), written by
# plot_hydrographs_best.py. The model is issued every 6 h (stride 1),
# independently of observation availability, so stacking any fixed lead time
# across windows yields a dense 6-hourly series. No timestamp reconstruction
# is needed — the times are read directly from the NPZ.

# Number of forecast lead steps to display/score in the default (non-
# interactive) hydrograph.
#   1  -> use only the first lead time (the analysis). For data-assimilated
#         runs (AIFL-DA) this is the post-assimilation state and snaps to the
#         observations; it is also the fair, consistent comparison across runs.
#   20 -> use the full forecast (all lead times); mixes lead 6 h..5 days.
# The interactive lead selector only exposes the KEEP_LEADS subset to reduce
# storage quota usage on ECMWF Sites.
DISPLAY_LEAD_STEPS = 1
KEEP_LEADS = [0, 1, 2, 3, 7, 19]  # lead times 1,2,3,4,8,20 (0-indexed: 0,1,2,3,7,19)


def _load_npz_forecast(sid, mode="free"):
    """Load the forecast-portion matrices and their valid times from an NPZ.

    Returns ``(sim_mat, obs_mat, times_mat)`` each of shape
    ``(n_windows, FORECAST_STEPS)`` where column ``L`` (0-based) is lead time
    ``L + 1`` (``(L+1) * 6 h`` after the analysis). ``times_mat`` is
    ``datetime64[ns]``. Returns ``(None, None, None)`` if the file or the
    ``forecast_times`` key is missing.

    The windows are NOT globally time-sorted (the dataset emits them in a
    rotated order); callers must sort by ``times_mat`` before plotting.
    """
    npz_path = PRED_DIR / f"station_{sid}_{mode}.npz"
    if not npz_path.exists():
        return None, None, None
    d = np.load(npz_path)
    if "forecast_times" not in d.files:
        return None, None, None
    sim_mat = d["sim_physical"][:, ASSIMILATION_STEPS:OUTPUT_STEPS]   # (nw, 20)
    obs_mat = d["obs_physical"][:, ASSIMILATION_STEPS:OUTPUT_STEPS]   # (nw, 20)
    times_mat = d["forecast_times"].astype("datetime64[ns]")          # (nw, 20)
    return sim_mat, obs_mat, times_mat


def _fmt_times(dt64_arr):
    """Format a datetime64[ns] array as ['YYYY-MM-DD HH:MM', ...]."""
    return [np.datetime_as_string(t, unit="m").replace("T", " ")
            for t in dt64_arr]


def load_npz_predictions(sid, mode="free"):
    """Return the default display series (first DISPLAY_LEAD_STEPS leads).

    Returns ``(sim_flat, obs_flat, times_flat)`` sorted chronologically, or
    ``(None, None, None)`` if unavailable. With ``DISPLAY_LEAD_STEPS == 1``
    this is the analysis (lead 1) of every 6-hourly-issued window; the model
    is issued every 6 h regardless of observation availability.
    """
    sim_mat, obs_mat, times_mat = _load_npz_forecast(sid, mode=mode)
    if sim_mat is None:
        return None, None, None
    k = DISPLAY_LEAD_STEPS
    sim_flat = sim_mat[:, :k].reshape(-1)
    obs_flat = obs_mat[:, :k].reshape(-1)
    times_flat = times_mat[:, :k].reshape(-1)
    order = np.argsort(times_flat)
    return sim_flat[order], obs_flat[order], times_flat[order]


def build_lead_series(sid, npz_mode):
    """Build the per-lead-time series bundle for the interactive lead selector.

    Compact on-disk layout to stay within the ECMWF Sites storage quota:
    the lead-1 valid-time axis is stored once as ``base_time`` and the browser
    reconstructs lead L's axis as ``base_time + L * step_hours``; only per-lead
    ``sim`` and ``kge`` are stored (per-lead ``obs`` is dropped, as the
    dashboard draws observations from the shared observation payload). Each
    lead's ``sim`` is one point per 6-hourly issue-time window. Returns
    ``None`` if the NPZ (or its ``forecast_times`` key) is unavailable.
    """
    sim_mat, obs_mat, times_mat = _load_npz_forecast(sid, mode=npz_mode)
    if sim_mat is None:
        return None

    n_leads = sim_mat.shape[1]
    # Every lead shares the same issue-time ordering: lead L's valid time is the
    # analysis (lead-1) valid time shifted forward by a constant L * 6 h. So we
    # sort once by lead 0 and reorder every lead identically, then store the
    # lead-1 valid-time axis ONCE as ``base_time``. The browser reconstructs
    # lead L's time axis as ``base_time + L * step_hours``. Per-lead ``obs`` is
    # NOT stored: the dashboard draws observations from the shared observation
    # payload, so emitting 20 redundant copies here only bloats the JSON
    # (~20x) and exhausts the ECMWF Sites storage quota.
    order = np.argsort(times_mat[:, 0])
    base_time = _fmt_times(times_mat[order, 0])

    sim_lists, kge_list = [], []
    for lead in KEEP_LEADS:
        if lead >= n_leads:
            continue
        s = sim_mat[order, lead]
        o = obs_mat[order, lead]
        sim_lists.append(
            [None if np.isnan(v) else round(float(v), 3) for v in s]
        )
        k, _, _, _, _ = kge_prime(o, s)
        kge_list.append(None if np.isnan(k) else round(float(k), 3))

    return {
        "n_leads": len(KEEP_LEADS),
        "step_hours": 6,
        "base_time": base_time,
        "sim": sim_lists,
        "kge": kge_list,
        "lead_indices": KEEP_LEADS,
    }


def write_experiment(expver, metrics_df, meta, obs_ts, sim_ts,
                     out_root, run_label, t0, t1, npz_mode=None):
    """Write stations_catalog.json + per-station JSON for one experiment."""

    exp_dir = out_root / expver / run_label
    stations_dir = exp_dir / "stations"
    stations_dir.mkdir(parents=True, exist_ok=True)

    catalog = []
    metrics_rows = []
    n_written = 0

    # Build lookup: ObsID -> metrics row
    met_lookup = {int(row["ObsID"]): row for _, row in metrics_df.iterrows()}

    for idx, (_, mrow) in enumerate(meta.iterrows()):
        sid = int(mrow["ObsID"])
        sid_str = str(sid)

        if sid not in met_lookup:
            continue

        met = met_lookup[sid]
        lon = float(mrow["StationLon"])
        lat = float(mrow["StationLat"])
        name = str(mrow.get("StationName", f"Station {sid}"))
        river = str(mrow.get("River", ""))
        country = str(mrow.get("Country code", ""))
        area = float(mrow.get("DrainingArea.km2.LDD", 0))

        kge_val = float(met.get("kge_prime", np.nan))
        corr_val = float(met.get("r", np.nan))
        gamma_val = float(met.get("gamma", np.nan))
        beta_val = float(met.get("beta", np.nan))
        rmse_val = float(met.get("rmse", np.nan))

        # ── Hydrograph data ──────────────────────────────────────
        obs_col = obs_ts.get(sid_str)
        if obs_col is None:
            continue

        # Simulated discharge — NPZ predictions (AIFL/AIFL_DA) or sim_ts (EFAS)
        sim_times = None
        sim_vals = None
        obs_times = None
        obs_vals = None
        missing_obs_times = []
        lead_series = None
        comparison_label = "6-hourly values on common dates"

        if npz_mode is not None:
            sim_flat, obs_flat_npz, times_flat = load_npz_predictions(
                sid, mode=npz_mode
            )
            if sim_flat is not None and len(sim_flat) > 10:
                # All-lead bundle for the interactive lead-time selector
                lead_series = build_lead_series(sid, npz_mode)

                # Model is issued 6-hourly regardless of obs; keep all model points.
                # Observations may have gaps; track them separately for visualization.
                sim_valid = ~np.isnan(sim_flat)
                if sim_valid.sum() >= 10:
                    vt = times_flat[sim_valid]
                    sim_times = _fmt_times(vt)
                    sim_vals = np.round(sim_flat[sim_valid], 3).tolist()
                    obs_times = sim_times

                    # Align obs to model times; mark missing obs for red-dot visualization
                    obs_at_sim = obs_flat_npz[sim_valid]
                    obs_vals = [
                        None if np.isnan(v) else round(float(v), 3)
                        for v in obs_at_sim
                    ]
                    missing_obs_times = [
                        t for t, o in zip(sim_times, obs_vals) if o is None
                    ]

                    if DISPLAY_LEAD_STEPS == 1:
                        comparison_label = (
                            "analysis (6 h lead time), one point per "
                            "assimilation window"
                        )
                    else:
                        comparison_label = (
                            f"forecast leads 1-{DISPLAY_LEAD_STEPS} "
                            "(6-hourly)"
                        )

                    # Recompute KGE' + components from obs vs sim where obs is valid
                    valid_both = ~np.isnan(obs_at_sim) & ~np.isnan(sim_flat[sim_valid])
                    if valid_both.sum() >= 10:
                        k, r, g, b, rm = kge_prime(
                            obs_at_sim[valid_both], sim_flat[sim_valid][valid_both]
                        )
                        if not np.isnan(k):
                            kge_val, corr_val, gamma_val, beta_val, rmse_val = (
                                k, r, g, b, rm
                            )

        if sim_times is None and sim_ts is not None:
            # EFAS: observations from parquet sliced to the display window
            mask = (obs_ts["time"] >= t0) & (obs_ts["time"] <= t1)
            obs_slice = obs_ts.loc[mask, ["time", sid_str]].dropna(subset=[sid_str])
            if len(obs_slice) < 10:
                continue
            obs_times = obs_slice["time"].dt.strftime("%Y-%m-%d %H:%M").tolist()
            obs_vals = obs_slice[sid_str].round(3).tolist()
            comparison_label = "6-hourly values on common dates"

            sim_col = sim_ts.get(sid_str)
            if sim_col is not None:
                sim_mask = (sim_ts["time"] >= t0) & (sim_ts["time"] <= t1)
                sim_slice = sim_ts.loc[sim_mask, ["time", sid_str]].copy()
                sim_slice = sim_slice.dropna(subset=[sid_str])
                sim_times = sim_slice["time"].dt.strftime("%Y-%m-%d %H:%M").tolist()
                sim_vals = sim_slice[sid_str].round(3).tolist()

        if sim_times is None or obs_times is None:
            # No predictions available — skip this station
            continue

        matched_days = len(obs_times) // 4  # 6h -> daily approx

        station_id = f"station_{idx:06d}"
        station_file = f"stations/{station_id}.json"

        station_json = {
            "station_index": idx,
            "station_id": station_id,
            "source_station_id": sid,
            "expver": expver,
            "resolution_arcmin": 0,
            "valid_time_shift_hours": 0,
            "time": sim_times,
            "model_discharge": sim_vals,
            "obs": {
                "time": obs_times,
                "discharge": obs_vals,
                "missing_times": missing_obs_times,
            },
            "metrics": {                "n": matched_days,
                "kge": round(kge_val, 3) if not np.isnan(kge_val) else None,
                "correlation": round(corr_val, 3) if not np.isnan(corr_val) else None,
                "kge_r": round(corr_val, 3) if not np.isnan(corr_val) else None,
                "kge_gamma": round(gamma_val, 3) if not np.isnan(gamma_val) else None,
                "kge_beta": round(beta_val, 3) if not np.isnan(beta_val) else None,
                "rmse": round(rmse_val, 3) if not np.isnan(rmse_val) else None,
                "rmse_unit": "m³ s⁻¹",
                "comparison": comparison_label,
            },
            "name": name,
            "river": river,
            "country_code": country,
            "lon": lon,
            "lat": lat,
            "upstream_area_km2": area,
        }

        if lead_series is not None:
            station_json["lead_series"] = lead_series

        with open(stations_dir / f"{station_id}.json", "w") as f:
            json.dump(station_json, f)

        catalog.append({
            "station_index": idx,
            "station_id": station_id,
            "source_station_id": sid,
            "expver": expver,
            "resolution_arcmin": 0,
            "valid_time_shift_hours": 0,
            "file": station_file,
            "name": name,
            "river": river,
            "country_code": country,
            "lon": lon,
            "lat": lat,
            "upstream_area_km2": area,
            "kge": round(kge_val, 3) if not np.isnan(kge_val) else None,
            "correlation": round(corr_val, 3) if not np.isnan(corr_val) else None,
            "rmse": round(rmse_val, 3) if not np.isnan(rmse_val) else None,
            "matched_days": matched_days,
        })

        metrics_rows.append({
            "station_index": idx,
            "source_station_id": sid,
            "name": name,
            "lon": lon,
            "lat": lat,
            "kge": round(kge_val, 3) if not np.isnan(kge_val) else "",
            "correlation": round(corr_val, 3) if not np.isnan(corr_val) else "",
            "rmse": round(rmse_val, 3) if not np.isnan(rmse_val) else "",
        })

        n_written += 1

    # Write catalog
    with open(exp_dir / "stations_catalog.json", "w") as f:
        json.dump(catalog, f)

    # Write metrics CSV
    if metrics_rows:
        keys = metrics_rows[0].keys()
        with open(exp_dir / "global_station_metrics.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(metrics_rows)

    print(f"  {expver}: {n_written} stations written to {exp_dir}")
    return n_written


def main():
    args = parse_args()
    out_root = args.output_root
    t0 = pd.Timestamp(args.date_start)
    t1 = pd.Timestamp(args.date_end) + pd.Timedelta(days=1)
    run_label = f"{args.date_start}_{args.date_end}_{RESOLUTION}arcmin"

    print("=" * 60)
    print("Prepare JRC AIFL / EFAS data for ifs-riverbench dashboard")
    print("=" * 60)

    # ── Load station metadata ────────────────────────────────────
    print("\nLoading station metadata...")
    meta = pd.read_csv(OBS_META)
    print(f"  {len(meta)} stations in metadata")

    # ── Load metrics ─────────────────────────────────────────────
    print("Loading AIFL metrics...")
    met_free = pd.read_csv(METRICS_FREE)
    met_da = pd.read_csv(METRICS_DA)
    print(f"  AIFL free: {len(met_free)} stations")
    print(f"  AIFL-DA:   {len(met_da)} stations")

    # ── Load 6h observations ─────────────────────────────────────
    print("Loading 6-hourly observations...")
    obs_ts = pd.read_parquet(OBS_TS)
    obs_ts["time"] = pd.to_datetime(obs_ts["time"])
    print(f"  {len(obs_ts)} timesteps, {len(obs_ts.columns)-1} stations")

    # ── Load EFAS 6.0 (daily CSV) ────────────────────────────────
    efas_ts = None
    if EFAS_CSV.exists():
        print(f"Loading EFAS 6.0 from {EFAS_CSV}...")
        efas_raw = pd.read_csv(EFAS_CSV, parse_dates=["Timestamp"])
        efas_raw = efas_raw.rename(columns={"Timestamp": "time"})
        efas_ts = efas_raw
        print(f"  {len(efas_raw)} timesteps")
    else:
        print(f"  EFAS CSV not found at {EFAS_CSV}, skipping EFAS experiment")

    # ── Compute EFAS metrics at JRC station locations ────────────
    efas_metrics = None
    if efas_ts is not None:
        print("Computing EFAS 6.0 metrics vs observations...")
        efas_met_rows = []
        mask_obs = (obs_ts["time"] >= t0) & (obs_ts["time"] <= t1)

        for _, mrow in meta.iterrows():
            sid = int(mrow["ObsID"])
            sid_str = str(sid)
            if sid_str not in obs_ts.columns or sid_str not in efas_ts.columns:
                continue

            # Resample obs to daily to match EFAS
            obs_daily = obs_ts.loc[mask_obs, ["time", sid_str]].set_index("time").resample("D").mean().dropna()
            efas_mask = (efas_ts["time"] >= t0) & (efas_ts["time"] <= t1)
            efas_daily = efas_ts.loc[efas_mask, ["time", sid_str]].set_index("time").dropna()

            common = obs_daily.index.intersection(efas_daily.index)
            if len(common) < 10:
                continue

            o = obs_daily.loc[common, sid_str].values.astype(float)
            s = efas_daily.loc[common, sid_str].values.astype(float)
            kge_val, r_val, gamma_val, beta_val, rmse_val = kge_prime(o, s)

            efas_met_rows.append({
                "ObsID": sid,
                "kge_prime": kge_val,
                "r": r_val,
                "gamma": gamma_val,
                "beta": beta_val,
                "rmse": rmse_val,
            })

        efas_metrics = pd.DataFrame(efas_met_rows)
        print(f"  EFAS 6.0: {len(efas_metrics)} stations with metrics")

    # ── Write each experiment ────────────────────────────────────
    print("\nWriting dashboard data...")

    write_experiment("AIFL", met_free, meta, obs_ts, None,
                     out_root, run_label, t0, t1, npz_mode="free")

    write_experiment("AIFL_DA", met_da, meta, obs_ts, None,
                     out_root, run_label, t0, t1, npz_mode="da")

    if efas_metrics is not None:
        write_experiment("EFAS_6.0", efas_metrics, meta, obs_ts, efas_ts,
                         out_root, run_label, t0, t1, npz_mode=None)

    print("\n" + "=" * 60)
    print("Done. Now build the dashboard:")
    print(f"  python3 02_build_dashboard.py \\")
    print(f"    --expver AIFL AIFL_DA" +
          (" EFAS_6.0" if efas_metrics is not None else "") + " \\")
    print(f"    --date-start {args.date_start} --date-end {args.date_end} \\")
    print(f"    --resolution {RESOLUTION} \\")
    print(f"    --metric kge --colour-mode best_metric \\")
    print(f"    --output-dir site_bundle --data-root {out_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()
