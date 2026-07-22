#!/usr/bin/env python3
"""
Benchmark CaMa-Flood river discharge against GloFAS/LISFLOOD discharge.

Default benchmark:
  experiments: j6n9, izay
  dates      : 2016-04-01 .. 2016-04-05 inclusive
  variable   : CaMa-Flood river discharge param 235270
               daily archive: date=valid day, step=24
               monthly archive: date=YYYYMM01, step=24*day_of_month
  reference  : GloFAS v5.0/LISFLOOD daily river discharge param 235270
               class=gf, stream=rfsd, type=sfo, forcing=ecmf-era5, step=24

This version also creates:
  * daily metrics for Global + continental domains
  * PNG daily spatial time-series plots of RMSE, MAE, mean error and correlation for each domain
  * average global mean-error and RMSE maps for each experiment
  * difference maps for every experiment relative to the last expver, treated as control

Example:
  python3 benchmark_cmf_vs_glofas.py
  python3 benchmark_cmf_vs_glofas.py --start-date 20160401 --end-date 20160405
  python3 benchmark_cmf_vs_glofas.py --expvers j6n9 j7ab izay --river-threshold 2  # izay is control
"""

from __future__ import annotations

import argparse
import csv
import pandas as pd
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import metview as mv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from netCDF4 import Dataset
    HAVE_NETCDF4 = True
except Exception:
    HAVE_NETCDF4 = False


# ============================================================
# Domain definitions
# bbox = [lat_min, lon_min, lat_max, lon_max]
# ============================================================

CONTINENT_BBOX = {
    "Africa":        [-35.0,  -20.0, 38.0,  55.0],
    "Europe":        [34.0,  -25.0,  72.0,  45.0],
    "Asia":          [-10.0,   25.0, 82.0, 180.0],
    "NorthAmerica":  [5.0, -170.0,   83.0, -50.0],
    "SouthAmerica":  [-60.0,  -85.0, 15.0, -30.0],
    "Oceania":       [-50.0,  110.0, 10.0, 180.0],
}

DOMAINS = {"Global": None, **CONTINENT_BBOX}

# ============================================================
# Archive layout
# ============================================================

MONTHLY_ARCHIVE_EXPVERS = {
    "j6ft",   # MSWEP3 hourly precipitation
    "j65z",   # MSWEP3 hourly precipitation
    "j6fu",   # MSWEP3 daily precipitation
    "j6fs",   # MSWEP3 monthly precipitation
    "j6gq",   # EFAS 6-hourly precipitation 
    "iyp3",   # 50r1 bugfix ERA5 control
    "j6zm",   # 49r1 Oper forcing
    "j734",   # 50r1 Oper forcing
    "iwya",   # 50r1 ERA5 forcing
    "j7xs",   # 50r1 GP4Hydro forcing
}

DAILY_ARCHIVE_EXPVERS = {
    "j6nx",   # GloFAS assimilation 1y fit parameters
    "j6n9",   # GloFAS assimilation 5y fit+tuned parameters
    "j6xo",   # GloFAS assimilation 5y fit+reduced parameters
    "izay",   # Oper forced control
}

# ============================================================
# Experiment metadata
# ============================================================

EXPERIMENT_LABELS = {
    "j6ft": "MSWEP3 hourly precipitation",
    "j65z": "MSWEP3 hourly precipitation",
    "j6fu": "MSWEP3 daily precipitation",
    "j6fs": "MSWEP3 monthly precipitation",
    "j6gq": "EFAS 6-hourly precipitation",
    "j6nx": "GloFAS AN with 1-year fit parameters",
    "j6n9": "GloFAS AN with 5-year fit + tuned parameters",
    "j6xo": "GloFAS AN with 5-year fit + reduced parameters",
    "izay": "Oper-forced control",
    "iyp3": "5Or1 ERA5 new runoff",
    "j6zm": "49r1 Oper forcing",
    "j734": "50r1 Oper forcing",
    "j7xs": "50r1 GP4Hydro",
    "iwya": "50r1 ERA5 Control",
}

# ============================================================
# Date helpers
# ============================================================

def parse_yyyymmdd(s: str) -> datetime:
    s = s.strip().replace("-", "")
    return datetime.strptime(s, "%Y%m%d")

def yyyymmdd(dt: datetime) -> str:
    return dt.strftime("%Y%m%d")

def first_day_of_month(date: str) -> str:
    d = parse_yyyymmdd(date)
    return d.strftime("%Y%m01")

def monthly_step_hours(date: str) -> int:
    """Return CaMa monthly-archive step for a valid date: 24, 48, ..., 744."""
    d = parse_yyyymmdd(date)
    return 24 * d.day

def date_range(start: str, end: str) -> list[str]:
    d0 = parse_yyyymmdd(start)
    d1 = parse_yyyymmdd(end)
    if d1 < d0:
        raise ValueError(f"end-date {end} is before start-date {start}")
    out = []
    d = d0
    while d <= d1:
        out.append(yyyymmdd(d))
        d += timedelta(days=1)
    return out

def exp_label(expver):
    return EXPERIMENT_LABELS.get(expver, expver)

def date_label(date: str) -> str:
    return parse_yyyymmdd(date).strftime("%d %b")

def _dynamic_tick_indices(n: int, max_ticks: int = 14) -> np.ndarray:
    """Return a sparse set of x indices for readable daily time-series axes."""
    if n <= 0:
        return np.asarray([], dtype=int)
    if n <= max_ticks:
        return np.arange(n, dtype=int)

    step = int(np.ceil(n / max_ticks))
    idx = np.arange(0, n, step, dtype=int)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return idx

def _dynamic_date_labels(dates: list[str], idx: np.ndarray) -> list[str]:
    """Compact labels for daily benchmark plots, including years for long periods."""
    if len(dates) == 0 or idx.size == 0:
        return []

    d0 = parse_yyyymmdd(dates[0])
    d1 = parse_yyyymmdd(dates[-1])
    ndays = max((d1 - d0).days + 1, len(dates))

    labels = []
    for i in idx:
        d = parse_yyyymmdd(dates[int(i)])
        if ndays > 730:
            labels.append(d.strftime("%b %Y"))
        elif ndays > 120:
            labels.append(d.strftime("%d %b %Y"))
        else:
            labels.append(d.strftime("%d %b"))
    return labels

def apply_dynamic_date_ticks(ax, dates: list[str], max_ticks: int = 14) -> None:
    """Apply sparse, readable x-axis date ticks to an axis with x=0..N-1."""
    if not dates:
        return
    idx = _dynamic_tick_indices(len(dates), max_ticks=max_ticks)
    ax.set_xticks(idx)
    ax.set_xticklabels(_dynamic_date_labels(dates, idx), rotation=30, ha="right")

# ============================================================
# Field helpers
# ============================================================

def get_values(field) -> np.ndarray:
    return np.asarray(field.values(), dtype=np.float32)

def clean_field(x: np.ndarray, name: str, missing_abs_threshold: float = 1.0e19) -> np.ndarray:
    """Convert non-finite and huge missing values to NaN."""
    x = np.asarray(x, dtype=np.float32)
    bad = (~np.isfinite(x)) | (np.abs(x) > missing_abs_threshold)
    nbad = int(np.count_nonzero(bad))
    if nbad > 0:
        print(f"{name}: replacing {nbad:,} missing/extreme values with NaN")
        x = x.copy()
        x[bad] = np.nan
    return x

def _regular_1d_from_field(lats: np.ndarray, lons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return 1-D latitude and longitude coordinate vectors preserving field scan order."""
    ny, nx = infer_regular_grid(lats, lons)
    lat2d = lats.reshape(ny, nx)
    lon2d = lons.reshape(ny, nx)
    return lat2d[:, 0].astype(np.float64), lon2d[0, :].astype(np.float64)

def _nearest_indices(src_1d: np.ndarray, target_1d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbour indices from monotonic 1-D source coordinates to target coordinates."""
    src = np.asarray(src_1d, dtype=np.float64)
    tgt = np.asarray(target_1d, dtype=np.float64)

    # Work in ascending order for searchsorted, then convert back to original indices.
    if src[0] > src[-1]:
        src_asc = src[::-1]
        reversed_src = True
    else:
        src_asc = src
        reversed_src = False

    idx = np.searchsorted(src_asc, tgt)
    idx0 = np.clip(idx - 1, 0, len(src_asc) - 1)
    idx1 = np.clip(idx,     0, len(src_asc) - 1)
    choose1 = np.abs(src_asc[idx1] - tgt) < np.abs(src_asc[idx0] - tgt)
    idx_asc = np.where(choose1, idx1, idx0)

    if reversed_src:
        idx_orig = len(src) - 1 - idx_asc
    else:
        idx_orig = idx_asc

    in_range = (tgt >= np.nanmin(src) - 1e-6) & (tgt <= np.nanmax(src) + 1e-6)
    return idx_orig.astype(np.int64), in_range

def remap_glofas_to_cama_domain(glofas_field, cama_field, method: str = "nearest") -> np.ndarray:
    """
    Return GloFAS discharge on the CaMa grid.

    Cases handled:
      1. Same grid size: direct values.
      2. Same resolution but GloFAS lacks Antarctica: latitude padding.
      3. Different regular lat/lon grids, e.g. GloFAS 3 arcmin and CaMa 15 arcmin:
         nearest-neighbour sampling from GloFAS to CaMa cell centres.

    For discharge, nearest-neighbour is the safest first benchmark choice because
    discharge is a channel variable, not an areal density. Averaging 5x5 GloFAS
    cells would smear rivers and can artificially lower peaks.
    """
    if method != "nearest":
        raise ValueError(f"Unsupported GloFAS remap method: {method}. Currently supported: nearest")

    q_glo_raw = get_values(glofas_field)
    q_cama = get_values(cama_field)

    lat_glo, lon_glo = get_lat_lon(glofas_field)
    lat_cama, lon_cama = get_lat_lon(cama_field)

    if q_glo_raw.size == q_cama.size:
        print("GloFAS and CaMa-Flood have same grid size.")
        return q_glo_raw

    print("GloFAS and CaMa-Flood have different grid sizes.")
    print("GloFAS points:", f"{q_glo_raw.size:,}")
    print("CaMa points  :", f"{q_cama.size:,}")
    print("GloFAS lat min/max:", float(np.nanmin(lat_glo)), float(np.nanmax(lat_glo)))
    print("CaMa lat min/max  :", float(np.nanmin(lat_cama)), float(np.nanmax(lat_cama)))

    ny_g, nx_g = infer_regular_grid(lat_glo, lon_glo)
    ny_c, nx_c = infer_regular_grid(lat_cama, lon_cama)
    print(f"GloFAS grid: ny={ny_g:,} nx={nx_g:,}")
    print(f"CaMa grid  : ny={ny_c:,} nx={nx_c:,}")

    # Old case: same longitude count and GloFAS only lacks southern latitudes.
    if nx_g == nx_c and q_glo_raw.size < q_cama.size and np.nanmin(lat_glo) > np.nanmin(lat_cama):
        q_glo_global = np.full(q_cama.shape, np.nan, dtype=np.float32)
        south = float(np.nanmin(lat_glo))
        north = float(np.nanmax(lat_glo))
        mask = (lat_cama >= south - 1e-6) & (lat_cama <= north + 1e-6)
        print("GloFAS insert mask points:", f"{np.count_nonzero(mask):,}")
        if np.count_nonzero(mask) != q_glo_raw.size:
            raise ValueError("Cannot safely insert GloFAS into CaMa global grid.")
        q_glo_global[mask] = q_glo_raw
        return q_glo_global

    # General regular-grid nearest-neighbour remap.
    latg_1d, long_1d = _regular_1d_from_field(lat_glo, lon_glo)
    latc_1d, lonc_1d = _regular_1d_from_field(lat_cama, lon_cama)

    ilat, valid_lat = _nearest_indices(latg_1d, latc_1d)
    ilon, valid_lon = _nearest_indices(long_1d, lonc_1d)

    qg2d = q_glo_raw.reshape(ny_g, nx_g)
    out2d = np.full((ny_c, nx_c), np.nan, dtype=np.float32)

    valid_rows = np.where(valid_lat)[0]
    valid_cols = np.where(valid_lon)[0]
    print("Nearest remap valid CaMa rows:", f"{valid_rows.size:,}/{ny_c:,}")
    print("Nearest remap valid CaMa cols:", f"{valid_cols.size:,}/{nx_c:,}")

    if valid_rows.size > 0 and valid_cols.size > 0:
        out2d[np.ix_(valid_rows, valid_cols)] = qg2d[np.ix_(ilat[valid_rows], ilon[valid_cols])]

    print("Nearest-remapped GloFAS valid points:", f"{np.count_nonzero(np.isfinite(out2d)):,}")
    return out2d.reshape(-1)

def get_lat_lon(field) -> tuple[np.ndarray, np.ndarray]:
    lats = np.asarray(mv.latitudes(field), dtype=np.float32)
    lons = np.asarray(mv.longitudes(field), dtype=np.float32)
    return lats, lons

def infer_regular_grid(lats: np.ndarray, lons: np.ndarray) -> tuple[int, int]:
    ulat = np.unique(lats)
    ulon = np.unique(lons)
    ny = len(ulat)
    nx = len(ulon)
    if ny * nx != lats.size:
        raise ValueError(
            f"Cannot infer regular grid: ny={ny}, nx={nx}, npts={lats.size}"
        )
    return ny, nx


# ============================================================
# Static-network upstream-area consistency
# ============================================================

def _normalise_target_longitudes(target_lon: np.ndarray, source_lon: np.ndarray) -> np.ndarray:
    """Express target longitudes in the convention used by source_lon."""
    out = np.asarray(target_lon, dtype=np.float64).copy()
    src_min = float(np.nanmin(source_lon))
    src_max = float(np.nanmax(source_lon))
    if src_min >= 0.0 and src_max > 180.0:
        out = np.mod(out, 360.0)
    elif src_min < 0.0 and src_max <= 180.0:
        out = ((out + 180.0) % 360.0) - 180.0
    return out


def read_static_field_on_target_grid(
    path: str | Path,
    varname: str,
    target_lat1d: np.ndarray,
    target_lon1d: np.ndarray,
) -> np.ndarray:
    """Read a regular-grid NetCDF field at nearest target cell centres.

    Only the source rows needed by the target grid are read, avoiding loading the
    full 3-arcmin global field into memory.
    """
    if not HAVE_NETCDF4:
        raise RuntimeError("netCDF4 is required for the upstream-area filter")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with Dataset(path, "r") as ds:
        if varname not in ds.variables:
            raise KeyError(f"Variable {varname!r} is missing from {path}")

        src_lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
        src_lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
        tgt_lon = _normalise_target_longitudes(target_lon1d, src_lon)

        ilat, valid_lat = _nearest_indices(src_lat, target_lat1d)
        ilon, valid_lon = _nearest_indices(src_lon, tgt_lon)

        out = np.full((target_lat1d.size, target_lon1d.size), np.nan, dtype=np.float32)
        v = ds.variables[varname]

        valid_cols = np.where(valid_lon)[0]
        if valid_cols.size == 0:
            return out

        # Cache repeated source rows if the target is finer than the source.
        row_cache: dict[int, np.ndarray] = {}
        for j in np.where(valid_lat)[0]:
            js = int(ilat[j])
            if js not in row_cache:
                row = np.ma.filled(v[js, :], np.nan).astype(np.float32, copy=False)
                row_cache[js] = row
            out[j, valid_cols] = row_cache[js][ilon[valid_cols]]

    out[(~np.isfinite(out)) | (out <= 0.0)] = np.nan
    return out


def build_uparea_consistency_mask(
    cama_lats: np.ndarray,
    cama_lons: np.ndarray,
    uparea_15min_file: str | Path,
    uparea_03min_file: str | Path,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mask where 3' and 15' upstream drainage areas agree.

    The relative mismatch is symmetric:
        abs(A03 - A15) / max(A03, A15)
    and a point is retained when this value is <= tolerance.
    """
    if tolerance < 0.0:
        raise ValueError("uparea tolerance must be non-negative")

    ny, nx = infer_regular_grid(cama_lats, cama_lons)
    lat1d, lon1d = _regular_1d_from_field(cama_lats, cama_lons)

    print("Reading 15-arcmin upstream area:", uparea_15min_file)
    area15 = read_static_field_on_target_grid(
        uparea_15min_file, "uparea", lat1d, lon1d
    )
    print("Reading/remapping 3-arcmin upstream area:", uparea_03min_file)
    area03 = read_static_field_on_target_grid(
        uparea_03min_file, "uparea", lat1d, lon1d
    )

    valid = np.isfinite(area15) & np.isfinite(area03) & (area15 > 0.0) & (area03 > 0.0)
    rel_diff = np.full((ny, nx), np.nan, dtype=np.float32)
    denom = np.maximum(area15[valid], area03[valid])
    rel_diff[valid] = np.abs(area03[valid] - area15[valid]) / denom
    mask = valid & (rel_diff <= tolerance)

    nvalid = int(np.count_nonzero(valid))
    nkeep = int(np.count_nonzero(mask))
    print("Upstream-area consistency")
    print("-------------------------")
    print(f"Tolerance                 : {100.0 * tolerance:.2f}%")
    print(f"Valid area pairs          : {nvalid:,}")
    print(f"Retained area pairs       : {nkeep:,}")
    print(f"Retained fraction         : {100.0 * nkeep / nvalid:.2f}%" if nvalid else "Retained fraction         : n/a")
    if nvalid:
        vals = rel_diff[valid]
        print("Mismatch percentiles [%]  :", np.nanpercentile(vals, [50, 75, 90, 95, 99]) * 100.0)
    print()

    return mask.reshape(-1), area15.reshape(-1), area03.reshape(-1)

# ============================================================
# Metrics
# ============================================================

def metrics(obs: np.ndarray, model: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    valid = mask & np.isfinite(obs) & np.isfinite(model)
    n = int(np.count_nonzero(valid))

    if n == 0:
        return {
            "n": 0,
            "bias": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "corr": np.nan,
            "nse": np.nan,
            "kge": np.nan,
            "obs_mean": np.nan,
            "model_mean": np.nan,
            "obs_median": np.nan,
            "model_median": np.nan,
        }

    o = obs[valid].astype(np.float64)
    m = model[valid].astype(np.float64)
    d = m - o

    bias = float(np.mean(d))
    rmse = float(np.sqrt(np.mean(d * d)))
    mae = float(np.mean(np.abs(d)))

    obs_mean = float(np.mean(o))
    model_mean = float(np.mean(m))
    obs_std = float(np.std(o))
    model_std = float(np.std(m))

    if n >= 2 and obs_std > 0.0 and model_std > 0.0:
        corr = float(np.corrcoef(o, m)[0, 1])
    else:
        corr = np.nan

    # Nash-Sutcliffe Efficiency
    if n >= 2:
        denom = float(np.sum((o - obs_mean) ** 2))
        if denom > 0.0:
            nse = float(1.0 - np.sum((m - o) ** 2) / denom)
        else:
            nse = np.nan
    else:
        nse = np.nan

    # Kling-Gupta Efficiency, 2009 formulation
    if (
        n >= 2
        and np.isfinite(corr)
        and obs_std > 0.0
        and model_std > 0.0
        and obs_mean != 0.0
    ):
        alpha = model_std / obs_std
        beta = model_mean / obs_mean
        kge = float(1.0 - np.sqrt((corr - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))
    else:
        kge = np.nan

    return {
        "n": n,
        "bias": bias,
        "rmse": rmse,
        "mae": mae,
        "corr": corr,
        "nse": nse,
        "kge": kge,
        "obs_mean": obs_mean,
        "model_mean": model_mean,
        "obs_median": float(np.median(o)),
        "model_median": float(np.median(m)),
    }

def print_metrics(label: str, m: dict[str, float | int]) -> None:
    print(label)
    print("-" * len(label))
    print("N      :", f"{m['n']:,}")
    print("BIAS   :", m["bias"])
    print("RMSE   :", m["rmse"])
    print("MAE    :", m["mae"])
    print("CORR   :", m["corr"])
    print("NSE    :", m["nse"])
    print("KGE    :", m["kge"])
    print("OBS μ  :", m["obs_mean"])
    print("MOD μ  :", m["model_mean"])
    print()

def domain_mask_for_bbox(lats: np.ndarray, lons: np.ndarray, bbox: list[float] | None) -> np.ndarray:
    if bbox is None:
        return np.ones(lats.shape, dtype=bool)

    lat_min, lon_min, lat_max, lon_max = bbox
    return (
        (lats >= lat_min) & (lats <= lat_max)
        & (lons >= lon_min) & (lons <= lon_max)
    )

# ============================================================
# MARS/cache helpers
# ============================================================

def retrieve_or_read(path: Path, request: dict, force: bool = False):
    if path.exists() and not force:
        print("Using cached file:", path)
        return mv.read(str(path))

    print("MARS request:")
    print(request)
    fs = mv.retrieve(request)
    fs.write(str(path))
    print("Saved:", path)
    return fs

def retrieve_glofas(
    date: str,
    outdir: Path,
    force: bool = False,
    configuration: str = "v5.0",
    forcing: str = "ecmf-era5",
    expver: str = "1",
):
    """Retrieve/read GloFAS v5 daily river discharge for one valid date.

    The request follows the GloFAS v5 MARS archive layout:
      class=gf, stream=rfsd, type=sfo, model=lisflood,
      configuration=v5.0, forcing=ecmf-era5, timespan=24h,
      date=<valid date>, time=00, step=24, param=235270.

    The configuration and forcing are included in the cache filename so that
    legacy GloFAS files or alternative configurations cannot be reused by
    mistake.
    """
    safe_configuration = configuration.replace("/", "-")
    safe_forcing = forcing.replace("/", "-")
    outfile = outdir / f"glofas_{safe_configuration}_{safe_forcing}_{date}.grib"

    req = {
        "class": "gf",
        "configuration": configuration,
        "date": date,
        "expver": expver,
        "forcing": forcing,
        "levtype": "sfc",
        "model": "lisflood",
        "param": "235270",
        "step": 24,
        "stream": "rfsd",
        "time": "00:00:00",
        "timespan": "24h",
        "type": "sfo",
    }
    return retrieve_or_read(outfile, req, force=force)

def retrieve_cama_discharge_daily(date: str, expver: str, outdir: Path, force: bool = False):
    """Retrieve/read normal daily archive: date=valid date, step=24."""
    outfile = outdir / f"camaflood_dis_{expver}_{date}.grib"
    req = {
        "class": "rd",
        "stream": "oper",
        "type": "fc",
        "levtype": "sfc",
        "expver": expver,
        "date": [date],
        "time": 0,
        "step": 24,
        "param": ["235270"],
    }
    return retrieve_or_read(outfile, req, force=force)

def retrieve_cama_discharge_monthly(date: str, expver: str, outdir: Path, force: bool = False):
    """
    Retrieve/read monthly-chunk archive.

    For a valid date YYYYMMDD, the archive date is fixed to YYYYMM01 and
    the daily discharge is stored at step = 24 * day_of_month.
    Examples: valid 20250101 -> date=20250101, step=24;
              valid 20250102 -> date=20250101, step=48;
              valid 20250131 -> date=20250101, step=744.
    """
    month_date = first_day_of_month(date)
    step = monthly_step_hours(date)
    outfile = outdir / f"camaflood_dis_{expver}_{date}_monthly_d{month_date}_s{step}.grib"
    req = {
        "class": "rd",
        "stream": "oper",
        "type": "fc",
        "levtype": "sfc",
        "expver": expver,
        "date": [month_date],
        "time": 0,
        "step": step,
        "param": ["235270"],
    }
    return retrieve_or_read(outfile, req, force=force)

def retrieve_cama_discharge(date, expver, outdir, force=False, archive_mode="auto"):

    if archive_mode == "auto":
        if expver in MONTHLY_ARCHIVE_EXPVERS:
            archive_mode = "monthly"
        elif expver in DAILY_ARCHIVE_EXPVERS:
            archive_mode = "daily"
        else:
            raise ValueError(f"Unknown archive mode for expver {expver}")

    if archive_mode == "daily":
        return retrieve_cama_discharge_daily(date, expver, outdir, force=force)

    if archive_mode == "monthly":
        return retrieve_cama_discharge_monthly(date, expver, outdir, force=force)

    raise ValueError(f"Unsupported archive_mode: {archive_mode}")

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        print("No rows to write for", path)
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print("Saved CSV:", path)

# ============================================================
# Plot helpers
# ============================================================

def rows_to_float(rows: list[dict], domain: str, expver: str, metric_key: str):
    dates = []
    values = []
    for r in rows:
        if r["domain"] == domain and r["expver"] == expver:
            dates.append(r["date"])
            try:
                values.append(float(r[metric_key]))
            except Exception:
                values.append(np.nan)
    return dates, np.asarray(values, dtype=np.float64)


def rows_to_count(rows: list[dict], domain: str, expver: str, count_key: str = "n"):
    dates = []
    values = []
    for r in rows:
        if r.get("domain") == domain and r.get("expver") == expver:
            dates.append(str(r.get("date")))
            try:
                values.append(float(r.get(count_key, np.nan)))
            except Exception:
                values.append(np.nan)
    return dates, np.asarray(values, dtype=np.float64)

def plot_domain_timeseries(rows: list[dict], expvers: list[str], 
        domains: list[str], outdir: Path, start_date, end_date):
    plot_dir = outdir / "plots_timeseries"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Daily diagnostics are spatial statistics computed independently for each day.
    # RMSE, MAE, mean error and spatial correlation have direct interpretation.
    # Daily spatial NSE/KGE are intentionally omitted; temporal NSE/KGE remain
    # available in the period maps and pooled benchmark summaries.
    metric_specs = [
        ("rmse", "RMSE [m³ s⁻¹]"),
        ("mae", "MAE [m³ s⁻¹]"),
        ("bias_model_minus_glofas", "Mean error [m³ s⁻¹]"),
        ("correlation", "Spatial correlation"),
    ]

    for domain in domains:
        fig, axes = plt.subplots(4, 1, figsize=(10, 13), sharex=True, constrained_layout=True)
        tick_dates = None

        for ax, (metric_key, ylabel) in zip(axes, metric_specs):
            any_data = False
            count_ax = ax.twinx()
            count_ax.patch.set_alpha(0.0)
            count_series = []
            for expver in expvers:
                dates, vals = rows_to_float(rows, domain, expver, metric_key)
                if len(dates) == 0:
                    continue
                if tick_dates is None:
                    tick_dates = dates
                labelexp = EXPERIMENT_LABELS.get(expver, expver)
                x = np.arange(len(dates))
                line, = ax.plot(x, vals, linewidth=1.8, label=f"{expver} - {labelexp}")

                count_dates, counts = rows_to_count(rows, domain, expver, count_key="n")
                if len(count_dates) == len(dates):
                    count_series.append((x, counts, line.get_color()))
                any_data = True

            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)

            if count_series:
                width = 0.8 / max(1, len(count_series))
                for i, (xv, counts, color) in enumerate(count_series):
                    offset = (i - (len(count_series) - 1) / 2.0) * width
                    count_ax.bar(xv + offset, counts, width=width, alpha=0.14, color=color, edgecolor="none")
                flat_counts = np.concatenate([c[np.isfinite(c)] for _, c, _ in count_series if np.any(np.isfinite(c))])
                if flat_counts.size > 0:
                    count_ax.set_ylim(0.0, 1.05 * float(np.nanmax(flat_counts)))
                count_ax.set_ylabel("N points", color="#475569")
                count_ax.tick_params(axis="y", colors="#475569", labelsize=8)
                count_ax.grid(False)

            if any_data:
                ax.legend()

        axes[0].set_title(f"Daily spatial diagnostics — {domain}\n" f"( {start_date} – {end_date} )", fontsize=12)
        axes[-1].set_xlabel("Date")
        if tick_dates is not None:
            apply_dynamic_date_ticks(axes[-1], tick_dates, max_ticks=14)

        outfile = plot_dir / f"timeseries_metrics_{domain}.png"
        fig.savefig(outfile, dpi=150)
        plt.close(fig)
        print("Saved plot:", outfile)

        # Standalone MAE plot so dashboard can expose MAE as a dedicated column.
        fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
        count_ax = ax.twinx()
        count_ax.patch.set_alpha(0.0)
        count_series = []
        any_data = False
        mae_tick_dates = None

        for expver in expvers:
            dates, vals = rows_to_float(rows, domain, expver, "mae")
            if len(dates) == 0:
                continue
            if mae_tick_dates is None:
                mae_tick_dates = dates
            x = np.arange(len(dates))
            labelexp = EXPERIMENT_LABELS.get(expver, expver)
            line, = ax.plot(x, vals, linewidth=2.0, label=f"{expver} - {labelexp}")

            count_dates, counts = rows_to_count(rows, domain, expver, count_key="n")
            if len(count_dates) == len(dates):
                count_series.append((x, counts, line.get_color()))
            any_data = True

        ax.set_title(
            f"Daily MAE vs GloFAS — {domain}\n"
            f"( {start_date} – {end_date} )",
            fontsize=12,
        )
        ax.set_ylabel("MAE [m³ s⁻¹]")
        ax.set_xlabel("Date")
        ax.grid(True, alpha=0.3)

        if count_series:
            width = 0.8 / max(1, len(count_series))
            for i, (xv, counts, color) in enumerate(count_series):
                offset = (i - (len(count_series) - 1) / 2.0) * width
                count_ax.bar(xv + offset, counts, width=width, alpha=0.14, color=color, edgecolor="none")
            flat_counts = np.concatenate([c[np.isfinite(c)] for _, c, _ in count_series if np.any(np.isfinite(c))])
            if flat_counts.size > 0:
                count_ax.set_ylim(0.0, 1.05 * float(np.nanmax(flat_counts)))
            count_ax.set_ylabel("N points", color="#475569")
            count_ax.tick_params(axis="y", colors="#475569", labelsize=8)
            count_ax.grid(False)

        if any_data:
            ax.legend()
            if mae_tick_dates is not None:
                apply_dynamic_date_ticks(ax, mae_tick_dates, max_ticks=14)

        mae_outfile = plot_dir / f"timeseries_mae_{domain}.png"
        fig.savefig(mae_outfile, dpi=150)
        plt.close(fig)
        print("Saved plot:", mae_outfile)

def plot_relative_improvement_timeseries(rows: list[dict], expvers: list[str], control_expver: str,
        domains: list[str], outdir: Path, start_date, end_date):
    """Plot daily relative RMSE improvement against the control for each domain.

    Positive values mean the experiment has lower RMSE than the control.
    """
    plot_dir = outdir / "plots_timeseries"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for domain in domains:
        fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
        any_data = False
        tick_dates = None
        count_ax = ax.twinx()
        count_ax.patch.set_alpha(0.0)
        count_series = []

        for expver in expvers:
            if expver == control_expver:
                continue

            dates = []
            vals = []
            for r in rows:
                if r.get("domain") == domain and r.get("expver") == expver:
                    dates.append(str(r["date"]))
                    try:
                        vals.append(float(r["rmse_improvement_pct_vs_control"]))
                    except Exception:
                        vals.append(np.nan)

            if len(dates) == 0:
                continue

            x = np.arange(len(dates))
            if tick_dates is None:
                tick_dates = dates
            labelexp = EXPERIMENT_LABELS.get(expver, expver)
            line, = ax.plot(x, np.asarray(vals, dtype=np.float64), linewidth=2.0, label=f"{expver} - {labelexp}")

            count_dates, counts = rows_to_count(rows, domain, expver, count_key="n_expver")
            if len(count_dates) == len(dates):
                count_series.append((x, counts, line.get_color()))
            any_data = True

        ax.axhline(0.0, linewidth=1.0, color="k", alpha=0.6)
        ax.set_title(
            f"Relative RMSE improvement vs control — {domain}\n"
            f"control = {control_expver} - {EXPERIMENT_LABELS.get(control_expver, control_expver)}; "
            f"{start_date} – {end_date}",
            fontsize=12,
        )
        ax.set_ylabel("RMSE improvement vs control [%]")
        ax.set_xlabel("Date")
        ax.grid(True, alpha=0.3)

        if count_series:
            width = 0.8 / max(1, len(count_series))
            for i, (xv, counts, color) in enumerate(count_series):
                offset = (i - (len(count_series) - 1) / 2.0) * width
                count_ax.bar(xv + offset, counts, width=width, alpha=0.14, color=color, edgecolor="none")
            flat_counts = np.concatenate([c[np.isfinite(c)] for _, c, _ in count_series if np.any(np.isfinite(c))])
            if flat_counts.size > 0:
                count_ax.set_ylim(0.0, 1.05 * float(np.nanmax(flat_counts)))
            count_ax.set_ylabel("N points", color="#475569")
            count_ax.tick_params(axis="y", colors="#475569", labelsize=8)
            count_ax.grid(False)

        if any_data:
            ax.legend()
            if tick_dates is not None:
                apply_dynamic_date_ticks(ax, tick_dates, max_ticks=14)

        outfile = plot_dir / f"timeseries_relative_rmse_improvement_{domain}.png"
        fig.savefig(outfile, dpi=150)
        plt.close(fig)
        print("Saved plot:", outfile)


def plot_control_difference_timeseries(rows: list[dict], expvers: list[str], control_expver: str,
        domains: list[str], outdir: Path, start_date, end_date):
    """Plot daily differences against the control for skill metrics.

    These are absolute differences, not relative percentages, because
    correlation, NSE and KGE are bounded skill scores where percentage changes
    can be misleading. Positive values mean the experiment is better than the
    control for correlation/NSE/KGE. For bias, negative absolute-bias difference
    means the experiment has smaller absolute bias than the control.
    """
    plot_dir = outdir / "plots_timeseries"
    plot_dir.mkdir(parents=True, exist_ok=True)

    metric_specs = [
        ("bias_difference_vs_control", "Bias difference [m³ s⁻¹]", "ΔBias = Bias(exp) - Bias(control)"),
        ("abs_bias_difference_vs_control", "Absolute-bias difference [m³ s⁻¹]", "Δ|Bias| < 0 is better"),
        ("correlation_difference_vs_control", "ΔCorrelation", "Positive = better"),
    ]

    for domain in domains:
        fig, axes = plt.subplots(len(metric_specs), 1, figsize=(11, 10), sharex=True, constrained_layout=True)

        tick_dates = None
        for ax, (metric_key, ylabel, subtitle) in zip(axes, metric_specs):
            any_data = False
            count_ax = ax.twinx()
            count_ax.patch.set_alpha(0.0)
            count_series = []

            for expver in expvers:
                if expver == control_expver:
                    continue

                dates = []
                vals = []
                for r in rows:
                    if r.get("domain") == domain and r.get("expver") == expver:
                        dates.append(str(r["date"]))
                        try:
                            vals.append(float(r[metric_key]))
                        except Exception:
                            vals.append(np.nan)

                if len(dates) == 0:
                    continue

                x = np.arange(len(dates))
                if tick_dates is None:
                    tick_dates = dates
                labelexp = EXPERIMENT_LABELS.get(expver, expver)
                line, = ax.plot(x, np.asarray(vals, dtype=np.float64), linewidth=2.0, label=f"{expver} - {labelexp}")

                count_dates, counts = rows_to_count(rows, domain, expver, count_key="n_expver")
                if len(count_dates) == len(dates):
                    count_series.append((x, counts, line.get_color()))
                any_data = True

            ax.axhline(0.0, linewidth=1.0, color="k", alpha=0.6)
            ax.set_ylabel(ylabel)
            ax.set_title(subtitle, fontsize=10)
            ax.grid(True, alpha=0.3)

            if count_series:
                width = 0.8 / max(1, len(count_series))
                for i, (xv, counts, color) in enumerate(count_series):
                    offset = (i - (len(count_series) - 1) / 2.0) * width
                    count_ax.bar(xv + offset, counts, width=width, alpha=0.14, color=color, edgecolor="none")
                flat_counts = np.concatenate([c[np.isfinite(c)] for _, c, _ in count_series if np.any(np.isfinite(c))])
                if flat_counts.size > 0:
                    count_ax.set_ylim(0.0, 1.05 * float(np.nanmax(flat_counts)))
                count_ax.set_ylabel("N points", color="#475569")
                count_ax.tick_params(axis="y", colors="#475569", labelsize=8)
                count_ax.grid(False)

            if any_data:
                ax.legend()

        axes[0].set_title(
            f"Metric differences vs control — {domain}\n"
            f"control = {control_expver} - {EXPERIMENT_LABELS.get(control_expver, control_expver)}; "
            f"{start_date} – {end_date}",
            fontsize=12,
        )
        axes[-1].set_xlabel("Date")
        if tick_dates is not None:
            apply_dynamic_date_ticks(axes[-1], tick_dates, max_ticks=14)

        outfile = plot_dir / f"timeseries_metric_differences_vs_control_{domain}.png"
        fig.savefig(outfile, dpi=150)
        plt.close(fig)
        print("Saved plot:", outfile)

def relative_rmse_improvement_map(rmse_exp: np.ndarray, rmse_control: np.ndarray) -> np.ndarray:
    """Return 100 * (RMSE_control - RMSE_exp) / RMSE_control.

    Positive values mean the experiment improves over the control. Values are
    clipped to [-200, 200] only for plotting/NetCDF readability; cells with
    zero/invalid control RMSE are masked.
    """
    out = np.full(rmse_exp.shape, np.nan, dtype=np.float32)
    good = (
        np.isfinite(rmse_exp)
        & np.isfinite(rmse_control)
        & (rmse_control > 0.0)
    )
    out[good] = (100.0 * (rmse_control[good] - rmse_exp[good]) / rmse_control[good]).astype(np.float32)
    out[~np.isfinite(out)] = np.nan
    return np.clip(out, -200.0, 200.0)


REL_IMPROVEMENT_LEVELS = [-200, -100, -50, -20, -10, 10, 20, 50, 100, 200]

def robust_limits(x: np.ndarray, symmetric: bool = False, p: float = 99.0):
    v = x[np.isfinite(x)]
    if v.size == 0:
        return None, None
    if symmetric:
        vmax = float(np.nanpercentile(np.abs(v), p))
        if vmax <= 0:
            vmax = float(np.nanmax(np.abs(v)))
        return -vmax, vmax
    vmax = float(np.nanpercentile(v, p))
    vmin = float(np.nanpercentile(v, 1.0))
    if vmax <= vmin:
        vmax = float(np.nanmax(v))
        vmin = float(np.nanmin(v))
    return vmin, vmax

def _coord_edges_1d(c: np.ndarray) -> np.ndarray:
    """Return cell-edge coordinates from cell-centre coordinates."""
    c = np.asarray(c, dtype=np.float64)
    edges = np.empty(c.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (c[:-1] + c[1:])
    edges[0] = c[0] - 0.5 * (c[1] - c[0])
    edges[-1] = c[-1] + 0.5 * (c[-1] - c[-2])
    return edges

def plot_map(
    flat,
    ny,
    nx,
    title,
    outfile,
    cmap,
    symmetric=False,
    bbox=None,
    cbar_label="m³ s⁻¹",
    lats=None,
    lons=None,
    vmin=None,
    vmax=None,
    clip_range=None,
    exact_pixels=False,
    levels=None,
    args=None,
):
    """
    Plot a regular lat/lon map.

    For continental exact-pixel maps, this function subsets the data to the
    requested bbox before calling pcolormesh. This avoids building a full global
    pcolormesh and prevents memory blow-up.
    """

    import gc

    if getattr(args, "no_map_png", False):
        print("Skipping PNG map:", outfile)
        return
    
    data = np.asarray(flat, dtype=np.float32).reshape(ny, nx)

    lat1d_plot = None
    lon1d_plot = None

    # ============================================================
    # Subset BEFORE plotting exact-pixel continental maps
    # ============================================================
    if exact_pixels and bbox is not None and lats is not None and lons is not None:
        lat_min, lon_min, lat_max, lon_max = bbox

        lat1d, lon1d = _regular_1d_from_field(lats, lons)

        row_mask = (lat1d >= lat_min) & (lat1d <= lat_max)
        col_mask = (lon1d >= lon_min) & (lon1d <= lon_max)

        if not np.any(row_mask) or not np.any(col_mask):
            print("WARNING: empty continent subset for", outfile)
            return

        data = data[np.ix_(row_mask, col_mask)]
        lat1d_plot = lat1d[row_mask]
        lon1d_plot = lon1d[col_mask]

    # ============================================================
    # Optional clipping for plotting only
    # ============================================================
    if clip_range is not None:
        data = np.clip(data, clip_range[0], clip_range[1])

    data = np.ma.masked_invalid(data)

    norm = None
    cbar_extend = "neither"
    if levels is not None:
        from matplotlib.colors import BoundaryNorm, ListedColormap

        levels = np.asarray(levels, dtype=np.float64)
        if levels.ndim != 1 or levels.size < 3:
            raise ValueError("levels must be a 1-D array with at least 3 values")

        # Special colour table for relative RMSE improvement maps:
        # negative = degradation, neutral [-10, 10] = white, positive = improvement.
        if levels.size == 10 and np.allclose(
            levels, [-200, -100, -50, -20, -10, 10, 20, 50, 100, 200]
        ):
            colors = [
                "#67001f",  # <= -100 strong deterioration
                "#b2182b",
                "#d6604d",
                "#f4a582",
                "#ffffff",  # -10 .. 10 neutral
                "#92c5de",
                "#4393c3",
                "#2166ac",
                "#053061",  # >= 100 strong improvement
            ]
            cmap = ListedColormap(colors, name="relative_improvement_discrete")
            cmap.set_under(colors[0])
            cmap.set_over(colors[-1])
        norm = BoundaryNorm(levels, ncolors=cmap.N if hasattr(cmap, "N") else 256, clip=False)
        cbar_extend = "both"
        vmin = None
        vmax = None

    # ============================================================
    # Colour limits
    # ============================================================
    if vmin is None or vmax is None:
        vmin_auto, vmax_auto = robust_limits(
            data.filled(np.nan),
            symmetric=symmetric,
            p=99.0,
        )

        if vmin is None:
            vmin = vmin_auto
        if vmax is None:
            vmax = vmax_auto

    # ============================================================
    # Map extent and figure size
    # ============================================================
    if bbox is None:
        map_extent = [-180, 180, -90, 90]
        image_extent = [-180, 180, -90, 90]
        figsize = (13, 6)
    else:
        lat_min, lon_min, lat_max, lon_max = bbox
        map_extent = [lon_min, lon_max, lat_min, lat_max]
        image_extent = map_extent
        figsize = (9, 7)

    # Build plotting kwargs once. Matplotlib does not allow passing
    # norm together with vmin/vmax.
    plot_kwargs_base = dict(cmap=cmap)
    if norm is not None:
        plot_kwargs_base["norm"] = norm
    else:
        plot_kwargs_base["vmin"] = vmin
        plot_kwargs_base["vmax"] = vmax

    # ============================================================
    # Plot with cartopy if available
    # ============================================================
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        fig = plt.figure(figsize=figsize, constrained_layout=True)
        ax = plt.axes(projection=ccrs.PlateCarree())
        plot_kwargs = plot_kwargs_base.copy()

        if exact_pixels and lats is not None and lons is not None:
            if lat1d_plot is None or lon1d_plot is None:
                lat1d_plot, lon1d_plot = _regular_1d_from_field(lats, lons)

            lon_edges = _coord_edges_1d(lon1d_plot)
            lat_edges = _coord_edges_1d(lat1d_plot)

            im = ax.pcolormesh(
                lon_edges,
                lat_edges,
                data,
                transform=ccrs.PlateCarree(),
                shading="flat", 
                **plot_kwargs,
                rasterized=True,
            )

        else:
            im = ax.imshow(
                data,
                origin="upper",
                extent=image_extent,
                transform=ccrs.PlateCarree(),
                interpolation="nearest", 
                **plot_kwargs,
                aspect="auto",
            )

        ax.coastlines(resolution="110m", linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
        ax.set_extent(map_extent, crs=ccrs.PlateCarree())
        ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4)

    # ============================================================
    # Fallback without cartopy
    # ============================================================
    except Exception:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        plot_kwargs = plot_kwargs_base.copy()

        if exact_pixels and lats is not None and lons is not None:
            if lat1d_plot is None or lon1d_plot is None:
                lat1d_plot, lon1d_plot = _regular_1d_from_field(lats, lons)

            lon_edges = _coord_edges_1d(lon1d_plot)
            lat_edges = _coord_edges_1d(lat1d_plot)

            im = ax.pcolormesh(
                lon_edges,
                lat_edges,
                data,
                shading="flat", 
                **plot_kwargs,
                rasterized=True,
            )

        else:
            im = ax.imshow(
                data,
                origin="upper",
                extent=image_extent,
                interpolation="nearest", 
                **plot_kwargs,
                aspect="auto",
            )

        ax.set_xlim(map_extent[0], map_extent[1])
        ax.set_ylim(map_extent[2], map_extent[3])
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, alpha=0.25)

    # ============================================================
    # Save and clean memory
    # ============================================================
    ax.set_title(title, fontsize=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, extend=cbar_extend)
    if levels is not None:
        cbar.set_ticks(levels)
    cbar.set_label(cbar_label)

    fig.savefig(outfile, dpi=120)
    plt.close(fig)

    del fig, ax, im, data
    gc.collect()

    print("Saved map:", outfile)


def write_map_nc(path: Path, field: np.ndarray, ny: int, nx: int, varname: str, long_name: str, lats: np.ndarray, lons: np.ndarray, units: str = "m3 s-1"):
    if not HAVE_NETCDF4:
        return
    lat1d, lon1d = _regular_1d_from_field(lats, lons)
    with Dataset(path, "w") as ds:
        ds.createDimension("lat", ny)
        ds.createDimension("lon", nx)
        lat = ds.createVariable("lat", "f4", ("lat",))
        lon = ds.createVariable("lon", "f4", ("lon",))
        lat[:] = lat1d.astype(np.float32)
        lon[:] = lon1d.astype(np.float32)
        v = ds.createVariable(varname, "f4", ("lat", "lon"), zlib=True, complevel=4, fill_value=np.nan)
        v.long_name = long_name
        v.units = units
        v[:, :] = field.reshape(ny, nx)
        ds.sync()
    print("Saved NetCDF:", path)


def finalize_and_plot_maps(acc: dict, expvers: list[str], ny: int, nx: int, outdir: Path, control_expver: str, lats: np.ndarray, lons: np.ndarray, start_date, end_date, continent_maps=False):
    """
    Plot mean-error, RMSE, NSE and KGE maps per experiment and difference maps
    relative to the control experiment.

    Continental maps use exact native grid pixels via pcolormesh, with no image
    interpolation.
    """
    plot_dir = outdir / "plots_maps"
    plot_dir.mkdir(parents=True, exist_ok=True)

    map_fields = {}

    for expver in expvers:
        count = acc[expver]["count"]
        sum_error = acc[expver]["sum_error"]
        sum_sq_error = acc[expver]["sum_sq_error"]

        sum_obs = acc[expver]["sum_obs"]
        sum_obs_sq = acc[expver]["sum_obs_sq"]
        sum_model = acc[expver]["sum_model"]
        sum_model_sq = acc[expver]["sum_model_sq"]
        sum_obs_model = acc[expver]["sum_obs_model"]

        mean_error = np.full(count.shape, np.nan, dtype=np.float32)
        rmse = np.full(count.shape, np.nan, dtype=np.float32)
        nse = np.full(count.shape, np.nan, dtype=np.float32)
        kge = np.full(count.shape, np.nan, dtype=np.float32)

        valid = count > 0
        mean_error[valid] = (sum_error[valid] / count[valid]).astype(np.float32)
        rmse[valid] = np.sqrt(sum_sq_error[valid] / count[valid]).astype(np.float32)

        valid_skill = count > 1

        obs_mean = np.full(count.shape, np.nan, dtype=np.float64)
        model_mean = np.full(count.shape, np.nan, dtype=np.float64)
        obs_var = np.full(count.shape, np.nan, dtype=np.float64)
        model_var = np.full(count.shape, np.nan, dtype=np.float64)

        obs_mean[valid_skill] = sum_obs[valid_skill] / count[valid_skill]
        model_mean[valid_skill] = sum_model[valid_skill] / count[valid_skill]

        obs_var[valid_skill] = (
            sum_obs_sq[valid_skill] / count[valid_skill]
            - obs_mean[valid_skill] ** 2
        )
        model_var[valid_skill] = (
            sum_model_sq[valid_skill] / count[valid_skill]
            - model_mean[valid_skill] ** 2
        )

        obs_var = np.where(obs_var > 0.0, obs_var, np.nan)
        model_var = np.where(model_var > 0.0, model_var, np.nan)

        nse_denom = sum_obs_sq - count * obs_mean ** 2
        valid_nse = valid_skill & np.isfinite(nse_denom) & (nse_denom > 0.0)
        nse[valid_nse] = (
            1.0 - sum_sq_error[valid_nse] / nse_denom[valid_nse]
        ).astype(np.float32)

        obs_std = np.sqrt(obs_var)
        model_std = np.sqrt(model_var)
        cov = np.full(count.shape, np.nan, dtype=np.float64)
        valid_cov = ( valid_skill & np.isfinite(obs_mean) & np.isfinite(model_mean) & (count > 1))
        cov[valid_cov] = ( sum_obs_model[valid_cov] / count[valid_cov]
            - obs_mean[valid_cov] * model_mean[valid_cov])

        corr_map = np.full(count.shape, np.nan, dtype=np.float64)
        valid_corr = ( valid_cov & np.isfinite(obs_std) & np.isfinite(model_std) & (obs_std > 0.0) & (model_std > 0.0))
        corr_map[valid_corr] = ( cov[valid_corr] / (obs_std[valid_corr] * model_std[valid_corr]))

        valid_kge = (
            valid_skill
            & np.isfinite(corr_map)
            & np.isfinite(obs_std)
            & np.isfinite(model_std)
            & (obs_std > 0.0)
            & (model_std > 0.0)
            & np.isfinite(obs_mean)
            & (obs_mean != 0.0)
        )

        alpha = model_std / obs_std
        beta = model_mean / obs_mean

        kge[valid_kge] = (
            1.0
            - np.sqrt(
                (corr_map[valid_kge] - 1.0) ** 2
                + (alpha[valid_kge] - 1.0) ** 2
                + (beta[valid_kge] - 1.0) ** 2
            )
        ).astype(np.float32)

        mean_error[(count == 0) | (~np.isfinite(mean_error))] = np.nan
        rmse[(count == 0) | (~np.isfinite(rmse)) | (rmse <= 0.0)] = np.nan
        nse[(count <= 1) | (~np.isfinite(nse))] = np.nan
        kge[(count <= 1) | (~np.isfinite(kge))] = np.nan

        corr = corr_map.astype(np.float32)
        corr[(count <= 1) | (~np.isfinite(corr))] = np.nan

        map_fields[(expver, "mean_error")] = mean_error
        map_fields[(expver, "rmse")] = rmse
        map_fields[(expver, "correlation")] = corr
        map_fields[(expver, "nse")] = nse
        map_fields[(expver, "kge")] = kge

        labelexp = EXPERIMENT_LABELS.get(expver, expver)

        # Global maps
        plot_map(
            mean_error, ny, nx,
            f"Mean error vs GloFAS\n{expver} — {labelexp}\n( {start_date} - {end_date} )",
            plot_dir / f"mean_error_map_{expver}.png",
            cmap="RdBu_r",
            symmetric=True,
            lats=lats,
            lons=lons,
        )

        plot_map(
            rmse, ny, nx,
            f"RMSE vs GloFAS\n{expver} — {labelexp}\n( {start_date} - {end_date} )",
            plot_dir / f"rmse_map_{expver}.png",
            cmap="cividis",
            symmetric=False,
            lats=lats,
            lons=lons,
        )

        plot_map(
            corr, ny, nx,
            f"Correlation vs GloFAS\n{expver} — {labelexp}\n( {start_date} - {end_date} )",
            plot_dir / f"correlation_map_{expver}.png",
            cmap="RdYlGn",
            symmetric=False,
            vmin=-1.0,
            vmax=1.0,
            clip_range=(-1.0, 1.0),
            cbar_label="Correlation",
            lats=lats,
            lons=lons,
        )

        plot_map(
            nse, ny, nx,
            f"NSE vs GloFAS\n{expver} — {labelexp}\n( {start_date} - {end_date} )",
            plot_dir / f"nse_map_{expver}.png",
            cmap="RdYlGn",
            symmetric=False,
            vmin=-1.0,
            vmax=1.0,
            clip_range=(-1.0, 1.0),
            cbar_label="NSE",
            lats=lats,
            lons=lons,
        )

        plot_map(
            kge, ny, nx,
            f"KGE vs GloFAS\n{expver} — {labelexp}\n( {start_date} - {end_date} )",
            plot_dir / f"kge_map_{expver}.png",
            cmap="RdYlGn",
            symmetric=False,
            vmin=-1.0,
            vmax=1.0,
            clip_range=(-1.0, 1.0),
            cbar_label="KGE",
            lats=lats,
            lons=lons,
        )

        write_map_nc(
            outdir / f"mean_error_map_{expver}.nc",
            mean_error, ny, nx,
            "mean_error", f"Mean error {expver} minus GloFAS", lats, lons,
        )
        write_map_nc(
            outdir / f"rmse_map_{expver}.nc",
            rmse, ny, nx,
            "rmse", f"RMSE {expver} versus GloFAS", lats, lons,
        )
        write_map_nc(
            outdir / f"correlation_map_{expver}.nc",
            corr, ny, nx,
            "correlation", f"Correlation {expver} versus GloFAS", lats, lons, units="1",
        )
        write_map_nc(
            outdir / f"nse_map_{expver}.nc",
            nse, ny, nx,
            "nse", f"NSE {expver} versus GloFAS", lats, lons, units="1",
        )
        write_map_nc(
            outdir / f"kge_map_{expver}.nc",
            kge, ny, nx,
            "kge", f"KGE {expver} versus GloFAS", lats, lons, units="1",
        )

        if continent_maps:
            continent_dir = plot_dir / "continents"
            continent_dir.mkdir(parents=True, exist_ok=True)

            for domain_name, bbox in CONTINENT_BBOX.items():
                plot_map(
                    mean_error, ny, nx,
                    f"Mean error vs GloFAS — {domain_name}\n{expver} — {labelexp}\n( {start_date} - {end_date} )",
                    continent_dir / f"mean_error_map_{expver}_{domain_name}.png",
                    cmap="RdBu_r",
                    symmetric=True,
                    bbox=bbox,
                    lats=lats,
                    lons=lons,
                    exact_pixels=True,
                )
                plot_map(
                    rmse, ny, nx,
                    f"RMSE vs GloFAS — {domain_name}\n{expver} — {labelexp}\n( {start_date} - {end_date} )",
                    continent_dir / f"rmse_map_{expver}_{domain_name}.png",
                    cmap="cividis",
                    symmetric=False,
                    bbox=bbox,
                    lats=lats,
                    lons=lons,
                    exact_pixels=True,
                )
                plot_map(
                    corr, ny, nx,
                    f"Correlation vs GloFAS — {domain_name}\n{expver} — {labelexp}\n( {start_date} - {end_date} )",
                    continent_dir / f"correlation_map_{expver}_{domain_name}.png",
                    cmap="RdYlGn",
                    symmetric=False,
                    bbox=bbox,
                    vmin=-1.0,
                    vmax=1.0,
                    clip_range=(-1.0, 1.0),
                    cbar_label="Correlation",
                    lats=lats,
                    lons=lons,
                    exact_pixels=True,
                )
                plot_map(
                    nse, ny, nx,
                    f"NSE vs GloFAS — {domain_name}\n{expver} — {labelexp}\n( {start_date} - {end_date} )",
                    continent_dir / f"nse_map_{expver}_{domain_name}.png",
                    cmap="RdYlGn",
                    symmetric=False,
                    bbox=bbox,
                    vmin=-1.0,
                    vmax=1.0,
                    clip_range=(-1.0, 1.0),
                    cbar_label="NSE",
                    lats=lats,
                    lons=lons,
                    exact_pixels=True,
                )
                plot_map(
                    kge, ny, nx,
                    f"KGE vs GloFAS — {domain_name}\n{expver} — {labelexp}\n( {start_date} - {end_date} )",
                    continent_dir / f"kge_map_{expver}_{domain_name}.png",
                    cmap="RdYlGn",
                    symmetric=False,
                    bbox=bbox,
                    vmin=-1.0,
                    vmax=1.0,
                    clip_range=(-1.0, 1.0),
                    cbar_label="KGE",
                    lats=lats,
                    lons=lons,
                    exact_pixels=True,
                )

    if control_expver not in expvers:
        raise ValueError(f"control_expver={control_expver} is not in expvers={expvers}")

    # Difference maps: each experiment relative to control.
    for expver in expvers:
        if expver == control_expver:
            continue

        bias_diff = map_fields[(expver, "mean_error")] - map_fields[(control_expver, "mean_error")]
        rmse_diff = map_fields[(expver, "rmse")] - map_fields[(control_expver, "rmse")]
        rmse_rel_improvement = relative_rmse_improvement_map(
            map_fields[(expver, "rmse")],
            map_fields[(control_expver, "rmse")],
        )
        corr_diff = map_fields[(expver, "correlation")] - map_fields[(control_expver, "correlation")]
        nse_diff = map_fields[(expver, "nse")] - map_fields[(control_expver, "nse")]
        kge_diff = map_fields[(expver, "kge")] - map_fields[(control_expver, "kge")]

        labelexp = EXPERIMENT_LABELS.get(expver, expver)
        labctl = EXPERIMENT_LABELS.get(control_expver, control_expver)

        # Write difference NetCDFs BEFORE plotting. This makes --plots-only
        # robust even if a later PNG plotting call fails or is interrupted.
        write_map_nc(
            outdir / f"mean_error_difference_{expver}_minus_{control_expver}.nc",
            bias_diff, ny, nx,
            "mean_error_difference",
            f"Mean-error difference {expver} minus control {control_expver}",
            lats, lons,
        )
        write_map_nc(
            outdir / f"rmse_difference_{expver}_minus_{control_expver}.nc",
            rmse_diff, ny, nx,
            "rmse_difference",
            f"RMSE difference {expver} minus control {control_expver}",
            lats, lons,
        )
        write_map_nc(
            outdir / f"rmse_relative_improvement_{expver}_vs_{control_expver}.nc",
            rmse_rel_improvement, ny, nx,
            "rmse_relative_improvement_pct",
            f"Relative RMSE improvement {expver} versus control {control_expver}",
            lats, lons,
            units="%",
        )
        write_map_nc(
            outdir / f"correlation_difference_{expver}_minus_{control_expver}.nc",
            corr_diff, ny, nx,
            "correlation_difference",
            f"Correlation difference {expver} minus control {control_expver}",
            lats, lons,
            units="1",
        )
        write_map_nc(
            outdir / f"nse_difference_{expver}_minus_{control_expver}.nc",
            nse_diff, ny, nx,
            "nse_difference",
            f"NSE difference {expver} minus control {control_expver}",
            lats, lons,
            units="1",
        )
        write_map_nc(
            outdir / f"kge_difference_{expver}_minus_{control_expver}.nc",
            kge_diff, ny, nx,
            "kge_difference",
            f"KGE difference {expver} minus control {control_expver}",
            lats, lons,
            units="1",
        )

        plot_map(
            bias_diff, ny, nx,
            f"Mean-error difference vs control ( {start_date} - {end_date} )\n"
            f"{expver} — {labelexp} minus\n"
            f"{control_expver} — {labctl}",
            plot_dir / f"mean_error_difference_{expver}_minus_{control_expver}.png",
            cmap="RdBu_r",
            symmetric=True,
            lats=lats,
            lons=lons,
        )

        plot_map(
            rmse_diff, ny, nx,
            f"RMSE difference vs control ( {start_date} - {end_date} )\n"
            f"{expver} — {labelexp} minus\n"
            f"{control_expver} — {labctl}",
            plot_dir / f"rmse_difference_{expver}_minus_{control_expver}.png",
            cmap="RdBu_r",
            symmetric=True,
            lats=lats,
            lons=lons,
        )

        plot_map(
            rmse_rel_improvement, ny, nx,
            f"Relative RMSE improvement vs control [%] ( {start_date} - {end_date} )\n"
            f"{expver} — {labelexp} vs\n"
            f"{control_expver} — {labctl}\n"
            f"Positive = better than control; neutral band = ±10%",
            plot_dir / f"rmse_relative_improvement_{expver}_vs_{control_expver}.png",
            cmap=plt.get_cmap("RdBu"),
            symmetric=False,
            vmin=-200.0,
            vmax=200.0,
            clip_range=(-200.0, 200.0),
            cbar_label="RMSE improvement vs control [%]",
            lats=lats,
            lons=lons,
            levels=REL_IMPROVEMENT_LEVELS,
        )

        plot_map(
            corr_diff, ny, nx,
            f"Correlation difference vs control ( {start_date} - {end_date} )\n"
            f"{expver} — {labelexp} minus\n"
            f"{control_expver} — {labctl}",
            plot_dir / f"correlation_difference_{expver}_minus_{control_expver}.png",
            cmap="RdBu",
            symmetric=True,
            vmin=-0.5,
            vmax=0.5,
            clip_range=(-0.5, 0.5),
            cbar_label="ΔCorrelation",
            lats=lats,
            lons=lons,
        )

        plot_map(
            nse_diff, ny, nx,
            f"NSE difference vs control ( {start_date} - {end_date} )\n"
            f"{expver} — {labelexp} minus\n"
            f"{control_expver} — {labctl}",
            plot_dir / f"nse_difference_{expver}_minus_{control_expver}.png",
            cmap="RdBu_r",
            symmetric=True,
            vmin=-0.5,
            vmax=0.5,
            clip_range=(-0.5, 0.5),
            cbar_label="ΔNSE",
            lats=lats,
            lons=lons,
        )

        plot_map(
            kge_diff, ny, nx,
            f"KGE difference vs control ( {start_date} - {end_date} )\n"
            f"{expver} — {labelexp} minus\n"
            f"{control_expver} — {labctl}",
            plot_dir / f"kge_difference_{expver}_minus_{control_expver}.png",
            cmap="RdBu_r",
            symmetric=True,
            vmin=-0.5,
            vmax=0.5,
            clip_range=(-0.5, 0.5),
            cbar_label="ΔKGE",
            lats=lats,
            lons=lons,
        )

        if continent_maps:
            continent_dir = plot_dir / "continents"
            continent_dir.mkdir(parents=True, exist_ok=True)

            for domain_name, bbox in CONTINENT_BBOX.items():
                plot_map(
                    bias_diff, ny, nx,
                    f"Mean-error difference vs control — {domain_name}\n"
                    f"{expver} — {labelexp} minus\n"
                    f"{control_expver} — {labctl}\n ( {start_date} - {end_date} )",
                    continent_dir / f"mean_error_difference_{expver}_minus_{control_expver}_{domain_name}.png",
                    cmap="RdBu_r",
                    symmetric=True,
                    bbox=bbox,
                    lats=lats,
                    lons=lons,
                    exact_pixels=True,
                )
                plot_map(
                    rmse_diff, ny, nx,
                    f"RMSE difference vs control — {domain_name}\n"
                    f"{expver} — {labelexp} minus\n"
                    f"{control_expver} — {labctl}\n( {start_date} - {end_date} )",
                    continent_dir / f"rmse_difference_{expver}_minus_{control_expver}_{domain_name}.png",
                    cmap="RdBu_r",
                    symmetric=True,
                    bbox=bbox,
                    lats=lats,
                    lons=lons,
                    exact_pixels=True,
                )
                plot_map(
                    rmse_rel_improvement, ny, nx,
                    f"Relative RMSE improvement vs control — {domain_name} [%]\n"
                    f"{expver} — {labelexp} vs\n"
                    f"{control_expver} — {labctl}\n"
                    f"( {start_date} - {end_date} ); positive = better",
                    continent_dir / f"rmse_relative_improvement_{expver}_vs_{control_expver}_{domain_name}.png",
                    cmap=plt.get_cmap("RdBu"),
                    symmetric=False,
                    bbox=bbox,
                    vmin=-200.0,
                    vmax=200.0,
                    clip_range=(-200.0, 200.0),
                    cbar_label="RMSE improvement vs control [%]",
                    lats=lats,
                    lons=lons,
                    exact_pixels=True,
                    levels=REL_IMPROVEMENT_LEVELS,
                )
                plot_map(
                    corr_diff, ny, nx,
                    f"Correlation difference vs control — {domain_name}\n"
                    f"{expver} — {labelexp} minus\n"
                    f"{control_expver} — {labctl}\n( {start_date} - {end_date} )",
                    continent_dir / f"correlation_difference_{expver}_minus_{control_expver}_{domain_name}.png",
                    cmap="RdBu",
                    symmetric=True,
                    bbox=bbox,
                    vmin=-0.5,
                    vmax=0.5,
                    clip_range=(-0.5, 0.5),
                    cbar_label="ΔCorrelation",
                    lats=lats,
                    lons=lons,
                    exact_pixels=True,
                )
                plot_map(
                    nse_diff, ny, nx,
                    f"NSE difference vs control — {domain_name}\n"
                    f"{expver} — {labelexp} minus\n"
                    f"{control_expver} — {labctl}\n( {start_date} - {end_date} )",
                    continent_dir / f"nse_difference_{expver}_minus_{control_expver}_{domain_name}.png",
                    cmap="RdBu_r",
                    symmetric=True,
                    bbox=bbox,
                    vmin=-0.5,
                    vmax=0.5,
                    clip_range=(-0.5, 0.5),
                    cbar_label="ΔNSE",
                    lats=lats,
                    lons=lons,
                    exact_pixels=True,
                )
                plot_map(
                    kge_diff, ny, nx,
                    f"KGE difference vs control — {domain_name}\n"
                    f"{expver} — {labelexp} minus\n"
                    f"{control_expver} — {labctl}\n( {start_date} - {end_date} )",
                    continent_dir / f"kge_difference_{expver}_minus_{control_expver}_{domain_name}.png",
                    cmap="RdBu_r",
                    symmetric=True,
                    bbox=bbox,
                    vmin=-0.5,
                    vmax=0.5,
                    clip_range=(-0.5, 0.5),
                    cbar_label="ΔKGE",
                    lats=lats,
                    lons=lons,
                    exact_pixels=True,
                )

        write_map_nc(
            outdir / f"mean_error_difference_{expver}_minus_{control_expver}.nc",
            bias_diff, ny, nx,
            "mean_error_difference",
            f"Mean-error difference {expver} minus control {control_expver}",
            lats, lons,
        )
        write_map_nc(
            outdir / f"rmse_difference_{expver}_minus_{control_expver}.nc",
            rmse_diff, ny, nx,
            "rmse_difference",
            f"RMSE difference {expver} minus control {control_expver}",
            lats, lons,
        )
        write_map_nc(
            outdir / f"rmse_relative_improvement_{expver}_vs_{control_expver}.nc",
            rmse_rel_improvement, ny, nx,
            "rmse_relative_improvement_pct",
            f"Relative RMSE improvement {expver} versus control {control_expver}",
            lats, lons,
            units="%",
        )
        write_map_nc(
            outdir / f"correlation_difference_{expver}_minus_{control_expver}.nc",
            corr_diff, ny, nx,
            "correlation_difference",
            f"Correlation difference {expver} minus control {control_expver}",
            lats, lons,
            units="1",
        )
        write_map_nc(
            outdir / f"nse_difference_{expver}_minus_{control_expver}.nc",
            nse_diff, ny, nx,
            "nse_difference",
            f"NSE difference {expver} minus control {control_expver}",
            lats, lons,
            units="1",
        )
        write_map_nc(
            outdir / f"kge_difference_{expver}_minus_{control_expver}.nc",
            kge_diff, ny, nx,
            "kge_difference",
            f"KGE difference {expver} minus control {control_expver}",
            lats, lons,
            units="1",
        )


def read_map_nc(path: Path) -> tuple[np.ndarray, int, int, np.ndarray, np.ndarray]:
    """
    Read a benchmark map NetCDF written by write_map_nc().
    Returns flat_field, ny, nx, lats_flat, lons_flat.
    """
    if not HAVE_NETCDF4:
        raise RuntimeError("netCDF4 is required for --plots-only map regeneration")

    with Dataset(path, "r") as ds:
        varnames = [v for v in ds.variables if v not in ("lat", "lon")]
        if len(varnames) != 1:
            raise ValueError(f"Expected one data variable in {path}, found {varnames}")

        varname = varnames[0]
        field2d = np.asarray(ds.variables[varname][:], dtype=np.float32)
        lat1d = np.asarray(ds.variables["lat"][:], dtype=np.float32)
        lon1d = np.asarray(ds.variables["lon"][:], dtype=np.float32)

    ny, nx = field2d.shape
    lon2d, lat2d = np.meshgrid(lon1d, lat1d)

    return field2d.reshape(-1), ny, nx, lat2d.reshape(-1), lon2d.reshape(-1)



def ensure_difference_map_netcdfs(
    outdir: Path,
    expvers: list[str],
    control_expver: str,
) -> None:
    """Create missing difference-map NetCDFs from per-experiment map NetCDFs.

    This makes --plots-only robust: if the original benchmark run produced
    mean_error_map/rmse_map/correlation_map/nse_map/kge_map files but was
    interrupted before the difference NetCDFs were written, the missing
    difference files can still be regenerated without retrieving GRIBs again.
    """
    if not HAVE_NETCDF4:
        print("WARNING: cannot create difference NetCDFs because netCDF4 is unavailable")
        return

    base_specs = {
        "mean_error": {
            "base_prefix": "mean_error_map",
            "diff_prefix": "mean_error_difference",
            "varname": "mean_error_difference",
            "units": "m3 s-1",
            "long_name": "Mean-error difference",
        },
        "rmse": {
            "base_prefix": "rmse_map",
            "diff_prefix": "rmse_difference",
            "varname": "rmse_difference",
            "units": "m3 s-1",
            "long_name": "RMSE difference",
        },
        "correlation": {
            "base_prefix": "correlation_map",
            "diff_prefix": "correlation_difference",
            "varname": "correlation_difference",
            "units": "1",
            "long_name": "Correlation difference",
        },
        "nse": {
            "base_prefix": "nse_map",
            "diff_prefix": "nse_difference",
            "varname": "nse_difference",
            "units": "1",
            "long_name": "NSE difference",
        },
        "kge": {
            "base_prefix": "kge_map",
            "diff_prefix": "kge_difference",
            "varname": "kge_difference",
            "units": "1",
            "long_name": "KGE difference",
        },
    }

    for expver in expvers:
        if expver == control_expver:
            continue

        # Standard difference fields: experiment minus control.
        for metric, spec in base_specs.items():
            out_nc = outdir / f"{spec['diff_prefix']}_{expver}_minus_{control_expver}.nc"
            if out_nc.exists():
                continue

            exp_nc = outdir / f"{spec['base_prefix']}_{expver}.nc"
            ctl_nc = outdir / f"{spec['base_prefix']}_{control_expver}.nc"
            if not exp_nc.exists() or not ctl_nc.exists():
                print(
                    f"WARNING: cannot create {out_nc.name}; missing base NetCDF(s): "
                    f"{exp_nc.name if not exp_nc.exists() else ''} "
                    f"{ctl_nc.name if not ctl_nc.exists() else ''}"
                )
                continue

            exp_field, ny, nx, lats, lons = read_map_nc(exp_nc)
            ctl_field, ny_ctl, nx_ctl, _, _ = read_map_nc(ctl_nc)
            if (ny, nx) != (ny_ctl, nx_ctl):
                print(f"WARNING: cannot create {out_nc.name}; grid mismatch")
                continue

            diff = exp_field - ctl_field
            write_map_nc(
                out_nc,
                diff.astype(np.float32),
                ny,
                nx,
                spec["varname"],
                f"{spec['long_name']} {expver} minus control {control_expver}",
                lats,
                lons,
                units=spec["units"],
            )

        # Relative RMSE improvement: special filename and formula.
        out_nc = outdir / f"rmse_relative_improvement_{expver}_vs_{control_expver}.nc"
        if not out_nc.exists():
            exp_nc = outdir / f"rmse_map_{expver}.nc"
            ctl_nc = outdir / f"rmse_map_{control_expver}.nc"
            if exp_nc.exists() and ctl_nc.exists():
                exp_rmse, ny, nx, lats, lons = read_map_nc(exp_nc)
                ctl_rmse, ny_ctl, nx_ctl, _, _ = read_map_nc(ctl_nc)
                if (ny, nx) == (ny_ctl, nx_ctl):
                    rel = relative_rmse_improvement_map(exp_rmse, ctl_rmse)
                    write_map_nc(
                        out_nc,
                        rel.astype(np.float32),
                        ny,
                        nx,
                        "rmse_relative_improvement_pct",
                        f"Relative RMSE improvement {expver} versus control {control_expver}",
                        lats,
                        lons,
                        units="%",
                    )
                else:
                    print(f"WARNING: cannot create {out_nc.name}; grid mismatch")
            else:
                print(
                    f"WARNING: cannot create {out_nc.name}; missing base RMSE NetCDF(s): "
                    f"{exp_nc.name if not exp_nc.exists() else ''} "
                    f"{ctl_nc.name if not ctl_nc.exists() else ''}"
                )

def _find_required_map_nc(outdir: Path, pattern: str) -> Path | None:
    matches = sorted(outdir.glob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        print("WARNING: multiple matches for", pattern)
        for m in matches:
            print("  ", m)
        print("Using:", matches[0])
    return matches[0]


def regenerate_maps_from_nc(
    outdir: Path,
    expvers: list[str],
    control_expver: str,
    start_date: str,
    end_date: str,
    continent_maps: bool = False,
) -> None:
    """
    Regenerate PNG map plots from existing NetCDF map files.
    This is used by --plots-only and avoids MARS/GloFAS/CaMa retrieval.
    """
    plot_dir = outdir / "plots_maps"
    plot_dir.mkdir(parents=True, exist_ok=True)

    metric_specs = {
        "mean_error": {
            "prefix": "mean_error_map",
            "title": "Mean error vs GloFAS",
            "cmap": "RdBu_r",
            "symmetric": True,
            "cbar_label": "m³ s⁻¹",
            "vmin": None,
            "vmax": None,
            "clip_range": None,
        },
        "rmse": {
            "prefix": "rmse_map",
            "title": "RMSE vs GloFAS",
            "cmap": "cividis",
            "symmetric": False,
            "cbar_label": "m³ s⁻¹",
            "vmin": None,
            "vmax": None,
            "clip_range": None,
        },
        "correlation": {
            "prefix": "correlation_map",
            "title": "Correlation vs GloFAS",
            "cmap": "RdYlGn",
            "symmetric": False,
            "cbar_label": "Correlation",
            "vmin": -1.0,
            "vmax": 1.0,
            "clip_range": (-1.0, 1.0),
        },
        "nse": {
            "prefix": "nse_map",
            "title": "NSE vs GloFAS",
            "cmap": "RdYlGn",
            "symmetric": False,
            "cbar_label": "NSE",
            "vmin": -1.0,
            "vmax": 1.0,
            "clip_range": (-1.0, 1.0),
        },
        "kge": {
            "prefix": "kge_map",
            "title": "KGE vs GloFAS",
            "cmap": "RdYlGn",
            "symmetric": False,
            "cbar_label": "KGE",
            "vmin": -1.0,
            "vmax": 1.0,
            "clip_range": (-1.0, 1.0),
        },
    }

    diff_specs = {
        "mean_error": {
            "prefix": "mean_error_difference",
            "title": "Mean-error difference vs control",
            "cmap": "RdBu_r",
            "symmetric": True,
            "cbar_label": "m³ s⁻¹",
            "vmin": None,
            "vmax": None,
            "clip_range": None,
        },
        "rmse": {
            "prefix": "rmse_difference",
            "title": "RMSE difference vs control",
            "cmap": "RdBu_r",
            "symmetric": True,
            "cbar_label": "m³ s⁻¹",
            "vmin": None,
            "vmax": None,
            "clip_range": None,
            "levels": None,
        },
        "rmse_relative_improvement": {
            "prefix": "rmse_relative_improvement",
            "title": "Relative RMSE improvement vs control [%]",
            "cmap": plt.get_cmap("RdBu"),
            "symmetric": False,
            "cbar_label": "RMSE improvement vs control [%]",
            "vmin": -200.0,
            "vmax": 200.0,
            "clip_range": (-200.0, 200.0),
            "levels": REL_IMPROVEMENT_LEVELS,
        },
        "correlation": {
            "prefix": "correlation_difference",
            "title": "Correlation difference vs control",
            "cmap": "RdBu",
            "symmetric": True,
            "cbar_label": "ΔCorrelation",
            "vmin": -0.5,
            "vmax": 0.5,
            "clip_range": (-0.5, 0.5),
        },
        "nse": {
            "prefix": "nse_difference",
            "title": "NSE difference vs control",
            "cmap": "RdBu_r",
            "symmetric": True,
            "cbar_label": "ΔNSE",
            "vmin": -0.5,
            "vmax": 0.5,
            "clip_range": (-0.5, 0.5),
        },
        "kge": {
            "prefix": "kge_difference",
            "title": "KGE difference vs control",
            "cmap": "RdBu_r",
            "symmetric": True,
            "cbar_label": "ΔKGE",
            "vmin": -0.5,
            "vmax": 0.5,
            "clip_range": (-0.5, 0.5),
        },
    }

    for expver in expvers:
        labelexp = EXPERIMENT_LABELS.get(expver, expver)

        for metric, spec in metric_specs.items():
            nc_path = _find_required_map_nc(outdir, f"{spec['prefix']}_{expver}.nc")
            if nc_path is None:
                print("WARNING: missing NetCDF for", metric, expver)
                continue

            field, ny, nx, lats, lons = read_map_nc(nc_path)

            plot_map(
                field, ny, nx,
                f"{spec['title']}\n{expver} — {labelexp}\n( {start_date} - {end_date} )",
                plot_dir / f"{spec['prefix']}_{expver}.png",
                cmap=spec["cmap"],
                symmetric=spec["symmetric"],
                cbar_label=spec["cbar_label"],
                lats=lats,
                lons=lons,
                vmin=spec["vmin"],
                vmax=spec["vmax"],
                clip_range=spec["clip_range"],
                levels=spec.get("levels"),
            )

            if continent_maps:
                continent_dir = plot_dir / "continents"
                continent_dir.mkdir(parents=True, exist_ok=True)

                for domain_name, bbox in CONTINENT_BBOX.items():
                    plot_map(
                        field, ny, nx,
                        f"{spec['title']} — {domain_name}\n"
                        f"{expver} — {labelexp}\n"
                        f"( {start_date} - {end_date} )",
                        continent_dir / f"{spec['prefix']}_{expver}_{domain_name}.png",
                        cmap=spec["cmap"],
                        symmetric=spec["symmetric"],
                        bbox=bbox,
                        cbar_label=spec["cbar_label"],
                        lats=lats,
                        lons=lons,
                        vmin=spec["vmin"],
                        vmax=spec["vmax"],
                        clip_range=spec["clip_range"],
                        exact_pixels=True,
                        levels=spec.get("levels"),
                    )

    # Ensure all difference NetCDFs exist before plotting them. This also
    # repairs older/incomplete runs where only *_map_*.nc files were created.
    ensure_difference_map_netcdfs(outdir, expvers, control_expver)

    for expver in expvers:
        if expver == control_expver:
            continue

        labelexp = EXPERIMENT_LABELS.get(expver, expver)
        labctl = EXPERIMENT_LABELS.get(control_expver, control_expver)

        for metric, spec in diff_specs.items():
            nc_path = _find_required_map_nc(
                outdir,
                (f"{spec['prefix']}_{expver}_vs_{control_expver}.nc"
                 if metric == "rmse_relative_improvement"
                 else f"{spec['prefix']}_{expver}_minus_{control_expver}.nc"),
            )
            if nc_path is None:
                print("WARNING: missing difference NetCDF for", metric, expver)
                continue

            field, ny, nx, lats, lons = read_map_nc(nc_path)

            plot_map(
                field, ny, nx,
                f"{spec['title']} ( {start_date} - {end_date} )\n"
                f"{expver} — {labelexp} minus\n"
                f"{control_expver} — {labctl}",
                (plot_dir / f"{spec['prefix']}_{expver}_vs_{control_expver}.png"
                 if metric == "rmse_relative_improvement"
                 else plot_dir / f"{spec['prefix']}_{expver}_minus_{control_expver}.png"),
                cmap=spec["cmap"],
                symmetric=spec["symmetric"],
                cbar_label=spec["cbar_label"],
                lats=lats,
                lons=lons,
                vmin=spec["vmin"],
                vmax=spec["vmax"],
                clip_range=spec["clip_range"],
                levels=spec.get("levels"),
            )

            if continent_maps:
                continent_dir = plot_dir / "continents"
                continent_dir.mkdir(parents=True, exist_ok=True)

                for domain_name, bbox in CONTINENT_BBOX.items():
                    plot_map(
                        field, ny, nx,
                        f"{spec['title']} — {domain_name}\n"
                        f"{expver} — {labelexp} minus \n"
                        f"{control_expver} — {labctl} \n"
                        f"( {start_date} - {end_date} )",
                        (continent_dir / f"{spec['prefix']}_{expver}_vs_{control_expver}_{domain_name}.png"
                         if metric == "rmse_relative_improvement"
                         else continent_dir / f"{spec['prefix']}_{expver}_minus_{control_expver}_{domain_name}.png"),
                        cmap=spec["cmap"],
                        symmetric=spec["symmetric"],
                        bbox=bbox,
                        cbar_label=spec["cbar_label"],
                        lats=lats,
                        lons=lons,
                        vmin=spec["vmin"],
                        vmax=spec["vmax"],
                        clip_range=spec["clip_range"],
                        exact_pixels=True,
                        levels=spec.get("levels"),
                    )


def make_control_difference_rows(daily_rows: list[dict], expvers: list[str], control_expver: str) -> list[dict]:
    """
    Build a daily/domain table comparing each experiment against the control.

    Convention:
      rmse_difference_vs_control = rmse_experiment - rmse_control
      rmse_improvement_pct_vs_control = 100 * (rmse_control - rmse_experiment) / rmse_control

    Therefore positive improvement means the experiment is better than control.
    """
    by_key = {}
    for r in daily_rows:
        by_key[(r["date"], r["domain"], r["expver"])] = r

    out = []
    dates = sorted({r["date"] for r in daily_rows})
    domains = list(DOMAINS.keys())

    for date in dates:
        for domain in domains:
            ctrl = by_key.get((date, domain, control_expver))
            if ctrl is None:
                continue

            ctrl_rmse = float(ctrl["rmse"])
            ctrl_bias = float(ctrl["bias_model_minus_glofas"])
            ctrl_corr = float(ctrl["correlation"])
            ctrl_nse = float(ctrl.get("nse", np.nan))
            ctrl_kge = float(ctrl.get("kge", np.nan))

            for expver in expvers:
                if expver == control_expver:
                    continue
                r = by_key.get((date, domain, expver))
                if r is None:
                    continue

                rmse = float(r["rmse"])
                bias = float(r["bias_model_minus_glofas"])
                corr = float(r["correlation"])
                nse = float(r.get("nse", np.nan))
                kge = float(r.get("kge", np.nan))

                if np.isfinite(ctrl_rmse) and ctrl_rmse != 0.0 and np.isfinite(rmse):
                    rmse_improvement_pct = 100.0 * (ctrl_rmse - rmse) / ctrl_rmse
                else:
                    rmse_improvement_pct = np.nan

                out.append({
                    "date": date,
                    "domain": domain,
                    "expver": expver,
                    "control_expver": control_expver,
                    "n_expver": r["n"],
                    "n_control": ctrl["n"],
                    "rmse": rmse,
                    "rmse_control": ctrl_rmse,
                    "rmse_difference_vs_control": rmse - ctrl_rmse,
                    "rmse_improvement_pct_vs_control": rmse_improvement_pct,
                    "bias_model_minus_glofas": bias,
                    "bias_control": ctrl_bias,
                    "bias_difference_vs_control": bias - ctrl_bias,
                    "abs_bias_difference_vs_control": abs(bias) - abs(ctrl_bias),
                    "correlation": corr,
                    "correlation_control": ctrl_corr,
                    "correlation_difference_vs_control": corr - ctrl_corr,
                    "nse": nse,
                    "nse_control": ctrl_nse,
                    "nse_difference_vs_control": nse - ctrl_nse,
                    "kge": kge,
                    "kge_control": ctrl_kge,
                    "kge_difference_vs_control": kge - ctrl_kge,
                })

    return out



def _safe_float(x):
    try:
        v = float(x)
    except Exception:
        return np.nan
    return v if np.isfinite(v) else np.nan


def _fmt(v, nd=3, suffix=""):
    v = _safe_float(v)
    if not np.isfinite(v):
        return "nan"
    return f"{v:.{nd}f}{suffix}"


def _fmt_signed(v, nd=3, suffix=""):
    v = _safe_float(v)
    if not np.isfinite(v):
        return "nan"
    return f"{v:+.{nd}f}{suffix}"


def _mean_of(rows, key):
    vals = np.array([_safe_float(r.get(key, np.nan)) for r in rows], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    return float(np.nanmean(vals))


def _sum_of(rows, key):
    vals = np.array([_safe_float(r.get(key, np.nan)) for r in rows], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    return float(np.nansum(vals))


def print_final_benchmark_summary(
    daily_rows: list[dict],
    control_diff_rows: list[dict],
    summary_rows: list[dict],
    summary_control_rows: list[dict],
    expvers: list[str],
    control_expver: str,
    start_date: str,
    end_date: str,
) -> None:
    """Print a compact copy-pasteable numerical recap.

    The summary is designed for quick interpretation and for pasting into a
    notebook, issue, report, or ChatGPT discussion.  It uses pooled Global
    summary metrics when available, and daily domain-vs-control rows for the
    continental recap.
    """

    domains = list(DOMAINS.keys())
    exp_list = [e for e in expvers if e != control_expver]
    summary_by_exp = {str(r.get("expver")): r for r in summary_rows}
    summary_ctrl_by_exp = {str(r.get("expver")): r for r in summary_control_rows}

    print()
    print("=" * 88)
    print("FINAL BENCHMARK SUMMARY — COPY/PASTE BLOCK")
    print("=" * 88)
    print(f"Period        : {start_date} - {end_date}")
    print(f"Control       : {control_expver} — {EXPERIMENT_LABELS.get(control_expver, control_expver)}")
    print(f"Experiments   : {', '.join(exp_list)}")
    print(f"Domains       : {', '.join(domains)}")
    print()

    # ------------------------------------------------------------------
    # Global pooled metrics
    # ------------------------------------------------------------------
    print("GLOBAL POOLED METRICS")
    print("---------------------")
    ctrl_global = summary_by_exp.get(control_expver)
    if ctrl_global is not None:
        print(
            f"Control {control_expver}: "
            f"RMSE={_fmt(ctrl_global.get('rmse'),2)}; "
            f"Bias={_fmt(ctrl_global.get('bias_model_minus_glofas'),2)}; "
            f"Corr={_fmt(ctrl_global.get('correlation'),3)}; "
            f"NSE={_fmt(ctrl_global.get('nse'),3)}; "
            f"KGE={_fmt(ctrl_global.get('kge'),3)}; "
            f"N={int(_safe_float(ctrl_global.get('n', 0))):,}"
        )
    else:
        print("Control pooled summary unavailable.")

    for expver in exp_list:
        r = summary_by_exp.get(expver)
        d = summary_ctrl_by_exp.get(expver)
        if r is None:
            print(f"{expver}: pooled summary unavailable")
            continue
        print(
            f"{expver} — {EXPERIMENT_LABELS.get(expver, expver)}: "
            f"RMSE={_fmt(r.get('rmse'),2)} "
            f"({_fmt_signed(d.get('rmse_improvement_pct_vs_control') if d else np.nan,2,'%')} vs control); "
            f"Bias={_fmt(r.get('bias_model_minus_glofas'),2)} "
            f"(ΔBias={_fmt_signed(d.get('bias_difference_vs_control') if d else np.nan,2)}; "
            f"Δ|Bias|={_fmt_signed(d.get('abs_bias_difference_vs_control') if d else np.nan,2)}); "
            f"Corr={_fmt(r.get('correlation'),3)} "
            f"(Δ={_fmt_signed(d.get('correlation_difference_vs_control') if d else np.nan,3)}); "
            f"NSE={_fmt(r.get('nse'),3)} "
            f"(Δ={_fmt_signed(d.get('nse_difference_vs_control') if d else np.nan,3)}); "
            f"KGE={_fmt(r.get('kge'),3)} "
            f"(Δ={_fmt_signed(d.get('kge_difference_vs_control') if d else np.nan,3)})"
        )

    # ------------------------------------------------------------------
    # Domain recap from daily-vs-control rows
    # ------------------------------------------------------------------
    print()
    print("DOMAIN MEAN DAILY PERFORMANCE VS CONTROL")
    print("----------------------------------------")
    print(
        "domain,expver,mean_RMSE_improvement_pct,mean_ΔBias,mean_Δ|Bias|,"
        "mean_ΔCorr,mean_ΔNSE,mean_ΔKGE,mean_n_exp,mean_n_control"
    )

    domain_summaries = []
    for domain in domains:
        for expver in exp_list:
            rows = [
                r for r in control_diff_rows
                if str(r.get("domain")) == domain and str(r.get("expver")) == expver
            ]
            if not rows:
                continue
            item = {
                "domain": domain,
                "expver": expver,
                "rmse_improvement": _mean_of(rows, "rmse_improvement_pct_vs_control"),
                "bias_diff": _mean_of(rows, "bias_difference_vs_control"),
                "abs_bias_diff": _mean_of(rows, "abs_bias_difference_vs_control"),
                "corr_diff": _mean_of(rows, "correlation_difference_vs_control"),
                "nse_diff": _mean_of(rows, "nse_difference_vs_control"),
                "kge_diff": _mean_of(rows, "kge_difference_vs_control"),
                "n_exp": _mean_of(rows, "n_expver"),
                "n_control": _mean_of(rows, "n_control"),
            }
            domain_summaries.append(item)
            print(
                f"{domain},{expver},"
                f"{_fmt_signed(item['rmse_improvement'],2,'%')},"
                f"{_fmt_signed(item['bias_diff'],2)},"
                f"{_fmt_signed(item['abs_bias_diff'],2)},"
                f"{_fmt_signed(item['corr_diff'],4)},"
                f"{_fmt_signed(item['nse_diff'],4)},"
                f"{_fmt_signed(item['kge_diff'],4)},"
                f"{_fmt(item['n_exp'],0)},"
                f"{_fmt(item['n_control'],0)}"
            )

    # ------------------------------------------------------------------
    # Best/worst by RMSE improvement
    # ------------------------------------------------------------------
    print()
    print("BEST/WORST DOMAIN BY MEAN DAILY RMSE IMPROVEMENT")
    print("------------------------------------------------")
    for expver in exp_list:
        items = [x for x in domain_summaries if x["expver"] == expver and np.isfinite(x["rmse_improvement"])]
        if not items:
            print(f"{expver}: unavailable")
            continue
        best = max(items, key=lambda x: x["rmse_improvement"])
        worst = min(items, key=lambda x: x["rmse_improvement"])
        print(
            f"{expver}: best={best['domain']} ({_fmt_signed(best['rmse_improvement'],2,'%')}), "
            f"worst={worst['domain']} ({_fmt_signed(worst['rmse_improvement'],2,'%')})"
        )

    # ------------------------------------------------------------------
    # Bias warning flags
    # ------------------------------------------------------------------
    print()
    print("BIAS FLAGS")
    print("----------")
    print("Negative Δ|Bias| means the experiment reduced absolute bias; positive means bias worsened.")
    for item in domain_summaries:
        if np.isfinite(item["abs_bias_diff"]) and item["abs_bias_diff"] > 0.0:
            print(
                f"BIAS_WORSE: domain={item['domain']} expver={item['expver']} "
                f"Δ|Bias|={_fmt_signed(item['abs_bias_diff'],2)} "
                f"RMSE_improvement={_fmt_signed(item['rmse_improvement'],2,'%')}"
            )

    print("=" * 88)
    print("END FINAL BENCHMARK SUMMARY")
    print("=" * 88)
    print()

# ============================================================
# Main
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark CaMa-Flood experiment river discharge against GloFAS and plot metrics/maps."
    )
    parser.add_argument("--start-date", default="20160401", help="Start date YYYYMMDD, inclusive")
    parser.add_argument("--end-date", default="20160405", help="End date YYYYMMDD, inclusive")
    parser.add_argument("--expvers", nargs="+", default=["j6n9", "izay"], help="Experiment expver list; the last item is treated as the control")
    parser.add_argument("--outdir", default="/perm/pad/glofas/benchmark", help="Output/cache directory")
    parser.add_argument("--river-threshold", type=float, default=2.0, help="River mask threshold in m3/s")
    parser.add_argument("--obs-min-discharge", type=float, default=None, help="If set, score/map only points where reference discharge exceeds this value in m3/s, e.g. 50")
    parser.add_argument("--continent-maps", action="store_true", help="Also write zoomed map PNGs for continental domains")
    parser.add_argument("--glofas-remap", default="nearest", choices=["nearest"], help="How to map GloFAS to the CaMa grid when resolutions differ")
    parser.add_argument("--glofas-configuration", default="v5.0", help="GloFAS MARS configuration, default: v5.0")
    parser.add_argument("--glofas-forcing", default="ecmf-era5", help="GloFAS forcing dataset, default: ecmf-era5")
    parser.add_argument("--glofas-expver", default="1", help="GloFAS archive expver, default: 1")
    parser.add_argument("--uparea-15min-file", default="/home/rdx/data/50r1/camaflood/static_network_nc_v2.1/glb_15min/ncdata.nc", help="CaMa-Flood 15-arcmin static network NetCDF containing uparea")
    parser.add_argument("--uparea-03min-file", default="/home/rdx/data/50r1/camaflood/static_network_nc_v2.1/glb_03min/ncdata.nc", help="CaMa-Flood 3-arcmin static network NetCDF containing uparea")
    parser.add_argument("--uparea-tolerance", type=float, default=0.10, help="Maximum symmetric relative upstream-area mismatch; default 0.10 (10%%). Use a negative value to disable the filter.")
    parser.add_argument("--cama-archive-mode", default="auto", choices=["auto", "daily", "monthly"], help="CaMa archive layout: daily uses date=valid day step=24; monthly uses date=YYYYMM01 step=24*day; auto tries daily then monthly")
    parser.add_argument("--force-retrieve", action="store_true", help="Ignore cached GRIB files")
    parser.add_argument("--no-river-mask", action="store_true", help="Use all valid points, not only river-threshold points")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG plot generation")
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument("--no-map-netcdf", action="store_true", help="Skip writing map NetCDF files")
    parser.add_argument("--no-map-png", action="store_true", help="Write map NetCDFs but skip PNG map plotting.")
    args = parser.parse_args()

    global HAVE_NETCDF4
    if args.no_map_netcdf:
        HAVE_NETCDF4 = False

    t0 = time.time()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dates = date_range(args.start_date, args.end_date)
    control_expver = args.expvers[-1]

    print("Benchmark settings")
    print("------------------")
    print("dates:", dates)
    print("expvers:", args.expvers)
    print("control_expver:", control_expver, "(last item in --expvers)")
    print("outdir:", outdir)
    print("river_threshold:", args.river_threshold)
    print("obs_min_discharge:", args.obs_min_discharge)
    print("continent_maps:", args.continent_maps)
    print("glofas_remap:", args.glofas_remap)
    print("glofas_configuration:", args.glofas_configuration)
    print("glofas_forcing:", args.glofas_forcing)
    print("glofas_expver:", args.glofas_expver)
    print("uparea_15min_file:", args.uparea_15min_file)
    print("uparea_03min_file:", args.uparea_03min_file)
    print("uparea_tolerance:", args.uparea_tolerance, "(disabled if < 0)")
    print("cama_archive_mode:", args.cama_archive_mode)
    print("force_retrieve:", args.force_retrieve)
    print("no_river_mask:", args.no_river_mask)
    print()

    if args.plots_only:
        daily_csv = (
            outdir
            / f"benchmark_daily_domains_{args.start_date}_{args.end_date}.csv"
        )

        summary_csv = (
            outdir
            / f"benchmark_summary_{args.start_date}_{args.end_date}.csv"
        )

        daily_vs_control_csv = (
            outdir
            / f"benchmark_daily_vs_control_{args.start_date}_{args.end_date}.csv"
        )

        summary_vs_control_csv = (
            outdir
            / f"benchmark_summary_vs_control_{args.start_date}_{args.end_date}.csv"
        )

        for f in [
            daily_csv,
            summary_csv,
            daily_vs_control_csv,
            summary_vs_control_csv,
        ]:
            print("Checking:", f)
            if not f.exists():
                raise FileNotFoundError(f)

        print("Reading existing benchmark files")

        daily_df = pd.read_csv(daily_csv)
        summary_df = pd.read_csv(summary_csv)

        daily_vs_control_df = pd.read_csv(daily_vs_control_csv)
        summary_vs_control_df = pd.read_csv(summary_vs_control_csv)

        daily_df["date"] = daily_df["date"].astype(str)
        daily_rows = daily_df.to_dict("records")

        if not args.no_plots:
            print("Regenerating time-series plots from CSV")
            plot_domain_timeseries(
                daily_rows,
                args.expvers,
                list(DOMAINS.keys()),
                outdir,
                args.start_date,
                args.end_date,
            )
            plot_relative_improvement_timeseries(
                daily_vs_control_df.to_dict("records"),
                args.expvers,
                control_expver,
                list(DOMAINS.keys()),
                outdir,
                args.start_date,
                args.end_date,
            )
            plot_control_difference_timeseries(
                daily_vs_control_df.to_dict("records"),
                args.expvers,
                control_expver,
                list(DOMAINS.keys()),
                outdir,
                args.start_date,
                args.end_date,
            )

            print("Regenerating map plots from existing NetCDF files")
            regenerate_maps_from_nc(
                outdir=outdir,
                expvers=args.expvers,
                control_expver=control_expver,
                start_date=args.start_date,
                end_date=args.end_date,
                continent_maps=args.continent_maps,
            )

        print_final_benchmark_summary(
            daily_rows=daily_rows,
            control_diff_rows=daily_vs_control_df.to_dict("records"),
            summary_rows=summary_df.to_dict("records"),
            summary_control_rows=summary_vs_control_df.to_dict("records"),
            expvers=args.expvers,
            control_expver=control_expver,
            start_date=args.start_date,
            end_date=args.end_date,
        )

        print()
        print("Plots-only mode completed")
        print("Elapsed seconds:", time.time() - t0)
        return 0

    daily_rows: list[dict] = []
    pooled_obs: dict[str, list[np.ndarray]] = {e: [] for e in args.expvers}
    pooled_mod: dict[str, list[np.ndarray]] = {e: [] for e in args.expvers}

    lats = None
    lons = None
    ny = None
    nx = None
    domain_masks: dict[str, np.ndarray] | None = None
    uparea_match_mask: np.ndarray | None = None
    uparea15_on_cama: np.ndarray | None = None
    uparea03_on_cama: np.ndarray | None = None
    map_acc: dict[str, dict[str, np.ndarray]] = {}

    for date in dates:
        print()
        print("=" * 100)
        print("DATE", date)
        print("=" * 100)

        glofas = retrieve_glofas(
            date,
            outdir,
            force=args.force_retrieve,
            configuration=args.glofas_configuration,
            forcing=args.glofas_forcing,
            expver=args.glofas_expver,
        )
        q_glofas_cache = None

        for expver in args.expvers:
            print()
            print("-" * 80)
            print(f"Experiment {expver}, date {date}")
            print("-" * 80)

            try:
                cama = retrieve_cama_discharge(
                    date,
                    expver,
                    outdir,
                    force=args.force_retrieve,
                    archive_mode=args.cama_archive_mode,
                )
            except Exception as e:
                print()
                print("WARNING: CaMa retrieval/read failed")
                print("----------------------------------")
                print("date  :", date)
                print("expver:", expver)
                print("mode  :", args.cama_archive_mode)
                print("error :", e)
                print("Skipping this experiment/date and continuing.")
                continue

            q_cama = clean_field(get_values(cama[0]), f"q_cama_{expver}_{date}")

            if lats is None:
                lats, lons = get_lat_lon(cama[0])
                ny, nx = infer_regular_grid(lats, lons)
                print("Grid:", f"ny={ny:,}", f"nx={nx:,}", f"npts={lats.size:,}")

                domain_masks = {
                    name: domain_mask_for_bbox(lats, lons, bbox)
                    for name, bbox in DOMAINS.items()
                }
                for name, m in domain_masks.items():
                    print(f"Domain {name:13s}: {np.count_nonzero(m):,} grid points")

                if args.uparea_tolerance >= 0.0:
                    uparea_match_mask, uparea15_on_cama, uparea03_on_cama = build_uparea_consistency_mask(
                        lats,
                        lons,
                        args.uparea_15min_file,
                        args.uparea_03min_file,
                        args.uparea_tolerance,
                    )
                    for name, dm in domain_masks.items():
                        n_domain_valid = int(np.count_nonzero(dm & np.isfinite(uparea15_on_cama) & np.isfinite(uparea03_on_cama)))
                        n_domain_keep = int(np.count_nonzero(dm & uparea_match_mask))
                        frac = 100.0 * n_domain_keep / n_domain_valid if n_domain_valid else np.nan
                        print(f"Uparea match {name:13s}: {n_domain_keep:,}/{n_domain_valid:,} ({frac:.2f}%)")
                    print()
                else:
                    print("Upstream-area consistency filter: disabled")
                    print()

                for e in args.expvers:
                    map_acc[e] = {
                        "sum_error": np.zeros(lats.size, dtype=np.float64),
                        "sum_sq_error": np.zeros(lats.size, dtype=np.float64),
                        "sum_obs": np.zeros(lats.size, dtype=np.float64),
                        "sum_obs_sq": np.zeros(lats.size, dtype=np.float64),
                        "sum_model": np.zeros(lats.size, dtype=np.float64),
                        "sum_model_sq": np.zeros(lats.size, dtype=np.float64),
                        "sum_obs_model": np.zeros(lats.size, dtype=np.float64),
                        "count": np.zeros(lats.size, dtype=np.int32),
                    }

            if q_glofas_cache is None:
                q_glofas_cache = clean_field(
                    remap_glofas_to_cama_domain(glofas[0], cama[0], method=args.glofas_remap),
                    f"q_glofas_{date}",
                )

            q_glofas = q_glofas_cache

            if q_glofas.size != q_cama.size:
                raise ValueError(
                    f"Size mismatch after padding for {expver} {date}: "
                    f"GloFAS={q_glofas.size}, CaMa={q_cama.size}"
                )

            base_mask = np.isfinite(q_glofas) & np.isfinite(q_cama)
            if uparea_match_mask is not None:
                base_mask &= uparea_match_mask

            if args.obs_min_discharge is not None:
                river_mask = base_mask & (q_glofas > args.obs_min_discharge)
                mask_name = f"obs_gt_{args.obs_min_discharge:g}"
            elif args.no_river_mask:
                river_mask = base_mask
                mask_name = "all_valid"
            else:
                river_mask = base_mask & (
                    (np.abs(q_glofas) > args.river_threshold)
                    | (np.abs(q_cama) > args.river_threshold)
                )
                mask_name = "river_mask"

            if uparea_match_mask is not None:
                mask_name += f"_uparea{100.0 * args.uparea_tolerance:g}pct"

            # Daily metrics for all domains.
            assert domain_masks is not None
            for domain_name, domain_mask in domain_masks.items():
                mask = river_mask & domain_mask
                met = metrics(q_glofas, q_cama, mask)
                print_metrics(f"Daily metrics {expver} {date} {domain_name} ({mask_name})", met)

                daily_rows.append({
                    "date": date,
                    "domain": domain_name,
                    "expver": expver,
                    "mask": mask_name,
                    "n": met["n"],
                    "bias_model_minus_glofas": met["bias"],
                    "rmse": met["rmse"],
                    "mae": met["mae"],
                    "correlation": met["corr"],
                    "nse": met["nse"],
                    "kge": met["kge"],
                    "glofas_mean": met["obs_mean"],
                    "model_mean": met["model_mean"],
                    "glofas_median": met["obs_median"],
                    "model_median": met["model_median"],
                })

            # Pooled global/river-mask samples for summary.
            valid = river_mask & np.isfinite(q_glofas) & np.isfinite(q_cama)
            if np.any(valid):
                pooled_obs[expver].append(q_glofas[valid].astype(np.float32))
                pooled_mod[expver].append(q_cama[valid].astype(np.float32))

            # Average error/RMSE map accumulation. Use the same benchmark mask as the scores.
            err_valid = river_mask
            err = np.zeros(q_cama.shape, dtype=np.float32)
            err[err_valid] = q_cama[err_valid] - q_glofas[err_valid]
            acc = map_acc[expver]
            obs_v = q_glofas[err_valid].astype(np.float64)
            mod_v = q_cama[err_valid].astype(np.float64)
            acc["sum_error"][err_valid] += err[err_valid].astype(np.float64)
            acc["sum_sq_error"][err_valid] += (err[err_valid].astype(np.float64) ** 2)
            acc["sum_obs"][err_valid] += obs_v
            acc["sum_obs_sq"][err_valid] += obs_v ** 2
            acc["sum_model"][err_valid] += mod_v
            acc["sum_model_sq"][err_valid] += mod_v ** 2
            acc["sum_obs_model"][err_valid] += obs_v * mod_v
            acc["count"][err_valid] += 1

            del q_cama, base_mask, river_mask, err, err_valid, cama

        del q_glofas_cache, glofas

    # Summary metrics pooled over global river-mask points.
    summary_rows: list[dict] = []

    print()
    print("=" * 100)
    print("POOLED SUMMARY")
    print("=" * 100)

    for expver in args.expvers:
        if pooled_obs[expver]:
            obs = np.concatenate(pooled_obs[expver])
            mod = np.concatenate(pooled_mod[expver])
            mask = np.isfinite(obs) & np.isfinite(mod)
            met = metrics(obs, mod, mask)
        else:
            met = metrics(np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=bool))

        print_metrics(f"Pooled global metrics {expver} {args.start_date}-{args.end_date}", met)

        summary_rows.append({
            "start_date": args.start_date,
            "end_date": args.end_date,
            "domain": "Global",
            "expver": expver,
            "n": met["n"],
            "bias_model_minus_glofas": met["bias"],
            "rmse": met["rmse"],
            "mae": met["mae"],
            "correlation": met["corr"],
            "nse": met["nse"],
            "kge": met["kge"],
            "glofas_mean": met["obs_mean"],
            "model_mean": met["model_mean"],
            "glofas_median": met["obs_median"],
            "model_median": met["model_median"],
        })


    # Pooled summary comparison versus control.
    summary_by_exp = {r["expver"]: r for r in summary_rows}
    summary_control_rows = []
    ctrl = summary_by_exp.get(control_expver)
    if ctrl is not None:
        ctrl_rmse = float(ctrl["rmse"])
        ctrl_bias = float(ctrl["bias_model_minus_glofas"])
        ctrl_corr = float(ctrl["correlation"])
        ctrl_nse = float(ctrl.get("nse", np.nan))
        ctrl_kge = float(ctrl.get("kge", np.nan))
        for expver in args.expvers:
            if expver == control_expver:
                continue
            r = summary_by_exp.get(expver)
            if r is None:
                continue
            rmse = float(r["rmse"])
            bias = float(r["bias_model_minus_glofas"])
            corr = float(r["correlation"])
            nse = float(r.get("nse", np.nan))
            kge = float(r.get("kge", np.nan))
            if np.isfinite(ctrl_rmse) and ctrl_rmse != 0.0 and np.isfinite(rmse):
                rmse_improvement_pct = 100.0 * (ctrl_rmse - rmse) / ctrl_rmse
            else:
                rmse_improvement_pct = np.nan
            summary_control_rows.append({
                "start_date": args.start_date,
                "end_date": args.end_date,
                "domain": "Global",
                "expver": expver,
                "control_expver": control_expver,
                "n_expver": r["n"],
                "n_control": ctrl["n"],
                "rmse": rmse,
                "rmse_control": ctrl_rmse,
                "rmse_difference_vs_control": rmse - ctrl_rmse,
                "rmse_improvement_pct_vs_control": rmse_improvement_pct,
                "bias_model_minus_glofas": bias,
                "bias_control": ctrl_bias,
                "bias_difference_vs_control": bias - ctrl_bias,
                "abs_bias_difference_vs_control": abs(bias) - abs(ctrl_bias),
                "correlation": corr,
                "correlation_control": ctrl_corr,
                "correlation_difference_vs_control": corr - ctrl_corr,
                "nse": nse,
                "nse_control": ctrl_nse,
                "nse_difference_vs_control": nse - ctrl_nse,
                "kge": kge,
                "kge_control": ctrl_kge,
                "kge_difference_vs_control": kge - ctrl_kge,
            })

    daily_csv = outdir / f"benchmark_daily_domains_{args.start_date}_{args.end_date}.csv"
    summary_csv = outdir / f"benchmark_summary_{args.start_date}_{args.end_date}.csv"
    write_csv(daily_csv, daily_rows)
    write_csv(summary_csv, summary_rows)

    control_diff_rows = make_control_difference_rows(daily_rows, args.expvers, control_expver)
    control_diff_csv = outdir / f"benchmark_daily_vs_control_{args.start_date}_{args.end_date}.csv"
    write_csv(control_diff_csv, control_diff_rows)

    summary_control_csv = outdir / f"benchmark_summary_vs_control_{args.start_date}_{args.end_date}.csv"
    write_csv(summary_control_csv, summary_control_rows)

    if not args.no_plots:
        plot_domain_timeseries(daily_rows, args.expvers, 
                list(DOMAINS.keys()), outdir, args.start_date, args.end_date)
        plot_relative_improvement_timeseries(control_diff_rows, args.expvers, control_expver,
                list(DOMAINS.keys()), outdir, args.start_date, args.end_date)
        plot_control_difference_timeseries(control_diff_rows, args.expvers, control_expver,
                list(DOMAINS.keys()), outdir, args.start_date, args.end_date)
        assert ny is not None and nx is not None
        finalize_and_plot_maps(map_acc, args.expvers, ny, nx, outdir, control_expver, 
                lats, lons, args.start_date, args.end_date, continent_maps=args.continent_maps)

    print_final_benchmark_summary(
        daily_rows=daily_rows,
        control_diff_rows=control_diff_rows,
        summary_rows=summary_rows,
        summary_control_rows=summary_control_rows,
        expvers=args.expvers,
        control_expver=control_expver,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    print()
    print("Completed")
    print("Elapsed seconds:", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
