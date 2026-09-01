#!/usr/bin/env python3
"""
03_prepare_sites_bundle.py

Create a single deployable directory containing:
- all generated dashboard HTML files
- dashboard_data directory
- an index.html switchboard to open dashboards by metric/mode
"""

from pathlib import Path
import argparse
import json
import shutil
import re
import os
import time


def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare a single sites bundle directory with dashboards and data."
    )
    p.add_argument(
        "--workflow-dir",
        type=Path,
        default=Path("."),
        help="Workflow directory containing generated dashboards and dashboard_data.",
    )
    p.add_argument(
        "--bundle-dirname",
        default="site_bundle",
        help="Output subdirectory under workflow-dir. Default: site_bundle",
    )
    p.add_argument(
        "--dashboard-source-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing generated dashboard HTML files. "
            "Default: workflow-dir, with auto-fallback to workflow-dir/bundle-dirname."
        ),
    )
    p.add_argument(
        "--html-pattern",
        default="global_station_dashboard_*.html",
        help="Pattern for dashboard html files to include.",
    )
    p.add_argument(
        "--include-extra-html",
        action="store_true",
        help="Include non dashboard html files (for example validation pages).",
    )
    p.add_argument(
        "--skip-dashboard-data-copy",
        action="store_true",
        help=(
            "Skip syncing dashboard_data into the bundle directory. "
            "Useful when only HTML/index changed and data is already present."
        ),
    )
    return p.parse_args()


def parse_dashboard_filename(name: str):
    """
    Parse:
      global_station_dashboard_<expvers>_<date_start>_<date_end>_<resolution>arcmin_<metric>_<mode>.html
    """
    pattern = re.compile(
        r"^global_station_dashboard_(?P<expvers>.+?)_"
        r"(?P<start>\d{8})_(?P<end>\d{8})_"
        r"(?P<res>\d+arcmin)_"
        r"(?P<metric>kge|correlation)_"
        r"(?P<mode>best_metric|best_experiment|experiment_difference)\.html$"
    )
    m = pattern.match(name)
    if not m:
        return None
    return m.groupdict()


def split_expvers(blob, known_expvers):
    """Split an underscore-joined expver blob (e.g. "iwya_glofas_v5") back
    into individual experiment names, by greedily matching the longest
    known expver name at each position. A plain blob.split("_") breaks as
    soon as any expver name itself contains an underscore (e.g. glofas_v5),
    which is exactly the case here."""
    names = sorted(known_expvers, key=len, reverse=True)
    result = []
    rest = blob
    while rest:
        for name in names:
            if rest == name or rest.startswith(name + "_"):
                result.append(name)
                rest = rest[len(name):].lstrip("_")
                break
        else:
            # Unrecognised remainder: fall back to a single raw token so
            # nothing is silently dropped.
            result.append(rest)
            break
    return result


def build_index(dashboard_files, known_expvers):
    """Single-page dashboard switchboard: metric/view buttons (and, for
    experiment_difference, a comparison-pair selector) swap which existing
    dashboard file is shown in an embedded frame, rather than merging all
    dashboards' data into one page (that would multiply an already-heavy
    ~15-18MB file by 8)."""
    entries = []
    for f in sorted(dashboard_files, key=lambda x: x.name):
        meta = parse_dashboard_filename(f.name)
        if meta is None:
            continue
        expvers = split_expvers(meta["expvers"], known_expvers)
        entries.append(
            {
                "file": f.name,
                "metric": meta["metric"],
                "mode": meta["mode"],
                "expvers": expvers,
            }
        )

    metrics = sorted({e["metric"] for e in entries})
    modes = [m for m in ("best_metric", "best_experiment", "experiment_difference") if any(e["mode"] == m for e in entries)]

    # dashboards[metric][mode] = filename, for best_metric/best_experiment.
    # dashboards[metric]["experiment_difference"][pair_key] = filename, and
    # pairs[pair_key] = human label, for experiment_difference.
    dashboards = {m: {} for m in metrics}
    pairs = {}
    for e in entries:
        if e["mode"] == "experiment_difference":
            ref, target = e["expvers"][0], e["expvers"][-1]
            pair_key = f"{target}_vs_{ref}"
            pairs[pair_key] = f"{target} vs {ref}"
            dashboards[e["metric"]].setdefault("experiment_difference", {})[pair_key] = e["file"]
        else:
            dashboards[e["metric"]][e["mode"]] = e["file"]

    dashboards_json = json.dumps(dashboards)
    # Put GloFAS comparisons last: it's an external reference product, not
    # an IFS experiment, so an IFS-vs-IFS comparison is the more useful
    # default when one exists (plain alphabetical sorts "glofas_v5" first).
    pair_order = sorted(pairs, key=lambda k: ("glofas" in k, k))
    pairs_json = json.dumps([{"key": k, "label": pairs[k]} for k in pair_order])

    default_metric = "kge" if "kge" in metrics else metrics[0]
    default_mode = "experiment_difference" if "experiment_difference" in modes else modes[0]
    default_pair = pair_order[0] if pair_order else None

    mode_labels = {"best_metric": "Best metric", "best_experiment": "Best experiment", "experiment_difference": "Difference"}
    metric_buttons = "\n".join(
        f'<button class="ctrl-btn metric-btn" data-metric="{m}">{m.upper() if m == "kge" else m.capitalize()}</button>'
        for m in metrics
    )
    mode_buttons = "\n".join(
        f'<button class="ctrl-btn mode-btn" data-mode="{m}">{mode_labels.get(m, m)}</button>' for m in modes
    )
    pair_buttons = "\n".join(
        f'<button class="ctrl-btn pair-btn" data-pair="{k}">{pairs[k]}</button>' for k in pair_order
    )

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>IFS Riverbench Dashboards</title>
<style>
html, body {{
  margin: 0;
  height: 100%;
  font-family: Arial, sans-serif;
  background: #f7f7f7;
  color: #1f2937;
}}
header {{
  padding: 10px 20px;
  background: #1f2937;
  color: white;
}}
header h1 {{
  margin: 0;
  font-size: 20px;
}}
header p {{
  margin: 2px 0 0;
  font-size: 13px;
  color: #cbd5e1;
}}
#controls {{
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px 18px;
  padding: 10px 20px;
  background: white;
  border-bottom: 1px solid #ccc;
}}
.ctrl-group {{ display: flex; align-items: center; gap: 6px; }}
.ctrl-label {{ font-size: 12px; color: #4b5563; text-transform: uppercase; letter-spacing: 0.04em; margin-right: 2px; }}
.ctrl-btn {{
  font-size: 13px;
  padding: 5px 12px;
  border-radius: 5px;
  border: 1px solid #ccc;
  background: #f3f4f6;
  color: #4b5563;
  cursor: pointer;
}}
.ctrl-btn.active {{ background: #1f2937; color: #fff; border-color: #1f2937; }}
#pair-group {{ display: none; }}
#open-new-tab {{ margin-left: auto; font-size: 13px; color: #1f2937; font-weight: bold; text-decoration: none; }}
#open-new-tab:hover {{ text-decoration: underline; }}
#frame-wrap {{ position: absolute; top: 96px; bottom: 0; left: 0; right: 0; }}
iframe {{ width: 100%; height: 100%; border: none; }}
@media (max-width: 900px) {{
  #frame-wrap {{ top: 132px; }}
}}
</style>
</head>
<body>
<header>
  <h1>IFS Riverbench Dashboards</h1>
  <p>River discharge model experiments vs. observations, by station</p>
</header>
<div id=\"controls\">
  <div class=\"ctrl-group\">
    <span class=\"ctrl-label\">Metric</span>
    {metric_buttons}
  </div>
  <div class=\"ctrl-group\">
    <span class=\"ctrl-label\">View</span>
    {mode_buttons}
  </div>
  <div class=\"ctrl-group\" id=\"pair-group\">
    <span class=\"ctrl-label\">Comparison</span>
    {pair_buttons}
  </div>
  <a id=\"open-new-tab\" href=\"#\" target=\"_blank\">Open in new tab &#8599;</a>
</div>
<div id=\"frame-wrap\">
  <iframe id=\"dashboard-frame\" title=\"Dashboard\"></iframe>
</div>
<script>
const DASHBOARDS = {dashboards_json};
const PAIRS = {pairs_json};

let state = {{
  metric: "{default_metric}",
  mode: "{default_mode}",
  pair: {f'"{default_pair}"' if default_pair else "null"}
}};

function currentFile() {{
  const byMode = DASHBOARDS[state.metric] || {{}};
  if (state.mode === "experiment_difference") {{
    const byPair = byMode.experiment_difference || {{}};
    return byPair[state.pair] || null;
  }}
  return byMode[state.mode] || null;
}}

function render() {{
  document.querySelectorAll(".metric-btn").forEach(b => b.classList.toggle("active", b.dataset.metric === state.metric));
  document.querySelectorAll(".mode-btn").forEach(b => b.classList.toggle("active", b.dataset.mode === state.mode));
  document.querySelectorAll(".pair-btn").forEach(b => b.classList.toggle("active", b.dataset.pair === state.pair));
  document.getElementById("pair-group").style.display = state.mode === "experiment_difference" ? "flex" : "none";

  const file = currentFile();
  const frame = document.getElementById("dashboard-frame");
  const link = document.getElementById("open-new-tab");
  if (file) {{
    if (frame.getAttribute("src") !== file) frame.setAttribute("src", file);
    link.setAttribute("href", file);
  }}
}}

document.querySelectorAll(".metric-btn").forEach(b => b.addEventListener("click", () => {{ state.metric = b.dataset.metric; render(); }}));
document.querySelectorAll(".mode-btn").forEach(b => b.addEventListener("click", () => {{ state.mode = b.dataset.mode; render(); }}));
document.querySelectorAll(".pair-btn").forEach(b => b.addEventListener("click", () => {{ state.pair = b.dataset.pair; render(); }}));

render();
</script>
</body>
</html>
"""


def copy_tree_merge(src: Path, dst: Path):
    """
    Recursively copy src into dst, overwriting files in place.

    This avoids deleting large directory trees on NFS first, which can fail
    intermittently with "Directory not empty" during rmtree.
    """
    src = src.resolve()
    dst = dst.resolve()

    if not src.exists():
        raise FileNotFoundError(f"Source directory not found: {src}")

    dst.mkdir(parents=True, exist_ok=True)

    copied_files = 0
    skipped_files = 0

    for root, dirs, files in os.walk(str(src)):
        root_path = Path(root)
        rel = root_path.relative_to(src)
        target_root = dst / rel
        target_root.mkdir(parents=True, exist_ok=True)

        for d in dirs:
            (target_root / d).mkdir(parents=True, exist_ok=True)

        for f in files:
            src_file = root_path / f
            dst_file = target_root / f

            if dst_file.exists():
                try:
                    src_stat = src_file.stat()
                    dst_stat = dst_file.stat()

                    if (
                        src_stat.st_size == dst_stat.st_size
                        and int(src_stat.st_mtime) == int(dst_stat.st_mtime)
                    ):
                        skipped_files += 1
                        continue
                except OSError:
                    pass

            shutil.copy2(src_file, dst_file)
            copied_files += 1

    return copied_files, skipped_files


def main():
    t0 = time.time()
    args = parse_args()
    workflow_dir = args.workflow_dir.resolve()
    bundle_dir = workflow_dir / args.bundle_dirname
    source_dir = (
        args.dashboard_source_dir.resolve()
        if args.dashboard_source_dir is not None
        else workflow_dir
    )

    if not workflow_dir.exists():
        raise FileNotFoundError(f"workflow dir not found: {workflow_dir}")

    dashboard_data = workflow_dir / "dashboard_data"
    if not dashboard_data.exists():
        raise FileNotFoundError(f"dashboard_data not found: {dashboard_data}")

    dashboard_files = sorted(source_dir.glob(args.html_pattern))

    if (
        not dashboard_files
        and args.dashboard_source_dir is None
        and source_dir == workflow_dir
        and bundle_dir.exists()
    ):
        source_dir = bundle_dir
        dashboard_files = sorted(source_dir.glob(args.html_pattern))

    if args.include_extra_html:
        extra = sorted(source_dir.glob("*.html"))
        known = {p.name for p in dashboard_files}
        dashboard_files.extend([p for p in extra if p.name not in known])

    if not dashboard_files:
        raise RuntimeError(f"No dashboard files found with pattern: {args.html_pattern}")

    in_place = source_dir.resolve() == bundle_dir.resolve()

    if not in_place:
        if bundle_dir.exists():
            shutil.rmtree(bundle_dir)
        bundle_dir.mkdir(parents=True, exist_ok=True)

        for src in dashboard_files:
            shutil.copy2(src, bundle_dir / src.name)
    else:
        bundle_dir.mkdir(parents=True, exist_ok=True)

    copied_files = 0
    skipped_files = 0

    if args.skip_dashboard_data_copy:
        print("Skipping dashboard_data sync (--skip-dashboard-data-copy).")
    else:
        bundle_dashboard_data = bundle_dir / "dashboard_data"
        copied_files, skipped_files = copy_tree_merge(dashboard_data, bundle_dashboard_data)

    known_expvers = sorted(p.name for p in dashboard_data.iterdir() if p.is_dir())
    index_html = build_index(dashboard_files, known_expvers)
    (bundle_dir / "index.html").write_text(index_html, encoding="utf-8")

    print(f"Prepared bundle: {bundle_dir}")
    print(f"Dashboards copied: {len(dashboard_files)}")
    if args.skip_dashboard_data_copy:
        print("Data copied: skipped")
    else:
        print(f"Data copied: {bundle_dir / 'dashboard_data'}")
        print(f"Data files copied: {copied_files}")
        print(f"Data files unchanged (skipped): {skipped_files}")
    print(f"Index page: {bundle_dir / 'index.html'}")
    print(f"Done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
