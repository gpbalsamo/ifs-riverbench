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


def human_mode(mode: str) -> str:
    if mode == "best_metric":
        return "Best Metric"
    if mode == "best_experiment":
        return "Best Experiment"
    if mode == "experiment_difference":
        return "Experiment Difference"
    return mode


def build_index(dashboard_files):
    rows = []
    for f in sorted(dashboard_files, key=lambda x: x.name):
        meta = parse_dashboard_filename(f.name)
        if meta is None:
            metric = "unknown"
            mode = "other"
            run = "-"
            experiments = "-"
        else:
            metric = meta["metric"]
            mode = meta["mode"]
            run = f"{meta['start']} to {meta['end']} ({meta['res']})"
            experiments = meta["expvers"].replace("_", ", ")

        rows.append(
            {
                "file": f.name,
                "metric": metric,
                "mode": human_mode(mode),
                "run": run,
                "experiments": experiments,
            }
        )

    lines = []
    for r in rows:
        lines.append(
            "<tr>"
            f"<td><a href=\"{r['file']}\">{r['file']}</a></td>"
            f"<td>{r['metric']}</td>"
            f"<td>{r['mode']}</td>"
            f"<td>{r['run']}</td>"
            f"<td>{r['experiments']}</td>"
            "</tr>"
        )

    table_rows = "\n".join(lines)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>IFS Riverbench Dashboards</title>
<style>
body {{
  margin: 0;
  font-family: Arial, sans-serif;
  background: #f7f7f7;
  color: #1f2937;
}}
header {{
  padding: 12px 20px;
  background: #1f2937;
  color: white;
}}
header h1 {{
  margin: 0;
  font-size: 22px;
}}
header p {{
  margin: 4px 0 0;
  font-size: 14px;
  color: #cbd5e1;
}}
main {{
  max-width: 1200px;
  margin: 40px auto;
  padding: 0 20px;
}}
.card {{
  background: white;
  border: 1px solid #ccc;
  border-radius: 6px;
  padding: 18px 20px;
  margin-bottom: 16px;
}}
.table-wrap {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; background: #fff; }}
th, td {{ border-bottom: 1px solid #ccc; padding: 10px 8px; text-align: left; vertical-align: top; font-size: 14px; }}
th {{ font-size: 12px; color: #4b5563; text-transform: uppercase; letter-spacing: 0.04em; }}
a {{ color: #1f2937; font-weight: bold; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
.badge {{ display:inline-block; padding:2px 7px; border:1px solid #ccc; border-radius:999px; font-size:12px; background:#f3f4f6; color: #4b5563; }}
</style>
</head>
<body>
<header>
  <h1>IFS Riverbench Dashboards</h1>
  <p>River discharge model experiments vs. observations, by station</p>
</header>

<main>
  <div class=\"card\">
    <p><span class=\"badge\">Tip</span> Open dashboards through a web server or Sites for full station hydrograph loading.</p>
    <div class=\"table-wrap\">
      <table>
        <thead>
          <tr>
            <th>Dashboard</th>
            <th>Metric</th>
            <th>Mode</th>
            <th>Run</th>
            <th>Experiments</th>
          </tr>
        </thead>
        <tbody>
          {table_rows}
        </tbody>
      </table>
    </div>
  </div>
</main>
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

    index_html = build_index(dashboard_files)
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
