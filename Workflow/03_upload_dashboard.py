#!/usr/bin/env python3
"""
03_upload_dashboard.py

Upload ifs-riverbench dashboard bundle to ECMWF Sites.

This script uploads:
    1. all dashboard HTML files (including index.html) from workflow-dir
    2. the dashboard_data directory recursively (unless --html-only)

Authentication
--------------
Set your ECMWF Sites API token as an environment variable before running:

  export ECMWF_SITES_TOKEN="..."

Then run:

  python3 03_upload_dashboard.py

Optional examples:

  python3 03_upload_dashboard.py --dry-run
    python3 03_upload_dashboard.py --workflow-dir /perm/USER/ifs-riverbench/Workflow/site_bundle
  python3 03_upload_dashboard.py --html-pattern "global_station_dashboard_*.html"
  python3 03_upload_dashboard.py --html-only
"""

from pathlib import Path
import argparse
import os
import sys

from sites.sdk import SitesClient
from sites.sdk.sites import Site, Authenticator


DEFAULT_WORKFLOW_DIR = Path(f"/perm/{os.environ['USER']}/ifs-riverbench/Workflow/site_bundle")
DEFAULT_DASHBOARD_DATA_DIRNAME = "dashboard_data"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload ifs-riverbench dashboard HTML and dashboard_data to ECMWF Sites."
    )

    parser.add_argument(
        "--workflow-dir",
        type=Path,
        default=DEFAULT_WORKFLOW_DIR,
        help=(
            "Directory containing dashboard HTML files and dashboard_data. "
            f"Default: {DEFAULT_WORKFLOW_DIR}"
        ),
    )

    parser.add_argument(
        "--html-pattern",
        default="*.html",
        help='Glob pattern for HTML files to upload. Default: "*.html".',
    )

    parser.add_argument(
        "--dashboard-data-dirname",
        default=DEFAULT_DASHBOARD_DATA_DIRNAME,
        help='Name of dashboard data directory inside workflow-dir. Default: "dashboard_data".',
    )

    parser.add_argument(
        "--space",
        default=os.environ['USER'],
        help=f'ECMWF Sites space. Default: "{os.environ["USER"]}".',
    )

    parser.add_argument(
        "--site-name",
        default="riverbench",
        help='ECMWF Sites name. Default: "riverbench".',
    )

    parser.add_argument(
        "--remote-html-dir",
        default=".",
        help='Remote directory for HTML files. Default: ".".',
    )

    parser.add_argument(
        "--remote-dashboard-data-dir",
        default="dashboard_data",
        help='Remote directory for dashboard_data. Default: "dashboard_data".',
    )

    parser.add_argument(
        "--if-exists-backup",
        action="store_true",
        help="If set, keep a backup when overwriting remote files.",
    )

    parser.add_argument(
        "--skip-dashboard-data",
        action="store_true",
        help="Upload only HTML files and skip dashboard_data.",
    )

    parser.add_argument(
        "--html-only",
        action="store_true",
        help=(
            "Upload only HTML files. This is an explicit alias for "
            "--skip-dashboard-data."
        ),
    )

    parser.add_argument(
        "--list-before",
        action="store_true",
        help="List remote files before upload.",
    )

    parser.add_argument(
        "--list-after",
        action="store_true",
        help="List remote files after upload.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without uploading.",
    )

    return parser.parse_args()


def get_token():
    token = os.environ.get("ECMWF_SITES_TOKEN")

    if not token:
        raise RuntimeError(
            "Missing ECMWF Sites token.\n"
            "Set it first, for example:\n"
            "  export ECMWF_SITES_TOKEN='...'\n"
        )

    return token


def main():
    args = parse_args()

    workflow_dir = args.workflow_dir.resolve()
    dashboard_data_dir = workflow_dir / args.dashboard_data_dirname
    upload_dashboard_data = not (args.skip_dashboard_data or args.html_only)

    if not workflow_dir.exists():
        raise FileNotFoundError(f"Workflow directory not found: {workflow_dir}")

    html_files = sorted(workflow_dir.glob(args.html_pattern))

    if len(html_files) == 0:
        raise RuntimeError(
            f"No HTML files found in {workflow_dir} with pattern {args.html_pattern!r}"
        )

    print("")
    print("============================================================")
    print("Upload ifs-riverbench dashboard")
    print("============================================================")
    print(f"Workflow directory       : {workflow_dir}")
    print(f"HTML pattern             : {args.html_pattern}")
    print(f"HTML files found         : {len(html_files)}")
    print(f"Dashboard data directory : {dashboard_data_dir}")
    print(f"Site                     : {args.space}/{args.site_name}")
    print(f"Remote HTML directory    : {args.remote_html_dir}")
    print(f"Remote dashboard_data    : {args.remote_dashboard_data_dir}")
    print(f"Upload dashboard_data    : {upload_dashboard_data}")
    print(f"Backup existing files    : {args.if_exists_backup}")
    print(f"Dry run                  : {args.dry_run}")
    print("============================================================")
    print("")

    print("HTML files to upload:")
    for html_file in html_files:
        print(f"  - {html_file.name}")

    if upload_dashboard_data:
        if dashboard_data_dir.exists():
            print(f"\nDashboard data directory will be uploaded: {dashboard_data_dir}")
        else:
            raise FileNotFoundError(
                f"dashboard_data directory not found: {dashboard_data_dir}"
            )
    else:
        print("\nHTML-only mode: dashboard_data upload will be skipped.")

    if args.dry_run:
        print("\nDry run: no upload performed.")
        return

    token = get_token()

    site_authenticator = Authenticator.from_token(token=token)
    site = Site.from_space_and_name(space=args.space, name=args.site_name)
    client = SitesClient(authenticator=site_authenticator)
    site_content_manager = client.content(site=site)

    if args.list_before:
        print("\nRemote listing before upload:")
        print(site_content_manager.list(recursive=False))

    print("\nUploading HTML files...")
    for i, html_file in enumerate(html_files, start=1):
        print(f"[{i}/{len(html_files)}] Uploading {html_file.name} -> {args.remote_html_dir}")
        result = site_content_manager.upload(
            local_path=str(html_file),
            remote_path=args.remote_html_dir,
            recursive=False,
            if_exists_backup=args.if_exists_backup,
        )
        print(result)

    if upload_dashboard_data:
        print(
            f"\nUploading dashboard_data recursively -> "
            f"{args.remote_dashboard_data_dir}"
        )
        result = site_content_manager.upload(
            local_path=str(dashboard_data_dir),
            remote_path=args.remote_dashboard_data_dir,
            recursive=True,
            if_exists_backup=args.if_exists_backup,
        )
        print(result)
    else:
        print("\nSkipping dashboard_data upload.")

    if args.list_after:
        print("\nRemote listing after upload:")
        print(site_content_manager.list(recursive=False))

    print("")
    print("============================================================")
    print("Upload complete")
    print("============================================================")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
