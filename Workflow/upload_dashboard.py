#!/usr/bin/env python3
"""
upload_dashboard.py

Upload an ifs-riverbench dashboard bundle (kurosiwo-dashboard,
modis2016-dashboard, or de2026-dashboard) to ECMWF Sites.

This script uploads the chosen dashboard directory recursively.

Authentication
--------------
Set your ECMWF Sites API token as an environment variable before running
(e.g. in ~/.profile):

  export ECMWF_RIVERBENCH_TOKEN="..."

Then run:

  python3 upload_dashboard.py --dashboard-dirname kurosiwo-dashboard

Optional examples:

  python3 upload_dashboard.py --dashboard-dirname modis2016-dashboard --dry-run
  python3 upload_dashboard.py --dashboard-dirname de2026-dashboard --workflow-dir /perm/USER/ifs-riverbench
"""

from pathlib import Path
import argparse
import os
import sys

from sites.sdk import SitesClient
from sites.sdk.sites import Site, Authenticator


DEFAULT_WORKFLOW_DIR = Path(f"/perm/{os.environ['USER']}/ifs-riverbench")
# Actual dashboard directories produced by kurosiwo_dashboard.py /
# modis2016_dashboard.py / de2026_dashboard.py -- note hyphens, not
# underscores. There's no sensible single default now that there are
# three, so --dashboard-dirname is required below instead of defaulting
# to one of them silently.
DASHBOARD_DIRNAME_CHOICES = ["kurosiwo-dashboard", "modis2016-dashboard", "de2026-dashboard"]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload ifs-riverbench dashboard to ECMWF Sites."
    )

    parser.add_argument(
        "--workflow-dir",
        type=Path,
        default=DEFAULT_WORKFLOW_DIR,
        help=(
            "Directory containing dashboard files and data. "
            f"Default: {DEFAULT_WORKFLOW_DIR}"
        ),
    )

    parser.add_argument(
        "--dashboard-dirname",
        required=True,
        choices=DASHBOARD_DIRNAME_CHOICES,
        help="Name of the dashboard directory inside workflow-dir to upload.",
    )

    parser.add_argument(
        "--remote-dashboard-dir",
        default=None,
        help=(
            "Remote path (relative to the site root) to upload into. "
            "Default: same name as --dashboard-dirname, so the site mirrors "
            "the local kurosiwo-dashboard/modis2016-dashboard/de2026-dashboard layout."
        ),
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
        "--if-exists-backup",
        action="store_true",
        help="If set, keep a backup when overwriting remote files.",
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
    token = os.environ.get("ECMWF_RIVERBENCH_TOKEN")

    if not token:
        raise RuntimeError(
            "Missing ECMWF Sites token.\n"
            "Set it first, for example:\n"
            "  export ECMWF_RIVERBENCH_TOKEN='...'\n"
        )

    return token


def main():
    args = parse_args()

    workflow_dir = args.workflow_dir.resolve()
    dashboard_dir = workflow_dir / args.dashboard_dirname
    remote_dashboard_dir = args.remote_dashboard_dir or args.dashboard_dirname

    if not workflow_dir.exists():
        raise FileNotFoundError(f"Workflow directory not found: {workflow_dir}")
    if not dashboard_dir.is_dir():
        raise FileNotFoundError(f"Dashboard directory not found: {dashboard_dir}")

    print("")
    print("============================================================")
    print("Upload ifs-riverbench dashboard")
    print("============================================================")
    print(f"Workflow directory       : {workflow_dir}")
    print(f"Dashboard data directory : {dashboard_dir}")
    print(f"Site                     : {args.space}/{args.site_name}")
    print(f"Remote path              : {remote_dashboard_dir}")
    print(f"Backup existing files    : {args.if_exists_backup}")
    print(f"Dry run                  : {args.dry_run}")
    print("============================================================")
    print("")

    if args.dry_run:
        files = sorted(p for p in dashboard_dir.glob("**/*") if p.is_file())
        print(f"Dry run: {len(files)} file(s) would be uploaded to '{args.space}/{args.site_name}/{remote_dashboard_dir}':")
        for f in files:
            print(f"  {f.relative_to(dashboard_dir)}")
        return

    token = get_token()

    site_authenticator = Authenticator.from_token(token=token)
    site = Site.from_space_and_name(space=args.space, name=args.site_name)
    client = SitesClient(authenticator=site_authenticator)
    site_content_manager = client.content(site=site)

    if args.list_before:
        print("\nRemote listing before upload:")
        print(site_content_manager.list(recursive=False))


    print(
            f"\nUploading dashboard recursively -> "
            f"{remote_dashboard_dir}"
    )
    result = site_content_manager.upload(
            local_path=str(dashboard_dir),
            remote_path=remote_dashboard_dir,
            recursive=True,
            if_exists_backup=args.if_exists_backup,
    )
    print(result)

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
