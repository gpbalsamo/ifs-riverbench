#!/usr/bin/env python3
"""
04_upload_dashboard.py

Upload the ifs-riverbench dashboard bundle (built by 03_prepare_sites_bundle.py)
to ECMWF Sites as a single recursive directory upload.

This shells out to the `sitesctl` CLI (module load sites) rather than the
Python `sites.sdk` package: that package name is not the ECMWF-internal SDK on
public PyPI (it resolves to an unrelated third-party package), so `sitesctl`
is the only working upload path in a plain pip/conda environment.

Authentication
--------------
Set your ECMWF Sites API token as an environment variable before running
(e.g. in ~/.profile):

  export ECMWF_RIVERBENCH_TOKEN="..."

Then run (after `module load sites`):

  python3 04_upload_dashboard.py

Optional examples:

  python3 04_upload_dashboard.py --dry-run
  python3 04_upload_dashboard.py --dashboard-dirname site_bundle --remote-dashboard-dir discharge-dashboard
"""

from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys


DEFAULT_WORKFLOW_DIR = Path(f"/perm/{os.environ['USER']}/ifs-riverbench/Workflow")
DEFAULT_DASHBOARD_DIRNAME = "site_bundle"
DEFAULT_REMOTE_DASHBOARD_DIR = "discharge-dashboard"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload ifs-riverbench dashboard bundle to ECMWF Sites via sitesctl."
    )

    parser.add_argument(
        "--workflow-dir",
        type=Path,
        default=DEFAULT_WORKFLOW_DIR,
        help=(
            "Directory containing the dashboard bundle directory. "
            f"Default: {DEFAULT_WORKFLOW_DIR}"
        ),
    )

    parser.add_argument(
        "--dashboard-dirname",
        default=DEFAULT_DASHBOARD_DIRNAME,
        help=f'Name of the bundle directory inside workflow-dir. Default: "{DEFAULT_DASHBOARD_DIRNAME}".',
    )

    parser.add_argument(
        "--remote-dashboard-dir",
        default=DEFAULT_REMOTE_DASHBOARD_DIR,
        help=(
            "Remote path (relative to the site root) to upload into, matching "
            "the sibling naming convention used by the other dashboard "
            f'collections on the site (e.g. "kurosiwo-dashboard"). Default: "{DEFAULT_REMOTE_DASHBOARD_DIR}".'
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
        help="List remote top-level files before upload.",
    )

    parser.add_argument(
        "--list-after",
        action="store_true",
        help="List remote top-level files after upload.",
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


def run_sitesctl(args, token, extra_args):
    if shutil.which("sitesctl") is None:
        raise RuntimeError(
            "sitesctl not found on PATH.\n"
            "Load the ECMWF sites module first, for example:\n"
            "  module load sites\n"
        )

    env = dict(os.environ)
    env["API_AUTHENTICATION_TOKEN"] = token

    cmd = [
        "sitesctl", "site",
        "--space", args.space,
        "--name", args.site_name,
        "--v2",
        *extra_args,
        "--force",
    ]
    subprocess.run(cmd, env=env, check=True)


def main():
    args = parse_args()

    workflow_dir = args.workflow_dir.resolve()
    dashboard_dir = workflow_dir / args.dashboard_dirname
    remote_dashboard_dir = args.remote_dashboard_dir

    if not workflow_dir.exists():
        raise FileNotFoundError(f"Workflow directory not found: {workflow_dir}")
    if not dashboard_dir.is_dir():
        raise FileNotFoundError(f"Dashboard directory not found: {dashboard_dir}")

    print("")
    print("============================================================")
    print("Upload ifs-riverbench dashboard")
    print("============================================================")
    print(f"Workflow directory        : {workflow_dir}")
    print(f"Dashboard bundle directory: {dashboard_dir}")
    print(f"Site                      : {args.space}/{args.site_name}")
    print(f"Remote path               : {remote_dashboard_dir}")
    print(f"Backup existing files     : {args.if_exists_backup}")
    print(f"Dry run                   : {args.dry_run}")
    print("============================================================")
    print("")

    if args.dry_run:
        files = sorted(p for p in dashboard_dir.glob("**/*") if p.is_file())
        print(f"Dry run: {len(files)} file(s) would be uploaded to '{args.space}/{args.site_name}/{remote_dashboard_dir}':")
        for f in files:
            print(f"  {f.relative_to(dashboard_dir)}")
        return

    token = get_token()

    if args.list_before:
        print("\nRemote listing before upload:")
        run_sitesctl(args, token, ["content", "list", "--output", "table"])

    print(f"\nUploading dashboard recursively -> {remote_dashboard_dir}")
    upload_args = [
        "content", "upload",
        "--source", str(dashboard_dir),
        "--destination", remote_dashboard_dir,
        "--recursive",
    ]
    if args.if_exists_backup:
        upload_args.append("--if-exists-backup")
    run_sitesctl(args, token, upload_args)

    if args.list_after:
        print("\nRemote listing after upload:")
        run_sitesctl(args, token, ["content", "list", "--output", "table"])

    print("")
    print("============================================================")
    print("Upload complete")
    print("============================================================")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: sitesctl exited with code {exc.returncode}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
