#!/usr/bin/env python3
"""
Clean up old experiments from ECMWF Sites to free storage quota.
Deletes old experiment directories (j6fu, j6gq, etc.) that are no longer needed.
"""

import os
import sys
from sites.sdk import SitesClient
from sites.sdk.sites import Site, Authenticator


def get_token():
    token = os.environ.get("ECMWF_SITES_TOKEN")
    if not token:
        raise RuntimeError(
            "Missing ECMWF Sites token.\n"
            "Set it first: export ECMWF_SITES_TOKEN='...'\n"
        )
    return token


def cleanup_old_experiments(
    space=os.environ.get("ECMWF_SITES_SPACE", "ecm7072"),
    site_name=os.environ.get("ECMWF_SITES_NAME", "riverbench_AIFL"),
):
    """Delete old experiment directories to free storage."""

    token = get_token()
    site_authenticator = Authenticator.from_token(token=token)
    site = Site.from_space_and_name(space=space, name=site_name)
    client = SitesClient(authenticator=site_authenticator)
    cm = client.content(site=site)

    # Old experiments to delete
    old_exps = ['j6fu', 'j6gq', 'iyp3', 'iwya', 'glofas_v4', 'j7xs']

    print(f"Cleaning up old experiments from {space}/{site_name}/dashboard_data/")
    print("=" * 60)

    for exp in old_exps:
        remote_path = f"dashboard_data/{exp}"
        try:
            print(f"Deleting {exp}...", end=" ")
            cm.delete(remote_path, recursive=True)
            print("✓ Done")
        except Exception as e:
            print(f"× Error: {e}")

    print("=" * 60)
    print("Cleanup complete. Run: python3 03_upload_dashboard.py")


if __name__ == "__main__":
    try:
        cleanup_old_experiments()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
