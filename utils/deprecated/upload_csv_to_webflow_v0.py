#!/usr/bin/env python3
"""
Upload BloomCast prediction CSVs (xgboost_7d / xgboost_15d) and a JSON time series
to Webflow Assets, optionally updating a CMS item with the hosted URLs.

This version:
- Still uploads xgboost CSVs
- Replaces the "gold slice CSV" upload with a JSON file containing only:
    CHLL_NN_TOTAL -> chlorophyll_a (mg/m^3)
    water_temperature (deg C)
    air_temperature (deg C)
    Watt_per_m2 -> radiation (Watt/m^2)
    precipitation (mm/day)
- Uses JSON structure:
  {
    "dates": [...],
    "series": {...},
    "units": {...},
    "updated_utc": "YYYY-MM-DDT00:00:00Z"
  }

Example:
  WEBFLOW_TOKEN=... python utils/upload_csv_to_webflow.py \\
    --date 2025-12-14 \\
    --base-dir /Users/cronjobs/src/BloomCast/results/preds \\
    --gold-csv /Users/cronjobs/src/BloomCast/src/pipeline/final_data/SJL_daily_df.csv \\
    --gold-days 60 \\
    --update-cms --cms-item-id <ITEM_ID>
"""

import os
import argparse
import hashlib
import requests
import sys
from datetime import datetime, timedelta, timezone
from pprint import pformat
import json
import tempfile

import pandas as pd

WEBFLOW_TOKEN = os.environ["WEBFLOW_TOKEN"]  # set this in your shell/cron env
SITE_ID = "67c0d993823cd09b06556c94"        # paste from Webflow


# -----------------------
# CLI
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Upload BloomCast assets to Webflow")
    parser.add_argument(
        "--date",
        required=True,
        help="Date stamp in filenames (YYYY-MM-DD), e.g., 2025-12-14. Used to locate source files.",
    )
    parser.add_argument(
        "--base-dir",
        default="/Users/cronjobs/src/BloomCast/results/preds",
        help="Directory containing the prediction CSVs",
    )
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Where to write a manifest JSON with uploaded URLs. Defaults to <base-dir>/latest_preds_manifest.json",
    )
    parser.add_argument(
        "--update-cms",
        action="store_true",
        help="Update Webflow CMS item with latest URLs",
    )
    parser.add_argument(
        "--cms-collection-id",
        default="69255944780d83b54f67a663",
        help="Webflow CMS collection ID to update (Pred Links collection ID)",
    )
    parser.add_argument(
        "--cms-item-id",
        default=None,
        help="Webflow CMS item ID to update (required when --update-cms)",
    )
    parser.add_argument(
        "--cms-name",
        default="pred-links",
        help="Name to send for the CMS item (required field in collection)",
    )
    parser.add_argument(
        "--cms-slug",
        default="pred-links",
        help="Slug to send for the CMS item (required field in collection)",
    )

    # Input “gold” CSV used to build the env JSON
    parser.add_argument(
        "--gold-csv",
        default="/Users/cronjobs/src/BloomCast/src/pipeline/final_data/SJL_daily_df.csv",
        help="Path to merged CSV used to build the env JSON",
    )
    parser.add_argument(
        "--gold-days",
        type=int,
        default=60,
        help="Number of trailing days to include (default: 60)",
    )
    parser.add_argument(
        "--env-dest-name",
        default=None,
        help="Override filename for uploaded env JSON (default: env_timeseries_last<days>d-<date>.json)",
    )
    parser.add_argument(
        "--env-manifest-key",
        default="env_timeseries",
        help="Manifest key to store env JSON URL under (default: env_timeseries)",
    )
    parser.add_argument(
    "--env-cms-field-slug",
    default="env-timeseries-url",
    help="CMS field slug to write env JSON URL into (default: env-timeseries-url).",
    )


    parser.add_argument(
        "--skip-env",
        action="store_true",
        help="Disable building/uploading the env JSON",
    )

    return parser.parse_args()


# -----------------------
# Webflow Asset Upload Helpers
# -----------------------
def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def create_asset_metadata(file_path: str, override_name: str | None = None) -> dict:
    file_name = override_name or os.path.basename(file_path)
    file_hash = md5_file(file_path)

    url = f"https://api.webflow.com/v2/sites/{SITE_ID}/assets"
    headers = {
        "Authorization": f"Bearer {WEBFLOW_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "fileName": file_name,
        "fileHash": file_hash,
        # "parentFolder": "folderIdHere"  # optional
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()


def upload_to_s3(upload_details: dict, file_path: str) -> None:
    upload_url = upload_details.get("uploadUrl") or upload_details.get("upload_url")
    details = upload_details.get("uploadDetails") or upload_details.get("upload_details") or {}
    fields = details.get("fields") if isinstance(details.get("fields"), dict) else details
    extra_headers = details.get("headers") if isinstance(details, dict) else {}

    if not upload_url or not fields:
        raise RuntimeError(
            "Unexpected upload_details shape. Got keys: "
            f"{list(upload_details.keys())}. Full payload:\n{pformat(upload_details)}"
        )

    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f)}
        resp = requests.post(upload_url, data=fields, files=files, headers=extra_headers, timeout=120)
        resp.raise_for_status()


# -----------------------
# Env JSON Builder
# -----------------------
ENV_COLUMN_MAP = {
    "chlorophyll_a": "CHLL_NN_TOTAL",
    "water_temperature": "water_temperature",
    "air_temperature": "air_temperature",
    "radiation": "Watt_per_m2",
    "precipitation": "precipitation",
}

ENV_UNITS = {
    "chlorophyll_a": "mg/m^3",
    "water_temperature": "deg C",
    "air_temperature": "deg C",
    "radiation": "Watt/m^2",
    "precipitation": "mm/day",
}


def build_env_json_window(src_path: str, days: int) -> str | None:
    """
    Read the merged CSV, trim to last `days` worth of data based on max(date),
    and write to a temporary JSON file with the expected structure.
    Returns temp JSON path or None.
    """
    if not os.path.exists(src_path):
        print(f"Env source CSV not found: {src_path}", file=sys.stderr)
        return None

    try:
        df = pd.read_csv(src_path)
    except Exception as e:
        print(f"Could not read env source CSV {src_path}: {e}", file=sys.stderr)
        return None

    if df.empty or "date" not in df.columns:
        print(f"Env source CSV missing data or 'date' column: {src_path}", file=sys.stderr)
        return None

    # Validate required columns
    missing_cols = [col for col in ENV_COLUMN_MAP.values() if col not in df.columns]
    if missing_cols:
        print(f"Env source CSV missing required columns: {missing_cols}", file=sys.stderr)
        return None

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"])
    if df.empty:
        print(f"Env source CSV has no valid dates: {src_path}", file=sys.stderr)
        return None

    end_ts = df["date"].max()
    start_ts = end_ts - timedelta(days=days - 1)

    window = df.loc[(df["date"] >= start_ts) & (df["date"] <= end_ts)].sort_values("date")
    if window.empty:
        print(f"No env rows in last {days} days (start {start_ts.date()}, end {end_ts.date()})", file=sys.stderr)
        return None

    dates = [d.strftime("%Y-%m-%d") for d in window["date"].tolist()]

    series = {}
    for out_key, src_col in ENV_COLUMN_MAP.items():
        # Convert to native Python floats/None for JSON
        vals = pd.to_numeric(window[src_col], errors="coerce").tolist()
        series[out_key] = [None if (v is None or (isinstance(v, float) and pd.isna(v))) else float(v) for v in vals]

    updated_utc = dt_midnight_utc(end_ts)

    payload = {
        "dates": dates,
        "series": series,
        "units": ENV_UNITS,
        "updated_utc": updated_utc,
    }

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json", prefix="env_timeseries_")
    with open(tmp.name, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.close()
    return tmp.name


def dt_midnight_utc(ts: pd.Timestamp) -> str:
    # Convert max date to midnight UTC string
    # end_ts is normalized date (no time), so midnight is implicit.
    d = ts.date()
    return f"{d.isoformat()}T00:00:00Z"


# -----------------------
# Main
# -----------------------
def main():
    args = parse_args()

    try:
        datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        sys.exit("Invalid --date format. Use YYYY-MM-DD (e.g., 2025-12-14).")

    suffix = f"_{args.date}"

    # Prediction CSVs (unchanged)
    files_to_upload: list[tuple[str, str, str]] = [
        (
            os.path.join(args.base_dir, f"xgboost_7d{suffix}.csv"),
            f"xgboost_7d-{args.date}.csv",
            "xgboost_7d",
        ),
        (
            os.path.join(args.base_dir, f"xgboost_15d{suffix}.csv"),
            f"xgboost_15d-{args.date}.csv",
            "xgboost_15d",
        ),
    ]

    env_tmp = None
    if not args.skip_env:
        env_tmp = build_env_json_window(args.gold_csv, args.gold_days)
        if env_tmp:
            env_dest = args.env_dest_name or f"env_timeseries_last{args.gold_days}d-{args.date}.json"
            files_to_upload.append((env_tmp, env_dest, args.env_manifest_key))

    manifest: dict[str, str] = {}
    manifest_path = args.manifest_path or os.path.join(args.base_dir, "latest_preds_manifest.json")

    for src_path, dest_name, manifest_key in files_to_upload:
        if not os.path.exists(src_path):
            print(f"Skipping missing file: {src_path}", file=sys.stderr)
            continue

        meta = create_asset_metadata(src_path, override_name=dest_name)
        print("Upload metadata received:\n", pformat(meta))

        upload_to_s3(meta, src_path)

        asset = meta.get("asset")
        hosted_url = meta.get("hostedUrl") or meta.get("assetUrl") or (asset and asset.get("url"))
        if hosted_url:
            manifest[manifest_key] = hosted_url

        if asset and "url" in asset:
            print(f"Uploaded URL for {dest_name}:", asset["url"])
        else:
            print(f"Upload done for {dest_name}, check Assets panel for the file.")

    # Cleanup temp env JSON
    if env_tmp and os.path.exists(env_tmp):
        try:
            os.remove(env_tmp)
        except Exception as e:
            print(f"Could not remove temp env JSON {env_tmp}: {e}", file=sys.stderr)

    # Write manifest
    if manifest:
        try:
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            print(f"Wrote manifest with uploaded URLs to {manifest_path}")
            print(json.dumps(manifest, indent=2))
        except Exception as e:
            print(f"Could not write manifest to {manifest_path}: {e}", file=sys.stderr)

    # Optional CMS update
    if args.update_cms:
        if not args.cms_item_id:
            sys.exit("--cms-item-id is required when --update-cms is set")
        if not manifest:
            sys.exit("No URLs uploaded; cannot update CMS")

        cms_fielddata = {
            "name": args.cms_name,
            "slug": args.cms_slug,
            "xgboost-1wk-url": manifest.get("xgboost_7d", ""),
            "xgboost-2wk-url": manifest.get("xgboost_15d", ""),
        }

        # If env JSON uploaded, include it too (only if the field exists in CMS)
        if args.env_manifest_key in manifest:
            cms_fielddata[args.env_cms_field_slug] = manifest.get(args.env_manifest_key, "")

        cms_payload = {"fieldData": cms_fielddata}

        cms_url = f"https://api.webflow.com/v2/collections/{args.cms_collection_id}/items/{args.cms_item_id}"
        headers = {
            "Authorization": f"Bearer {WEBFLOW_TOKEN}",
            "Content-Type": "application/json",
        }
        resp = requests.patch(cms_url, json=cms_payload, headers=headers, timeout=60)
        if not resp.ok:
            print(f"CMS update failed: {resp.status_code} {resp.text}", file=sys.stderr)
            resp.raise_for_status()
        else:
            print(f"Updated CMS item {args.cms_item_id} with latest URLs")

if __name__ == "__main__":
    main()