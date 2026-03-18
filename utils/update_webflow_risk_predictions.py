#!/usr/bin/env python3
"""
Update Webflow CMS risk prediction items from daily ML CSV outputs.

Reads three single-row CSVs:
  - 0-day horizon nowcast (today)
  - 7-day horizon (next week)
  - 15-day horizon (next two weeks)

Uses the CSV columns to populate:
  - fieldData["risk-label"] (human-readable: "Low risk", etc.)
  - fieldData["risk-level"] (Webflow option/reference ID)
  - fieldData["risk-score"] (numeric score from y_pred)
  - fieldData["last-updated"] (ISO datetime at midnight UTC)

Then publishes the updated items for the specified CMS locale.

Expected CSV columns:
  run_date, as_of_date, horizon_days, predicted_date, y_pred, label, model_dir
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from typing import Tuple, Dict, Any, List

import pandas as pd
import requests


# --------- YOUR CONFIG ---------
COLLECTION_ID = "693b53d9a199004dc462bf0e"
CMS_LOCALE_ID = "67c0de7c982e3ec7d13e3d3d"

ITEM_IDS = {
    "today": "693b77ee9d52c301fb732ecb",
    "next_week": "693b77aab073d4e13f7c9b97",
    "next_two_weeks": "693b5443905928a35ddd13fe",
}

RISK_LEVEL_VALUE_IDS = {
    "low": "78ff34acb3eb53149df1b0e27c11ab3a",
    "medium": "eb59d4723ff74b9ea2b48c4d6b125291",
    "high": "81bea1ab83103bd8a5e1bde5512a7872",
}

FIELD_RISK_LABEL = "risk-label"
FIELD_RISK_LEVEL = "risk-level"
FIELD_LAST_UPDATED = "last-updated"
FIELD_RISK_SCORE = "risk-score"  # confirm matches your Webflow field slug


def iso_midnight_z(date_str: str) -> str:
    d = dt.date.fromisoformat(date_str)
    return d.strftime("%Y-%m-%dT00:00:00.000Z")


def normalize_label(label: str) -> str:
    s = str(label).strip().lower()
    if s == "med":
        s = "medium"
    if s not in {"low", "medium", "high"}:
        raise ValueError(f"Unexpected label '{label}'. Expected low/medium/high.")
    return s


def display_risk_label(norm: str) -> str:
    return f"{norm.capitalize()} risk"


def read_single_prediction(csv_path: str) -> Tuple[str, str, float]:
    """
    Returns (run_date_yyyy_mm_dd, normalized_label, risk_score) from a single-row CSV.
    risk_score comes from y_pred.
    """
    df = pd.read_csv(csv_path)
    if df.shape[0] != 1:
        raise ValueError(f"{csv_path}: expected exactly 1 row, got {df.shape[0]}")

    required = {"run_date", "label", "y_pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing required columns: {sorted(missing)}")

    run_date = str(df.loc[0, "run_date"])[:10]
    label = normalize_label(df.loc[0, "label"])

    try:
        risk_score = float(df.loc[0, "y_pred"])
    except Exception as e:
        raise ValueError(f"{csv_path}: y_pred must be numeric, got {df.loc[0, 'y_pred']!r}") from e

    return run_date, label, risk_score


def _headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "accept": "application/json",
    }


def webflow_update_items_bulk(
    token: str,
    collection_id: str,
    items_payload: List[Dict[str, Any]],
    timeout: int = 30,
) -> Dict[str, Any]:
    url = f"https://api.webflow.com/v2/collections/{collection_id}/items"
    resp = requests.patch(url, headers=_headers(token), json={"items": items_payload}, timeout=timeout)
    if not resp.ok:
        raise RuntimeError(f"PATCH failed {resp.status_code}: {resp.text}")
    return resp.json()


def webflow_publish_items(
    token: str,
    collection_id: str,
    item_ids: List[str],
    cms_locale_ids: List[str],
    timeout: int = 30,
) -> Dict[str, Any]:
    url = f"https://api.webflow.com/v2/collections/{collection_id}/items/publish"
    payload = {"itemIds": item_ids, "cmsLocaleIds": cms_locale_ids}
    resp = requests.post(url, headers=_headers(token), json=payload, timeout=timeout)
    if not resp.ok:
        raise RuntimeError(f"PUBLISH failed {resp.status_code}: {resp.text}")
    return resp.json()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv0", required=True, help="Path to nowcast CSV (today).")
    parser.add_argument("--csv7", required=True, help="Path to 7-day CSV (next week).")
    parser.add_argument("--csv15", required=True, help="Path to 15-day CSV (next two weeks).")
    parser.add_argument("--token", default=os.getenv("WEBFLOW_TOKEN"), help="Webflow API token (or env WEBFLOW_TOKEN).")
    parser.add_argument("--publish", action="store_true", help="Publish updated items after patch.")
    parser.add_argument("--dry-run", action="store_true", help="Print payloads but do not call Webflow.")
    args = parser.parse_args()

    if not args.token:
        print("ERROR: Missing Webflow token. Pass --token or set WEBFLOW_TOKEN.", file=sys.stderr)
        return 2

    run_date_0, label_0, score_0 = read_single_prediction(args.csv0)
    run_date_7, label_7, score_7 = read_single_prediction(args.csv7)
    run_date_15, label_15, score_15 = read_single_prediction(args.csv15)

    # Keep "last-updated" consistent across cards: newest run_date among inputs.
    last_updated_date = max(run_date_0, run_date_7, run_date_15)
    last_updated_iso = iso_midnight_z(last_updated_date)

    print("Planned updates:")
    print(f"  today: label={label_0} score={score_0:.4f} -> {display_risk_label(label_0)}")
    print(f"  next_week: label={label_7} score={score_7:.4f} -> {display_risk_label(label_7)}")
    print(f"  next_two_weeks: label={label_15} score={score_15:.4f} -> {display_risk_label(label_15)}")
    print(f"  last-updated: {last_updated_iso}")
    print()

    items_payload = [
        {
            "id": ITEM_IDS["today"],
            "cmsLocaleId": CMS_LOCALE_ID,
            "fieldData": {
                FIELD_RISK_LABEL: display_risk_label(label_0),
                FIELD_RISK_LEVEL: RISK_LEVEL_VALUE_IDS[label_0],
                FIELD_RISK_SCORE: score_0,
                FIELD_LAST_UPDATED: last_updated_iso,
            },
        },
        {
            "id": ITEM_IDS["next_week"],
            "cmsLocaleId": CMS_LOCALE_ID,
            "fieldData": {
                FIELD_RISK_LABEL: display_risk_label(label_7),
                FIELD_RISK_LEVEL: RISK_LEVEL_VALUE_IDS[label_7],
                FIELD_RISK_SCORE: score_7,
                FIELD_LAST_UPDATED: last_updated_iso,
            },
        },
        {
            "id": ITEM_IDS["next_two_weeks"],
            "cmsLocaleId": CMS_LOCALE_ID,
            "fieldData": {
                FIELD_RISK_LABEL: display_risk_label(label_15),
                FIELD_RISK_LEVEL: RISK_LEVEL_VALUE_IDS[label_15],
                FIELD_RISK_SCORE: score_15,
                FIELD_LAST_UPDATED: last_updated_iso,
            },
        },
    ]

    if args.dry_run:
        print("DRY RUN payload:")
        print({"items": items_payload})
        return 0

    patch_url = f"https://api.webflow.com/v2/collections/{COLLECTION_ID}/items"
    print("PATCH URL:", patch_url)
    webflow_update_items_bulk(args.token, COLLECTION_ID, items_payload)
    print("PATCH OK")

    if args.publish:
        publish_url = f"https://api.webflow.com/v2/collections/{COLLECTION_ID}/items/publish"
        print("PUBLISH URL:", publish_url)
        webflow_publish_items(
            args.token,
            COLLECTION_ID,
            [ITEM_IDS["today"], ITEM_IDS["next_week"], ITEM_IDS["next_two_weeks"]],
            [CMS_LOCALE_ID],
        )
        print("PUBLISH OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
