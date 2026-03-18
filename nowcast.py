#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
nowcast.py

Create a single-row nowcast CSV (same schema as xgboost outputs) where:
- y_pred = mean(CHLL_NN_TOTAL) over the last N calendar days ending on as_of_date (inclusive)
- label is derived from thresholds (q1/q2)

Output columns:
  run_date, as_of_date, horizon_days, predicted_date, y_pred, label, model_dir

Defaults:
- n_days = 7
- as_of_date = latest date present in SJL_daily_df.csv
- chl_col = CHLL_NN_TOTAL

Example usage:
  # Compute a 5-day nowcast ending on the most recent date in the dataset
  python nowcast.py \
    --input SJL_daily_df.csv \
    --output-dir results \
    --n-days 5
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime, timedelta

import pandas as pd

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None


def local_today(tz_str: str) -> date:
    if ZoneInfo:
        return datetime.now(ZoneInfo(tz_str)).date()
    return datetime.now().date()


def label_from_thresholds(y: float, q1: float, q2: float) -> str:
    if y < q1:
        return "low"
    if y <= q2:
        return "medium"
    return "high"


def pick_default_as_of(df: pd.DataFrame) -> date:
    max_dt = pd.to_datetime(df["date"], errors="coerce").dropna().max()
    if pd.isna(max_dt):
        raise ValueError("Could not parse any valid dates from 'date' column.")
    return max_dt.date()


def parse_as_of(as_of_str: str | None, df: pd.DataFrame) -> date:
    if as_of_str is None:
        return pick_default_as_of(df)
    d = pd.to_datetime(as_of_str, errors="coerce")
    if pd.isna(d):
        raise ValueError("Invalid --as-of-date. Use YYYY-MM-DD.")
    return d.date()


def compute_calendar_mean(
    df: pd.DataFrame,
    chl_col: str,
    as_of: date,
    n_days: int,
    min_valid: int,
) -> float:
    if n_days <= 0:
        raise ValueError("--n-days must be >= 1")
    if min_valid <= 0:
        raise ValueError("--min-valid must be >= 1")

    start = pd.Timestamp(as_of - timedelta(days=n_days - 1))
    end = pd.Timestamp(as_of)

    window = df[(df["date"] >= start) & (df["date"] <= end)]
    if window.empty:
        raise ValueError(f"No rows found in date window {start.date()}..{end.date()}")

    vals = pd.to_numeric(window[chl_col], errors="coerce")
    valid = int(vals.notna().sum())
    if valid < min_valid:
        raise ValueError(
            f"Not enough valid chlorophyll values in last {n_days} days "
            f"({start.date()}..{end.date()}): have {valid}, require {min_valid}."
        )

    return float(vals.mean(skipna=True))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Create nowcast CSV from trailing calendar-day chlorophyll mean.")
    p.add_argument("--input", required=True, help="Path to SJL_daily_df.csv.")
    p.add_argument("--output-dir", default="results", help="Directory to write nowcast_YYYY-MM-DD.csv.")
    p.add_argument("--n-days", type=int, default=7, help="Trailing calendar days (default: 7).")
    p.add_argument("--min-valid", type=int, default=1,
                   help="Minimum number of non-NaN CHL values required in window (default: 1).")
    p.add_argument("--tz", default="America/Puerto_Rico", help="Timezone for run_date (default: America/Puerto_Rico).")
    p.add_argument("--as-of-date", default=None,
                   help="YYYY-MM-DD end date for window (inclusive). Default: latest date in CSV.")
    p.add_argument("--chl-col", default="CHLL_NN_TOTAL", help="Chlorophyll column (default: CHLL_NN_TOTAL).")
    p.add_argument("--q1", type=float, default=10.148629867177084, help="Low/medium threshold.")
    p.add_argument("--q2", type=float, default=15.377913418040292, help="Medium/high threshold.")
    p.add_argument("--model-dir", default=None, help="Optional model_dir string.")
    args = p.parse_args(argv)

    if not os.path.exists(args.input):
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        return 2

    df = pd.read_csv(args.input)

    if "date" not in df.columns:
        print("ERROR: input CSV must include a 'date' column.", file=sys.stderr)
        return 2
    if args.chl_col not in df.columns:
        print(f"ERROR: chlorophyll column '{args.chl_col}' not found.", file=sys.stderr)
        return 2

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.floor("D")
    df = df.dropna(subset=["date"]).sort_values("date")

    run_d = local_today(args.tz)
    as_of = parse_as_of(args.as_of_date, df)

    y = compute_calendar_mean(
        df=df,
        chl_col=args.chl_col,
        as_of=as_of,
        n_days=args.n_days,
        min_valid=args.min_valid,
    )

    label = label_from_thresholds(y, args.q1, args.q2)
    model_dir = args.model_dir or f"nowcast/calendar_{args.n_days}d_mean:{args.chl_col}"

    out_row = {
        "run_date": run_d.isoformat(),
        "as_of_date": as_of.isoformat(),
        "horizon_days": 0,
        "predicted_date": as_of.isoformat(),
        "y_pred": y,
        "label": label,
        "model_dir": model_dir,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"nowcast_{run_d.isoformat()}.csv")
    pd.DataFrame([out_row]).to_csv(out_path, index=False)

    print(f"Wrote: {out_path}")
    print(pd.DataFrame([out_row]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
