#!/usr/bin/env python3
"""Build a compact JSON summary of the last 4 trailing weekly chlorophyll means."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None

# Bootstrap imports when executed as `python src/pipeline/weekly_chlorophyll_summary_pipeline.py`.
_here = Path(__file__).resolve()
for parent in [_here.parent] + list(_here.parents):
    if (parent / "src").exists() and (parent / "utils").exists():
        sys.path.insert(0, str(parent))
        sys.path.insert(0, str(parent / "src"))
        break

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
LOGGER = logging.getLogger("weekly_chlorophyll_summary_pipeline")
DEFAULT_INPUT = Path("src/pipeline/final_data/SJL_daily_df.csv")
DEFAULT_OUTPUT_DIR = Path("results/weekly_chlorophyll_summary")
OUTPUT_FILENAME = "weekly_chlorophyll_summary.json"
DEFAULT_CHL_COL = "CHLL_NN_TOTAL"
DEFAULT_WINDOW_DAYS = 7
DEFAULT_WINDOW_COUNT = 4
DEFAULT_TIMEZONE = "America/Puerto_Rico"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help=f"Path to merged daily CSV (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory where JSON should be written (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--as-of-date",
        default=None,
        help="YYYY-MM-DD end date for week_1 (inclusive). Default: latest date in CSV.",
    )
    parser.add_argument(
        "--chl-col",
        default=DEFAULT_CHL_COL,
        help=f"Chlorophyll column (default: {DEFAULT_CHL_COL}).",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=DEFAULT_WINDOW_DAYS,
        help=f"Calendar days per trailing window (default: {DEFAULT_WINDOW_DAYS}).",
    )
    parser.add_argument(
        "--window-count",
        type=int,
        default=DEFAULT_WINDOW_COUNT,
        help=f"Number of trailing windows to emit (default: {DEFAULT_WINDOW_COUNT}).",
    )
    parser.add_argument(
        "--min-valid",
        type=int,
        default=1,
        help="Minimum number of non-NaN CHL values required in each window (default: 1).",
    )
    parser.add_argument(
        "--tz",
        default=DEFAULT_TIMEZONE,
        help=f"Timezone for run_date (default: {DEFAULT_TIMEZONE}).",
    )
    return parser.parse_args(argv)


def local_today(tz_str: str) -> date:
    if ZoneInfo:
        return datetime.now(ZoneInfo(tz_str)).date()
    return datetime.now().date()


def generated_at_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def pick_default_as_of(df: pd.DataFrame) -> date:
    max_dt = pd.to_datetime(df["date"], errors="coerce").dropna().max()
    if pd.isna(max_dt):
        raise ValueError("Could not parse any valid dates from 'date' column.")
    return max_dt.date()


def parse_as_of(as_of_str: str | None, df: pd.DataFrame) -> date:
    if as_of_str is None:
        return pick_default_as_of(df)
    parsed = pd.to_datetime(as_of_str, errors="coerce")
    if pd.isna(parsed):
        raise ValueError("Invalid --as-of-date. Use YYYY-MM-DD.")
    return parsed.date()


def load_daily_frame(input_path: Path, chl_col: str) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"Input path is not a file: {input_path}")

    df = pd.read_csv(input_path)
    if "date" not in df.columns:
        raise ValueError("Input CSV must include a 'date' column.")
    if chl_col not in df.columns:
        raise ValueError(f"Chlorophyll column '{chl_col}' not found.")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.floor("D")
    out = out.dropna(subset=["date"]).sort_values("date")
    return out


def compute_window_mean(
    df: pd.DataFrame,
    chl_col: str,
    start_date: date,
    end_date: date,
    min_valid: int,
) -> float:
    if min_valid <= 0:
        raise ValueError("--min-valid must be >= 1")
    if start_date > end_date:
        raise ValueError("Window start_date must be <= end_date.")

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    window = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]
    if window.empty:
        raise ValueError(f"No rows found in date window {start_date.isoformat()}..{end_date.isoformat()}")

    values = pd.to_numeric(window[chl_col], errors="coerce")
    valid = int(values.notna().sum())
    if valid < min_valid:
        raise ValueError(
            f"Not enough valid chlorophyll values in window "
            f"{start_date.isoformat()}..{end_date.isoformat()}: have {valid}, require {min_valid}."
        )

    return float(values.mean(skipna=True))


def build_trailing_windows(as_of: date, window_days: int, window_count: int) -> list[tuple[str, date, date]]:
    if window_days <= 0:
        raise ValueError("--window-days must be >= 1")
    if window_count <= 0:
        raise ValueError("--window-count must be >= 1")

    windows: list[tuple[str, date, date]] = []
    for index in range(window_count):
        end_date = as_of - timedelta(days=index * window_days)
        start_date = end_date - timedelta(days=window_days - 1)
        windows.append((f"week_{index + 1}", start_date, end_date))
    return windows


def build_weekly_summary_payload(
    input_path: Path,
    chl_col: str = DEFAULT_CHL_COL,
    as_of_date: str | None = None,
    window_days: int = DEFAULT_WINDOW_DAYS,
    window_count: int = DEFAULT_WINDOW_COUNT,
    min_valid: int = 1,
    run_date: date | None = None,
    generated_at: str | None = None,
    input_csv_label: str | None = None,
) -> dict[str, Any]:
    df = load_daily_frame(input_path=input_path, chl_col=chl_col)
    as_of = parse_as_of(as_of_date, df)
    run_d = run_date or local_today(DEFAULT_TIMEZONE)
    windows = build_trailing_windows(as_of=as_of, window_days=window_days, window_count=window_count)

    weeks = []
    for label, start_date, end_date in windows:
        chlorophyll = compute_window_mean(
            df=df,
            chl_col=chl_col,
            start_date=start_date,
            end_date=end_date,
            min_valid=min_valid,
        )
        weeks.append(
            {
                "label": label,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "chlorophyll": chlorophyll,
            }
        )

    return {
        "generated_at_utc": generated_at or generated_at_utc(),
        "run_date": run_d.isoformat(),
        "as_of_date": as_of.isoformat(),
        "input_csv": input_csv_label or str(input_path),
        "chl_col": chl_col,
        "window_days": window_days,
        "window_count": window_count,
        "weeks": weeks,
    }


def run(
    input_path: Path = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    as_of_date: str | None = None,
    chl_col: str = DEFAULT_CHL_COL,
    window_days: int = DEFAULT_WINDOW_DAYS,
    window_count: int = DEFAULT_WINDOW_COUNT,
    min_valid: int = 1,
    tz: str = DEFAULT_TIMEZONE,
) -> Path:
    LOGGER.info("Computing trailing weekly chlorophyll summary from %s", input_path)
    payload = build_weekly_summary_payload(
        input_path=input_path,
        chl_col=chl_col,
        as_of_date=as_of_date,
        window_days=window_days,
        window_count=window_count,
        min_valid=min_valid,
        run_date=local_today(tz),
        input_csv_label=str(input_path),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_FILENAME
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"), ensure_ascii=True)
        handle.write("\n")

    LOGGER.info("Wrote weekly chlorophyll summary to %s", output_path)
    return output_path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    args = parse_args(argv)
    try:
        run(
            input_path=Path(args.input),
            output_dir=Path(args.output_dir),
            as_of_date=args.as_of_date,
            chl_col=args.chl_col,
            window_days=args.window_days,
            window_count=args.window_count,
            min_valid=args.min_valid,
            tz=args.tz,
        )
    except Exception as exc:
        LOGGER.error("Weekly chlorophyll summary pipeline failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
