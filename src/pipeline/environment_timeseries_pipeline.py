#!/usr/bin/env python3
"""Build a compact frontend-ready environmental time series JSON artifact."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# Bootstrap imports when executed as `python src/pipeline/environment_timeseries_pipeline.py`.
_here = Path(__file__).resolve()
for parent in [_here.parent] + list(_here.parents):
    if (parent / "src").exists() and (parent / "utils").exists():
        sys.path.insert(0, str(parent))
        sys.path.insert(0, str(parent / "src"))
        break

from src.pipeline.latest_conditions_pipeline import FIELD_SPECS

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
LOGGER = logging.getLogger("environment_timeseries_pipeline")
DEFAULT_INPUT = Path("src/pipeline/final_data/SJL_daily_df.csv")
DEFAULT_OUTPUT_DIR = Path("results/environment_timeseries")
OUTPUT_FILENAME = "environment_timeseries.json"
WINDOW_DAYS = 60
SERIES_FIELD_ORDER = (
    "chlorophyll_a",
    "air_temperature",
    "precipitation",
    "radiation",
    "tidal_range",
    "water_temperature",
    "wind_speed",
)
FIELD_DECIMALS = {
    "chlorophyll_a": 2,
    "air_temperature": 2,
    "precipitation": 2,
    "radiation": 2,
    "tidal_range": 3,
    "water_temperature": 2,
    "wind_speed": 2,
}


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
        help="YYYY-MM-DD end date for the inclusive 60-day window. Default: latest date in CSV.",
    )
    return parser.parse_args(argv)


def _generated_at_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _validate_input(input_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"Input path is not a file: {input_path}")


def _pick_default_as_of(df: pd.DataFrame) -> date:
    max_dt = df["date"].dropna().max()
    if pd.isna(max_dt):
        raise ValueError("Could not parse any valid dates from 'date' column.")
    return max_dt.date()


def _parse_as_of(as_of_str: str | None, df: pd.DataFrame) -> date:
    if as_of_str is None:
        return _pick_default_as_of(df)
    parsed = pd.to_datetime(as_of_str, errors="coerce")
    if pd.isna(parsed):
        raise ValueError("Invalid --as-of-date. Use YYYY-MM-DD.")
    return parsed.date()


def _load_frame(input_path: Path) -> pd.DataFrame:
    _validate_input(input_path)

    df = pd.read_csv(input_path)
    if "date" not in df.columns:
        raise ValueError("Input CSV must include a 'date' column.")

    missing_columns = [
        FIELD_SPECS[field_name]["source_column"]
        for field_name in SERIES_FIELD_ORDER
        if FIELD_SPECS[field_name]["source_column"] not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Input CSV is missing required columns: {', '.join(sorted(missing_columns))}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.floor("D")
    out = out.dropna(subset=["date"]).sort_values("date")
    return out


def _build_daily_series(df: pd.DataFrame, as_of: date) -> tuple[date, date, list[dict[str, Any]]]:
    end_date = as_of
    start_date = as_of - timedelta(days=WINDOW_DAYS - 1)
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    filtered = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].copy()
    renamed_columns: dict[str, pd.Series] = {}
    for field_name in SERIES_FIELD_ORDER:
        source_column = FIELD_SPECS[field_name]["source_column"]
        renamed_columns[field_name] = pd.to_numeric(filtered[source_column], errors="coerce")

    series_df = pd.DataFrame({"date": filtered["date"], **renamed_columns})
    aggregated = series_df.groupby("date", as_index=False)[list(SERIES_FIELD_ORDER)].mean()
    aggregated = aggregated.sort_values("date")
    aggregated = aggregated.dropna(subset=list(SERIES_FIELD_ORDER), how="all")

    records: list[dict[str, Any]] = []
    for row in aggregated.itertuples(index=False):
        record = {"date": row.date.date().isoformat()}
        for field_name in SERIES_FIELD_ORDER:
            value = getattr(row, field_name)
            record[field_name] = _format_output_value(field_name=field_name, value=value)
        records.append(record)

    return start_date, end_date, records


def _format_output_value(field_name: str, value: Any) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), FIELD_DECIMALS[field_name])


def build_environment_timeseries_payload(
    input_path: Path,
    as_of_date: str | None = None,
    generated_at: str | None = None,
    input_csv_label: str | None = None,
) -> dict[str, Any]:
    df = _load_frame(input_path)
    as_of = _parse_as_of(as_of_date, df)
    start_date, end_date, records = _build_daily_series(df=df, as_of=as_of)

    fields = {
        field_name: {"units": FIELD_SPECS[field_name]["units"]}
        for field_name in SERIES_FIELD_ORDER
    }
    return {
        "generated_at_utc": generated_at or _generated_at_utc(),
        "input_csv": input_csv_label or str(input_path),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "fields": fields,
        "series": records,
    }


def run(
    input_path: Path = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    as_of_date: str | None = None,
) -> Path:
    LOGGER.info("Computing environmental time series from %s", input_path)
    payload = build_environment_timeseries_payload(
        input_path=input_path,
        as_of_date=as_of_date,
        input_csv_label=str(input_path),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_FILENAME
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"), ensure_ascii=True)
        handle.write("\n")

    LOGGER.info("Wrote environmental time series to %s", output_path)
    return output_path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    args = parse_args(argv)
    try:
        run(
            input_path=Path(args.input),
            output_dir=Path(args.output_dir),
            as_of_date=args.as_of_date,
        )
    except Exception as exc:
        LOGGER.error("Environment time series pipeline failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
