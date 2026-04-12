#!/usr/bin/env python3
"""Write the latest independently-available daily conditions from SJL_daily_df."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# Bootstrap imports when executed as `python src/pipeline/latest_conditions_pipeline.py`.
_here = Path(__file__).resolve()
for parent in [_here.parent] + list(_here.parents):
    if (parent / "src").exists() and (parent / "utils").exists():
        sys.path.insert(0, str(parent))
        sys.path.insert(0, str(parent / "src"))
        break

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
LOGGER = logging.getLogger("latest_conditions_pipeline")
OUTPUT_FILENAME = "latest_conditions.json"

FIELD_SPECS = {
    "air_temperature": {"source_column": "air_temperature", "units": "deg C"},
    "water_temperature": {"source_column": "water_temperature", "units": "deg C"},
    "wind_speed": {"source_column": "AWND", "units": "m/s"},
    "radiation": {"source_column": "Watt_per_m2", "units": "W/m^2"},
    "precipitation": {"source_column": "precipitation", "units": "mm/day"},
    "chlorophyll_a": {"source_column": "CHLL_NN_TOTAL", "units": "mg/m^3"},
    "tidal_range": {"source_column": "tidal_range", "units": "m"},
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to merged daily CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory where JSON should be written.")
    return parser.parse_args(argv)


def _generated_at_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _validate_input(input_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"Input path is not a file: {input_path}")


def _load_frame(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    if "date" not in df.columns:
        raise ValueError("Input CSV must include a 'date' column.")

    missing_columns = [
        spec["source_column"] for spec in FIELD_SPECS.values() if spec["source_column"] not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Input CSV is missing required columns: {', '.join(sorted(missing_columns))}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    out = out.dropna(subset=["date"]).sort_values("date")
    return out


def _latest_value_for_column(df: pd.DataFrame, source_column: str, units: str) -> dict[str, Any]:
    numeric = pd.to_numeric(df[source_column], errors="coerce")
    valid_rows = df.loc[numeric.notna(), ["date"]].copy()
    valid_rows[source_column] = numeric.loc[numeric.notna()]
    if valid_rows.empty:
        return {
            "value": None,
            "units": units,
            "source_column": source_column,
            "as_of": None,
            "missing": True,
        }

    latest_row = valid_rows.iloc[-1]
    return {
        "value": float(latest_row[source_column]),
        "units": units,
        "source_column": source_column,
        "as_of": latest_row["date"].isoformat(),
        "missing": False,
    }


def build_latest_conditions_payload(input_path: Path) -> dict[str, Any]:
    df = _load_frame(input_path)
    latest_conditions = {
        field_name: _latest_value_for_column(
            df=df,
            source_column=spec["source_column"],
            units=spec["units"],
        )
        for field_name, spec in FIELD_SPECS.items()
    }
    return {
        "generated_at_utc": _generated_at_utc(),
        "latest_conditions": latest_conditions,
    }


def run(input_path: Path, output_dir: Path) -> Path:
    _validate_input(input_path)
    LOGGER.info("Computing latest daily conditions from %s", input_path)

    payload = build_latest_conditions_payload(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_FILENAME
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    LOGGER.info("Wrote latest conditions to %s", output_path)
    return output_path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    args = parse_args(argv)
    try:
        run(Path(args.input), Path(args.output_dir))
    except Exception as exc:
        LOGGER.error("Latest conditions pipeline failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
