#!/usr/bin/env python3
"""Run BloomCast operational HAB forecasting from a merged daily CSV."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Bootstrap imports when executed as `python src/pipeline/hab_forecast_pipeline.py`.
_here = Path(__file__).resolve()
for parent in [_here.parent] + list(_here.parents):
    if (parent / "src").exists() and (parent / "utils").exists():
        sys.path.insert(0, str(parent))
        sys.path.insert(0, str(parent / "src"))
        break

from bloomcast.hab_forecasting import predict_operational_package

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
LOGGER = logging.getLogger("hab_forecast_pipeline")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to merged HAB input CSV.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where operational HAB forecast JSON will be written.",
    )
    return parser.parse_args(argv)


def _validate_input(input_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"Input path is not a file: {input_path}")


def run(input_path: Path, output_dir: Path) -> Path:
    _validate_input(input_path)
    LOGGER.info("Running HAB operational forecast for %s", input_path)

    package = predict_operational_package(csv_path=input_path)
    prediction_date = package["week_1"]["regression"]["prediction_date"]
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "input_csv": str(input_path),
        "prediction_date": prediction_date,
        "forecast_package": package,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "operational_hab_forecast.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    LOGGER.info("Wrote HAB operational forecast to %s", output_path)
    return output_path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    args = parse_args(argv)
    try:
        run(Path(args.input), Path(args.output_dir))
    except Exception as exc:
        LOGGER.error("HAB operational forecast failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
