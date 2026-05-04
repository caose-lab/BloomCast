#!/usr/bin/env python3
"""Build a weekly gridded chlorophyll JSON product from recent scene CSVs."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Bootstrap imports when executed as `python src/pipeline/weekly_chlorophyll_map_pipeline.py`.
_here = Path(__file__).resolve()
for parent in [_here.parent] + list(_here.parents):
    if (parent / "src").exists() and (parent / "utils").exists():
        sys.path.insert(0, str(parent))
        sys.path.insert(0, str(parent / "src"))
        break

from src.pipeline.chl_daily_coverage import _delogify_chl, _filter_df_by_polygon, _load_polygon_from_kml

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
LOGGER = logging.getLogger("weekly_chlorophyll_map_pipeline")
OUTPUT_FILENAME = "weekly_chlorophyll_map.json"
DEFAULT_SOURCE_DIR = Path("src/pipeline/data/chl_data")
DEFAULT_POLYGON_FILE = Path("src/pipeline/polygon/LSJmasking.kml")
DEFAULT_GRID_SIZE_DEG = 0.0001
TIMESTAMP_PATTERN = re.compile(r"^\d{8}T\d{6}$")
REQUIRED_COLUMNS = ("latitude", "longitude", "CHL_NN")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        default=str(DEFAULT_SOURCE_DIR),
        help="Directory containing timestamped chlorophyll scene CSVs.",
    )
    parser.add_argument(
        "--polygon",
        default=str(DEFAULT_POLYGON_FILE),
        help="KML polygon used to mask lagoon points.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the weekly chlorophyll JSON should be written.",
    )
    parser.add_argument(
        "--grid-size-deg",
        type=float,
        default=DEFAULT_GRID_SIZE_DEG,
        help="Uniform grid cell size in degrees (default: %(default)s).",
    )
    return parser.parse_args(argv)


def _generated_at_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_scene_timestamp(path: Path) -> datetime | None:
    stem = path.stem
    if not TIMESTAMP_PATTERN.fullmatch(stem):
        return None
    return datetime.strptime(stem, "%Y%m%dT%H%M%S")


def list_scene_files(source_dir: Path) -> list[Path]:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    if not source_dir.is_dir():
        raise ValueError(f"Source path is not a directory: {source_dir}")

    files = [path for path in source_dir.glob("*.csv") if parse_scene_timestamp(path) is not None]
    if not files:
        raise ValueError(f"No timestamped chlorophyll CSV files found in {source_dir}")
    return sorted(files)


def select_latest_7_calendar_days(files: list[Path]) -> tuple[list[Path], date, date, list[str]]:
    if not files:
        raise ValueError("At least one scene file is required.")

    stamped_files = [(path, parse_scene_timestamp(path)) for path in files]
    stamped_files = [(path, ts) for path, ts in stamped_files if ts is not None]
    if not stamped_files:
        raise ValueError("No timestamped scene files were provided.")

    end_date = max(ts.date() for _, ts in stamped_files)
    start_date = end_date - timedelta(days=6)
    included_dates = [(start_date + timedelta(days=offset)).isoformat() for offset in range(7)]
    selected = [
        path
        for path, ts in stamped_files
        if ts is not None and start_date <= ts.date() <= end_date
    ]
    return sorted(selected), start_date, end_date, included_dates


def _load_scene_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=list(REQUIRED_COLUMNS))
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Scene CSV {path} is missing required columns: {', '.join(missing)}")

    frame = frame.copy()
    frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
    frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
    frame["CHL_NN"] = pd.to_numeric(frame["CHL_NN"], errors="coerce")
    frame = frame.dropna(subset=["latitude", "longitude"])
    frame = _delogify_chl(frame, cols=("CHL_NN",))
    return frame.rename(columns={"CHL_NN": "chlorophyll"})


def load_weekly_masked_observations(
    source_dir: Path,
    polygon_path: Path,
) -> tuple[pd.DataFrame, date, date, list[str], int]:
    files = list_scene_files(source_dir)
    selected_files, start_date, end_date, included_dates = select_latest_7_calendar_days(files)
    polygon = _load_polygon_from_kml(str(polygon_path))

    frames: list[pd.DataFrame] = []
    for path in selected_files:
        scene = _load_scene_frame(path)
        masked = _filter_df_by_polygon(scene, polygon)
        if masked.empty:
            continue
        frames.append(masked[["latitude", "longitude", "chlorophyll"]])

    if frames:
        observations = pd.concat(frames, ignore_index=True)
    else:
        observations = pd.DataFrame(columns=["latitude", "longitude", "chlorophyll"])

    return observations, start_date, end_date, included_dates, len(selected_files)


def aggregate_uniform_grid(
    observations: pd.DataFrame,
    polygon_path: Path,
    cell_size_deg: float = DEFAULT_GRID_SIZE_DEG,
) -> pd.DataFrame:
    if cell_size_deg <= 0:
        raise ValueError("Grid cell size must be positive.")

    if observations.empty:
        return pd.DataFrame(columns=["lat", "lon", "chlorophyll"])

    polygon = _load_polygon_from_kml(str(polygon_path))
    min_lon, min_lat, _, _ = polygon.bounds

    frame = observations.copy()
    frame["chlorophyll"] = pd.to_numeric(frame["chlorophyll"], errors="coerce")
    frame = frame.dropna(subset=["chlorophyll"])
    if frame.empty:
        return pd.DataFrame(columns=["lat", "lon", "chlorophyll"])

    frame["lon_index"] = np.floor((frame["longitude"] - min_lon) / cell_size_deg).astype(int)
    frame["lat_index"] = np.floor((frame["latitude"] - min_lat) / cell_size_deg).astype(int)

    grouped = (
        frame.groupby(["lat_index", "lon_index"], as_index=False)["chlorophyll"]
        .mean()
        .sort_values(["lat_index", "lon_index"])
    )
    grouped["lat"] = min_lat + (grouped["lat_index"] + 0.5) * cell_size_deg
    grouped["lon"] = min_lon + (grouped["lon_index"] + 0.5) * cell_size_deg

    result = grouped[["lat", "lon", "chlorophyll"]].copy()
    result["lat"] = result["lat"].astype(float)
    result["lon"] = result["lon"].astype(float)
    result["chlorophyll"] = result["chlorophyll"].astype(float)
    return result


def build_weekly_chlorophyll_payload(
    source_dir: Path,
    polygon_path: Path,
    cell_size_deg: float = DEFAULT_GRID_SIZE_DEG,
) -> dict[str, Any]:
    observations, start_date, end_date, included_dates, source_file_count = load_weekly_masked_observations(
        source_dir=source_dir,
        polygon_path=polygon_path,
    )
    grid = aggregate_uniform_grid(
        observations=observations,
        polygon_path=polygon_path,
        cell_size_deg=cell_size_deg,
    )
    cells = [
        {
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "chlorophyll": float(row["chlorophyll"]),
        }
        for _, row in grid.iterrows()
    ]
    return {
        "generated_at_utc": _generated_at_utc(),
        "source_directory": str(source_dir.resolve()),
        "polygon_file": str(polygon_path.resolve()),
        "start_timestamp": f"{start_date.isoformat()}T00:00:00",
        "end_timestamp": f"{end_date.isoformat()}T23:59:59",
        "included_dates": included_dates,
        "source_file_count": source_file_count,
        "grid_resolution_degrees": cell_size_deg,
        "cell_count": len(cells),
        "cells": cells,
    }


def run(source_dir: Path, polygon_path: Path, output_dir: Path, cell_size_deg: float = DEFAULT_GRID_SIZE_DEG) -> Path:
    payload = build_weekly_chlorophyll_payload(
        source_dir=source_dir,
        polygon_path=polygon_path,
        cell_size_deg=cell_size_deg,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_FILENAME
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    LOGGER.info("Wrote weekly chlorophyll map to %s", output_path)
    return output_path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    args = parse_args(argv)
    try:
        run(
            source_dir=Path(args.source_dir),
            polygon_path=Path(args.polygon),
            output_dir=Path(args.output_dir),
            cell_size_deg=args.grid_size_deg,
        )
    except Exception as exc:
        LOGGER.error("Weekly chlorophyll map pipeline failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
