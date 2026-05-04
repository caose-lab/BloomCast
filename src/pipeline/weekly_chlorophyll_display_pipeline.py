#!/usr/bin/env python3
"""Build a smoothed display-oriented weekly chlorophyll lagoon surface JSON."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter
from shapely import contains_xy

# Bootstrap imports when executed as `python src/pipeline/weekly_chlorophyll_display_pipeline.py`.
_here = Path(__file__).resolve()
for parent in [_here.parent] + list(_here.parents):
    if (parent / "src").exists() and (parent / "utils").exists():
        sys.path.insert(0, str(parent))
        sys.path.insert(0, str(parent / "src"))
        break

from src.pipeline.chl_daily_coverage import _load_polygon_from_kml
from src.pipeline.weekly_chlorophyll_map_pipeline import DEFAULT_POLYGON_FILE, DEFAULT_SOURCE_DIR, load_weekly_masked_observations

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
LOGGER = logging.getLogger("weekly_chlorophyll_display_pipeline")
OUTPUT_FILENAME = "weekly_chlorophyll_display_surface.json"
DEFAULT_DISPLAY_GRID_SIZE_DEG = 0.00015
INTERPOLATION_METHOD = "idw"
SMOOTHING_METHOD = "gaussian_filter"
TARGET_SMOOTHING_METERS = 300.0
LAT_LON_PRECISION = 5
CHLOROPHYLL_PRECISION = 3


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
        help="KML polygon used to clip lagoon output cells.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the display-surface JSON should be written.",
    )
    parser.add_argument(
        "--grid-size-deg",
        type=float,
        default=DEFAULT_DISPLAY_GRID_SIZE_DEG,
        help="Display grid spacing in degrees (default: %(default)s).",
    )
    return parser.parse_args(argv)


def _generated_at_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _build_display_grid(polygon_path: Path, grid_size_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if grid_size_deg <= 0:
        raise ValueError("Grid size must be positive.")
    polygon = _load_polygon_from_kml(str(polygon_path))
    min_lon, min_lat, max_lon, max_lat = polygon.bounds
    lon_values = np.arange(min_lon, max_lon + grid_size_deg, grid_size_deg)
    lat_values = np.arange(min_lat, max_lat + grid_size_deg, grid_size_deg)
    lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)
    polygon_mask = contains_xy(polygon, lon_grid, lat_grid)
    return lon_grid, lat_grid, polygon_mask


def _compute_sigma_pixels(grid_size_deg: float, polygon_path: Path, target_smoothing_meters: float = TARGET_SMOOTHING_METERS) -> float:
    polygon = _load_polygon_from_kml(str(polygon_path))
    min_lon, min_lat, max_lon, max_lat = polygon.bounds
    mean_lat = (min_lat + max_lat) / 2.0
    meters_per_degree_lat = 111_320.0
    meters_per_degree_lon = meters_per_degree_lat * math.cos(math.radians(mean_lat))
    average_meters_per_degree = math.sqrt(meters_per_degree_lat * meters_per_degree_lon)
    return target_smoothing_meters / (grid_size_deg * average_meters_per_degree)


def _idw_interpolate(
    obs_lon: np.ndarray,
    obs_lat: np.ndarray,
    obs_values: np.ndarray,
    target_lon: np.ndarray,
    target_lat: np.ndarray,
    power: float = 2.0,
) -> np.ndarray:
    obs_points = np.column_stack([obs_lon, obs_lat])
    targets = np.column_stack([target_lon, target_lat])
    out = np.empty(targets.shape[0], dtype=float)
    for idx, point in enumerate(targets):
        diff = obs_points - point
        distance = np.sqrt(np.sum(diff * diff, axis=1))
        zero_distance = distance <= 1e-12
        if np.any(zero_distance):
            out[idx] = float(obs_values[zero_distance][0])
            continue
        weights = 1.0 / np.power(distance, power)
        out[idx] = float(np.sum(weights * obs_values) / np.sum(weights))
    return out


def _gaussian_smooth_masked(surface: np.ndarray, polygon_mask: np.ndarray, sigma: float) -> np.ndarray:
    values = np.where(polygon_mask & np.isfinite(surface), surface, 0.0)
    weights = (polygon_mask & np.isfinite(surface)).astype(float)
    smooth_values = gaussian_filter(values, sigma=sigma, mode="nearest")
    smooth_weights = gaussian_filter(weights, sigma=sigma, mode="nearest")
    out = np.full(surface.shape, np.nan, dtype=float)
    valid = polygon_mask & (smooth_weights > 1e-12)
    out[valid] = smooth_values[valid] / smooth_weights[valid]
    return out


def build_display_surface_frame(
    source_dir: Path,
    polygon_path: Path,
    grid_size_deg: float = DEFAULT_DISPLAY_GRID_SIZE_DEG,
) -> tuple[Any, ...]:
    observations, start_date, end_date, included_dates, source_file_count = load_weekly_masked_observations(
        source_dir=source_dir,
        polygon_path=polygon_path,
    )
    frame = observations.copy()
    frame["longitude"] = frame["longitude"].astype(float)
    frame["latitude"] = frame["latitude"].astype(float)
    frame["chlorophyll"] = frame["chlorophyll"].astype(float)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["longitude", "latitude", "chlorophyll"])
    if frame.empty:
        raise ValueError("No valid weekly chlorophyll observations available for display surface generation.")

    lon_grid, lat_grid, polygon_mask = _build_display_grid(polygon_path, grid_size_deg)
    smoothing_sigma_pixels = _compute_sigma_pixels(grid_size_deg, polygon_path)
    target_lon = lon_grid[polygon_mask]
    target_lat = lat_grid[polygon_mask]
    interpolated = _idw_interpolate(
        frame["longitude"].to_numpy(),
        frame["latitude"].to_numpy(),
        frame["chlorophyll"].to_numpy(),
        target_lon,
        target_lat,
    )
    idw_surface = np.full(lon_grid.shape, np.nan, dtype=float)
    idw_surface[polygon_mask] = interpolated
    smoothed = _gaussian_smooth_masked(idw_surface, polygon_mask, sigma=smoothing_sigma_pixels)
    smoothed = np.where(np.isfinite(smoothed), np.maximum(smoothed, 0.0), np.nan)

    cells = [
        {
            "lat": round(float(lat), LAT_LON_PRECISION),
            "lon": round(float(lon), LAT_LON_PRECISION),
            "chlorophyll": round(float(chl), CHLOROPHYLL_PRECISION),
        }
        for lat, lon, chl in zip(lat_grid[polygon_mask], lon_grid[polygon_mask], smoothed[polygon_mask], strict=False)
        if np.isfinite(chl)
    ]

    return cells, start_date, end_date, included_dates, source_file_count, smoothing_sigma_pixels


def build_display_surface_payload(
    source_dir: Path,
    polygon_path: Path,
    grid_size_deg: float = DEFAULT_DISPLAY_GRID_SIZE_DEG,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    cells, start_date, end_date, included_dates, source_file_count, smoothing_sigma_pixels = build_display_surface_frame(
        source_dir=source_dir,
        polygon_path=polygon_path,
        grid_size_deg=grid_size_deg,
    )
    return {
        "generated_at_utc": generated_at_utc or _generated_at_utc(),
        "start_timestamp": f"{start_date.isoformat()}T00:00:00",
        "end_timestamp": f"{end_date.isoformat()}T23:59:59",
        "included_dates": included_dates,
        "source_file_count": source_file_count,
        "interpolation_method": INTERPOLATION_METHOD,
        "smoothing_method": SMOOTHING_METHOD,
        "smoothing_sigma_pixels": smoothing_sigma_pixels,
        "target_smoothing_meters": TARGET_SMOOTHING_METERS,
        "coordinate_precision_decimal_places": LAT_LON_PRECISION,
        "chlorophyll_precision_decimal_places": CHLOROPHYLL_PRECISION,
        "grid_resolution_degrees": grid_size_deg,
        "polygon_file": str(polygon_path.resolve()),
        "surface_description": "Smoothed display surface for lagoon map rendering derived from weekly chlorophyll observations; not higher-resolution measured data.",
        "cells": cells,
    }


def run(
    source_dir: Path,
    polygon_path: Path,
    output_dir: Path,
    grid_size_deg: float = DEFAULT_DISPLAY_GRID_SIZE_DEG,
) -> Path:
    payload = build_display_surface_payload(
        source_dir=source_dir,
        polygon_path=polygon_path,
        grid_size_deg=grid_size_deg,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_FILENAME
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    LOGGER.info("Wrote weekly chlorophyll display surface to %s", output_path)
    return output_path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    args = parse_args(argv)
    try:
        run(
            source_dir=Path(args.source_dir),
            polygon_path=Path(args.polygon),
            output_dir=Path(args.output_dir),
            grid_size_deg=args.grid_size_deg,
        )
    except Exception as exc:
        LOGGER.error("Weekly chlorophyll display surface pipeline failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
