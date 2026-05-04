#!/usr/bin/env python3
"""Render a local preview image for weekly_chlorophyll_map.json."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

# Bootstrap imports when executed as `python src/pipeline/weekly_chlorophyll_map_preview.py`.
_here = Path(__file__).resolve()
for parent in [_here.parent] + list(_here.parents):
    if (parent / "src").exists() and (parent / "utils").exists():
        sys.path.insert(0, str(parent))
        sys.path.insert(0, str(parent / "src"))
        break

from src.pipeline.chl_daily_coverage import _load_polygon_from_kml

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
LOGGER = logging.getLogger("weekly_chlorophyll_map_preview")
DEFAULT_POLYGON_FILE = Path("src/pipeline/polygon/LSJmasking.kml")
DEFAULT_OUTPUT_NAME = "weekly_chlorophyll_map_preview.png"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to weekly_chlorophyll_map.json.")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_NAME,
        help="Path to the preview PNG to write (default: %(default)s).",
    )
    parser.add_argument(
        "--polygon",
        default=str(DEFAULT_POLYGON_FILE),
        help="Optional KML polygon to overlay (default: %(default)s).",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=55.0,
        help="Marker size for grid cells in the preview (default: %(default)s).",
    )
    return parser.parse_args(argv)


def load_weekly_map_cells(input_path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    cells = payload.get("cells", [])
    frame = pd.DataFrame(cells, columns=["lat", "lon", "chlorophyll"])
    if not frame.empty:
        frame["lat"] = pd.to_numeric(frame["lat"], errors="coerce")
        frame["lon"] = pd.to_numeric(frame["lon"], errors="coerce")
        frame["chlorophyll"] = pd.to_numeric(frame["chlorophyll"], errors="coerce")
        frame = frame.dropna(subset=["lat", "lon", "chlorophyll"]).sort_values(["lat", "lon"])
    return payload, frame


def _plot_polygon_outline(ax: plt.Axes, polygon_path: Path) -> None:
    polygon = _load_polygon_from_kml(str(polygon_path))
    geoms = getattr(polygon, "geoms", [polygon])
    for geom in geoms:
        x_coords, y_coords = geom.exterior.xy
        ax.plot(x_coords, y_coords, color="black", linewidth=1.2, alpha=0.8, zorder=2)


def render_preview(
    input_path: Path,
    output_path: Path,
    polygon_path: Path | None = None,
    point_size: float = 55.0,
) -> Path:
    payload, frame = load_weekly_map_cells(input_path)
    if frame.empty:
        raise ValueError(f"No valid cells found in {input_path}")

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    scatter = ax.scatter(
        frame["lon"],
        frame["lat"],
        c=frame["chlorophyll"],
        cmap="viridis",
        s=point_size,
        edgecolors="none",
        zorder=3,
    )
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Chlorophyll (mg/m^3)")

    if polygon_path is not None and polygon_path.exists():
        _plot_polygon_outline(ax, polygon_path)

    mean_lat = float(frame["lat"].mean())
    aspect = 1.0 / max(math.cos(math.radians(mean_lat)), 1e-6)
    ax.set_aspect(aspect)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        "Weekly Chlorophyll Map Preview\n"
        f"{payload.get('start_timestamp', '?')} to {payload.get('end_timestamp', '?')}"
    )
    ax.grid(True, alpha=0.2, linewidth=0.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    LOGGER.info("Wrote weekly chlorophyll preview to %s", output_path)
    return output_path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    args = parse_args(argv)
    try:
        render_preview(
            input_path=Path(args.input),
            output_path=Path(args.output),
            polygon_path=Path(args.polygon) if args.polygon else None,
            point_size=args.point_size,
        )
    except Exception as exc:
        LOGGER.error("Weekly chlorophyll preview failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
